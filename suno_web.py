from __future__ import annotations

import json
import logging
import mimetypes
import os
import pathlib
import random
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse

# redis optional
try:
    import redis  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    redis = None  # type: ignore

try:
    from suno.downloader import download_file as suno_download_file
except Exception:  # pragma: no cover - optional helper
    suno_download_file = None  # type: ignore


logging.basicConfig(level=logging.INFO, format="%(levelname)s | suno-web | %(message)s")
log = logging.getLogger("suno-web")

app = FastAPI(title="Suno Callback Web")

REDIS_URL = os.getenv("REDIS_URL")
TG_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
ADMIN_IDS = [item.strip() for item in os.getenv("ADMIN_IDS", "").split(",") if item.strip()]
CALLBACK_SECRET = os.getenv("SUNO_CALLBACK_SECRET", "")
UA = os.getenv("SUNO_DOWNLOAD_UA") or "SunoCallbackBot/1.0 (+render)"
HEADERS = {"User-Agent": UA, "Accept": "*/*", "Connection": "close"}
DOWNLOAD_TRIES = int(os.getenv("SUNO_DOWNLOAD_TRIES", "3"))
DOWNLOAD_TIMEOUT = int(os.getenv("SUNO_DOWNLOAD_TIMEOUT", "60"))

rds = None
if REDIS_URL and redis is not None:
    try:
        rds = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        rds.ping()
        log.info("Redis connected")
    except Exception as exc:  # pragma: no cover - best effort
        log.warning(f"Redis unavailable: {exc}")


_seen: set[str] = set()


@app.get("/healthz")
def healthz() -> Dict[str, bool]:
    return {"ok": True}


@app.get("/")
def root() -> Dict[str, bool]:
    return {"ok": True}


@app.get("/suno-callback")
def ping_callback() -> Dict[str, Any]:
    return {"ok": True, "endpoint": "suno-callback"}


def _idem_key(task_id: str, cb_type: str) -> str:
    return f"suno:cb:{task_id}:{cb_type}"


def _idem_check_and_mark(key: str) -> bool:
    """Return True if key already processed."""

    try:
        if rds is not None:
            added = rds.setnx(key, "1")
            if not added:
                return True
            rds.expire(key, 86400)
            return False
    except Exception as exc:  # pragma: no cover - network/cache issue
        log.warning(f"Redis idempotency failed: {exc}")

    if key in _seen:
        return True
    _seen.add(key)
    return False


def _ensure_dir(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _guess_ext_from_headers(resp: requests.Response, url_path: str) -> str:
    ct = resp.headers.get("Content-Type", "").lower()
    if ct:
        ext = mimetypes.guess_extension(ct.split(";")[0].strip(), strict=False)
        if ext:
            return ext
        if "audio/mpeg" in ct:
            return ".mp3"
        if "audio/wav" in ct:
            return ".wav"
        if "image/jpeg" in ct:
            return ".jpg"
        if "image/png" in ct:
            return ".png"

    path_ext = pathlib.Path(url_path or "").suffix
    return path_ext if path_ext else ".bin"


def _download(url: str, destination: pathlib.Path) -> Optional[pathlib.Path]:
    if not url:
        return None

    parsed = urlparse(url)
    referer = ""
    if parsed.scheme and parsed.netloc:
        referer = f"{parsed.scheme}://{parsed.netloc}/"

    tries = max(1, DOWNLOAD_TRIES)
    delay = 1.0
    last_err: Optional[Exception] = None
    url_path = parsed.path
    dest_with_suffix = destination if destination.suffix else destination.with_suffix(".bin")

    for attempt in range(1, tries + 1):
        try:
            headers = HEADERS.copy()
            if referer:
                headers["Referer"] = referer

            with requests.get(
                url,
                headers=headers,
                stream=True,
                timeout=DOWNLOAD_TIMEOUT,
            ) as resp:
                if resp.status_code >= 400:
                    raise requests.HTTPError(f"HTTP {resp.status_code}")

                final_dest = dest_with_suffix
                if not destination.suffix:
                    guessed = _guess_ext_from_headers(resp, url_path)
                    final_dest = destination.with_suffix(guessed)

                _ensure_dir(final_dest.parent)
                with final_dest.open("wb") as fh:
                    for chunk in resp.iter_content(8192):
                        if chunk:
                            fh.write(chunk)
                return final_dest

        except Exception as exc:
            last_err = exc
            log.warning(
                "Download attempt %s/%s failed for %s: %s",
                attempt,
                tries,
                url,
                exc,
            )
            if attempt < tries:
                time.sleep(delay + random.uniform(0, 0.25))
                delay *= 1.5

    log.warning("Download failed permanently for %s: %s", url, last_err)

    if suno_download_file is not None:
        try:
            fallback_dest = destination
            if not fallback_dest.suffix:
                fallback_dest = dest_with_suffix
            _ensure_dir(fallback_dest.parent)
            downloaded = suno_download_file(url, fallback_dest.name, base_dir=fallback_dest.parent)
            if downloaded:
                return pathlib.Path(downloaded)
        except Exception as exc:  # pragma: no cover - optional helper fallback
            log.warning(f"Download helper fallback failed {url}: {exc}")

    return None


def _tg_send_message(text: str) -> List[str]:
    statuses: List[str] = []
    if not TG_TOKEN or not ADMIN_IDS:
        log.warning("No TELEGRAM_TOKEN/ADMIN_IDS; skip Telegram notify")
        return ["skipped"]

    for chat_id in ADMIN_IDS:
        try:
            response = requests.post(
                f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                json={"chat_id": chat_id, "text": text},
                timeout=20,
            )
            if response.ok:
                statuses.append(f"{chat_id}:ok")
            else:
                statuses.append(f"{chat_id}:http{response.status_code}")
        except Exception as exc:  # pragma: no cover - network issues
            statuses.append(f"{chat_id}:error")
            log.warning(f"sendMessage failed for {chat_id}: {exc}")
    return statuses


def _tg_send_file(filepath: pathlib.Path, caption: str = "") -> List[str]:
    statuses: List[str] = []
    if not TG_TOKEN or not ADMIN_IDS:
        return ["skipped"]

    suffix = filepath.suffix.lower()
    mime = "audio/mpeg" if suffix in {".mp3", ".wav"} else None
    method = "sendDocument"
    field_name = "document"
    if suffix in {".jpg", ".jpeg", ".png"}:
        method = "sendPhoto"
        field_name = "photo"

    for chat_id in ADMIN_IDS:
        try:
            with filepath.open("rb") as fh:
                files = {field_name: (filepath.name, fh, mime or "application/octet-stream")}
                data = {"chat_id": chat_id, "caption": caption}
                response = requests.post(
                    f"https://api.telegram.org/bot{TG_TOKEN}/{method}",
                    data=data,
                    files=files,
                    timeout=60,
                )
            if response.ok:
                statuses.append(f"{chat_id}:ok")
            else:
                statuses.append(f"{chat_id}:http{response.status_code}")
        except Exception as exc:  # pragma: no cover - network issues
            statuses.append(f"{chat_id}:error")
            log.warning(f"sendFile failed for {chat_id}: {exc}")
    return statuses
@app.post("/suno-callback")
async def suno_callback(
    request: Request,
    x_callback_token: Optional[str] = Header(default=None, convert_underscores=False),
):
    if CALLBACK_SECRET:
        if not x_callback_token or x_callback_token != CALLBACK_SECRET:
            log.warning("Forbidden: bad X-Callback-Token")
            return JSONResponse({"error": "forbidden"}, status_code=403)
    else:
        log.warning("SUNO_CALLBACK_SECRET is empty — allowing requests in DEV")

    try:
        payload = await request.json()
    except json.JSONDecodeError as exc:
        body = await request.body()
        log.error(f"Invalid JSON payload: {exc}; body={body!r}")
        return {"status": "received"}

    if not isinstance(payload, dict):
        log.error(f"Unexpected payload type: {type(payload)!r}")
        return {"status": "received"}

    code: Optional[int] = payload.get("code")
    msg: Optional[str] = payload.get("msg")
    data: Dict[str, Any] = payload.get("data") or {}
    cb_type_raw = data.get("callbackType") or data.get("callback_type") or ""
    cb_type = str(cb_type_raw).lower() or "unknown"
    task_id = data.get("task_id") or data.get("taskId") or "unknown"
    items: List[Dict[str, Any]] = data.get("data") or []

    idem_key = _idem_key(task_id, cb_type)
    if _idem_check_and_mark(idem_key):
        log.info(f"duplicate callback dropped | task={task_id} type={cb_type}")
        return {"status": "received"}

    downloaded_assets: List[str] = []
    tg_statuses: List[str] = []
    download_errors: List[str] = []

    log.info(
        f"callback received | code={code} task={task_id} type={cb_type} items={len(items)}"
    )

    tg_statuses.extend(_tg_send_message(f"Suno: этап {cb_type}, task={task_id}"))

    if cb_type in {"first", "complete"} and items:
        base_dir = pathlib.Path("/tmp/suno") / task_id
        for idx, track in enumerate(items, start=1):
            track_id = str(track.get("id") or idx)
            title = track.get("title") or "track"
            audio_url = track.get("audio_url") or track.get("audioUrl")
            image_url = track.get("image_url") or track.get("imageUrl")

            if audio_url:
                dest = base_dir / track_id
                downloaded = _download(audio_url, dest)
                if downloaded and downloaded.exists():
                    size = downloaded.stat().st_size
                    downloaded_assets.append(f"audio:{downloaded.name}:{size}")
                    tg_statuses.extend(
                        _tg_send_file(downloaded, caption=f"{title} (audio)")
                    )
                else:
                    log.warning(
                        f"audio download skipped | task={task_id} track={track_id} url={audio_url}"
                    )
                    downloaded_assets.append(f"audio-link:{audio_url}")
                    download_errors.append(f"audio:{track_id}:{audio_url}")
                    tg_statuses.extend(
                        _tg_send_message(f"Audio link (fallback): {audio_url}")
                    )

            if image_url:
                dest = base_dir / f"{track_id}_cover"
                downloaded = _download(image_url, dest)
                if downloaded and downloaded.exists():
                    size = downloaded.stat().st_size
                    downloaded_assets.append(f"image:{downloaded.name}:{size}")
                    tg_statuses.extend(
                        _tg_send_file(downloaded, caption=f"{title} (cover)")
                    )
                else:
                    log.warning(
                        f"image download skipped | task={task_id} track={track_id} url={image_url}"
                    )
                    downloaded_assets.append(f"image-link:{image_url}")
                    download_errors.append(f"image:{track_id}:{image_url}")
                    tg_statuses.extend(
                        _tg_send_message(f"Image link (fallback): {image_url}")
                    )

    if download_errors:
        err_preview = "\n".join(download_errors[:10])
        tg_statuses.extend(
            _tg_send_message(
                "Suno: проблемы со скачиванием файлов" + (f"\n{err_preview}" if err_preview else "")
            )
        )

    log_line = {
        "task_id": task_id,
        "callbackType": cb_type,
        "code": code,
        "assets": downloaded_assets,
        "errors": download_errors,
        "telegram": tg_statuses or ["not-sent"],
        "message": msg,
    }
    log.info(f"processed | {json.dumps(log_line, ensure_ascii=False)}")

    return {"status": "received"}
