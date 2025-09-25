"""FastAPI application receiving callbacks from Suno."""
from __future__ import annotations

import json
import logging
import mimetypes
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse

from settings import LOG_LEVEL
from suno.schemas import SunoTask
from suno.service import SunoService

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(levelname)s | suno-web | %(message)s",
)
for noisy in ("httpx", "urllib3", "uvicorn", "gunicorn"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger("suno-web")

app = FastAPI(title="Suno Callback Web")

CALLBACK_SECRET = os.getenv("SUNO_CALLBACK_SECRET", "")
DOWNLOAD_TIMEOUT = float(os.getenv("SUNO_DOWNLOAD_TIMEOUT", "60"))
DOWNLOAD_TRIES = int(os.getenv("SUNO_DOWNLOAD_TRIES", "3"))
UA = os.getenv("SUNO_DOWNLOAD_UA") or "best-veo3-bot/1.0"
HEADERS = {"User-Agent": UA, "Accept": "*/*", "Connection": "close"}
BASE_DIR = Path("/tmp/suno")

service = SunoService()
_seen_callbacks: set[str] = set()


@app.get("/healthz")
def healthz() -> Dict[str, bool]:
    return {"ok": True}


@app.get("/")
def root() -> Dict[str, bool]:
    return {"ok": True}


def _idem_key(task_id: str, cb_type: str) -> str:
    return f"{task_id}:{cb_type}" if task_id else cb_type


def _should_skip(key: str) -> bool:
    if not key:
        return False
    if key in _seen_callbacks:
        return True
    _seen_callbacks.add(key)
    return False


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _guess_ext(resp: requests.Response, url_path: str, suggested: Optional[str]) -> str:
    if suggested:
        clean = suggested if suggested.startswith(".") else f".{suggested}"
        return clean
    ct = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
    if ct:
        ext = mimetypes.guess_extension(ct, strict=False)
        if ext:
            return ext
        if ct == "audio/mpeg":
            return ".mp3"
        if ct == "audio/wav":
            return ".wav"
    path_ext = Path(url_path or "").suffix
    return path_ext if path_ext else ".bin"


def _download(url: str, destination: Path, *, suggested_ext: Optional[str]) -> Tuple[Optional[Path], Optional[str]]:
    if not url:
        return None, None
    tries = max(1, DOWNLOAD_TRIES)
    delay = 1.0
    last_error: Optional[str] = None
    parsed_path = Path(url).name
    for attempt in range(1, tries + 1):
        try:
            with requests.get(url, headers=HEADERS, stream=True, timeout=DOWNLOAD_TIMEOUT) as resp:
                status = resp.status_code
                if status in {403, 404}:
                    return None, f"http{status}"
                if status >= 500:
                    last_error = f"http{status}"
                    raise requests.HTTPError(status)
                if status >= 400:
                    return None, f"http{status}"
                final_dest = destination
                if not destination.suffix:
                    ext = _guess_ext(resp, parsed_path, suggested_ext)
                    final_dest = destination.with_suffix(ext)
                _ensure_dir(final_dest.parent)
                with final_dest.open("wb") as fh:
                    for chunk in resp.iter_content(8192):
                        if chunk:
                            fh.write(chunk)
                return final_dest, None
        except Exception as exc:  # pragma: no cover - network errors
            last_error = last_error or str(exc)
            if attempt < tries:
                time.sleep(delay + random.uniform(0, 0.25))
                delay *= 1.5
    return None, last_error


def _process_tracks(task_id: str, task: SunoTask) -> Tuple[List[str], List[str]]:
    downloaded: List[str] = []
    errors: List[str] = []
    if not task_id:
        return downloaded, errors
    base_dir = BASE_DIR / task_id
    for idx, track in enumerate(task.items, start=1):
        track_id = track.id or str(idx)
        title = track.title or f"track-{idx}"
        if track.audio_url:
            dest = base_dir / track_id
            path, err = _download(track.audio_url, dest, suggested_ext=track.ext)
            if path:
                size = path.stat().st_size
                downloaded.append(f"audio:{path.name}:{size}")
            else:
                downloaded.append(f"audio-link:{track.audio_url}")
                if err:
                    errors.append(f"audio:{track_id}:{err}")
        if track.image_url:
            dest = base_dir / f"{track_id}_cover"
            path, err = _download(track.image_url, dest, suggested_ext=None)
            if path:
                size = path.stat().st_size
                downloaded.append(f"image:{path.name}:{size}")
            else:
                downloaded.append(f"image-link:{track.image_url}")
                if err:
                    errors.append(f"image:{track_id}:{err}")
    return downloaded, errors


@app.post("/suno-callback")
async def suno_callback(
    request: Request,
    x_callback_token: Optional[str] = Header(default=None, convert_underscores=False),
):
    if CALLBACK_SECRET:
        if not x_callback_token or x_callback_token != CALLBACK_SECRET:
            log.warning("Forbidden: bad X-Callback-Token")
            return JSONResponse({"error": "forbidden"}, status_code=403)
    try:
        payload = await request.json()
    except json.JSONDecodeError as exc:
        body = await request.body()
        log.error("Invalid JSON payload: %s body=%s", exc, body[:200])
        return {"status": "received"}
    if not isinstance(payload, dict):
        log.error("Unexpected payload type: %s", type(payload))
        return {"status": "received"}

    data = payload.get("data") or {}
    callback_type = str(data.get("callbackType") or data.get("callback_type") or "").lower() or "unknown"
    task_id = str(data.get("task_id") or data.get("taskId") or "").strip()
    code = payload.get("code")
    items = data.get("data") or []

    key = _idem_key(task_id, callback_type)
    if _should_skip(key):
        log.info("duplicate callback dropped | task=%s type=%s", task_id or "?", callback_type)
        return {"status": "received"}

    log.info(
        "callback received | code=%s task=%s type=%s items=%s",
        code,
        task_id or "?",
        callback_type,
        len(items) if isinstance(items, list) else 0,
    )

    task = SunoTask.from_payload(payload)
    if task_id and not task.task_id:
        task.task_id = task_id
    task.status = callback_type or task.status

    downloaded_assets, download_errors = _process_tracks(task.task_id, task)

    log_line = {
        "task_id": task.task_id,
        "callbackType": callback_type,
        "code": code,
        "assets": downloaded_assets,
        "errors": download_errors,
        "message": payload.get("msg"),
    }
    log.info("processed | %s", json.dumps(log_line, ensure_ascii=False))

    try:
        service.handle_callback(task)
    except Exception:  # pragma: no cover - defensive
        log.exception("SunoService.handle_callback failed for task %s", task.task_id)

    return {"status": "received"}
