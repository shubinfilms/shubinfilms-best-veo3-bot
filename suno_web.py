"""FastAPI application receiving callbacks from Suno."""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse

from redis_utils import rds
from settings import LOG_LEVEL, REDIS_PREFIX, SUNO_CALLBACK_SECRET
from suno.schemas import CallbackEnvelope, SunoTask
from suno.service import SunoService


logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(levelname)s | suno-web | %(message)s",
)
for noisy in ("httpx", "urllib3", "uvicorn", "gunicorn"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger("suno-web")


app = FastAPI(title="Suno Callback Web")
service = SunoService()

_CALLBACK_TTL = 24 * 60 * 60
_DOWNLOAD_TRIES = 3
_BACKOFF_SCHEDULE = (1, 3, 7)
_BASE_DIR = Path(os.getenv("SUNO_STORAGE", "/tmp/suno"))
_memory_idempotency: dict[str, float] = {}


@app.get("/")
def root() -> dict[str, bool]:
    return {"ok": True}


@app.get("/healthz")
def healthz() -> dict[str, bool]:
    return {"ok": True}


@app.get("/callbackz")
def callbackz() -> dict[str, str | bool]:
    return {"ok": True, "endpoint": "/suno-callback"}


def _idempotency_key(task: str, cb_type: str) -> str:
    task_part = task or "unknown"
    type_part = cb_type or "unknown"
    return f"{REDIS_PREFIX}:cb:{task_part}:{type_part}"


def _register_once(key: str) -> bool:
    if not key:
        return True
    if rds is not None:
        try:
            stored = rds.set(key, "1", nx=True, ex=_CALLBACK_TTL)
            if stored:
                return True
            return False
        except Exception as exc:  # pragma: no cover - Redis failure fallback
            log.warning("idempotency redis error | key=%s err=%s", key, exc)
    now = time.time()
    expires_at = now + _CALLBACK_TTL
    current = _memory_idempotency.get(key)
    if current and current > now:
        return False
    _memory_idempotency[key] = expires_at
    return True


def _ensure_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _apply_extension(base: Path, url: str) -> Path:
    parsed = urlparse(url)
    suffix = Path(parsed.path or "").suffix
    if suffix:
        return base.with_suffix(suffix)
    return base


def _download(url: str, dest: Path) -> str:
    if not url:
        return url
    for attempt in range(1, _DOWNLOAD_TRIES + 1):
        try:
            with requests.get(url, stream=True, timeout=(10, 30)) as resp:
                status = resp.status_code
                if status in {403, 408} or 500 <= status < 600:
                    if attempt == _DOWNLOAD_TRIES:
                        log.warning("asset download failed | code=%s url=%s", status, url)
                        return url
                    delay = _BACKOFF_SCHEDULE[min(attempt - 1, len(_BACKOFF_SCHEDULE) - 1)]
                    time.sleep(delay)
                    continue
                if status >= 400:
                    log.warning("asset download non-retryable | code=%s url=%s", status, url)
                    return url
                _ensure_directory(dest)
                with dest.open("wb") as fh:
                    for chunk in resp.iter_content(8192):
                        if chunk:
                            fh.write(chunk)
                return str(dest)
        except requests.RequestException as exc:
            if attempt == _DOWNLOAD_TRIES:
                log.warning("asset download exception | url=%s err=%s", url, exc)
                return url
            delay = _BACKOFF_SCHEDULE[min(attempt - 1, len(_BACKOFF_SCHEDULE) - 1)]
            time.sleep(delay)
    return url


def _prepare_assets(task: SunoTask) -> None:
    if not task.task_id:
        return
    base_dir = _BASE_DIR / task.task_id
    for index, track in enumerate(task.items, start=1):
        track_id = track.id or str(index)
        if track.audio_url:
            target = _apply_extension(base_dir / track_id, track.audio_url)
            track.audio_url = _download(track.audio_url, target)
        if track.image_url:
            target = _apply_extension(base_dir / f"{track_id}_cover", track.image_url)
            track.image_url = _download(track.image_url, target)


@app.post("/suno-callback")
async def suno_callback(
    request: Request,
    x_callback_token: Optional[str] = Header(default=None),
):
    provided = x_callback_token or request.query_params.get("token")
    if SUNO_CALLBACK_SECRET and provided != SUNO_CALLBACK_SECRET:
        log.warning("forbidden callback | provided=%s", provided)
        return JSONResponse({"error": "forbidden"}, status_code=403)

    try:
        payload = await request.json()
    except json.JSONDecodeError:
        body = await request.body()
        log.error("invalid json payload: %s", body[:200])
        return JSONResponse({"status": "ignored"}, status_code=400)

    envelope = CallbackEnvelope.model_validate(payload)
    task = SunoTask.from_envelope(envelope)
    key = _idempotency_key(task.task_id, task.callback_type)
    if not _register_once(key):
        log.info("duplicate callback ignored | key=%s", key)
        return {"ok": True, "duplicate": True}

    _prepare_assets(task)
    service.handle_callback(task)
    return {"ok": True}


__all__ = ["app"]
