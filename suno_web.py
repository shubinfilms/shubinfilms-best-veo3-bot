"""FastAPI application receiving callbacks from Suno."""
from __future__ import annotations

import json
import logging
import os
import signal
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from requests.adapters import HTTPAdapter
from urllib3.util import Timeout

from logging_utils import configure_logging
from metrics import (
    render_metrics,
    suno_callback_download_fail_total,
    suno_callback_total,
    suno_latency_seconds,
)
from redis_utils import rds
from settings import (
    HTTP_POOL_CONNECTIONS,
    HTTP_POOL_PER_HOST,
    HTTP_TIMEOUT_CONNECT,
    HTTP_TIMEOUT_READ,
    HTTP_TIMEOUT_TOTAL,
    REDIS_PREFIX,
    SUNO_API_BASE,
    SUNO_CALLBACK_SECRET,
    SUNO_CALLBACK_URL,
    SUNO_ENABLED,
    TMP_CLEANUP_HOURS,
)
from suno.schemas import CallbackEnvelope, SunoTask
from suno.service import SunoService
from suno.tempfiles import cleanup_old_directories, task_directory

configure_logging("suno-web")
log = logging.getLogger("suno-web")

app = FastAPI(title="Suno Callback Web", docs_url=None, redoc_url=None)
service = SunoService()

_MAX_JSON_BYTES = 512 * 1024
_ALLOWED_CORS_PATHS = {"/healthz", "/callbackz"}
_ACTIVE_LOCK = threading.Lock()
_ACTIVE_REQUESTS = 0
_SHUTDOWN = threading.Event()

_adapter = HTTPAdapter(pool_connections=HTTP_POOL_CONNECTIONS, pool_maxsize=HTTP_POOL_PER_HOST)
_session = requests.Session()
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)
_timeout = Timeout(connect=HTTP_TIMEOUT_CONNECT, read=HTTP_TIMEOUT_READ, total=HTTP_TIMEOUT_TOTAL)
_ENV = (os.getenv("APP_ENV") or "prod").strip() or "prod"
_WEB_LABELS = {"env": _ENV, "service": "web"}


def _active_count() -> int:
    with _ACTIVE_LOCK:
        return _ACTIVE_REQUESTS


def _handle_sigterm(signum: int, frame: Optional[object]) -> None:  # pragma: no cover - signal handling
    log.warning("sigterm received", extra={"meta": {"active_requests": _active_count()}})
    _SHUTDOWN.set()
    deadline = time.time() + 10.0
    while time.time() < deadline:
        if _active_count() == 0:
            break
        time.sleep(0.1)
    os._exit(0)


signal.signal(signal.SIGTERM, _handle_sigterm)


@app.on_event("startup")
async def _startup_event() -> None:
    cleanup_old_directories()
    summary = {
        "suno_enabled": SUNO_ENABLED,
        "api_base": SUNO_API_BASE,
        "callback_configured": bool(SUNO_CALLBACK_URL),
        "tmp_cleanup_hours": TMP_CLEANUP_HOURS,
    }
    log.info("configuration summary", extra={"meta": summary})
    if not SUNO_ENABLED:
        log.warning("suno disabled; callback endpoints not registered")


@app.middleware("http")
async def _middleware(request: Request, call_next):  # type: ignore[override]
    if request.method == "OPTIONS":
        if request.url.path in _ALLOWED_CORS_PATHS:
            response = Response(status_code=204)
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "GET"
            response.headers["Access-Control-Allow-Headers"] = "accept"
            return response
        return Response(status_code=405)

    if _SHUTDOWN.is_set():
        return JSONResponse({"status": "shutting_down"}, status_code=503)

    global _ACTIVE_REQUESTS
    with _ACTIVE_LOCK:
        _ACTIVE_REQUESTS += 1
    try:
        response = await call_next(request)
    finally:
        with _ACTIVE_LOCK:
            _ACTIVE_REQUESTS -= 1

    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    origin = request.headers.get("origin")
    if origin and request.method == "GET" and request.url.path in _ALLOWED_CORS_PATHS:
        response.headers["Access-Control-Allow-Origin"] = "*"
    return response


@app.get("/")
def root() -> dict[str, bool]:
    return {"ok": True}


@app.get("/healthz")
def healthz() -> dict[str, bool]:
    return {"ok": True}


@app.get("/callbackz")
def callbackz() -> dict[str, str | bool]:
    return {"ok": True, "endpoint": "/suno-callback"}


@app.get("/metrics", include_in_schema=False)
def metrics_endpoint() -> Response:
    payload = render_metrics()
    return Response(content=payload, media_type="text/plain; version=0.0.4; charset=utf-8")


def _idempotency_key(task: str, cb_type: str) -> str:
    task_part = task or "unknown"
    type_part = cb_type or "unknown"
    return f"{REDIS_PREFIX}:cb:{task_part}:{type_part}"


def _register_once(key: str) -> bool:
    if not key:
        return True
    if rds is not None:
        try:
            stored = rds.set(key, "1", nx=True, ex=24 * 60 * 60)
            if stored:
                return True
            return False
        except Exception as exc:  # pragma: no cover - Redis failure fallback
            log.warning("idempotency redis error", extra={"meta": {"key": key, "err": str(exc)}})
    now = time.time()
    expires_at = now + 24 * 60 * 60
    current = _memory_idempotency.get(key)
    if current and current > now:
        return False
    _memory_idempotency[key] = expires_at
    return True


_memory_idempotency: dict[str, float] = {}


def _ensure_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _apply_extension(base: Path, url: str) -> Path:
    parsed = urlparse(url)
    suffix = Path(parsed.path or "").suffix
    if suffix:
        return base.with_suffix(suffix)
    return base


def _record_download_failure(reason: str, url: str) -> None:
    suno_callback_download_fail_total.labels(reason=reason).inc()
    log.warning("asset download failed", extra={"meta": {"reason": reason, "url": url}})


def _parse_iso8601(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _download(url: str, dest: Path) -> str:
    if not url:
        return url
    for attempt in range(1, 4):
        try:
            with _session.get(url, stream=True, timeout=_timeout) as resp:
                status = resp.status_code
                if status in {403, 408} or 500 <= status < 600:
                    if attempt == 3:
                        _record_download_failure(f"status_{status}", url)
                        return url
                    time.sleep((1, 3, 7)[min(attempt - 1, 2)])
                    continue
                if status >= 400:
                    _record_download_failure(f"status_{status}", url)
                    return url
                _ensure_directory(dest)
                with dest.open("wb") as fh:
                    for chunk in resp.iter_content(8192):
                        if chunk:
                            fh.write(chunk)
                return str(dest)
        except requests.RequestException:
            if attempt == 3:
                _record_download_failure("network", url)
                return url
            time.sleep((1, 3, 7)[min(attempt - 1, 2)])
    return url


def _prepare_assets(task: SunoTask) -> None:
    if not task.task_id:
        return
    base_dir = task_directory(task.task_id)
    for index, track in enumerate(task.items, start=1):
        track_id = track.id or str(index)
        if track.audio_url:
            target = _apply_extension(base_dir / track_id, track.audio_url)
            track.audio_url = _download(track.audio_url, target)
        if track.image_url:
            target = _apply_extension(base_dir / f"{track_id}_cover", track.image_url)
            track.image_url = _download(track.image_url, target)


if SUNO_ENABLED:

    @app.post("/suno-callback")
    async def suno_callback(
        request: Request,
        x_callback_secret: Optional[str] = Header(default=None, alias="X-Callback-Secret"),
    ):
        provided = (
            x_callback_secret
            or request.headers.get("X-Callback-Token")
            or request.query_params.get("secret")
            or request.query_params.get("token")
        )
        if SUNO_CALLBACK_SECRET and provided != SUNO_CALLBACK_SECRET:
            log.warning("forbidden callback", extra={"meta": {"provided": bool(provided)}})
            suno_callback_total.labels(status="forbidden", **_WEB_LABELS).inc()
            return JSONResponse({"error": "forbidden"}, status_code=403)

        body = await request.body()
        if len(body) > _MAX_JSON_BYTES:
            log.warning("payload too large", extra={"meta": {"size": len(body)}})
            raise HTTPException(status_code=413, detail="payload too large")

        try:
            payload = json.loads(body or b"{}")
        except json.JSONDecodeError:
            log.error(
                "invalid json payload",
                extra={"meta": {"preview": body[:200].decode("utf-8", errors="replace")}},
            )
            return JSONResponse({"status": "ignored"}, status_code=400)

        envelope = CallbackEnvelope.model_validate(payload)
        task = SunoTask.from_envelope(envelope)
        header_req_id = (
            request.headers.get("X-Request-ID")
            or request.headers.get("X-Req-Id")
            or request.headers.get("X-Req-ID")
        )
        req_id = header_req_id or service.get_request_id(task.task_id)
        start_ts = service.get_start_timestamp(task.task_id)
        if start_ts:
            started_at = _parse_iso8601(start_ts)
            if started_at is not None:
                elapsed = max(0.0, (datetime.now(timezone.utc) - started_at).total_seconds())
                suno_latency_seconds.labels(**_WEB_LABELS).observe(elapsed)
        key = _idempotency_key(task.task_id, task.callback_type)
        if not _register_once(key):
            log.info(
                "duplicate callback ignored",
                extra={"meta": {"key": key, "task_id": task.task_id, "req_id": req_id}},
            )
            suno_callback_total.labels(status="skipped", **_WEB_LABELS).inc()
            return {"ok": True, "duplicate": True}

        _prepare_assets(task)
        service.handle_callback(task, req_id=req_id)
        status_name = (task.callback_type or "").lower()
        if status_name in {"complete", "success"}:
            log.info(
                "suno callback success",
                extra={"meta": {"task_id": task.task_id, "req_id": req_id, "code": task.code}},
            )
        else:
            log.warning(
                "suno callback processed",
                extra={
                    "meta": {
                        "task_id": task.task_id,
                        "req_id": req_id,
                        "code": task.code,
                        "type": status_name,
                    }
                },
            )
        suno_callback_total.labels(status="ok", **_WEB_LABELS).inc()
        return {"ok": True}

else:
    log.warning("suno callback endpoint disabled by configuration")


__all__ = ["app"]
