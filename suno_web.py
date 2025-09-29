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
from typing import Any, Dict, Iterable, Mapping, Optional
from urllib.parse import urlparse

import requests
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from requests.adapters import HTTPAdapter
from urllib3.util import Timeout

from logging_utils import configure_logging, log_environment
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
    SUNO_API_TOKEN,
    SUNO_CALLBACK_SECRET,
    SUNO_CALLBACK_URL,
    SUNO_ENABLED,
    SUNO_GEN_PATH,
    SUNO_TASK_STATUS_PATH,
    SUNO_INSTR_PATH,
    SUNO_VOCAL_PATH,
    SUNO_READY,
    TMP_CLEANUP_HOURS,
    KIE_BASE_URL,
    resolve_outbound_ip,
    token_tail,
)
from suno.schemas import CallbackEnvelope, SunoTask
from suno.service import SunoService
from suno.tempfiles import cleanup_old_directories, task_directory

configure_logging("suno-web")
log = logging.getLogger("suno-web")
log_environment(log)

app = FastAPI(title="Suno Callback Web", docs_url=None, redoc_url=None)
service = SunoService()

_SUNO_AVAILABLE = bool(SUNO_READY)
_OUTBOUND_IP: Optional[str] = None

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
_EXPECTED_RENDER_BASE = (os.getenv("RENDER_EXTERNAL_URL") or "https://shubinfilms-best-veo3-bot.onrender.com").rstrip("/")
_EXPECTED_CALLBACK_URL = f"{_EXPECTED_RENDER_BASE}/suno-callback"


def _mask_tokens(text: str) -> str:
    secrets = [
        SUNO_CALLBACK_SECRET or "",
        os.getenv("SUNO_API_TOKEN") or "",
        os.getenv("TELEGRAM_TOKEN") or "",
    ]
    cleaned = text
    for token in secrets:
        trimmed = token.strip()
        if trimmed:
            cleaned = cleaned.replace(trimmed, "***")
    return cleaned


def _json_preview(payload: Any, *, limit: int = 700) -> str:
    try:
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        text = str(payload)
    text = _mask_tokens(text)
    if len(text) > limit:
        return f"{text[:limit]}…"
    return text


def _normalize_callback_payload(raw: Any) -> tuple[dict[str, Any], Dict[str, Any], list[str]]:
    if not isinstance(raw, Mapping):
        return {"code": None, "msg": None, "data": {}}, {}, []
    layers: list[Dict[str, Any]] = []
    current: Mapping[str, Any] = raw
    while True:
        materialized = dict(current)
        layers.append(materialized)
        next_layer: Optional[Mapping[str, Any]] = None
        for key in ("payload", "data"):
            candidate = materialized.get(key)
            if isinstance(candidate, Mapping):
                next_layer = candidate
                break
        if next_layer is None:
            break
        current = next_layer
    innermost = layers[-1]
    base_data_candidate = innermost.get("data") if isinstance(innermost.get("data"), Mapping) else None
    if isinstance(base_data_candidate, Mapping):
        base_data = dict(base_data_candidate)
    else:
        base_data = dict(innermost)
    code_value: Any = None
    msg_value: Any = None
    for layer in layers:
        if code_value is None and layer.get("code") not in (None, ""):
            code_value = layer.get("code")
        message_candidate = layer.get("msg") or layer.get("message")
        if msg_value is None and message_candidate not in (None, ""):
            msg_value = message_candidate
        for key, value in layer.items():
            if key in {"payload", "data", "code", "msg", "message"}:
                continue
            base_data.setdefault(key, value)
    if "results" in base_data and "items" not in base_data:
        base_data["items"] = base_data["results"]
    if "type" in base_data and "callbackType" not in base_data:
        base_data.setdefault("callbackType", base_data["type"])
    if "status" in base_data and "callbackType" not in base_data:
        base_data.setdefault("callbackType", base_data["status"])
    all_keys = sorted({key for layer in layers for key in layer.keys()})
    return {"code": code_value, "msg": msg_value, "data": base_data}, dict(layers[0]), all_keys


def _callback_status(data: Mapping[str, Any]) -> Optional[str]:
    for key in ("status", "callbackType", "callback_type", "type"):
        value = data.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def _first_identifier(data: Mapping[str, Any], keys: Iterable[str]) -> Optional[str]:
    for key in keys:
        value = data.get(key)
        if value not in (None, ""):
            text = str(value).strip()
            if text:
                return text
    return None


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
    global _OUTBOUND_IP
    _OUTBOUND_IP = resolve_outbound_ip()
    token_suffix = token_tail(SUNO_API_TOKEN or "")
    tail_display = f"****{token_suffix}" if token_suffix else "none"
    log.info(
        "ENV Suno: base=%s, gen=%s, token_tail=%s",
        SUNO_API_BASE or KIE_BASE_URL,
        SUNO_GEN_PATH,
        tail_display,
    )
    summary = {
        "suno_enabled": SUNO_ENABLED,
        "api_base": SUNO_API_BASE,
        "callback_configured": bool(SUNO_CALLBACK_URL),
        "tmp_cleanup_hours": TMP_CLEANUP_HOURS,
    }
    log.info("configuration summary", extra={"meta": summary})
    if SUNO_ENABLED and SUNO_CALLBACK_URL and SUNO_CALLBACK_URL.rstrip("/") != _EXPECTED_CALLBACK_URL:
        log.warning(
            "suno callback url mismatch",
            extra={
                "meta": {
                    "configured": SUNO_CALLBACK_URL,
                    "expected": _EXPECTED_CALLBACK_URL,
                }
            },
        )


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
def healthz() -> JSONResponse:
    payload = {"ok": True, "suno_enabled": _SUNO_AVAILABLE, "base": KIE_BASE_URL}
    return JSONResponse(payload)


@app.get("/callbackz")
def callbackz() -> dict[str, str | bool]:
    return {"ok": True, "endpoint": "/suno-callback"}


@app.get("/debug/env")
def debug_env(token: Optional[str] = None) -> JSONResponse:
    if not SUNO_CALLBACK_SECRET:
        raise HTTPException(status_code=503, detail="callback secret not configured")
    if token != SUNO_CALLBACK_SECRET:
        raise HTTPException(status_code=403, detail="forbidden")
    tail = token_tail(SUNO_API_TOKEN or "")
    payload = {
        "KIE_BASE_URL": KIE_BASE_URL,
        "SUNO_GEN_PATH": SUNO_GEN_PATH,
        "SUNO_TASK_STATUS_PATH": SUNO_TASK_STATUS_PATH,
        "SUNO_INSTR_PATH": SUNO_INSTR_PATH,
        "SUNO_VOCAL_PATH": SUNO_VOCAL_PATH,
        "SUNO_API_TOKEN_TAIL": f"****{tail}" if tail else "",
        "OUTBOUND_IP": _OUTBOUND_IP,
    }
    return JSONResponse(payload)


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


@app.get("/suno-callback")
async def suno_callback_get() -> Response:
    return Response(status_code=405)


@app.post("/suno-callback")
async def suno_callback(
    request: Request,
    x_callback_secret: Optional[str] = Header(default=None, alias="X-Callback-Secret"),
):
    expected_secret = (SUNO_CALLBACK_SECRET or "").strip()
    provided = x_callback_secret or request.headers.get("X-Callback-Token")
    if expected_secret:
        if provided != expected_secret:
            log.warning(
                "forbidden callback",
                extra={"meta": {"provided": bool(provided), "has_secret": True}},
            )
            suno_callback_total.labels(status="forbidden", **_WEB_LABELS).inc()
            return Response(status_code=403)

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

    log.info(
        "Received SUNO callback",
        extra={
            "meta": {
                "phase": "callback",
                "content_length": len(body),
                "path": str(request.url.path),
            }
        },
    )

    normalized_payload, flat_payload, key_list = _normalize_callback_payload(payload)
    envelope = CallbackEnvelope.model_validate(normalized_payload)
    data_map: Mapping[str, Any] = envelope.data or {}
    task = SunoTask.from_envelope(envelope)
    items_count = len(task.items or [])
    callback_type = (task.callback_type or "").lower() or (_callback_status(data_map) or "unknown").lower()
    payload_req_id = _first_identifier(data_map, ("req_id", "requestId", "request_id"))
    if not payload_req_id:
        payload_req_id = _first_identifier(flat_payload, ("req_id", "requestId", "request_id"))
    payload_task_id = _first_identifier(data_map, ("task_id", "taskId"))
    if not payload_task_id:
        payload_task_id = _first_identifier(flat_payload, ("task_id", "taskId"))
    if payload_task_id and not task.task_id:
        task = task.model_copy(update={"task_id": payload_task_id})
    header_req_id = (
        request.headers.get("X-Request-ID")
        or request.headers.get("X-Req-Id")
        or request.headers.get("X-Req-ID")
    )
    initial_req_id = header_req_id or payload_req_id
    status_hint = callback_type or "unknown"
    preview = _json_preview(payload)
    if not initial_req_id or status_hint == "unknown":
        log.warning(
            "Suno callback missing identifiers",
            extra={
                "meta": {
                    "phase": "callback",
                    "status": status_hint,
                    "req_id": initial_req_id,
                    "preview": preview,
                }
            },
        )
    if not task.task_id and initial_req_id:
        mapped_task = service.get_task_id_by_request(initial_req_id)
        if mapped_task:
            task = task.model_copy(update={"task_id": mapped_task})
    req_id = initial_req_id or service.get_request_id(task.task_id)
    if not req_id and payload_req_id:
        req_id = payload_req_id
    if not task.task_id and req_id:
        mapped_task = service.get_task_id_by_request(req_id)
        if mapped_task:
            task = task.model_copy(update={"task_id": mapped_task})
    if not req_id and task.task_id:
        req_id = service.get_request_id(task.task_id)
    summary_meta = {
        "phase": "callback",
        "req_id": req_id or initial_req_id or payload_req_id,
        "task_id": task.task_id or payload_task_id,
        "status": status_hint,
        "items": items_count,
        "type": callback_type,
    }
    log.info("suno callback", extra={"meta": summary_meta})
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
            extra={
                "meta": {
                    "phase": "callback",
                    "key": key,
                    "task_id": task.task_id,
                    "req_id": req_id,
                }
            },
        )
        suno_callback_total.labels(status="skipped", **_WEB_LABELS).inc()
        return {"ok": True, "duplicate": True}

    process_status = "ok"
    try:
        _prepare_assets(task)
    except Exception as exc:  # pragma: no cover - defensive
        process_status = "error"
        log.exception(
            "suno callback asset prep failed",
            extra={"meta": {"task_id": task.task_id, "req_id": req_id, "err": str(exc)}},
        )
    try:
        service.handle_callback(task, req_id=req_id)
    except Exception as exc:  # pragma: no cover - defensive
        process_status = "error"
        log.exception(
            "suno callback handler failed",
            extra={"meta": {"task_id": task.task_id, "req_id": req_id, "err": str(exc)}},
        )
    else:
        meta = {
            "phase": "callback",
            "task_id": task.task_id,
            "req_id": req_id,
            "code": task.code,
            "status": status_hint,
            "items": items_count,
            "type": callback_type,
        }
        log.info("suno callback processed", extra={"meta": meta})
    suno_callback_total.labels(status=process_status, **_WEB_LABELS).inc()
    return {"ok": process_status == "ok"}


__all__ = ["app"]
