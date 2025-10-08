from __future__ import annotations

import inspect
import logging
import uuid
from typing import Any, Awaitable, Callable, Mapping, MutableMapping, Optional

from telegram.ext import ContextTypes

from helpers.errors import send_user_error

logger = logging.getLogger("veo.fast")

_RETRY_KEY = "veo_fast_retry_callbacks"


class VEOFastError(RuntimeError):
    """Base class for VEO fast errors."""


class VEOFastInvalidInput(VEOFastError):
    """Raised when required prompt or image is missing."""


class VEOFastTimeout(VEOFastError):
    """Raised when a VEO fast job timed out."""


class VEOFastHTTPError(VEOFastError):
    """Raised for HTTP errors returned by the backend."""

    def __init__(
        self,
        status_code: int,
        payload: Optional[Mapping[str, Any]] = None,
        *,
        req_id: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = int(status_code)
        self.payload = dict(payload or {})
        self.req_id = req_id
        self.reason = reason


class VEOFastBackendError(VEOFastError):
    """Raised when backend communication failed for other reasons."""

    def __init__(self, message: str, *, req_id: Optional[str] = None) -> None:
        super().__init__(message)
        self.reason = message
        self.req_id = req_id


RetryHandler = Callable[[], Awaitable[Any] | Any]


def _get_retry_storage(context: ContextTypes.DEFAULT_TYPE) -> MutableMapping[str, Any]:
    chat_data = getattr(context, "chat_data", None)
    if not isinstance(chat_data, MutableMapping):
        raise RuntimeError("chat_data is not available for retry storage")
    storage = chat_data.get(_RETRY_KEY)
    if not isinstance(storage, MutableMapping):
        storage = {}
        chat_data[_RETRY_KEY] = storage
    return storage


def _register_retry_handler(
    context: ContextTypes.DEFAULT_TYPE, handler: RetryHandler
) -> Optional[str]:
    if handler is None:
        return None
    try:
        storage = _get_retry_storage(context)
    except RuntimeError:
        logger.debug("veo.fast.retry_storage_unavailable")
        return None
    retry_id = f"veo_fast:retry:{uuid.uuid4().hex}"
    storage[retry_id] = handler
    return retry_id


def _is_policy_violation(payload: Mapping[str, Any]) -> bool:
    reason = str(payload.get("reason") or payload.get("code") or "").lower()
    if reason and any(token in reason for token in ("policy", "blocked", "safety")):
        return True
    message = str(payload.get("message") or payload.get("msg") or "").lower()
    if message and any(token in message for token in ("policy", "harm", "unsafe", "blocked")):
        return True
    error_text = str(payload.get("error") or payload.get("detail") or "").lower()
    return any(token in error_text for token in ("policy", "harm", "unsafe", "blocked"))


def _resolve_reason(payload: Mapping[str, Any], fallback: Optional[str]) -> Optional[str]:
    for key in ("message", "msg", "detail", "error", "reason"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return fallback


async def handle_veo_fast_error(
    context: ContextTypes.DEFAULT_TYPE,
    *,
    chat_id: Optional[int],
    user_id: Optional[int],
    error: Exception,
    retry_handler: Optional[RetryHandler] = None,
    mode: str = "veo_fast",
) -> None:
    """Map backend errors to user-facing messages and send them."""

    kind = "backend_fail"
    reason: Optional[str] = None
    req_id: Optional[str] = None
    retry_cb: Optional[str] = None

    if isinstance(error, VEOFastInvalidInput):
        kind = "invalid_input"
        retry_handler = None
        reason = str(error) or "invalid_input"
    elif isinstance(error, VEOFastTimeout):
        kind = "timeout"
        reason = str(error) or "timeout"
    elif isinstance(error, VEOFastHTTPError):
        req_id = error.req_id
        reason = error.reason or _resolve_reason(error.payload, None)
        status = int(error.status_code)
        if status in {400, 403, 429} and _is_policy_violation(error.payload):
            kind = "content_policy"
        elif status >= 500:
            kind = "backend_fail"
        else:
            kind = "backend_fail"
    elif isinstance(error, VEOFastBackendError):
        kind = "backend_fail"
        reason = error.reason
        req_id = error.req_id
    else:
        reason = getattr(error, "reason", None)

    if retry_handler is not None:
        retry_cb = _register_retry_handler(context, retry_handler)
    details = {
        "chat_id": chat_id,
        "user_id": user_id,
        "mode": mode,
        "reason": reason,
        "req_id": req_id,
    }
    await send_user_error(context, kind, details=details, retry_cb=retry_cb)


async def trigger_retry_callback(
    context: ContextTypes.DEFAULT_TYPE, retry_id: str
) -> bool:
    """Execute a previously registered retry handler."""

    try:
        storage = _get_retry_storage(context)
    except RuntimeError:
        return False
    handler = storage.pop(retry_id, None)
    if handler is None:
        return False
    result = handler()
    if inspect.isawaitable(result):
        await result
    return True


__all__ = [
    "VEOFastBackendError",
    "VEOFastError",
    "VEOFastHTTPError",
    "VEOFastInvalidInput",
    "VEOFastTimeout",
    "handle_veo_fast_error",
    "trigger_retry_callback",
]
