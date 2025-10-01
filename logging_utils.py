"""Utilities for structured JSON logging with safe contextual payloads."""
from __future__ import annotations

import json
import logging
import os
import re
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional

from settings import LOG_JSON, LOG_LEVEL, MAX_IN_LOG_BODY

_SECRET_ENV_KEYS = {
    "DATABASE_URL",
    "REDIS_URL",
    "TELEGRAM_TOKEN",
    "OPENAI_API_KEY",
    "KIE_API_KEY",
    "SUNO_API_TOKEN",
    "SUNO_CALLBACK_SECRET",
}
for key, value in os.environ.items():
    upper = key.upper()
    if upper.endswith(("_TOKEN", "_KEY", "_SECRET")) or upper in _SECRET_ENV_KEYS:
        if value:
            _SECRET_ENV_KEYS.add(upper)

_SECRET_VALUES_LOCK = threading.Lock()
_SECRET_VALUES = {
    value
    for name, value in os.environ.items()
    if value and (name.upper().endswith(("_TOKEN", "_KEY", "_SECRET")) or name.upper() in _SECRET_ENV_KEYS)
}


def refresh_secret_cache() -> None:
    """Reload the cached secret values from the environment."""

    with _SECRET_VALUES_LOCK:
        _SECRET_VALUES.clear()
        for name, value in os.environ.items():
            if not value:
                continue
            upper = name.upper()
            if upper.endswith(("_TOKEN", "_KEY", "_SECRET")) or upper in {"DATABASE_URL", "REDIS_URL"}:
                _SECRET_VALUES.add(value)


def _truncate(value: str) -> str:
    if len(value) <= MAX_IN_LOG_BODY:
        return value
    return value[:MAX_IN_LOG_BODY] + "…(truncated)"


_TOKEN_QUERY_RE = re.compile(r"(token=)([^&\s]+)", re.IGNORECASE)


def _redact_text(value: str) -> str:
    if not value:
        return value
    with _SECRET_VALUES_LOCK:
        secrets = list(_SECRET_VALUES)
    for secret in secrets:
        if secret and secret in value:
            value = value.replace(secret, "***")
    value = _TOKEN_QUERY_RE.sub(r"\1***", value)
    return value


def _sanitize(value: Any) -> Any:
    if isinstance(value, str):
        return _truncate(_redact_text(value))
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
        return _truncate(_redact_text(text))
    if isinstance(value, Mapping):
        return {str(key): _sanitize(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize(item) for item in value]
    return value


RESERVED_LOG_KEYS = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "message",
    "asctime",
}


class JsonFormatter(logging.Formatter):
    """Format log records into JSON with structured metadata."""

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        message = _truncate(_redact_text(message))
        meta: dict[str, Any] = {}
        ctx = getattr(record, "ctx", None)
        if isinstance(ctx, Mapping):
            meta.update(_sanitize(dict(ctx)))
        elif ctx is not None:
            meta["ctx"] = _sanitize(ctx)

        meta.setdefault("logger", record.name)
        meta.setdefault("module", record.module)
        meta.setdefault("pid", os.getpid())

        if record.exc_info:
            try:
                exc_text = self.formatException(record.exc_info)
            except Exception:  # pragma: no cover - defensive
                exc_text = "exception"
            meta["exc_info"] = _truncate(_redact_text(exc_text))

        data = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "msg": message,
            "meta": meta,
        }
        return json.dumps(data, ensure_ascii=False)


_CONFIGURED = False
_CONFIG_LOCK = threading.Lock()
_RECORD_FACTORY_INSTALLED = False


def log_environment(logger: logging.Logger, *, redact: bool = True) -> None:
    """Log the current environment variables in a safe way.

    Secrets are always redacted regardless of the ``redact`` flag. When
    ``redact`` is true, additional scrubbing (token patterns, truncation) is
    applied to the values before logging.
    """

    safe_env: dict[str, str] = {}
    for name in sorted(os.environ):
        value = os.environ.get(name)
        if value is None:
            continue
        upper = name.upper()
        if upper.endswith(("_TOKEN", "_KEY", "_SECRET")) or upper in _SECRET_ENV_KEYS:
            safe_env[name] = "***"
            continue
        safe_env[name] = _redact_text(value) if redact else value

    logger.info("environment", extra=build_log_extra(meta={"env": safe_env}))


def _normalize_extra(extra: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    if not extra:
        return {"ctx": {}}

    ctx_payload: dict[str, Any] = {}
    if "ctx" in extra and isinstance(extra["ctx"], Mapping):
        ctx_payload.update(extra["ctx"])

    for key, value in extra.items():
        if key == "ctx":
            continue
        if key in RESERVED_LOG_KEYS:
            safe_key = f"field_{key}"
        else:
            safe_key = key
        if isinstance(value, Mapping):
            ctx_payload[safe_key] = dict(value)
        else:
            ctx_payload[safe_key] = value

    normalized: dict[str, Any] = {"ctx": ctx_payload}
    meta_value = ctx_payload.get("meta")
    if meta_value is not None:
        normalized["meta"] = meta_value
    return normalized


class SafeLogger(logging.Logger):
    """Custom logger that ensures ``extra`` payloads are safely wrapped."""

    def _log(
        self,
        level: int,
        msg: Any,
        args: Any,
        exc_info=None,
        extra: Optional[Mapping[str, Any]] = None,
        stack_info: bool = False,
        stacklevel: int = 1,
    ) -> None:  # type: ignore[override]
        normalized_extra: Optional[dict[str, Any]]
        if isinstance(extra, Mapping):
            if "ctx" in extra and len(extra) == 1 and isinstance(extra["ctx"], Mapping):
                normalized_extra = {"ctx": dict(extra["ctx"])}
            else:
                normalized_extra = _normalize_extra(extra)
        elif extra is not None:
            normalized_extra = _normalize_extra({"payload": extra})
        else:
            normalized_extra = None

        super()._log(
            level,
            msg,
            args,
            exc_info=exc_info,
            extra=normalized_extra,
            stack_info=stack_info,
            stacklevel=stacklevel,
        )


class CtxLogger(logging.LoggerAdapter):
    """Logger adapter that wraps ``extra`` payloads under ``ctx`` key."""

    def process(
        self,
        msg: str,
        kwargs: Dict[str, Any],
    ) -> tuple[str, Dict[str, Any]]:  # type: ignore[override]
        combined: dict[str, Any] = {}

        base_extra = getattr(self, "extra", None)
        if isinstance(base_extra, Mapping):
            combined.update(base_extra)

        call_extra = kwargs.pop("extra", None)
        if isinstance(call_extra, Mapping):
            combined.update(call_extra)
        elif call_extra is not None:
            combined["payload"] = call_extra

        normalized = _normalize_extra(combined)
        new_kwargs = dict(kwargs)
        new_kwargs["extra"] = normalized
        return msg, new_kwargs


SafeLoggerAdapter = CtxLogger


if logging.getLoggerClass() is not SafeLogger:
    logging.setLoggerClass(SafeLogger)


def build_log_extra(
    update: Optional[Any] = None,
    context: Optional[Any] = None,
    *,
    command: Optional[str] = None,
    meta: Optional[Mapping[str, Any]] = None,
    **fields: Any,
) -> dict[str, Any]:
    """Construct a safe ``extra`` payload for logging.

    The resulting mapping always contains a single top-level key ``ctx`` and never
    attempts to overwrite reserved :class:`logging.LogRecord` attributes.
    """

    ctx_payload: dict[str, Any] = {}

    env_name = os.getenv("ENV", os.getenv("ENVIRONMENT", "prod"))
    if env_name:
        ctx_payload.setdefault("env", env_name)

    if update is not None:
        user = getattr(update, "effective_user", None)
        chat = getattr(update, "effective_chat", None)
        if user is not None:
            ctx_payload["user_id"] = getattr(user, "id", None)
        if chat is not None:
            ctx_payload["chat_id"] = getattr(chat, "id", None)

        ctx_payload["update_type"] = type(update).__name__

        if command is None:
            message = getattr(update, "effective_message", None)
            text = getattr(message, "text", None)
            if isinstance(text, str) and text.startswith("/"):
                command = text.split()[0]

    if command:
        ctx_payload["command"] = command.lstrip("/")

    if context is not None:
        try:
            current_state = getattr(context, "chat_data", {}).get("state")  # type: ignore[assignment]
        except Exception:
            current_state = None
        if current_state is not None:
            ctx_payload.setdefault("state", current_state)

    if meta is not None:
        if isinstance(meta, Mapping):
            ctx_payload["meta"] = dict(meta)
        else:
            ctx_payload["meta"] = meta

    for key, value in fields.items():
        if key in RESERVED_LOG_KEYS or key == "ctx":
            safe_key = f"field_{key}"
        else:
            safe_key = key
        ctx_payload[safe_key] = value

    return {"ctx": ctx_payload}


def _install_record_factory_defaults() -> None:
    global _RECORD_FACTORY_INSTALLED
    if _RECORD_FACTORY_INSTALLED:
        return

    _RECORD_FACTORY_INSTALLED = True
    # No-op placeholder kept for backwards compatibility. Custom factories
    # that inject reserved attributes would lead to ``KeyError`` when callers
    # provide extras with the same names, so we intentionally avoid modifying
    # the default factory here.


def get_logger(name: str = "veo3-bot", *, extra: Optional[Mapping[str, Any]] = None) -> CtxLogger:
    """Return a :class:`CtxLogger` for the requested logger name."""

    base_extra = dict(extra or {})
    return CtxLogger(logging.getLogger(name), base_extra)


def configure_logging(app_name: str) -> None:
    """Configure root logging to emit JSON logs with secret redaction."""

    if not LOG_JSON:
        logging.basicConfig(
            level=getattr(logging, LOG_LEVEL, logging.INFO),
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s | ctx=%(ctx)s",
        )
        _install_record_factory_defaults()
        return

    global _CONFIGURED
    if _CONFIGURED:
        return

    with _CONFIG_LOCK:
        if _CONFIGURED:
            return
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        root = logging.getLogger()
        root.handlers.clear()
        root.addHandler(handler)
        root.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
        logging.captureWarnings(True)
        for noisy in ("httpx", "urllib3", "aiogram", "telegram", "uvicorn", "gunicorn", "pydantic"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
        _CONFIGURED = True
        _install_record_factory_defaults()


__all__ = [
    "JsonFormatter",
    "CtxLogger",
    "SafeLoggerAdapter",
    "build_log_extra",
    "configure_logging",
    "get_logger",
    "log_environment",
    "refresh_secret_cache",
]
