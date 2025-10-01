"""Utilities for structured JSON logging with secret redaction."""
from __future__ import annotations

import json
import logging
import os
import re
import threading
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

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


class JsonFormatter(logging.Formatter):
    """Format log records into JSON with structured metadata."""

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        message = _truncate(_redact_text(message))
        meta: dict[str, Any] = {}
        extra_meta = getattr(record, "meta", None)
        if isinstance(extra_meta, Mapping):
            meta.update(_sanitize(dict(extra_meta)))
        elif extra_meta is not None:
            meta["extra"] = _sanitize(extra_meta)

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

    logger.info("environment", extra={"meta": {"env": safe_env}})


class CtxLogger(logging.LoggerAdapter):
    """Logger adapter that injects safe context fields into log records."""

    RESERVED = {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "lineno",
        "exc_info",
        "func",
        "sinfo",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "process",
        "processName",
        "stack_info",
    }

    CONTEXT_FIELDS = ("cmd", "user_id", "chat_id", "state")

    def process(
        self,
        msg: str,
        kwargs: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:  # type: ignore[override]
        """Sanitize ``extra`` payloads and enforce safe context keys."""

        combined: dict[str, Any] = {}
        if isinstance(self.extra, Mapping):
            combined.update(self.extra)

        call_extra = kwargs.get("extra")
        if isinstance(call_extra, Mapping):
            combined.update(call_extra)
        elif call_extra is not None:
            combined["meta"] = call_extra

        safe_extra: dict[str, Any] = {}
        for field in self.CONTEXT_FIELDS:
            safe_extra[field] = combined.get(field)

        meta_payload: dict[str, Any] = {}
        if "meta" in combined and isinstance(combined["meta"], Mapping):
            meta_payload.update(combined["meta"])
        elif "meta" in combined:
            meta_payload["meta"] = combined["meta"]

        for key, value in combined.items():
            if key in self.CONTEXT_FIELDS or key == "meta":
                continue
            safe_key = key
            if safe_key in self.RESERVED or safe_key in self.CONTEXT_FIELDS:
                safe_key = f"extra_{safe_key}"
            safe_extra[safe_key] = value

        if meta_payload:
            safe_extra["meta"] = meta_payload

        new_kwargs = dict(kwargs)
        new_kwargs["extra"] = safe_extra
        return msg, new_kwargs


SafeLoggerAdapter = CtxLogger


def _install_record_factory_defaults() -> None:
    global _RECORD_FACTORY_INSTALLED
    if _RECORD_FACTORY_INSTALLED:
        return

    _RECORD_FACTORY_INSTALLED = True
    factory = logging.getLogRecordFactory()

    def _factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = factory(*args, **kwargs)
        for field in CtxLogger.CONTEXT_FIELDS:
            if not hasattr(record, field):
                setattr(record, field, None)
        if not hasattr(record, "meta"):
            setattr(record, "meta", None)
        return record

    logging.setLogRecordFactory(_factory)


def get_logger(name: str = "veo3-bot", *, extra: Optional[Mapping[str, Any]] = None) -> CtxLogger:
    """Return a :class:`CtxLogger` for the requested logger name."""

    base_extra = dict(extra or {})
    return CtxLogger(logging.getLogger(name), base_extra)


def configure_logging(app_name: str) -> None:
    """Configure root logging to emit JSON logs with secret redaction."""

    if not LOG_JSON:
        logging.basicConfig(
            level=getattr(logging, LOG_LEVEL, logging.INFO),
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s | "
            "cmd=%(cmd)s user_id=%(user_id)s chat_id=%(chat_id)s state=%(state)s",
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
    "configure_logging",
    "get_logger",
    "log_environment",
    "refresh_secret_cache",
]
