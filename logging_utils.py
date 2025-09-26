"""Utilities for structured JSON logging with secret redaction."""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any, Mapping

from settings import LOG_JSON, LOG_LEVEL, MAX_IN_LOG_BODY

_SECRET_ENV_KEYS = {"DATABASE_URL", "REDIS_URL"}
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


def _redact_text(value: str) -> str:
    if not value:
        return value
    with _SECRET_VALUES_LOCK:
        secrets = list(_SECRET_VALUES)
    for secret in secrets:
        if secret and secret in value:
            value = value.replace(secret, "***")
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


def configure_logging(app_name: str) -> None:
    """Configure root logging to emit JSON logs with secret redaction."""

    if not LOG_JSON:
        logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
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


__all__ = [
    "JsonFormatter",
    "configure_logging",
    "refresh_secret_cache",
]
