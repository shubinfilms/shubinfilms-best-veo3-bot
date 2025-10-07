"""Utilities for structured JSON logging with secret redaction."""
from __future__ import annotations

import json
import logging
import os
import re
import threading
from datetime import datetime, timezone
from typing import Any, Mapping

from core.settings import settings

MAX_IN_LOG_BODY = int(settings.MAX_IN_LOG_BODY)

_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARN": logging.WARNING,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}

_JSON_ENABLED = bool(settings.LOG_JSON)
_DEFAULT_LEVEL = settings.LOG_LEVEL

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


def _resolve_level(name: str | None) -> int:
    if not name:
        name = _DEFAULT_LEVEL
    normalized = str(name).strip().upper() or "INFO"
    return _LEVEL_MAP.get(normalized, logging.INFO)


def init_logging(app_name: str, level: str | None = None, *, json_logs: bool | None = None) -> None:
    """Configure root logging according to runtime configuration."""

    effective_level = _resolve_level(level)
    use_json = _JSON_ENABLED if json_logs is None else bool(json_logs)

    global _CONFIGURED
    with _CONFIG_LOCK:
        if not _CONFIGURED:
            handler = logging.StreamHandler()
            if use_json:
                handler.setFormatter(JsonFormatter())
            else:
                handler.setFormatter(
                    logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
                )
            root = logging.getLogger()
            root.handlers.clear()
            root.addHandler(handler)
            root.setLevel(effective_level)
            logging.captureWarnings(True)
            for noisy in (
                "httpx",
                "urllib3",
                "aiogram",
                "telegram",
                "uvicorn",
                "gunicorn",
                "pydantic",
            ):
                logging.getLogger(noisy).setLevel(logging.WARNING)
            logging.getLogger("httpcore").setLevel(logging.INFO)
            _CONFIGURED = True
        else:
            logging.getLogger().setLevel(effective_level)

    logger = logging.getLogger(app_name)
    log_level = max(logging.INFO, effective_level)
    logger.log(
        log_level,
        "configuration summary",
        extra={"meta": settings.configuration_summary()},
    )
    logger.log(
        log_level,
        "configuration critical",
        extra={"meta": settings.critical_variables()},
    )


__all__ = [
    "JsonFormatter",
    "init_logging",
    "log_environment",
    "refresh_secret_cache",
]
