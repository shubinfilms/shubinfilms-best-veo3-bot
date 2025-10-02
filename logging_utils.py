"""Utilities for structured JSON logging with safe contextual payloads."""
from __future__ import annotations

import json
import logging
import os
import re
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, MutableMapping, Optional

_DEFAULT_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
_DEFAULT_LOG_JSON = os.getenv("LOG_JSON", "1").lower() not in {"0", "false", "no"}
_DEFAULT_MAX_BODY = int(os.getenv("MAX_IN_LOG_BODY", "2048"))

LOG_LEVEL: str = _DEFAULT_LOG_LEVEL
LOG_JSON: bool = _DEFAULT_LOG_JSON
MAX_IN_LOG_BODY: int = _DEFAULT_MAX_BODY

RESERVED = {
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
}


def build_log_extra(raw: dict | None = None, *, update: Any = None, **ctx: Any) -> dict:
    """Assemble a logging ``extra`` payload with contextual fields.

    ``raw`` is treated as the base ``meta`` dictionary. Additional keyword
    arguments and ``update``-derived values are merged on top while ensuring
    there are no collisions with standard :class:`logging.LogRecord` fields.
    """

    def _assign(target: dict[str, Any], key: str, value: Any) -> None:
        if value is None:
            return
        safe_key = key if key not in RESERVED else f"ctx_{key}"
        target[safe_key] = _sanitize(value)

    key_mapping = {
        "user": "ctx_user",
        "user_id": "ctx_user_id",
        "chat": "ctx_chat",
        "chat_id": "ctx_chat_id",
        "command": "ctx_command",
        "update_type": "ctx_update_type",
        "callback_data": "ctx_callback_data",
        "env": "ctx_env",
    }
    base: dict[str, Any] = {}
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            if key == "meta" and isinstance(value, Mapping):
                for meta_key, meta_value in value.items():
                    target_key = key_mapping.get(str(meta_key), str(meta_key))
                    _assign(base, target_key, meta_value)
            else:
                target_key = key_mapping.get(str(key), str(key))
                _assign(base, target_key, value)

    if update is not None:
        user = getattr(update, "effective_user", None)
        if user is not None:
            _assign(base, "ctx_user_id", getattr(user, "id", None))
            _assign(base, "ctx_username", getattr(user, "username", None))
        chat = getattr(update, "effective_chat", None)
        if chat is not None:
            _assign(base, "ctx_chat_id", getattr(chat, "id", None))
            _assign(base, "ctx_chat_type", getattr(chat, "type", None))
        _assign(base, "ctx_update_type", type(update).__name__)
        query = getattr(update, "callback_query", None)
        if query is not None:
            _assign(base, "ctx_callback_data", getattr(query, "data", None))
    for key, value in ctx.items():
        target_key = key_mapping.get(key, key)
        _assign(base, target_key, value)

    return {"extra": {"meta": base}}

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


def _reserved_record_keys() -> set[str]:
    sample = logging.LogRecord(
        name="sample",
        level=logging.INFO,
        pathname=__file__,
        lineno=0,
        msg="",
        args=(),
        exc_info=None,
    )
    # ``LogRecord`` in Python 3.13 performs strict checks against any attribute
    # already present in ``__dict__``.  To avoid ``KeyError`` we eagerly collect
    # every known attribute and treat them as reserved.
    reserved: set[str] = set(sample.__dict__.keys())
    # ``ctx``/``meta`` are also used internally by our formatter and therefore
    # need to be guarded against collision when the payload contains the same
    # keys.
    reserved.add("ctx")
    return reserved


RESERVED_LOG_KEYS = _reserved_record_keys()


def _needs_prefix(key: str) -> bool:
    return key in RESERVED_LOG_KEYS


def _prefix_key(key: str) -> str:
    candidate = f"ctx_{key}"
    while _needs_prefix(candidate):
        candidate = f"ctx_{candidate}"
    return candidate


def _sanitize_extra_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _sanitize_mapping(value)
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_extra_value(item) for item in value]
    return value


def _sanitize_mapping(data: Mapping[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for raw_key, value in data.items():
        key = str(raw_key)
        safe_key = key if not _needs_prefix(key) else _prefix_key(key)
        sanitized[safe_key] = _sanitize_extra_value(value)
    return sanitized


def _add_ctx_value(ctx_payload: Dict[str, Any], key: str, value: Any) -> None:
    safe_key = key if not _needs_prefix(key) else _prefix_key(key)
    ctx_payload[safe_key] = _sanitize_extra_value(value)


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

    logger.info("environment", **build_log_extra({"env": safe_env}))


def _normalize_extra(extra: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    if not extra:
        return {"ctx": {}}

    ctx_payload: Dict[str, Any] = {}
    base_ctx = extra.get("ctx")
    if isinstance(base_ctx, Mapping):
        for key, value in base_ctx.items():
            _add_ctx_value(ctx_payload, str(key), value)

    for key, value in extra.items():
        if key == "ctx":
            continue
        _add_ctx_value(ctx_payload, str(key), value)

    normalized: dict[str, Any] = {"ctx": ctx_payload}
    meta_value = ctx_payload.get("meta") if isinstance(ctx_payload, Mapping) else None
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


try:  # Late import to avoid creating ``Logger`` instances before patching
    from settings import LOG_JSON as _SET_LOG_JSON, LOG_LEVEL as _SET_LOG_LEVEL, MAX_IN_LOG_BODY as _SET_MAX_BODY

    LOG_JSON = _SET_LOG_JSON
    LOG_LEVEL = _SET_LOG_LEVEL
    MAX_IN_LOG_BODY = _SET_MAX_BODY
except Exception:
    # ``settings`` may fail to import in some unit tests. Fall back to env defaults.
    pass


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


def get_context_logger(
    context: Any | None = None,
    *,
    application: Any | None = None,
    fallback: str = "veo3-bot",
) -> logging.Logger:
    """Return the logger stored inside ``application.bot_data`` if available."""

    app = application or getattr(context, "application", None)
    if app is not None:
        bot_data = getattr(app, "bot_data", None)
        if isinstance(bot_data, MutableMapping):
            stored = bot_data.get("logger")
            if isinstance(stored, logging.Logger):
                return stored
            if isinstance(stored, CtxLogger):
                return stored
    return get_logger(fallback)


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
        logging.getLogger("telegram.request").setLevel(logging.DEBUG)
        _CONFIGURED = True
        _install_record_factory_defaults()


__all__ = [
    "JsonFormatter",
    "CtxLogger",
    "SafeLoggerAdapter",
    "build_log_extra",
    "configure_logging",
    "get_context_logger",
    "get_logger",
    "log_environment",
    "refresh_secret_cache",
]
