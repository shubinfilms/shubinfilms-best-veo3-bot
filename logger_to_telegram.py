"""Forward WARNING/ERROR logs to the admin via Telegram."""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

import requests

__all__ = ["TelegramLogHandler"]


def _env(name: str) -> Optional[str]:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _env_int(name: str) -> Optional[int]:
    raw = _env(name)
    if not raw:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


class TelegramLogHandler(logging.Handler):
    """Logging handler that mirrors records to a Telegram chat."""

    _API_URL_TEMPLATE = "https://api.telegram.org/bot{token}/sendMessage"

    def __init__(self, token: str, chat_id: int, *, level: int = logging.WARNING) -> None:
        super().__init__(level=level)
        self._token = token
        self._chat_id = chat_id
        self._session = requests.Session()
        self._lock = threading.Lock()
        self._formatter = logging.Formatter()

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        try:
            payload = self._render_payload(record)
            if not payload:
                return
            url = self._API_URL_TEMPLATE.format(token=self._token)
            with self._lock:
                response = self._session.post(url, json=payload, timeout=5)
            if response.status_code >= 400:
                return
        except Exception:
            # Never let logging failures bring the application down.
            return

    def _render_payload(self, record: logging.LogRecord) -> Optional[dict[str, object]]:
        icon = "❌" if record.levelno >= logging.ERROR else "⚠️"
        header = f"{icon} {record.levelname} • {record.name}"
        message = record.getMessage()
        location_parts = []
        if record.module:
            location_parts.append(record.module)
        if record.funcName:
            location_parts.append(record.funcName)
        if record.lineno:
            location_parts.append(str(record.lineno))
        location = "::".join(location_parts)

        lines = [header]
        if location:
            lines.append(f"Location: {location}")
        if message:
            lines.append(message)

        if record.exc_info:
            try:
                exc_text = self._formatter.formatException(record.exc_info)
            except Exception:
                exc_text = ""
            if exc_text:
                lines.append("")
                lines.append(exc_text)
        elif record.stack_info:
            lines.append("")
            lines.append(record.stack_info)

        text = "\n".join(lines).strip()
        if not text:
            return None
        if len(text) > 3900:
            text = text[:3900] + "…"

        return {
            "chat_id": self._chat_id,
            "text": text,
            "disable_web_page_preview": True,
        }


def _configure_handler() -> None:
    token = _env("TELEGRAM_TOKEN")
    if not token:
        return
    chat_id = _env_int("ADMIN_ID") or 878622103
    if not chat_id:
        return

    root = logging.getLogger()
    for handler in root.handlers:
        if isinstance(handler, TelegramLogHandler):
            return

    handler = TelegramLogHandler(token, chat_id, level=logging.WARNING)
    root.addHandler(handler)


_configure_handler()
