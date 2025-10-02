"""Helpers for structured logging with Telegram updates."""
from __future__ import annotations

from typing import Any, Optional

try:  # pragma: no cover - telegram is optional for type checking
    from telegram import Chat, Message, Update, User
except Exception:  # pragma: no cover - fallback for runtime without telegram
    Chat = Message = Update = User = object  # type: ignore[assignment]


def _get_user_meta(user: Optional[User]) -> dict[str, Any]:  # type: ignore[type-arg]
    if user is None or getattr(user, "id", None) is None:
        return {}
    meta: dict[str, Any] = {"user_id": getattr(user, "id", None)}
    language_code = getattr(user, "language_code", None)
    if isinstance(language_code, str) and language_code:
        meta["user_lang"] = language_code
    return meta


def _get_chat_meta(chat: Optional[Chat]) -> dict[str, Any]:  # type: ignore[type-arg]
    if chat is None or getattr(chat, "id", None) is None:
        return {}
    meta: dict[str, Any] = {"chat_id": getattr(chat, "id", None)}
    chat_type = getattr(chat, "type", None)
    if isinstance(chat_type, str) and chat_type:
        meta["chat_type"] = chat_type
    return meta


def _get_message_meta(message: Optional[Message]) -> dict[str, Any]:  # type: ignore[type-arg]
    if message is None or getattr(message, "message_id", None) is None:
        return {}
    meta: dict[str, Any] = {"message_id": getattr(message, "message_id", None)}
    return meta


def _build_update_extra(update: Optional[Update]) -> dict[str, Any]:  # type: ignore[type-arg]
    if update is None:
        return {}
    meta: dict[str, Any] = {}
    meta.update(_get_user_meta(getattr(update, "effective_user", None)))
    meta.update(_get_chat_meta(getattr(update, "effective_chat", None)))
    meta.update(_get_message_meta(getattr(update, "effective_message", None)))
    return meta


def build_log_extra(update: Optional[Update] = None, **kwargs: Any) -> dict[str, Any]:  # type: ignore[type-arg]
    meta = _build_update_extra(update)
    for key, value in kwargs.items():
        if value is None:
            continue
        meta[str(key)] = value
    return {"meta": meta}


__all__ = ["build_log_extra", "_build_update_extra"]
