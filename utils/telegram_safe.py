from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from telegram import Message, ReplyMarkup, Update
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import ContextTypes

from logging_utils import build_log_extra


log = logging.getLogger("bot.send")
_logger = logging.getLogger("telegram-safe")


@dataclass(slots=True)
class SafeSendResult:
    ok: bool
    message_id: Optional[int] = None
    description: Optional[str] = None
    error_code: Optional[int] = None
    message: Optional[Message] = None


async def safe_send_text(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    text: str,
    reply_markup: Optional[ReplyMarkup] = None,
    **kwargs: Any,
) -> SafeSendResult:
    chat_id = update.effective_chat.id if getattr(update, "effective_chat", None) else None
    bot = getattr(context, "bot", None)
    if bot is None:
        raise RuntimeError("Context has no bot instance")

    def _mark_sent() -> None:
        chat_data = getattr(context, "chat_data", None)
        if isinstance(chat_data, MutableMapping):
            chat_data["_last_command_reply_sent"] = True

    if chat_id is None:
        log.warning(
            "send.fail",
            extra={
                "meta": {
                    "chat_id": None,
                    "error": "chat_id is None",
                    "ctx_update_type": type(update).__name__,
                }
            },
        )
        return SafeSendResult(ok=False, description="chat_id is None")

    try:
        message = await bot.send_message(
            chat_id=chat_id,
            text=text,
            reply_markup=reply_markup,
            **kwargs,
        )
        log.info(
            "send.ok",
            extra={
                "meta": {
                    "chat_id": chat_id,
                    "message_id": getattr(message, "message_id", None),
                    "ctx_update_type": type(update).__name__,
                }
            },
        )
        _mark_sent()
        return SafeSendResult(
            ok=True,
            message_id=getattr(message, "message_id", None),
            message=message,
        )
    except Exception as exc:  # pragma: no cover - network issues
        description = getattr(exc, "message", None) or str(exc)
        error_code = getattr(exc, "status_code", None) or getattr(exc, "errno", None)
        log.warning(
            "send.fail",
            extra={
                "meta": {
                    "chat_id": chat_id,
                    "error": description,
                    "error_code": error_code,
                    "ctx_update_type": type(update).__name__,
                }
            },
            exc_info=True,
        )

        try:
            if chat_id is not None:
                await bot.send_message(chat_id=chat_id, text="⚠️ Не смог ответить из-за ошибки. Уже чиним.")
                _mark_sent()
        except Exception:  # pragma: no cover - defensive fallback
            pass

        return SafeSendResult(ok=False, description=description, error_code=error_code)


_MessageKey = Tuple[int, int]
_MessageHashes: dict[_MessageKey, Tuple[str, str]] = {}


def _serialize_markup(markup: Any) -> str:
    if markup is None:
        return ""
    try:
        payload = markup.to_dict()
    except AttributeError:
        payload = markup
    try:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(payload)


def _hash_payload(text: str, markup: Any) -> Tuple[str, str]:
    markup_serialized = _serialize_markup(markup)
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    markup_hash = hashlib.sha256(markup_serialized.encode("utf-8")).hexdigest()
    return text_hash, markup_hash


def _store_hashes(key: _MessageKey, hashes: Tuple[str, str]) -> None:
    # Keep the cache bounded to avoid unbounded growth in long-lived processes.
    if len(_MessageHashes) > 2048:
        _MessageHashes.clear()
    _MessageHashes[key] = hashes


async def safe_edit_message(
    ctx: Any,
    chat_id: int,
    message_id: int,
    new_text: Optional[str] = None,
    reply_markup: Any = None,
    *,
    parse_mode: ParseMode = ParseMode.HTML,
    disable_web_page_preview: bool = True,
) -> bool:
    """Safely edit a Telegram message without triggering noisy errors."""

    bot = getattr(ctx, "bot", None)
    if bot is None:
        raise RuntimeError("Context has no bot instance")

    text_payload = "" if new_text is None else str(new_text)
    key: _MessageKey = (int(chat_id), int(message_id))
    new_hashes = _hash_payload(text_payload, reply_markup)

    if _MessageHashes.get(key) == new_hashes:
        _logger.info(
            "card_edit_noop",
            **build_log_extra({"chat_id": chat_id, "message_id": message_id}),
        )
        return False

    try:
        await bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text=text_payload,
            reply_markup=reply_markup,
            parse_mode=parse_mode,
            disable_web_page_preview=disable_web_page_preview,
        )
        _store_hashes(key, new_hashes)
        return True
    except BadRequest as exc:
        lowered = str(exc).lower()
        if "message is not modified" in lowered:
            _logger.info(
                "card_edit_ignored_same_content",
                **build_log_extra({"chat_id": chat_id, "message_id": message_id}),
            )
            _store_hashes(key, new_hashes)
            return False
        raise
