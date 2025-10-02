from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from telegram.constants import ParseMode
from telegram.error import BadRequest


_logger = logging.getLogger("telegram-safe")

_MessageKey = Tuple[int, int]
_MessageHashes: dict[_MessageKey, Tuple[str, str]] = {}


@dataclass(slots=True)
class SafeSendResult:
    """Result wrapper for safe send attempts."""

    ok: bool
    message: Any = None
    error: Optional[BaseException] = None
    message_id: Optional[int] = None

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return self.ok


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
            extra={"chat_id": chat_id, "message_id": message_id},
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
                extra={"chat_id": chat_id, "message_id": message_id},
            )
            _store_hashes(key, new_hashes)
            return False
        raise


async def safe_send_text(
    ctx: Any,
    *,
    chat_id: int,
    text: str,
    reply_markup: Any = None,
    parse_mode: ParseMode = ParseMode.MARKDOWN_V2,
    disable_web_page_preview: bool = True,
) -> SafeSendResult:
    """Safely send a text message without propagating Telegram errors."""

    bot = getattr(ctx, "bot", None)
    if bot is None:
        raise RuntimeError("Context has no bot instance")

    try:
        message = await bot.send_message(
            chat_id=chat_id,
            text=text,
            reply_markup=reply_markup,
            parse_mode=parse_mode,
            disable_web_page_preview=disable_web_page_preview,
        )
    except Exception as exc:  # pragma: no cover - exercised via tests
        return SafeSendResult(ok=False, message=None, error=exc, message_id=None)

    message_id = getattr(message, "message_id", None)
    return SafeSendResult(ok=True, message=message, error=None, message_id=message_id)
