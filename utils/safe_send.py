"""Utilities for safely delivering HTML-formatted Telegram messages."""

from __future__ import annotations

import html
import logging
import re
from html.parser import HTMLParser
from typing import Iterable, Optional

from telegram import Bot, Message
from telegram.constants import ParseMode
from telegram.error import BadRequest

from utils.html_render import html_to_plain
from logging_utils import build_log_extra

logger = logging.getLogger(__name__)

_SANITIZE_ALLOWED_TAGS = {"b", "i", "u", "s", "code", "pre", "a", "blockquote"}
_SANITIZE_ATTRS = {"a": {"href"}}
_TAG_REMAP = {"strong": "b", "em": "i"}


class _HTMLStripper(HTMLParser):
    """HTML sanitizer that keeps only Telegram-safe tags."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.result: list[str] = []
        self._skip_depth = 0
        self._stack: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[override]
        tag = _TAG_REMAP.get(tag.lower(), tag.lower())
        if tag == "br":
            self.result.append("\n")
            return
        if self._skip_depth:
            self._skip_depth += 1
            return
        if tag not in _SANITIZE_ALLOWED_TAGS:
            self._skip_depth = 1
            return
        filtered = []
        allowed_attrs = _SANITIZE_ATTRS.get(tag, set())
        for name, value in attrs:
            if value is None:
                continue
            if name not in allowed_attrs:
                continue
            if tag == "a" and value.lower().startswith("javascript:"):
                continue
            filtered.append((name, value))
        attr_str = "".join(
            f' {name}="{html.escape(value, quote=True)}"' for name, value in filtered
        )
        self.result.append(f"<{tag}{attr_str}>")
        self._stack.append(tag)

    def handle_startendtag(self, tag: str, attrs) -> None:  # type: ignore[override]
        self.handle_starttag(tag, attrs)
        if self._skip_depth:
            self._skip_depth -= 1

    def handle_endtag(self, tag: str) -> None:  # type: ignore[override]
        tag = _TAG_REMAP.get(tag.lower(), tag.lower())
        if tag == "br":
            return
        if self._skip_depth:
            self._skip_depth -= 1
            return
        while self._stack:
            top = self._stack.pop()
            if top == tag:
                self.result.append(f"</{tag}>")
                break

    def handle_data(self, data: str) -> None:  # type: ignore[override]
        if self._skip_depth:
            return
        if not data:
            return
        self.result.append(html.escape(data))

    def handle_entityref(self, name: str) -> None:  # type: ignore[override]
        if self._skip_depth:
            return
        self.result.append(f"&{name};")

    def handle_charref(self, name: str) -> None:  # type: ignore[override]
        if self._skip_depth:
            return
        self.result.append(f"&#{name};")

    def get_text(self) -> str:
        return "".join(self.result)


def sanitize_html(text: str) -> str:
    """Normalise HTML string to Telegram-safe subset."""

    if not text:
        return ""
    normalized = text.replace("\r", "")
    normalized = re.sub(r"<br\s*/?>", "\n", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"[\t\f]+", " ", normalized)
    parser = _HTMLStripper()
    try:
        parser.feed(normalized)
        parser.close()
    except Exception:  # pragma: no cover - defensive
        logger.exception("sanitize_html.failure")
        return html.escape(normalized)
    safe_text = parser.get_text()
    safe_text = re.sub(r" {2,}", " ", safe_text)
    safe_text = re.sub(r"\n{3,}", "\n\n", safe_text)
    return safe_text.strip()


def _chunk_text(text: str, *, limit: int) -> Iterable[str]:
    """Yield safe chunks for Telegram send operations."""

    if not text:
        yield ""
        return

    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + limit, text_len)
        if end < text_len:
            split = text.rfind("\n", start, end)
            if split <= start:
                split = text.rfind(" ", start, end)
            if split <= start:
                split = end
        else:
            split = end
        yield text[start:split]
        start = split


async def safe_send(
    bot: Bot,
    chat_id: int,
    text: str,
    *,
    reply_markup=None,
    chunk_limit: int = 3500,
) -> Optional[Message]:
    """Send HTML text safely, slicing long payloads into chunks.

    ``reply_markup`` is attached only to the last chunk.
    """

    last_message: Optional[Message] = None
    sanitized = sanitize_html(text)
    chunks = list(_chunk_text(sanitized, limit=chunk_limit))
    total = len(chunks)
    for index, chunk in enumerate(chunks):
        last_message = await bot.send_message(
            chat_id=chat_id,
            text=chunk,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
            reply_markup=reply_markup if index == total - 1 else None,
        )
    return last_message


async def safe_send_sticker(
    bot: Bot,
    chat_id: int,
    sticker: str,
    **kwargs,
) -> Optional[Message]:
    """Send a sticker with best-effort error propagation."""

    return await bot.send_sticker(chat_id=chat_id, sticker=sticker, **kwargs)


async def safe_delete_message(bot: Bot, chat_id: int, message_id: int) -> bool:
    """Delete a message while suppressing common Telegram errors."""

    try:
        await bot.delete_message(chat_id, message_id)
        return True
    except BadRequest as exc:
        message = str(exc).lower()
        if "message to delete not found" in message or "message can't be deleted" in message:
            logger.debug(
                "safe_delete.skip",
                **build_log_extra({"chat_id": chat_id, "message_id": message_id, "error": str(exc)}),
            )
            return False
        logger.debug(
            "safe_delete.bad_request",
            **build_log_extra({"chat_id": chat_id, "message_id": message_id, "error": str(exc)}),
        )
        return False
    except Exception as exc:  # pragma: no cover - network issues
        logger.warning(
            "safe_delete.error",
            **build_log_extra({"chat_id": chat_id, "message_id": message_id, "error": repr(exc)}),
        )
        return False


async def send_html_with_fallback(
    bot: Bot,
    chat_id: int,
    text: str,
    *,
    reply_markup=None,
    chunk_limit: int = 3500,
) -> Optional[Message]:
    """Send HTML text and fall back to plain text on parse errors."""

    sanitized = sanitize_html(text)
    try:
        return await safe_send(
            bot,
            chat_id,
            sanitized,
            reply_markup=reply_markup,
            chunk_limit=chunk_limit,
        )
    except BadRequest as exc:
        message = str(exc).lower()
        if "can't parse entities" not in message and "parse entities" not in message:
            raise
        logger.warning("pm.html_fallback", **build_log_extra({"exc": repr(exc)}))
        logger.info("pm.render.fallback")
        plain = html_to_plain(sanitized)
        if not plain:
            plain = re.sub(r"<[^>]+>", "", text)
            plain = html.unescape(plain)
        return await bot.send_message(
            chat_id=chat_id,
            text=plain,
            disable_web_page_preview=True,
            reply_markup=reply_markup,
        )


__all__ = [
    "safe_send",
    "safe_send_sticker",
    "send_html_with_fallback",
    "sanitize_html",
    "safe_delete_message",
]
