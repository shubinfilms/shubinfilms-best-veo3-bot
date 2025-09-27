"""Utilities for safely delivering HTML-formatted Telegram messages."""

from __future__ import annotations

import html
import logging
import re
from typing import Iterable, Optional

from telegram import Bot, Message
from telegram.constants import ParseMode
from telegram.error import BadRequest

from utils.html_render import html_to_plain

logger = logging.getLogger(__name__)


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
    chunks = list(_chunk_text(text, limit=chunk_limit))
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


async def send_html_with_fallback(
    bot: Bot,
    chat_id: int,
    text: str,
    *,
    reply_markup=None,
    chunk_limit: int = 3500,
) -> Optional[Message]:
    """Send HTML text and fall back to plain text on parse errors."""

    try:
        return await safe_send(
            bot,
            chat_id,
            text,
            reply_markup=reply_markup,
            chunk_limit=chunk_limit,
        )
    except BadRequest as exc:
        message = str(exc).lower()
        if "can't parse entities" not in message and "parse entities" not in message:
            raise
        logger.warning("pm.html_fallback", extra={"exc": repr(exc)})
        logger.info("pm.render.fallback")
        plain = html_to_plain(text)
        if not plain:
            plain = re.sub(r"<[^>]+>", "", text)
            plain = html.unescape(plain)
        return await bot.send_message(
            chat_id=chat_id,
            text=plain,
            disable_web_page_preview=True,
            reply_markup=reply_markup,
        )


__all__ = ["safe_send", "send_html_with_fallback"]
