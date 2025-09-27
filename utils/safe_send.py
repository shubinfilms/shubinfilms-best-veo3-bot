"""Utilities for safely delivering HTML-formatted Telegram messages."""

from __future__ import annotations

from typing import Iterable, Optional

from telegram import Bot, Message
from telegram.constants import ParseMode


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


__all__ = ["safe_send"]
