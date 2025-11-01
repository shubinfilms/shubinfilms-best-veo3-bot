"""Helpers for working with Telegram file attachments."""
from __future__ import annotations

import imghdr

from telegram import Message

MAX_MB = 25


async def tg_image_bytes(message: Message, bot) -> bytes:
    """Fetch raw bytes for an image contained in ``message``."""

    if message.document and (message.document.mime_type or "").startswith("image/"):
        file = await bot.get_file(message.document.file_id)
    else:
        file = await bot.get_file(message.photo[-1].file_id)
    return await file.download_as_bytearray()


def validate_image(data: bytes) -> None:
    """Ensure the image payload is supported and within limits."""

    if len(data) > MAX_MB * 1024 * 1024:
        raise ValueError("image too large")
    if imghdr.what(None, data) not in {"jpeg", "png", "webp"}:
        raise ValueError("unsupported image type")
