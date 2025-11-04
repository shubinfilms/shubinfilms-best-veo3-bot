"""Helpers for working with Telegram file attachments."""
from __future__ import annotations

from io import BytesIO
from typing import Optional, Set, Tuple

from PIL import Image
from telegram import Message

try:  # pragma: no cover - optional dependency
    import pillow_heif

    pillow_heif.register_heif_opener()
except Exception:  # pragma: no cover - best effort registration
    pass

MAX_MB = 25
ALLOWED_FORMATS: Set[str] = {"JPEG", "PNG", "WEBP"}


async def tg_image_bytes(message: Message, bot) -> bytes:
    """Fetch raw bytes for an image contained in ``message``."""

    if message.document and (message.document.mime_type or "").startswith("image/"):
        file = await bot.get_file(message.document.file_id)
    else:
        file = await bot.get_file(message.photo[-1].file_id)
    return await file.download_as_bytearray()


def identify_image(data: bytes) -> Tuple[Optional[str], Optional[str]]:
    """Detect image format and MIME type for ``data``.

    Returns ``(format, mime)`` similar to ``PIL.Image`` metadata or ``(None, None)``
    if the payload is not a valid image.
    """

    try:
        with Image.open(BytesIO(data)) as im:
            fmt = (im.format or "").upper()
            mime = Image.MIME.get(fmt)
            return fmt, mime
    except Exception:
        return None, None


def validate_image(
    data: bytes, *, allowed: Optional[Set[str]] = None
) -> Tuple[bool, Optional[str], Optional[str]]:
    """Check that ``data`` is a supported image payload.

    Returns a tuple ``(ok, format, mime)``. ``ok`` is ``True`` when the payload is
    within size limits and belongs to the ``allowed`` formats. ``format`` and
    ``mime`` follow PIL naming conventions and may be ``None`` when detection
    fails.
    """

    if len(data) > MAX_MB * 1024 * 1024:
        return False, None, None

    allowed_formats = {fmt.upper() for fmt in (allowed or ALLOWED_FORMATS)}
    fmt, mime = identify_image(data)
    if not fmt or not mime:
        return False, fmt, mime
    if allowed_formats and fmt.upper() not in allowed_formats:
        return False, fmt, mime
    return True, fmt, mime


__all__ = ["ALLOWED_FORMATS", "identify_image", "tg_image_bytes", "validate_image"]
