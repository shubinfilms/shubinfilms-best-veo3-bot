"""Utilities for detecting and validating image data."""

from __future__ import annotations

from io import BytesIO

from PIL import Image

SUPPORTED = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}


def detect_image_ext(data: bytes) -> str | None:
    """Return a file extension (without dot) for supported images.

    The detection uses Pillow formats to support Python 3.13 where ``imghdr``
    has been removed. ``None`` is returned when detection fails or when the
    image format is not whitelisted.
    """

    try:
        with Image.open(BytesIO(data)) as im:
            fmt = (im.format or "").upper()
            return SUPPORTED.get(fmt)
    except Exception:
        return None


def is_image(data: bytes) -> bool:
    """Return ``True`` when the byte sequence represents a valid image."""

    try:
        with Image.open(BytesIO(data)) as im:
            im.verify()
        return True
    except Exception:
        return False
