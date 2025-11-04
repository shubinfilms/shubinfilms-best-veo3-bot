from io import BytesIO

import pytest
from PIL import Image

from utils.files import validate_image


def _make_bytes(fmt: str) -> bytes:
    image = Image.new("RGB", (1, 1), color=(123, 222, 111))
    buffer = BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.mark.parametrize(
    "format_name,expected",
    [
        ("JPEG", "JPEG"),
        ("PNG", "PNG"),
        ("WEBP", "WEBP"),
    ],
)
def test_validate_ok(format_name, expected):
    ok, detected_fmt, mime = validate_image(_make_bytes(format_name))
    assert ok is True
    assert detected_fmt == expected
    assert mime and mime.startswith("image/")


def test_validate_bad_bytes():
    ok, fmt, mime = validate_image(b"not an image at all")
    assert ok is False
    assert fmt is None
    assert mime is None
