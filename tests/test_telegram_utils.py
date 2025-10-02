import asyncio
import sys
from pathlib import Path

import pytest
from telegram import InputMediaPhoto
from telegram.constants import ParseMode

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from telegram_utils import (
    safe_send,
    safe_send_document,
    safe_send_media_group,
    safe_send_photo,
    sanitize_html,
)


def test_sanitize_html_replaces_br_and_strips_tags() -> None:
    raw = "Hello<br/>world<br>!<div><span>keep</span> <b>bold</b></div><script>alert(1)</script><tg-spoiler>spoiler</tg-spoiler><"
    sanitized = sanitize_html(raw)
    assert "Hello\nworld\n!" in sanitized
    assert "<b>bold</b>" in sanitized
    assert "tg-spoiler" in sanitized
    assert "<span>" not in sanitized
    assert "script" not in sanitized
    assert "keep" in sanitized
    assert "&lt;" in sanitized


def test_safe_send_sanitizes_html_payload() -> None:
    captured: list[dict] = []

    async def fake_send(**kwargs):
        captured.append(kwargs)
        return object()

    asyncio.run(
        safe_send(
            fake_send,
            method_name="send_message",
            kind="test",
            chat_id=1,
            text="line1<br/>line2",
            parse_mode=ParseMode.HTML,
        )
    )

    assert captured
    payload = captured[0]
    assert payload["text"] == "line1\nline2"


def test_safe_send_photo_sanitizes_caption() -> None:
    captured: dict[str, str] = {}

    class _Bot:
        async def send_photo(self, **kwargs):
            captured.update(kwargs)
            return object()

    asyncio.run(
        safe_send_photo(
            _Bot(),
            chat_id=1,
            photo=b"data",
            caption="hello<br/>world",
            parse_mode=ParseMode.HTML,
        )
    )

    assert captured["caption"] == "hello\nworld"


def test_safe_send_document_propagates_errors() -> None:
    class _Bot:
        async def send_document(self, **kwargs):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        asyncio.run(
            safe_send_document(
                _Bot(),
                chat_id=1,
                document=b"doc",
            )
        )


def test_safe_send_media_group_sanitizes_first_caption() -> None:
    captured: dict[str, list] = {}

    class _Bot:
        async def send_media_group(self, **kwargs):
            captured.update(kwargs)
            return [object()]

    media = [
        InputMediaPhoto(media="id1", caption="line1<br/>line2"),
        InputMediaPhoto(media="id2", caption=None),
    ]

    asyncio.run(
        safe_send_media_group(
            _Bot(),
            chat_id=1,
            media=media,
            parse_mode=ParseMode.HTML,
        )
    )

    sent_media = captured.get("media")
    assert sent_media is not None
    assert sent_media[0].caption == "line1\nline2"
