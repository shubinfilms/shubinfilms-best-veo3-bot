import asyncio

from telegram.constants import ParseMode

from telegram_utils import sanitize_html, safe_send


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
