import asyncio

from telegram.constants import ParseMode

import logging
from types import SimpleNamespace

from logging_utils import get_logger
from telegram_utils import SafeSendResult, sanitize_html, safe_send, safe_send_text


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


def test_safe_send_text_logs_success(caplog) -> None:
    class DummyBot:
        async def send_message(self, **kwargs):
            return SimpleNamespace(message_id=777)

    logger = get_logger("test.safe_send")
    context = SimpleNamespace(
        bot=DummyBot(),
        application=SimpleNamespace(bot_data={"logger": logger}),
    )

    with caplog.at_level(logging.INFO, logger="test.safe_send"):
        result = asyncio.run(safe_send_text(context, 123, "hello", parse_mode=None))
    assert isinstance(result, SafeSendResult)
    assert result.ok is True
    assert any(record.message == "send.ok" for record in caplog.records)


def test_safe_send_text_logs_failure(caplog) -> None:
    class DummyBot:
        async def send_message(self, **kwargs):
            raise RuntimeError("boom")

    logger = get_logger("test.safe_send")
    context = SimpleNamespace(
        bot=DummyBot(),
        application=SimpleNamespace(bot_data={"logger": logger}),
    )

    with caplog.at_level(logging.WARNING, logger="test.safe_send"):
        result = asyncio.run(safe_send_text(context, 123, "hello", parse_mode=None))
    assert isinstance(result, SafeSendResult)
    assert result.ok is False
    assert any(record.message == "send.fail" for record in caplog.records)
