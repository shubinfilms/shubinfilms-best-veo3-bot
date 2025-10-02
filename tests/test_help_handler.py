import asyncio
import logging
import os
import sys
from types import SimpleNamespace

import pytest
from telegram import InlineKeyboardMarkup

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from handlers import help_command as help_handler  # noqa: E402
from handlers.help_handler import help_command, support_command  # noqa: E402
from settings import SUPPORT_USERNAME  # noqa: E402
from texts import help_text  # noqa: E402
from utils.telegram_safe import SafeSendResult  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_logging_handlers(monkeypatch):
    # Ensure logger uses fresh handlers during tests to avoid interference.
    logger = logging.getLogger("bot.commands.help")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    logger.propagate = True
    yield


def _build_update(language_code: str = "ru"):
    chat = SimpleNamespace(id=123, type="private")
    user = SimpleNamespace(id=555, language_code=language_code)
    message = SimpleNamespace(message_id=42, chat=chat)
    return SimpleNamespace(
        effective_chat=chat,
        effective_user=user,
        effective_message=message,
    )


def _run(coro):
    return asyncio.run(coro)


def test_help_command_ru_sends_localized_message(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="bot.commands.help")
    captured = {}

    async def fake_send(context, *, chat_id, text, reply_markup, **kwargs):
        captured["chat_id"] = chat_id
        captured["text"] = text
        captured["markup"] = reply_markup
        return SafeSendResult(ok=True, message_id=777)

    monkeypatch.setattr("handlers.help_handler.safe_send_text", fake_send)

    update = _build_update("ru")
    context = SimpleNamespace()

    _run(help_command(update, context))

    expected_text, expected_button = help_text("ru", SUPPORT_USERNAME)
    assert captured["chat_id"] == update.effective_chat.id
    assert captured["text"] == expected_text
    markup = captured["markup"]
    assert isinstance(markup, InlineKeyboardMarkup)
    button = markup.inline_keyboard[0][0]
    assert button.text == expected_button
    assert button.text == "Написать в поддержку"
    assert button.url == f"https://t.me/{SUPPORT_USERNAME}"

    messages = [record.msg for record in caplog.records if record.name == "bot.commands.help"]
    assert "command.dispatch" in messages
    assert "send.ok" in messages
    send_ok_records = [record for record in caplog.records if record.msg == "send.ok"]
    assert send_ok_records
    assert send_ok_records[0].meta.get("message_id") == 777  # type: ignore[union-attr]


def test_help_command_en_uses_english_copy(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="bot.commands.help")
    captured = {}

    async def fake_send(context, *, chat_id, text, reply_markup, **kwargs):
        captured["text"] = text
        captured["markup"] = reply_markup
        return SafeSendResult(ok=True, message_id=321)

    monkeypatch.setattr("handlers.help_handler.safe_send_text", fake_send)

    update = _build_update("en")
    context = SimpleNamespace()

    _run(help_command(update, context))

    expected_text, expected_button = help_text("en", SUPPORT_USERNAME)
    assert captured["text"] == expected_text
    button = captured["markup"].inline_keyboard[0][0]
    assert button.text == expected_button
    assert button.url.endswith(SUPPORT_USERNAME)


def test_support_alias_reuses_handler():
    assert support_command is help_command
    assert help_handler is help_command  # sanity: public shortcut matches


def test_help_command_logs_failure(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="bot.commands.help")

    async def fake_send(context, *, chat_id, text, reply_markup, **kwargs):
        return SafeSendResult(ok=False, error=RuntimeError("boom"))

    monkeypatch.setattr("handlers.help_handler.safe_send_text", fake_send)

    update = _build_update("ru")
    context = SimpleNamespace()

    _run(help_command(update, context))

    messages = [record.msg for record in caplog.records if record.name == "bot.commands.help"]
    assert "send.fail" in messages
    fail_records = [record for record in caplog.records if record.msg == "send.fail"]
    assert fail_records
    meta = fail_records[0].meta  # type: ignore[union-attr]
    assert meta.get("error_type") == "RuntimeError"
