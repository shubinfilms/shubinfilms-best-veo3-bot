"""Unit tests for the /help command handler."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("SUNO_API_TOKEN", "test-token")
os.environ.setdefault("LEDGER_BACKEND", "memory")
os.environ.setdefault("DATABASE_URL", "postgresql://user:pass@localhost/db")

import bot
from handlers import help_handler
from utils.telegram_safe import SafeSendResult


def _build_update(language_code: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        effective_user=SimpleNamespace(language_code=language_code, id=101),
        effective_chat=SimpleNamespace(id=202),
    )


def _context() -> SimpleNamespace:
    return SimpleNamespace(application=None, chat_data={})


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"

@pytest.mark.anyio
async def test_help_command_ru(monkeypatch: pytest.MonkeyPatch) -> None:
    logger = MagicMock()
    send_mock = AsyncMock(return_value=SafeSendResult(ok=True, message_id=555))

    monkeypatch.setattr(help_handler, "get_context_logger", lambda context: logger)
    monkeypatch.setattr(help_handler, "safe_send_text", send_mock)
    monkeypatch.setattr(help_handler, "_SUPPORT_USERNAME", "BestAi_Support", raising=False)
    monkeypatch.setattr(help_handler, "_SUPPORT_URL", "https://t.me/BestAi_Support", raising=False)

    update = _build_update("ru")
    context = _context()

    await help_handler.help_command(update, context)

    send_mock.assert_awaited_once()
    _, kwargs = send_mock.await_args
    assert "🆘 Поддержка" in kwargs["text"]
    assert "@BestAi_Support" in kwargs["text"]
    markup = kwargs["reply_markup"]
    button = markup.inline_keyboard[0][0]
    assert button.text == "Написать в поддержку"
    assert button.url == "https://t.me/BestAi_Support"

    assert any(call.args[0] == "command.dispatch" for call in logger.debug.call_args_list)
    assert any(call.args[0] == "send.ok" for call in logger.info.call_args_list)


@pytest.mark.anyio
async def test_help_command_en(monkeypatch: pytest.MonkeyPatch) -> None:
    logger = MagicMock()
    send_mock = AsyncMock(return_value=SafeSendResult(ok=True, message_id=777))

    monkeypatch.setattr(help_handler, "get_context_logger", lambda context: logger)
    monkeypatch.setattr(help_handler, "safe_send_text", send_mock)
    monkeypatch.setattr(help_handler, "_SUPPORT_USERNAME", "BestAi_Support", raising=False)
    monkeypatch.setattr(help_handler, "_SUPPORT_URL", "https://t.me/BestAi_Support", raising=False)

    update = _build_update("en")
    context = _context()

    await help_handler.help_command(update, context)

    _, kwargs = send_mock.await_args
    assert "🆘 Support" in kwargs["text"]
    assert "@BestAi_Support" in kwargs["text"]
    button = kwargs["reply_markup"].inline_keyboard[0][0]
    assert button.text == "Message Support"
    assert button.url == "https://t.me/BestAi_Support"


def test_support_alias_registered() -> None:
    for names, callback in bot.COMMAND_HANDLER_SPECS:
        if "help" in names:
            assert "support" in names
            assert callback is help_handler.help_command
            break
    else:  # pragma: no cover - defensive safeguard
        pytest.fail("/help handler not registered")


@pytest.mark.anyio
async def test_help_command_logs_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    logger = MagicMock()
    send_mock = AsyncMock(
        return_value=SafeSendResult(ok=False, message_id=None, description="boom", error_code=400)
    )

    monkeypatch.setattr(help_handler, "get_context_logger", lambda context: logger)
    monkeypatch.setattr(help_handler, "safe_send_text", send_mock)
    monkeypatch.setattr(help_handler, "_SUPPORT_USERNAME", "BestAi_Support", raising=False)
    monkeypatch.setattr(help_handler, "_SUPPORT_URL", "https://t.me/BestAi_Support", raising=False)

    update = _build_update("ru")
    context = _context()

    await help_handler.help_command(update, context)

    assert any(call.args[0] == "send.fail" for call in logger.warning.call_args_list)
