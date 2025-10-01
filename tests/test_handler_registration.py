"""Tests for Telegram handler registration and reply button wiring."""

import asyncio
import os
import sys
from types import SimpleNamespace
from typing import Dict
from unittest.mock import AsyncMock

from telegram.ext import AIORateLimiter, ApplicationBuilder, CommandHandler, MessageHandler

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("SUNO_API_TOKEN", "test-token")
os.environ.setdefault("LEDGER_BACKEND", "memory")
os.environ.setdefault("DATABASE_URL", "postgresql://user:pass@localhost/db")

import bot

from bot import (  # noqa: E402
    MENU_BTN_BALANCE,
    MENU_BTN_CHAT,
    MENU_BTN_IMAGE,
    MENU_BTN_PM,
    MENU_BTN_SUNO,
    MENU_BTN_VIDEO,
    REPLY_BUTTON_ROUTES,
    handle_balance_entry,
    handle_chat_entry,
    handle_image_entry,
    handle_music_entry,
    handle_video_entry,
    prompt_master_command,
    register_handlers,
    unknown_command,
)
def _build_application():
    application = (
        ApplicationBuilder().token("123:ABC").rate_limiter(AIORateLimiter()).build()
    )
    return application


def test_chat_and_prompt_master_handlers_registered() -> None:
    """/chat and /prompt_master commands must be registered."""

    application = _build_application()
    register_handlers(application)

    commands: set[str] = set()
    command_handlers: Dict[str, CommandHandler] = {}
    callback_patterns: set[str] = set()

    for handlers in application.handlers.values():
        for handler in handlers:
            if isinstance(handler, CommandHandler):
                commands.update(handler.commands)
                for command_name in handler.commands:
                    command_handlers[command_name] = handler
            pattern = getattr(handler, "pattern", None)
            if pattern is None:
                continue
            if hasattr(pattern, "pattern"):
                callback_patterns.add(pattern.pattern)
            else:
                callback_patterns.add(str(pattern))

    assert "chat" in commands
    assert "prompt_master" in commands

    expected_commands = {
        "start",
        "menu",
        "video",
        "image",
        "music",
        "buy",
        "lang",
        "help",
        "faq",
        "my_balance",
        "ping",
    }
    missing = expected_commands - commands
    assert not missing, f"commands not registered: {sorted(missing)}"

    def has_pre_reset(callback) -> bool:
        seen = set()
        current = callback
        while current and current not in seen:
            if getattr(current, "__pre_command_reset__", False):
                return True
            seen.add(current)
            current = getattr(current, "__wrapped__", None)
        return False

    for command_name in expected_commands:
        handler = command_handlers.get(command_name)
        assert handler is not None, f"handler missing for /{command_name}"
        assert has_pre_reset(handler.callback), f"/{command_name} must reset state"

    unknown_present = False
    for handlers in application.handlers.values():
        for handler in handlers:
            if isinstance(handler, MessageHandler) and getattr(handler, "callback", None) is unknown_command:
                unknown_present = True
                break
        if unknown_present:
            break

    assert unknown_present, "unknown command handler must be registered"

    assert any(pattern.startswith("^pm:") for pattern in callback_patterns)
    assert "^hub:.*" in callback_patterns

    command_groups = {
        group
        for group, handlers in application.handlers.items()
        for handler in handlers
        if isinstance(handler, CommandHandler)
    }
    assert command_groups == {0}

    on_text_group = None
    for group, handlers in application.handlers.items():
        for handler in handlers:
            if not isinstance(handler, MessageHandler):
                continue
            target = getattr(handler, "callback", None)
            while hasattr(target, "__wrapped__"):
                target = target.__wrapped__  # type: ignore[attr-defined]
            if getattr(target, "__name__", "") == "on_text":
                on_text_group = group
                break
        if on_text_group is not None:
            break

    assert on_text_group is not None, "fallback text handler must exist"
    assert on_text_group >= 10


def test_reply_button_routes_match_expected() -> None:
    """Large reply buttons should be wired to the correct handlers."""

    mapping: Dict[str, object] = dict(REPLY_BUTTON_ROUTES)
    assert mapping == {
        MENU_BTN_VIDEO: handle_video_entry,
        MENU_BTN_IMAGE: handle_image_entry,
        MENU_BTN_SUNO: handle_music_entry,
        MENU_BTN_PM: prompt_master_command,
        MENU_BTN_CHAT: handle_chat_entry,
        MENU_BTN_BALANCE: handle_balance_entry,
    }


def _find_command_handler(application, command: str) -> CommandHandler:
    for handlers in application.handlers.values():
        for handler in handlers:
            if isinstance(handler, CommandHandler) and command in handler.commands:
                return handler
    raise AssertionError(f"handler not found for /{command}")


def test_help_command_dispatches_and_resets(monkeypatch) -> None:
    application = _build_application()

    reset_mock = AsyncMock(return_value=0)
    monkeypatch.setattr(bot, "reset_user_state_safely", reset_mock)

    dispatched: list[object] = []

    async def fake_safe_dispatch(fn, update, context, *args, **kwargs):
        dispatched.append(fn)
        return None

    monkeypatch.setattr(bot, "safe_dispatch", fake_safe_dispatch)

    register_handlers(application)

    handler = _find_command_handler(application, "help")

    message = SimpleNamespace(text="/help")
    update = SimpleNamespace(
        effective_message=message,
        effective_user=SimpleNamespace(id=111),
        effective_chat=SimpleNamespace(id=222),
    )
    context = SimpleNamespace(application=application)

    asyncio.run(handler.callback(update, context))

    reset_mock.assert_awaited_once_with(user_id=111, chat_id=222)
    assert dispatched == [bot.on_help]


def test_unknown_command_replies_with_menu(monkeypatch) -> None:
    reset_mock = AsyncMock(return_value=0)
    monkeypatch.setattr(bot, "reset_user_state_safely", reset_mock)

    replies: list[str] = []

    class DummyMessage(SimpleNamespace):
        async def reply_text(self, text: str, **_kwargs) -> None:  # type: ignore[override]
            replies.append(text)

    message = DummyMessage(text="/abracadabra")
    update = SimpleNamespace(
        effective_message=message,
        effective_user=SimpleNamespace(id=555),
        effective_chat=SimpleNamespace(id=777),
    )
    context = SimpleNamespace(application=SimpleNamespace(logger=bot.log))

    asyncio.run(bot.unknown_command(update, context))

    reset_mock.assert_awaited_once_with(user_id=555, chat_id=777)
    assert replies == ["Неизвестная команда. Нажмите /menu"]
