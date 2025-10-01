"""Tests for Telegram handler registration and reply button wiring."""

import os
import sys
from typing import Dict

from telegram.ext import AIORateLimiter, ApplicationBuilder, CommandHandler, MessageHandler

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("SUNO_API_TOKEN", "test-token")
os.environ.setdefault("LEDGER_BACKEND", "memory")

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
)
def _build_application():
    return (
        ApplicationBuilder().token("123:ABC").rate_limiter(AIORateLimiter()).build()
    )


def test_chat_and_prompt_master_handlers_registered() -> None:
    """/chat and /prompt_master commands must be registered."""

    application = _build_application()
    register_handlers(application)

    commands: set[str] = set()
    callback_patterns: set[str] = set()

    for handlers in application.handlers.values():
        for handler in handlers:
            if isinstance(handler, CommandHandler):
                commands.update(handler.commands)
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
    }
    missing = expected_commands - commands
    assert not missing, f"commands not registered: {sorted(missing)}"

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
