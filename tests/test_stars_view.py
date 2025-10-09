import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("LEDGER_BACKEND", "memory")

import bot as bot_module
import handlers.profile as profile_handlers
from handlers import profile_simple
from handlers.stars import STARS_TIERS, build_stars_kb, render_stars_text


def test_render_stars_text_no_bonus():
    text = render_stars_text()
    lowered = text.lower()
    assert "бонус" not in lowered
    assert "скоро" not in lowered


def test_build_stars_kb_structure():
    keyboard = build_stars_kb()
    rows = keyboard.inline_keyboard
    assert len(rows) == 8
    for index, (stars, gems) in enumerate(STARS_TIERS):
        button = rows[index][0]
        assert button.text == f"⭐️ {stars} → 💎 {gems}"
        assert button.callback_data == f"stars:buy:{stars}"
    assert rows[6][0].url == "https://t.me/PremiumBot"
    assert rows[7][0].callback_data == "nav:back"


def test_buy_command_opens_stars(monkeypatch):
    async def run() -> None:
        ensure_mock = AsyncMock()
        monkeypatch.setattr(bot_module, "ensure_user_record", ensure_mock)
        open_mock = AsyncMock()
        monkeypatch.setattr(bot_module, "open_stars_menu", open_mock)

        chat = SimpleNamespace(id=777)
        message = SimpleNamespace(message_id=55, chat=chat, chat_id=chat.id)
        update = SimpleNamespace(effective_chat=chat, effective_message=message)
        ctx = SimpleNamespace()

        await bot_module.buy_command(update, ctx)

        ensure_mock.assert_awaited_once()
        open_mock.assert_awaited_once()
        kwargs = open_mock.call_args.kwargs
        assert kwargs["edit_message"] is False
        assert kwargs["source"] == "command"
        assert kwargs["chat_id"] == chat.id
        assert kwargs["message_id"] == message.message_id

    asyncio.run(run())


def test_profile_topup_uses_stars(monkeypatch):
    async def run() -> None:
        monkeypatch.setattr(profile_handlers, "_simple_profile_enabled", lambda: False)
        open_mock = AsyncMock()
        monkeypatch.setattr(profile_handlers, "open_stars_menu", open_mock)

        chat = SimpleNamespace(id=1001)
        message = SimpleNamespace(message_id=202, chat=chat, chat_id=chat.id)
        query = SimpleNamespace(message=message)
        update = SimpleNamespace(
            callback_query=query,
            effective_chat=chat,
            effective_message=message,
        )
        ctx = SimpleNamespace()

        await profile_handlers.on_profile_topup(update, ctx)

        open_mock.assert_awaited_once()
        kwargs = open_mock.call_args.kwargs
        assert kwargs["edit_message"] is True
        assert kwargs["source"] == "profile"

    asyncio.run(run())


def test_profile_simple_topup_uses_stars(monkeypatch):
    async def run() -> None:
        open_mock = AsyncMock()
        monkeypatch.setattr(profile_simple, "open_stars_menu", open_mock)

        chat = SimpleNamespace(id=900)
        message = SimpleNamespace(message_id=77, chat=chat, chat_id=chat.id)
        async def answer(*_args, **_kwargs):
            return None

        query = SimpleNamespace(message=message, answer=answer)
        update = SimpleNamespace(
            callback_query=query,
            effective_chat=chat,
            effective_message=message,
        )
        ctx = SimpleNamespace()

        await profile_simple.profile_topup(update, ctx)

        open_mock.assert_awaited_once()
        kwargs = open_mock.call_args.kwargs
        assert kwargs["edit_message"] is True
        assert kwargs["source"] == "profile"

    asyncio.run(run())
