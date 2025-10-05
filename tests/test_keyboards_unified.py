import asyncio
import os
import sys
import types
from pathlib import Path

import pytest

from telegram import InlineKeyboardMarkup

sys.path.append(str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("DATABASE_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("LEDGER_BACKEND", "memory")

import bot
from keyboards import menu_bottom_unified, menu_pay_unified
from handlers import prompt_master_handler


def test_menu_bottom_unified_layout():
    markup = menu_bottom_unified()
    rows = markup.inline_keyboard

    assert len(rows) == 6

    texts = [button.text for row in rows for button in row]
    assert texts == [
        "🎬 Генерация видео",
        "🎨 Генерация изображений",
        "🎵 Генерация музыки",
        "🧠 Prompt-Master",
        "💬 Обычный чат",
        "👥 Профиль",
    ]


def test_menu_pay_unified_layout():
    markup = menu_pay_unified()
    rows = markup.inline_keyboard

    assert len(rows) == 4

    texts = [button.text for row in rows for button in row]
    assert texts == [
        "⭐️ Телеграм Stars",
        "💳 Оплата картой",
        "🔐 Crypto",
        "⬅️ Назад",
    ]


class DummyBot:
    async def send_message(self, *args, **kwargs):
        return types.SimpleNamespace(message_id=99)

    async def edit_message_text(self, *args, **kwargs):
        return None

    async def edit_message_reply_markup(self, *args, **kwargs):
        return None

    async def delete_message(self, *args, **kwargs):
        return None


class DummyCtx:
    def __init__(self):
        self.bot = DummyBot()
        self.user_data = {"state": {}}


class DummyChat:
    def __init__(self, chat_id):
        self.id = chat_id


class DummyUpdate:
    def __init__(self, chat_id, user_id):
        self.effective_chat = DummyChat(chat_id)
        self.effective_user = types.SimpleNamespace(id=user_id)


@pytest.mark.parametrize(
    "callable_name, setup_state",
    [
        ("show_veo_card", {"aspect": "16:9", "model": "veo3_fast", "last_prompt": None, "last_image_url": None}),
        ("show_banana_card", {"banana_images": [], "last_prompt": None}),
        ("show_mj_format_card", {"aspect": "16:9", "last_prompt": None}),
    ],
)
def test_menu_bottom_called_for_cards(monkeypatch, callable_name, setup_state):
    ctx = DummyCtx()
    ctx.user_data["state"].update(setup_state)

    calls = []

    async def fake_safe_edit_or_send_menu(*args, **kwargs):
        return 1

    async def fake_upsert_card(*args, **kwargs):
        return 1

    def fake_menu_bottom_unified():
        calls.append(True)
        return InlineKeyboardMarkup([])

    monkeypatch.setattr(bot, "safe_edit_or_send_menu", fake_safe_edit_or_send_menu)
    monkeypatch.setattr(bot, "upsert_card", fake_upsert_card)
    monkeypatch.setattr(bot, "menu_bottom_unified", fake_menu_bottom_unified)

    def fake_state(context):
        return context.user_data.setdefault("state", {})

    monkeypatch.setattr(bot, "state", fake_state)

    func = getattr(bot, callable_name)

    async def _run() -> None:
        await func(1, ctx, force_new=True)

    asyncio.run(_run())

    assert calls


def test_menu_bottom_called_for_suno(monkeypatch):
    ctx = DummyCtx()
    ctx.user_data["state"].update({})

    calls = []

    async def fake_safe_edit_or_send_menu(*args, **kwargs):
        return 1

    async def fake_refresh_raw(*args, **kwargs):
        return 1

    def fake_menu_bottom_unified():
        calls.append(True)
        return InlineKeyboardMarkup([])

    monkeypatch.setattr(bot, "safe_edit_or_send_menu", fake_safe_edit_or_send_menu)
    monkeypatch.setattr(bot, "_refresh_suno_card_raw", fake_refresh_raw)
    monkeypatch.setattr(bot, "menu_bottom_unified", fake_menu_bottom_unified)

    def fake_state(context):
        return context.user_data.setdefault("state", {})

    monkeypatch.setattr(bot, "state", fake_state)

    async def _run() -> None:
        await bot.refresh_suno_card(ctx, 1, ctx.user_data["state"], price=10)

    asyncio.run(_run())

    assert calls


def test_menu_bottom_called_for_chat(monkeypatch):
    ctx = DummyCtx()
    ctx.user_data.setdefault("state", {})

    calls = []

    async def fake_safe_edit_or_send_menu(*args, **kwargs):
        return 1

    def fake_menu_bottom_unified():
        calls.append(True)
        return InlineKeyboardMarkup([])

    async def fake_safe_send_text(*args, **kwargs):
        return None

    async def fake_ensure_user_record(update):
        return None

    def fake_state(context):
        return context.user_data.setdefault("state", {})

    monkeypatch.setattr(bot, "safe_edit_or_send_menu", fake_safe_edit_or_send_menu)
    monkeypatch.setattr(bot, "menu_bottom_unified", fake_menu_bottom_unified)
    monkeypatch.setattr(bot, "safe_send_text", fake_safe_send_text)
    monkeypatch.setattr(bot, "ensure_user_record", fake_ensure_user_record)
    monkeypatch.setattr(bot, "set_mode", lambda *args, **kwargs: None)
    monkeypatch.setattr(bot, "_mode_set", lambda *args, **kwargs: None)
    monkeypatch.setattr(bot, "state", fake_state)

    update = DummyUpdate(chat_id=1, user_id=2)

    async def _run() -> None:
        await bot.chat_command(update, ctx)

    asyncio.run(_run())

    assert calls


def test_prompt_master_bottom_menu(monkeypatch):
    calls = []

    def fake_menu_bottom_unified():
        calls.append(True)
        return InlineKeyboardMarkup([])

    async def fake_safe_send(*args, **kwargs):
        return types.SimpleNamespace(message_id=42)

    ctx = DummyCtx()
    update = DummyUpdate(chat_id=1, user_id=2)
    state = prompt_master_handler._ensure_state(ctx)

    monkeypatch.setattr(prompt_master_handler, "menu_bottom_unified", fake_menu_bottom_unified)
    monkeypatch.setattr(prompt_master_handler, "safe_send", fake_safe_send)
    monkeypatch.setattr(ctx.bot, "send_message", fake_safe_send)
    monkeypatch.setattr(ctx.bot, "edit_message_text", fake_safe_send)

    async def _run() -> None:
        await prompt_master_handler._upsert_card(update, ctx, engine="veo", state=state, lang="ru")

    asyncio.run(_run())

    assert calls
