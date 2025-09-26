import asyncio
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import faq_content
from telegram import InlineKeyboardMarkup


@pytest.fixture
def bot_module(monkeypatch):
    monkeypatch.setenv("TELEGRAM_TOKEN", "test-token")
    monkeypatch.setenv("SUNO_API_BASE", "https://example.com")
    monkeypatch.setenv("SUNO_API_TOKEN", "dummy-token")
    monkeypatch.setenv("LEDGER_BACKEND", "memory")
    monkeypatch.setenv("DATABASE_URL", "postgres://test")
    module = importlib.import_module("bot")
    module = importlib.reload(module)
    return module


def test_faq_command_sends_root_menu(monkeypatch, bot_module):
    sends = []

    async def fake_safe_send(method, *, method_name, kind, **kwargs):
        sends.append({
            "method": method,
            "method_name": method_name,
            "kind": kind,
            "kwargs": kwargs,
        })

    async def fake_ensure_user_record(update):
        return None

    monkeypatch.setattr(bot_module, "tg_safe_send", fake_safe_send)
    monkeypatch.setattr(bot_module, "ensure_user_record", fake_ensure_user_record)

    chat_id = 777
    update = SimpleNamespace(effective_message=SimpleNamespace(chat_id=chat_id))
    ctx = SimpleNamespace(bot=SimpleNamespace(send_message=object()))

    asyncio.run(bot_module.faq_command(update, ctx))

    assert len(sends) == 1
    call = sends[0]
    assert call["method_name"] == "send_message"
    assert call["kind"] == "faq"

    payload = call["kwargs"]
    assert payload["chat_id"] == chat_id
    assert payload["parse_mode"] == bot_module.ParseMode.MARKDOWN_V2
    assert payload["text"] == bot_module._FAQ_ROOT_MESSAGE

    keyboard = payload["reply_markup"]
    assert isinstance(keyboard, InlineKeyboardMarkup)

    expected_layout = [
        ("🎬 Видео (VEO)", "faq:veo"),
        ("🎨 Изображения (MJ)", "faq:mj"),
        ("🧩 Banana", "faq:banana"),
        ("🎵 Музыка (Suno)", "faq:suno"),
        ("💎 Баланс и оплата", "faq:billing"),
        ("⚡ Токены и возвраты", "faq:refunds"),
        ("💬 Обычный чат", "faq:chat"),
        ("🧠 Prompt-Master", "faq:pm"),
        ("ℹ️ Общие вопросы", "faq:general"),
        ("⬅️ Назад (в главное меню бота)", "faq:home"),
    ]

    actual_layout = [
        (button.text, button.callback_data)
        for row in keyboard.inline_keyboard
        for button in row
    ]

    assert actual_layout == expected_layout


def test_faq_callback_section_edits_message(monkeypatch, bot_module):
    edits = []

    async def fake_ensure_user_record(update):
        return None

    async def fake_edit(edit_callable, text, **kwargs):
        edits.append({"text": text, "kwargs": kwargs})

    faq_views_total_mock = MagicMock()
    faq_views_total_handle = MagicMock()
    faq_views_total_mock.labels.return_value = faq_views_total_handle

    monkeypatch.setattr(bot_module, "ensure_user_record", fake_ensure_user_record)
    monkeypatch.setattr(bot_module, "_safe_edit_message_text", fake_edit)
    monkeypatch.setattr(bot_module, "faq_views_total", faq_views_total_mock)

    query = SimpleNamespace(
        data="faq:veo",
        message=SimpleNamespace(chat_id=42, message_id=100),
        edit_message_text=object(),
        answers=[],
    )

    async def fake_answer(text=None, show_alert=False):
        query.answers.append({"text": text, "show_alert": show_alert})

    query.answer = fake_answer

    update = SimpleNamespace(callback_query=query)
    ctx = SimpleNamespace(bot=SimpleNamespace())

    asyncio.run(bot_module.faq_callback_router(update, ctx))

    assert len(edits) == 1
    edit = edits[0]
    assert edit["text"] == bot_module._FAQ_SECTION_MESSAGES["veo"]

    kwargs = edit["kwargs"]
    assert kwargs["parse_mode"] == bot_module.ParseMode.MARKDOWN_V2
    assert kwargs["disable_web_page_preview"] is True

    reply_markup = kwargs["reply_markup"]
    assert isinstance(reply_markup, InlineKeyboardMarkup)
    assert reply_markup.inline_keyboard == faq_content.faq_back_kb().inline_keyboard

    faq_views_total_mock.labels.assert_called_once_with(section="veo", **bot_module._METRIC_LABELS)
    faq_views_total_handle.inc.assert_called_once()
    assert query.answers == [{"text": None, "show_alert": False}]


def test_faq_callback_root_returns_to_main_menu(monkeypatch, bot_module):
    edits = []

    async def fake_ensure_user_record(update):
        return None

    async def fake_edit(edit_callable, text, **kwargs):
        edits.append({"text": text, "kwargs": kwargs})

    faq_root_mock = MagicMock()
    faq_root_handle = MagicMock()
    faq_root_mock.labels.return_value = faq_root_handle

    monkeypatch.setattr(bot_module, "ensure_user_record", fake_ensure_user_record)
    monkeypatch.setattr(bot_module, "_safe_edit_message_text", fake_edit)
    monkeypatch.setattr(bot_module, "faq_root_views_total", faq_root_mock)

    query = SimpleNamespace(
        data="faq:root",
        message=SimpleNamespace(chat_id=55, message_id=200),
        edit_message_text=object(),
        answers=[],
    )

    async def fake_answer(text=None, show_alert=False):
        query.answers.append({"text": text, "show_alert": show_alert})

    query.answer = fake_answer

    update = SimpleNamespace(callback_query=query)
    ctx = SimpleNamespace(bot=SimpleNamespace())

    asyncio.run(bot_module.faq_callback_router(update, ctx))

    assert len(edits) == 1
    edit = edits[0]
    assert edit["text"] == bot_module._FAQ_ROOT_MESSAGE

    kwargs = edit["kwargs"]
    assert kwargs["parse_mode"] == bot_module.ParseMode.MARKDOWN_V2
    assert kwargs["disable_web_page_preview"] is True
    assert kwargs["reply_markup"].inline_keyboard == faq_content.faq_main_kb().inline_keyboard

    faq_root_mock.labels.assert_called_once_with(**bot_module._METRIC_LABELS)
    faq_root_handle.inc.assert_called_once()
    assert query.answers == [{"text": None, "show_alert": False}]


def test_faq_texts_are_escaped(bot_module):
    for key, section in faq_content.FAQ_SECTIONS.items():
        escaped = bot_module.md2_escape(section["text"])
        assert escaped == bot_module._FAQ_SECTION_MESSAGES[key]

    assert bot_module.md2_escape(faq_content.FAQ_ROOT_TEXT) == bot_module._FAQ_ROOT_MESSAGE
