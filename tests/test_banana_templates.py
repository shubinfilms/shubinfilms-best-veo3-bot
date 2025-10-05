import asyncio
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from telegram import InlineKeyboardMarkup

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def bot_module(monkeypatch):
    monkeypatch.setenv("TELEGRAM_TOKEN", "dummy-token")
    monkeypatch.setenv("SUNO_API_BASE", "https://example.com")
    monkeypatch.setenv("SUNO_API_TOKEN", "test-token")
    monkeypatch.setenv("LEDGER_BACKEND", "memory")
    monkeypatch.setenv("DATABASE_URL", "postgres://test")
    module = importlib.import_module("bot")
    return importlib.reload(module)


def _build_ctx():
    return SimpleNamespace(bot=None, user_data={})


def _build_update(data: str, chat_id: int = 777):
    answers: list[dict[str, object]] = []

    async def _answer(text: str | None = None, **kwargs):
        payload: dict[str, object] = {}
        if text is not None:
            payload["text"] = text
        payload.update(kwargs)
        answers.append(payload)

    async def _edit_message_text(*args, **kwargs):
        return None

    message = SimpleNamespace(chat_id=chat_id)
    query = SimpleNamespace(
        data=data,
        message=message,
        answer=_answer,
        edit_message_text=_edit_message_text,
    )
    update = SimpleNamespace(
        callback_query=query,
        effective_chat=SimpleNamespace(id=chat_id),
        effective_user=SimpleNamespace(id=42),
        _answers=answers,
    )
    return update


def test_banana_templates_menu(monkeypatch, bot_module):
    ctx = _build_ctx()
    update = _build_update("banana_templates")
    state_dict: dict[str, object] = {"_last_text_banana": "cached"}

    edits: list[dict[str, object]] = []

    async def fake_ensure_user_record(_update):
        return None

    async def fake_safe_edit(_callable, text, **kwargs):
        edits.append({"text": text, "reply_markup": kwargs.get("reply_markup")})

    monkeypatch.setattr(bot_module, "ensure_user_record", fake_ensure_user_record)
    monkeypatch.setattr(bot_module, "state", lambda _ctx: state_dict)
    monkeypatch.setattr(bot_module, "_safe_edit_message_text", fake_safe_edit)

    asyncio.run(bot_module.on_callback(update, ctx))

    assert not state_dict.get("_last_text_banana")
    assert edits, "expected edit to be performed"
    payload = edits[-1]
    assert "Готовые шаблоны" in payload["text"]
    markup = payload["reply_markup"]
    assert isinstance(markup, InlineKeyboardMarkup)
    buttons = [button.callback_data for row in markup.inline_keyboard for button in row]
    assert "btpl_bg_remove" in buttons
    assert "banana_back_to_card" in buttons
    assert update._answers and update._answers[-1] == {}


def test_banana_template_apply_sets_prompt(monkeypatch, bot_module):
    ctx = _build_ctx()
    update = _build_update("btpl_bg_studio")
    state_dict: dict[str, object] = {"last_prompt": "", "_last_text_banana": "cached"}

    async def fake_ensure_user_record(_update):
        return None

    called: list[tuple[int, dict[str, object]]] = []

    async def fake_show_banana_card(chat_id, _ctx, *, force_new=False):
        called.append((chat_id, {"force_new": force_new}))

    monkeypatch.setattr(bot_module, "ensure_user_record", fake_ensure_user_record)
    monkeypatch.setattr(bot_module, "state", lambda _ctx: state_dict)
    monkeypatch.setattr(bot_module, "show_banana_card", fake_show_banana_card)

    asyncio.run(bot_module.on_callback(update, ctx))

    assert state_dict["last_prompt"] == "замени фон на студийный (чистая белая/серая подложка)"
    assert not state_dict.get("_last_text_banana")
    assert called and called[-1][0] == update.effective_chat.id
    assert called[-1][1]["force_new"] is True
    assert update._answers and update._answers[-1]["text"] == "Шаблон подставлен ✅"


def test_banana_templates_back_to_card(monkeypatch, bot_module):
    ctx = _build_ctx()
    update = _build_update("banana_back_to_card")
    state_dict: dict[str, object] = {"_last_text_banana": "cached"}

    async def fake_ensure_user_record(_update):
        return None

    calls: list[tuple[int, dict[str, object]]] = []

    async def fake_show_banana_card(chat_id, _ctx, *, force_new=False):
        calls.append((chat_id, {"force_new": force_new}))

    monkeypatch.setattr(bot_module, "ensure_user_record", fake_ensure_user_record)
    monkeypatch.setattr(bot_module, "state", lambda _ctx: state_dict)
    monkeypatch.setattr(bot_module, "show_banana_card", fake_show_banana_card)

    asyncio.run(bot_module.on_callback(update, ctx))

    assert not state_dict.get("_last_text_banana")
    assert calls and calls[-1][0] == update.effective_chat.id
    assert calls[-1][1]["force_new"] is True
    assert update._answers and update._answers[-1] == {}
