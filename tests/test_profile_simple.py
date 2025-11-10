import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("FEATURE_PROFILE_SIMPLE", "true")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from unittest.mock import AsyncMock

from tests.suno_test_utils import FakeBot, bot_module  # noqa: E402
import handlers.profile_simple as profile_simple  # noqa: E402


def _make_context(bot: FakeBot) -> SimpleNamespace:
    return SimpleNamespace(bot=bot, chat_data={}, user_data={}, application=SimpleNamespace(bot_data={}))


def test_open_profile_sends_message_without_html_and_callbacks(monkeypatch):
    bot = FakeBot()
    ctx = _make_context(bot)

    monkeypatch.setattr(
        profile_simple,
        "get_user_balance_async",
        AsyncMock(return_value=123),
    )

    message = SimpleNamespace(chat=SimpleNamespace(id=101), chat_id=101)
    update = SimpleNamespace(
        effective_message=message,
        effective_chat=message.chat,
        effective_user=SimpleNamespace(id=101),
        callback_query=None,
    )

    asyncio.run(profile_simple.profile_open(update, ctx))

    assert bot.sent, "Profile open must send a message"
    payload = bot.sent[-1]
    assert payload.get("parse_mode") is None
    assert "Баланс: 123" in payload["text"]
    keyboard = payload["reply_markup"].inline_keyboard
    assert keyboard[0][0].callback_data == "btn:profile|view=topup"


def test_history_empty(monkeypatch):
    bot = FakeBot()
    ctx = _make_context(bot)

    monkeypatch.setattr(profile_simple, "get_history", lambda _uid: [])

    chat_id = 202
    ctx.chat_data["profile_card_message_id"] = 555

    answered = {"value": False}

    async def fake_answer():
        answered["value"] = True

    message = SimpleNamespace(chat=SimpleNamespace(id=chat_id), chat_id=chat_id, message_id=555)
    query = SimpleNamespace(data="btn:profile|view=history", message=message, answer=fake_answer)
    update = SimpleNamespace(
        callback_query=query,
        effective_chat=message.chat,
        effective_user=SimpleNamespace(id=chat_id),
        effective_message=message,
    )

    asyncio.run(profile_simple.profile_history(update, ctx))

    assert answered["value"], "Callback query should be answered"
    assert bot.deleted and bot.deleted[-1]["message_id"] == 555
    payload = bot.sent[-1]
    assert "История операций пока пуста." in payload["text"]
    assert payload["reply_markup"].inline_keyboard[0][0].callback_data == "btn:profile"


def test_invite_without_botname_fallback(monkeypatch):
    bot = FakeBot()
    ctx = _make_context(bot)

    monkeypatch.setattr(profile_simple.app_settings, "BOT_NAME", "")
    monkeypatch.setattr(profile_simple.app_settings, "BOT_USERNAME", "")

    answered = {"value": False}

    async def fake_answer():
        answered["value"] = True

    chat_id = 303
    message = SimpleNamespace(chat=SimpleNamespace(id=chat_id), chat_id=chat_id, message_id=10)
    query = SimpleNamespace(data="btn:profile|view=invite", message=message, answer=fake_answer)
    update = SimpleNamespace(
        callback_query=query,
        effective_chat=message.chat,
        effective_user=SimpleNamespace(id=chat_id),
        effective_message=message,
    )

    ctx.chat_data["profile_card_message_id"] = 10

    asyncio.run(profile_simple.profile_invite(update, ctx))

    assert answered["value"], "Callback query should be answered"
    payload = bot.sent[-1]
    assert "Скоро включим приглашения." in payload["text"]
    keyboard = payload["reply_markup"].inline_keyboard
    assert keyboard[0][0].callback_data == "btn:profile"
    assert "profile_card_message_id" not in ctx.chat_data


def test_topup_stub(monkeypatch):
    bot = FakeBot()
    ctx = _make_context(bot)

    async def fake_open(update, inner_ctx, *, edit_message, source):
        assert edit_message is True
        assert source == "profile"
        await inner_ctx.bot.send_message(
            chat_id=update.effective_chat.id,
            text="💎 Пополнение — скоро.",
        )

    monkeypatch.setattr(profile_simple, "open_stars_menu", fake_open)

    answered = {"value": False}

    async def fake_answer():
        answered["value"] = True

    chat_id = 404
    message = SimpleNamespace(chat=SimpleNamespace(id=chat_id), chat_id=chat_id, message_id=11)
    query = SimpleNamespace(data="btn:profile|view=topup", message=message, answer=fake_answer)
    update = SimpleNamespace(
        callback_query=query,
        effective_chat=message.chat,
        effective_user=SimpleNamespace(id=chat_id),
        effective_message=message,
    )

    asyncio.run(profile_simple.profile_topup(update, ctx))

    assert answered["value"], "Callback query should be answered"
    assert bot.sent[-1]["text"].startswith("💎 Пополнение — скоро.")


def test_back_returns_to_menu(monkeypatch):
    bot = FakeBot()
    ctx = _make_context(bot)

    called = {"value": False}

    async def fake_menu(update, inner_ctx, *, notify_chat_off):
        called["value"] = (update, inner_ctx, notify_chat_off)

    monkeypatch.setattr(bot_module, "handle_menu", fake_menu)

    answered = {"value": False}

    async def fake_answer():
        answered["value"] = True

    chat_id = 505
    ctx.chat_data["profile_card_message_id"] = 900

    message = SimpleNamespace(chat=SimpleNamespace(id=chat_id), chat_id=chat_id, message_id=900)
    query = SimpleNamespace(data="btn:profile|view=back", message=message, answer=fake_answer)
    update = SimpleNamespace(
        callback_query=query,
        effective_chat=message.chat,
        effective_user=SimpleNamespace(id=chat_id),
        effective_message=message,
    )

    asyncio.run(profile_simple.profile_back(update, ctx))

    assert answered["value"], "Callback query should be answered"
    assert called["value"] and called["value"][2] is False
    assert bot.deleted and bot.deleted[-1]["message_id"] == 900
    assert "profile_card_message_id" not in ctx.chat_data
