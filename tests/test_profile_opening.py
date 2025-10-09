import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.suno_test_utils import FakeBot, bot_module
import handlers.profile as profile_handlers


def _make_context() -> SimpleNamespace:
    bot = FakeBot()
    ctx = SimpleNamespace(
        bot=bot,
        chat_data={},
        user_data={},
        application=SimpleNamespace(bot_data={}),
    )
    ctx._user_id_and_data = (999, {})
    return ctx


def test_quick_button_opens_profile_once(monkeypatch):
    ctx = _make_context()

    calls: list[dict[str, object]] = []

    async def fake_open_card(update, ctx: SimpleNamespace, *, suppress_nav, source):
        chat = getattr(update, "effective_chat", None)
        chat_id = getattr(chat, "id", None)
        user = getattr(update, "effective_user", None)
        user_id = getattr(user, "id", None)
        calls.append({
            "chat_id": chat_id,
            "user_id": user_id,
            "source": source,
            "suppress_nav": suppress_nav,
        })
        ctx.chat_data["profile_msg_id"] = 111
        return profile_handlers.OpenedProfile(msg_id=111, reused=False)

    monkeypatch.setattr(profile_handlers, "open_profile_card", fake_open_card)

    chat = SimpleNamespace(id=123)
    message = SimpleNamespace(chat=chat, chat_id=123, message_id=7)
    update = SimpleNamespace(
        effective_chat=chat,
        effective_user=SimpleNamespace(id=555),
        effective_message=message,
    )

    async def scenario():
        await profile_handlers.open_profile(update, ctx, source="quick")
        await profile_handlers.open_profile(update, ctx, source="quick")

    asyncio.run(scenario())

    assert len(calls) == 1
    assert calls[0]["source"] == "quick"
    assert ctx.chat_data.get("profile_msg_id") == 111


def test_menu_card_opens_profile_and_reuses_message(monkeypatch):
    ctx = _make_context()

    core_calls: list[dict[str, object]] = []

    async def fake_core_open(update, context, *, suppress_nav, edit, force_new):
        core_calls.append({"edit": edit, "force_new": force_new})
        return 222

    monkeypatch.setattr(bot_module, "open_profile_card", fake_core_open)

    chat = SimpleNamespace(id=321)
    message = SimpleNamespace(chat=chat, chat_id=321, message_id=42)
    update = SimpleNamespace(
        effective_chat=chat,
        effective_user=SimpleNamespace(id=888),
        effective_message=message,
    )

    async def scenario():
        await profile_handlers.open_profile(update, ctx, source="menu")
        ctx.chat_data["profile_open_at"] = 0.0
        await profile_handlers.open_profile(update, ctx, source="menu")

    asyncio.run(scenario())

    assert ctx.chat_data.get("profile_msg_id") == 222
    assert len(core_calls) == 2
    assert core_calls[0] == {"edit": False, "force_new": True}
    assert core_calls[1] == {"edit": True, "force_new": False}


def test_profile_internal_buttons_do_not_spawn_new_messages(monkeypatch):
    ctx = _make_context()
    ctx.chat_data["profile_msg_id"] = 777

    calls: list[dict[str, object]] = []

    async def fake_edit_card(_ctx, chat_id, message_id, payload):
        calls.append({"chat_id": chat_id, "message_id": message_id, "text": payload.get("text")})
        return True

    async def fake_history(_user_id):
        return []

    monkeypatch.setattr(profile_handlers, "_edit_card", fake_edit_card)
    monkeypatch.setattr(profile_handlers, "_billing_history", fake_history)
    monkeypatch.setattr(profile_handlers, "_bot_name", lambda: "mybot")

    chat = SimpleNamespace(id=555)
    message = SimpleNamespace(chat=chat, chat_id=555, message_id=777)

    async def answer(_text=None):
        return None

    query = SimpleNamespace(message=message, answer=answer)
    update = SimpleNamespace(
        callback_query=query,
        effective_chat=chat,
        effective_user=SimpleNamespace(id=111, username="tester"),
        effective_message=message,
    )

    async def scenario():
        await profile_handlers.on_profile_topup(update, ctx)
        await profile_handlers.on_profile_history(update, ctx)
        await profile_handlers.on_profile_invite(update, ctx)
        await profile_handlers.on_profile_promo_start(update, ctx)
        await profile_handlers.on_profile_back(update, ctx)

    asyncio.run(scenario())

    assert all(call["message_id"] == 777 for call in calls)
    assert not ctx.bot.sent


def test_no_dialog_disabled_on_navigation(monkeypatch):
    ctx = _make_context()
    ctx.chat_data["suppress_dialog_notice"] = True
    state = bot_module.state(ctx)
    state["mode"] = "chat"

    asyncio.run(bot_module.reset_user_state(ctx, chat_id=123, notify_chat_off=True))

    assert not any(entry.get("text") == "🛑 Режим диалога отключён." for entry in ctx.bot.sent)


def test_dialog_disabled_only_after_plain_chat_exit():
    ctx = _make_context()
    ctx.chat_data["just_exited_plain_chat"] = True
    state = bot_module.state(ctx)
    state["mode"] = "chat"

    asyncio.run(bot_module.reset_user_state(ctx, chat_id=123, notify_chat_off=True))

    assert sum(1 for entry in ctx.bot.sent if entry.get("text") == "🛑 Режим диалога отключён.") == 1

    state = bot_module.state(ctx)
    state["mode"] = "chat"

    asyncio.run(bot_module.reset_user_state(ctx, chat_id=123, notify_chat_off=True))

    assert sum(1 for entry in ctx.bot.sent if entry.get("text") == "🛑 Режим диалога отключён.") == 1
