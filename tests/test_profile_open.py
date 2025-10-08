import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.suno_test_utils import FakeBot, bot_module
import handlers.profile as profile_handlers


def test_open_from_inline_no_dialog_notice(monkeypatch):
    bot = FakeBot()
    ctx = SimpleNamespace(
        bot=bot,
        chat_data={},
        user_data={},
        application=SimpleNamespace(bot_data={}),
    )
    ctx._user_id_and_data = (555, {})

    async def fake_core_open(update, context, *, suppress_nav, edit, force_new):
        context.chat_data["profile_msg_id"] = 200
        return 200

    monkeypatch.setattr(bot_module, "open_profile_card", fake_core_open)

    async def fake_answer():
        return None

    message = SimpleNamespace(chat=SimpleNamespace(id=321), chat_id=321, message_id=10)
    query = SimpleNamespace(message=message, from_user=SimpleNamespace(id=555), answer=fake_answer)
    update = SimpleNamespace(
        callback_query=query,
        effective_chat=message.chat,
        effective_user=query.from_user,
        effective_message=message,
    )

    asyncio.run(profile_handlers.on_profile_menu(update, ctx))

    assert ctx.chat_data.get("profile_msg_id") == 200
    assert not any(entry.get("text") == "🛑 Режим диалога отключён." for entry in bot.sent)
    assert getattr(ctx, "nav_event", False) is False


def test_open_from_quick_no_dialog_notice(monkeypatch):
    bot = FakeBot()
    ctx = SimpleNamespace(
        bot=bot,
        chat_data={},
        user_data={},
        application=SimpleNamespace(bot_data={}),
    )
    ctx._user_id_and_data = (777, {})

    calls: list[dict] = []

    async def fake_helper(
        chat_id,
        user_id,
        *,
        suppress_nav,
        update,
        ctx,
        source,
    ):
        calls.append(
            {
                "chat_id": chat_id,
                "user_id": user_id,
                "suppress_nav": suppress_nav,
                "source": source,
                "nav": getattr(ctx, "nav_event", False),
            }
        )
        ctx.chat_data["profile_msg_id"] = 300
        return profile_handlers.OpenedProfile(msg_id=300, reused=True)

    async def fake_disable(*_args, **_kwargs):
        return False

    async def fake_ensure(_update):
        return None

    monkeypatch.setattr(profile_handlers, "open_profile_card", fake_helper)
    monkeypatch.setattr(bot_module, "disable_chat_mode", fake_disable)
    monkeypatch.setattr(bot_module, "ensure_user_record", fake_ensure)

    message = SimpleNamespace(
        text="Профиль",
        chat_id=100,
        chat=SimpleNamespace(id=100),
    )
    message.replies = []

    async def reply_text(text, **_kwargs):
        message.replies.append(text)

    message.reply_text = reply_text

    update = SimpleNamespace(
        message=message,
        effective_message=message,
        effective_user=SimpleNamespace(id=777),
        effective_chat=message.chat,
    )

    asyncio.run(bot_module.on_text(update, ctx))

    assert calls and calls[0]["source"] == "quick"
    assert calls[0]["suppress_nav"] is True
    assert calls[0]["user_id"] == 777
    assert calls[0]["nav"] is True
    assert ctx.chat_data.get("profile_msg_id") == 300
    assert not any(entry.get("text") == "🛑 Режим диалога отключён." for entry in bot.sent)


def test_single_render_no_duplicates(monkeypatch):
    bot = FakeBot()
    ctx = SimpleNamespace(
        bot=bot,
        chat_data={},
        user_data={},
        application=SimpleNamespace(bot_data={}),
    )
    ctx._user_id_and_data = (404, {})

    chat = SimpleNamespace(id=999)
    message = SimpleNamespace(chat=chat, chat_id=chat.id, message_id=42)

    async def fake_core_open(update, context, *, suppress_nav, edit, force_new):
        if "profile_msg_id" in context.chat_data:
            return context.chat_data["profile_msg_id"]
        sent = await context.bot.send_message(chat_id=chat.id, text="profile")
        context.chat_data["profile_msg_id"] = sent.message_id
        return sent.message_id

    monkeypatch.setattr(bot_module, "open_profile_card", fake_core_open)

    update = SimpleNamespace(
        effective_chat=chat,
        effective_message=message,
        effective_user=SimpleNamespace(id=404),
        callback_query=None,
    )

    async def scenario():
        first = await profile_handlers.open_profile_card(
            chat.id,
            404,
            update=update,
            ctx=ctx,
            suppress_nav=True,
            source="inline",
        )
        second = await profile_handlers.open_profile_card(
            chat.id,
            404,
            update=update,
            ctx=ctx,
            suppress_nav=True,
            source="inline",
        )
        assert first.msg_id == second.msg_id == ctx.chat_data.get("profile_msg_id")

    asyncio.run(scenario())

    assert len(bot.sent) == 1
