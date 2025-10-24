import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bot
import handlers.profile as profile


def test_profile_root_first_paint_is_lazy(monkeypatch):
    send_called = asyncio.Event()
    referral_called = asyncio.Event()
    referral_completed = asyncio.Event()
    edit_calls: list[dict] = []

    async def fake_build_referral_link(user_id, ctx):
        referral_called.set()
        await send_called.wait()
        referral_completed.set()
        return "https://t.me/bot?start=ref"

    async def fake_profile_update_or_send(update, ctx, text, markup):
        send_called.set()
        return SimpleNamespace(message_id=123)

    async def fake_edit_markup(chat_id, message_id, reply_markup):
        edit_calls.append({"chat_id": chat_id, "message_id": message_id})
        return None

    monkeypatch.setattr(bot, "_build_referral_link", fake_build_referral_link)
    monkeypatch.setattr(profile, "profile_update_or_send", fake_profile_update_or_send)
    monkeypatch.setattr(
        bot,
        "balance_menu_kb",
        lambda referral_url=None: SimpleNamespace(inline_keyboard=[["btn"]]),
    )

    class DummyBot:
        async def edit_message_reply_markup(self, chat_id, message_id, reply_markup):
            await fake_edit_markup(chat_id, message_id, reply_markup)

    ctx = SimpleNamespace(bot=DummyBot(), chat_data={}, application=SimpleNamespace(logger=None))

    async def fake_query_answer(*args, **kwargs):
        return "ok"

    query = SimpleNamespace(
        data="btn:profile",
        message=SimpleNamespace(chat=SimpleNamespace(id=111), message_id=222),
        answer=fake_query_answer,
    )
    update = SimpleNamespace(
        callback_query=query,
        effective_chat=query.message.chat,
        effective_message=query.message,
        effective_user=SimpleNamespace(id=333),
    )

    async def scenario():
        await profile.handle_profile_view(update, ctx, "root")

    asyncio.run(scenario())

    assert send_called.is_set(), "profile send should complete"
    assert referral_called.is_set(), "referral fetch should start"
    assert referral_completed.is_set(), "referral fetch should finish"
    assert edit_calls, "referral update should edit markup"
