from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.suno_test_utils import FakeBot  # noqa: E402
import handlers.profile as profile_handlers  # noqa: E402


def _build_update(chat_id: int, message_id: int, user_id: int):
    async def fake_answer():
        return None

    message = SimpleNamespace(
        chat=SimpleNamespace(id=chat_id),
        chat_id=chat_id,
        message_id=message_id,
    )
    query = SimpleNamespace(
        message=message,
        from_user=SimpleNamespace(id=user_id),
        answer=fake_answer,
    )
    update = SimpleNamespace(
        callback_query=query,
        effective_chat=message.chat,
        effective_user=query.from_user,
    )
    return update


def test_inner_buttons_edit_same_message(monkeypatch):
    bot = FakeBot()
    ctx = SimpleNamespace(
        bot=bot,
        chat_data={"profile_msg_id": 900},
        user_data={},
        application=SimpleNamespace(bot_data={}),
    )

    edits: list[dict[str, int]] = []

    async def fake_edit(ctx_obj, chat_id, message_id, payload):
        edits.append({"chat_id": chat_id, "message_id": message_id, "title": payload.get("text", "")})
        return True

    monkeypatch.setattr("handlers.profile._edit_card", fake_edit)
    async def fake_history(_uid):
        return []

    monkeypatch.setattr(profile_handlers, "_billing_history", fake_history)
    monkeypatch.setattr(profile_handlers, "_topup_url", lambda: None)
    monkeypatch.setattr(profile_handlers, "_bot_name", lambda: "ExampleBot")
    monkeypatch.setattr(profile_handlers, "set_wait_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(profile_handlers, "_activate_promo_wait", lambda *args, **kwargs: None)
    monkeypatch.setattr(profile_handlers, "clear_promo_wait", lambda _ctx: None)

    update = _build_update(chat_id=100, message_id=900, user_id=500)

    asyncio.run(profile_handlers.on_profile_topup(update, ctx))
    asyncio.run(profile_handlers.on_profile_history(update, ctx))
    asyncio.run(profile_handlers.on_profile_promo_start(update, ctx))

    monkeypatch.setattr(
        "handlers.menu.build_main_menu_card",
        lambda: {"text": "Main", "reply_markup": None},
    )
    asyncio.run(profile_handlers.on_profile_back(update, ctx))

    assert edits, "no edits performed"
    assert all(entry["message_id"] == 900 for entry in edits)
    assert ctx.chat_data.get("profile_msg_id") is None
    assert not bot.sent
