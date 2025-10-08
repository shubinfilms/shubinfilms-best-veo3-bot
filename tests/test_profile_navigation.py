import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

import hub_router
from tests.suno_test_utils import FakeBot, bot_module
import handlers.profile as profile_handlers


def test_profile_open_no_duplicates(monkeypatch):
    bot = FakeBot()
    ctx = SimpleNamespace(
        bot=bot,
        chat_data={},
        user_data={},
        application=SimpleNamespace(bot_data={}),
    )
    ctx._user_id_and_data = (777, {})  # satisfy get_user_id

    snapshot = SimpleNamespace(value=100, display="100", warning=None)

    async def fake_referral(user_id, context):  # pragma: no cover - helper
        return None

    async def fake_edit_message(
        ctx_param,
        chat_id_param,
        message_id_param,
        text_param,
        reply_markup_param,
        **kwargs,
    ) -> bool:
        bot.edited.append(
            {
                "chat_id": chat_id_param,
                "message_id": message_id_param,
                "text": text_param,
                "reply_markup": reply_markup_param,
            }
        )
        return True

    monkeypatch.setattr(bot_module, "_resolve_balance_snapshot", lambda *_args, **_kwargs: snapshot)
    monkeypatch.setattr(bot_module, "_build_referral_link", fake_referral)
    monkeypatch.setattr(bot_module, "safe_edit_message", fake_edit_message)
    monkeypatch.setattr(
        bot_module,
        "balance_menu_kb",
        lambda **_: InlineKeyboardMarkup(
            [[InlineKeyboardButton("Пополнить", callback_data="profile:topup")]]
        ),
    )

    chat = SimpleNamespace(id=123)
    message = SimpleNamespace(chat=chat, chat_id=chat.id, message_id=55)
    update = SimpleNamespace(
        effective_chat=chat,
        effective_message=message,
        effective_user=SimpleNamespace(id=777),
        callback_query=None,
    )

    async def _scenario():
        first_mid = await bot_module.open_profile_card(update, ctx)
        assert first_mid is not None
        assert ctx.chat_data.get("profile_msg_id") == first_mid
        assert len(bot.sent) == 1

        second_mid = await bot_module.open_profile_card(update, ctx)
        assert second_mid == first_mid
        assert len(bot.sent) == 1
        assert ctx.chat_data.get("profile_msg_id") == first_mid

    asyncio.run(_scenario())


def test_profile_buttons_route(monkeypatch):
    bot = FakeBot()
    ctx = SimpleNamespace(
        bot=bot,
        application=SimpleNamespace(bot_data={}),
        chat_data={},
        user_data={},
    )

    calls: list[str] = []

    def _recorder(label: str):
        async def _handler(update, context):  # pragma: no cover - helper
            calls.append(label)

        return _handler

    monkeypatch.setattr(profile_handlers, "on_profile_topup", _recorder("topup"))
    monkeypatch.setattr(profile_handlers, "on_profile_history", _recorder("history"))
    monkeypatch.setattr(profile_handlers, "on_profile_invite", _recorder("invite"))
    monkeypatch.setattr(profile_handlers, "on_profile_promo_start", _recorder("promo"))
    monkeypatch.setattr(profile_handlers, "on_profile_menu", _recorder("menu"))

    async def fake_answer():
        return None

    message = SimpleNamespace(chat=SimpleNamespace(id=900), message_id=10)
    user = SimpleNamespace(id=501)

    async def _scenario():
        for payload, label in [
            ("profile:topup", "topup"),
            ("profile:history", "history"),
            ("profile:invite", "invite"),
            ("profile:promo", "promo"),
            ("profile:menu", "menu"),
        ]:
            calls.clear()
            query = SimpleNamespace(
                data=payload,
                message=message,
                from_user=user,
                answer=fake_answer,
            )
            update = SimpleNamespace(
                callback_query=query,
                effective_chat=message.chat,
                effective_user=user,
            )
            await hub_router.hub_router(update, ctx)
            assert calls == [label]

    asyncio.run(_scenario())


def test_nav_suppresses_dialog_notice(monkeypatch):
    bot = FakeBot()
    ctx = SimpleNamespace(bot=bot, chat_data={"nav_in_progress": True}, user_data={})

    async def fake_ensure(update):  # pragma: no cover - helper
        return None

    monkeypatch.setattr(bot_module, "ensure_user_record", fake_ensure)
    monkeypatch.setattr(profile_handlers, "is_waiting_for_promo", lambda _ctx: False)

    message = SimpleNamespace(text="произвольный текст", chat_id=777, chat=SimpleNamespace(id=777))
    update = SimpleNamespace(message=message, effective_message=message, effective_user=None)

    asyncio.run(bot_module.on_text(update, ctx))

    assert ctx.chat_data.get("nav_in_progress") is False
    assert not bot.sent
