import asyncio
import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from telegram.error import BadRequest

import handlers.profile as profile_handlers
from handlers.profile import NAV_UNTIL, PROFILE_MSG_ID
from tests.suno_test_utils import FakeBot, bot_module


class EditFailBot(FakeBot):
    async def edit_message_text(self, **kwargs):  # type: ignore[override]
        raise BadRequest("Message to edit not found")


class NotModifiedBot(FakeBot):
    async def edit_message_text(self, **kwargs):  # type: ignore[override]
        self.edited.append(kwargs)
        raise BadRequest("Message is not modified")


def _ctx_with_bot(bot):
    return SimpleNamespace(
        bot=bot,
        chat_data={},
        user_data={},
        application=SimpleNamespace(bot_data={}),
    )


def _make_callback_update(data: str, *, chat_id: int = 700, message_id: int = 400, user_id: int = 900):
    message = SimpleNamespace(
        chat=SimpleNamespace(id=chat_id),
        chat_id=chat_id,
        message_id=message_id,
    )

    async def answer(*_args, **_kwargs):
        return None

    query = SimpleNamespace(
        data=data,
        message=message,
        from_user=SimpleNamespace(id=user_id),
        answer=answer,
    )

    return SimpleNamespace(
        callback_query=query,
        effective_chat=message.chat,
        effective_message=message,
        effective_user=query.from_user,
    )


def test_profile_open_fallback_on_missing_msg(caplog):
    bot = EditFailBot()
    ctx = _ctx_with_bot(bot)
    ctx.chat_data[PROFILE_MSG_ID] = 123

    payload = {"text": "profile", "reply_markup": None, "parse_mode": None, "disable_web_page_preview": True}

    with caplog.at_level(logging.INFO):
        asyncio.run(profile_handlers.safe_send_or_edit_profile(ctx, 111, payload))

    assert len(bot.sent) == 1
    assert ctx.chat_data[PROFILE_MSG_ID] != 123
    assert any("profile.card.sent" in record.message for record in caplog.records)


def test_profile_edit_then_fallback_on_error(caplog):
    bot = NotModifiedBot()
    ctx = _ctx_with_bot(bot)
    ctx.chat_data[PROFILE_MSG_ID] = 555

    payload = {"text": "profile", "reply_markup": None, "parse_mode": None, "disable_web_page_preview": True}

    with caplog.at_level(logging.INFO):
        asyncio.run(profile_handlers.safe_send_or_edit_profile(ctx, 222, payload))

    assert len(bot.edited) == 1
    assert len(bot.sent) == 1
    assert ctx.chat_data[PROFILE_MSG_ID] != 555
    assert any("profile.card.edit_failed" in record.message for record in caplog.records)
    assert any("profile.card.sent" in record.message for record in caplog.records)


def test_profile_button_priority_over_generic_text(monkeypatch):
    bot = FakeBot()
    ctx = _ctx_with_bot(bot)
    ctx._user_id_and_data = (321, {})

    calls = []

    async def fake_open(update, context, *, source: str, suppress_nav: bool):
        calls.append({
            "source": source,
            "suppress_nav": suppress_nav,
            "nav_flag": context.chat_data.get("nav_event"),
        })
        context.chat_data[PROFILE_MSG_ID] = 777
        return profile_handlers.OpenedProfile(msg_id=777, reused=False)

    async def fake_disable(*_args, **_kwargs):
        return None

    async def fake_ensure(_update):
        return None

    monkeypatch.setattr(profile_handlers, "open_profile_card", fake_open)
    monkeypatch.setattr(bot_module, "disable_chat_mode", fake_disable)
    monkeypatch.setattr(bot_module, "ensure_user_record", fake_ensure)

    message = SimpleNamespace(
        text="Профиль",
        chat_id=321,
        chat=SimpleNamespace(id=321),
        replies=[],
    )

    async def reply_text(text, **_kwargs):
        message.replies.append(text)

    message.reply_text = reply_text

    update = SimpleNamespace(
        message=message,
        effective_message=message,
        effective_user=SimpleNamespace(id=321),
        effective_chat=message.chat,
    )

    asyncio.run(bot_module.on_text(update, ctx))

    assert calls, "profile handler was not invoked"
    assert calls[0]["source"] == "quick"
    assert calls[0]["suppress_nav"] is True
    assert ctx.chat_data.get(PROFILE_MSG_ID) == 777
    assert not any(entry.get("text") == "🛑 Режим диалога отключён." for entry in bot.sent)


def test_profile_callbacks_all_actions(monkeypatch, caplog):
    bot = FakeBot()
    ctx = _ctx_with_bot(bot)
    ctx.chat_data[PROFILE_MSG_ID] = 400

    captured = []

    async def fake_edit_card(_ctx, chat_id, message_id, payload):
        captured.append({
            "chat_id": chat_id,
            "message_id": message_id,
            "payload": payload,
        })
        return True

    monkeypatch.setattr(profile_handlers, "_edit_card", fake_edit_card)

    async def fake_history(_user_id):
        return [
            {"created_at": int(time.time()), "type": "credit", "amount": 10},
            {"created_at": int(time.time()), "type": "debit", "amount": -3},
        ]

    monkeypatch.setattr(profile_handlers, "_billing_history", fake_history)

    def fake_activate(ctx_obj, user_id, chat_id, message_id):
        ctx_obj.chat_data["promo_wait"] = {
            "user": user_id,
            "chat": chat_id,
            "message": message_id,
        }

    monkeypatch.setattr(profile_handlers, "_activate_promo_wait", fake_activate)

    # Topup with URL
    monkeypatch.setattr(profile_handlers, "_topup_url", lambda: "https://pay.example")
    update = _make_callback_update("profile:topup")
    asyncio.run(profile_handlers.on_profile_topup(update, ctx))
    assert captured and captured[-1]["payload"]["reply_markup"].inline_keyboard[0][0].url == "https://pay.example"

    # Topup without URL logs warning
    monkeypatch.setattr(profile_handlers, "_topup_url", lambda: "")
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        asyncio.run(profile_handlers.on_profile_topup(_make_callback_update("profile:topup"), ctx))
    assert any("profile.topup.no_url" in record.message for record in caplog.records)

    # History renders entries
    asyncio.run(profile_handlers.on_profile_history(_make_callback_update("profile:history"), ctx))
    history_payload = captured[-1]["payload"]
    assert "История" in history_payload["text"]

    # Invite with bot name
    monkeypatch.setattr(profile_handlers, "_bot_name", lambda: "TestBot")
    asyncio.run(profile_handlers.on_profile_invite(_make_callback_update("profile:invite"), ctx))
    invite_markup = captured[-1]["payload"]["reply_markup"].inline_keyboard[0][0]
    assert invite_markup.url and "TestBot" in invite_markup.url

    # Invite without bot name triggers warning
    monkeypatch.setattr(profile_handlers, "_bot_name", lambda: None)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        asyncio.run(profile_handlers.on_profile_invite(_make_callback_update("profile:invite"), ctx))
    assert any("profile.invite.no_bot_name" in record.message for record in caplog.records)

    # Promo card sets wait state
    asyncio.run(profile_handlers.on_profile_promo_start(_make_callback_update("profile:promo"), ctx))
    assert ctx.chat_data.get("promo_wait")

    # Back clears cached message id
    ctx.chat_data[PROFILE_MSG_ID] = 999
    asyncio.run(profile_handlers.on_profile_back(_make_callback_update("profile:back"), ctx))
    assert PROFILE_MSG_ID not in ctx.chat_data


def test_nav_suppression_flag():
    bot = FakeBot()
    ctx = _ctx_with_bot(bot)
    ctx.chat_data[NAV_UNTIL] = time.monotonic() + 1.5

    message = SimpleNamespace(
        text="hello",
        chat_id=123,
        chat=SimpleNamespace(id=123),
        replies=[],
    )

    async def reply_text(text, **_kwargs):
        message.replies.append(text)

    message.reply_text = reply_text

    update = SimpleNamespace(
        message=message,
        effective_message=message,
        effective_chat=message.chat,
        effective_user=SimpleNamespace(id=555),
    )

    asyncio.run(bot_module.on_text(update, ctx))

    assert not any(entry.get("text") == "🛑 Режим диалога отключён." for entry in bot.sent)
