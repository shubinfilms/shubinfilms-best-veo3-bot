import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.suno_test_utils import FakeBot
import handlers.profile as profile_handlers
from utils.input_state import WaitKind


def _build_callback_update(*, chat_id: int, message_id: int, user_id: int):
    async def fake_answer():
        return None

    message = SimpleNamespace(chat=SimpleNamespace(id=chat_id), chat_id=chat_id, message_id=message_id)
    query = SimpleNamespace(message=message, answer=fake_answer)
    update = SimpleNamespace(
        callback_query=query,
        effective_chat=message.chat,
        effective_user=SimpleNamespace(id=user_id),
    )
    return update


def test_history_empty(monkeypatch):
    bot = FakeBot()
    ctx = SimpleNamespace(
        bot=bot,
        chat_data={},
        user_data={},
        application=SimpleNamespace(bot_data={}),
    )

    async def fake_history(_uid):
        return []

    monkeypatch.setattr(profile_handlers, "_billing_history", fake_history)

    captured: dict[str, dict] = {}

    async def fake_edit(ctx_obj, chat_id, message_id, payload):
        captured["payload"] = payload
        ctx_obj.chat_data["nav_in_progress"] = ctx_obj.chat_data.get("nav_in_progress", False)
        return True

    monkeypatch.setattr(profile_handlers, "_edit_card", fake_edit)

    update = _build_callback_update(chat_id=10, message_id=77, user_id=123)

    asyncio.run(profile_handlers.on_profile_history(update, ctx))

    payload = captured.get("payload")
    assert payload is not None
    assert "История операций пока пуста" in payload["text"]
    assert not any(entry.get("text") == "🛑 Режим диалога отключён." for entry in bot.sent)


def test_invite_link_logged(monkeypatch, caplog):
    bot = FakeBot()
    ctx = SimpleNamespace(
        bot=bot,
        chat_data={},
        user_data={},
        application=SimpleNamespace(bot_data={}),
    )

    monkeypatch.setattr(profile_handlers, "_bot_name", lambda: "ExampleBot")

    captured: dict[str, dict] = {}

    async def fake_edit(ctx_obj, chat_id, message_id, payload):
        captured["payload"] = payload
        return True

    monkeypatch.setattr(profile_handlers, "_edit_card", fake_edit)

    update = _build_callback_update(chat_id=55, message_id=88, user_id=900)

    with caplog.at_level("INFO"):
        asyncio.run(profile_handlers.on_profile_invite(update, ctx))

    payload = captured.get("payload")
    assert payload is not None
    assert "https://t.me/ExampleBot?start=ref_900" in payload["text"]
    markup = payload.get("reply_markup")
    buttons = getattr(markup, "inline_keyboard", []) if markup else []
    assert any(
        getattr(button, "url", None) == "https://t.me/ExampleBot?start=ref_900"
        for row in buttons
        for button in row
    )
    assert not bot.sent
    assert any("profile.invite_link" in record.getMessage() for record in caplog.records)


def test_topup_stub(monkeypatch):
    bot = FakeBot()
    ctx = SimpleNamespace(
        bot=bot,
        chat_data={},
        user_data={},
        application=SimpleNamespace(bot_data={}),
    )

    captured: dict[str, dict] = {}

    async def fake_edit(ctx_obj, chat_id, message_id, payload):
        captured["payload"] = payload
        return True

    monkeypatch.setattr(profile_handlers, "_edit_card", fake_edit)

    update = _build_callback_update(chat_id=33, message_id=44, user_id=101)

    asyncio.run(profile_handlers.on_profile_topup(update, ctx))

    payload = captured.get("payload")
    assert payload is not None
    assert "Пополнение — в разработке." in payload["text"]
    assert not bot.sent


def test_promo_ask_code_sets_wait(monkeypatch):
    bot = FakeBot()
    ctx = SimpleNamespace(
        bot=bot,
        chat_data={},
        user_data={},
        application=SimpleNamespace(bot_data={}),
    )

    calls: list[tuple[int, object, int]] = []

    async def fake_edit(ctx_obj, chat_id, message_id, payload):
        ctx_obj.chat_data.setdefault("payload", payload)
        return True

    def fake_set_wait_state(user_id, state, *, ttl_seconds):
        calls.append((user_id, state, ttl_seconds))

    monkeypatch.setattr(profile_handlers, "_edit_card", fake_edit)
    monkeypatch.setattr(profile_handlers, "set_wait_state", fake_set_wait_state)

    update = _build_callback_update(chat_id=71, message_id=19, user_id=250)

    asyncio.run(profile_handlers.on_profile_promo_start(update, ctx))

    assert calls, "wait state was not set"
    assert calls[0][0] == 250
    assert calls[0][1].kind == WaitKind.PROMO_CODE
    assert ctx.chat_data.get(profile_handlers._PROMO_WAIT_KEY) == profile_handlers.PROMO_WAIT_KIND
    assert not any(entry.get("text") == "🛑 Режим диалога отключён." for entry in bot.sent)
