import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import BadRequest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import handlers.profile as profile_handlers


class DummyBot:
    def __init__(self, *, edit_exception: Exception | None = None):
        self.edit_calls: list[dict] = []
        self.send_calls: list[dict] = []
        self._edit_exception = edit_exception
        self._next_message_id = 1000

    async def edit_message_text(self, **kwargs):  # type: ignore[override]
        self.edit_calls.append(kwargs)
        if self._edit_exception is not None:
            raise self._edit_exception
        return SimpleNamespace(message_id=self._next_message_id, chat_id=kwargs.get("chat_id"))

    async def send_message(self, *args, **kwargs):  # type: ignore[override]
        self.send_calls.append({"args": args, "kwargs": kwargs})
        self._next_message_id += 1
        return SimpleNamespace(message_id=self._next_message_id, chat_id=kwargs.get("chat_id"))


def _make_ctx(bot: DummyBot | None = None):
    if bot is None:
        bot = DummyBot()
    return SimpleNamespace(
        bot=bot,
        chat_data={},
        user_data={},
        application=SimpleNamespace(bot_data={}),
    )


def _make_callback_update(view: str, *, chat_id: int = 1, message_id: int = 10, user_id: int = 20):
    async def answer(*_args, **_kwargs):
        return None

    message = SimpleNamespace(chat=SimpleNamespace(id=chat_id), chat_id=chat_id, message_id=message_id)
    query = SimpleNamespace(data=f"profile:{view}", message=message, answer=answer)
    return SimpleNamespace(
        callback_query=query,
        effective_chat=message.chat,
        effective_message=message,
        effective_user=SimpleNamespace(id=user_id),
    )


def test_profile_callbacks_route_all_views(monkeypatch):
    rendered: list[str] = []
    updated: list[str] = []

    async def fake_prepare(update, ctx):
        return {"snapshot": SimpleNamespace(display="10💎", warning=None, value=10), "snapshot_target": 1, "referral_url": None, "chat_id": 1}

    async def fake_history(_uid):
        return []

    monkeypatch.setattr(profile_handlers, "_prepare_root_payload", fake_prepare)
    monkeypatch.setattr(profile_handlers, "_payment_urls", lambda: {"card": "https://pay.example"})
    monkeypatch.setattr(profile_handlers, "get_user_transactions", fake_history)
    monkeypatch.setattr(profile_handlers, "_bot_name", lambda: "TestBot")

    def fake_render(ctx, view, data=None):
        rendered.append(view)
        return f"view:{view}", InlineKeyboardMarkup([[InlineKeyboardButton("⬅️", callback_data="profile:back")]])

    async def fake_update(update, ctx, text, markup, *, parse_mode=profile_handlers.ParseMode.HTML):
        updated.append(text)
        return SimpleNamespace(message_id=ctx.chat_data.get("profile_msg_id", 99))

    monkeypatch.setattr(profile_handlers, "render_profile_view", fake_render)
    monkeypatch.setattr(profile_handlers, "profile_update_or_send", fake_update)

    async def scenario():
        ctx = _make_ctx()
        for view in ["root", "topup", "history", "invite", "promo", "back"]:
            update = _make_callback_update(view)
            await profile_handlers.on_profile_cbq(update, ctx)

    asyncio.run(scenario())

    assert rendered == ["root", "topup", "history", "invite", "promo", "root"], rendered
    assert len(updated) == 6


def test_profile_edit_fallback_when_msg_missing():
    bot = DummyBot(edit_exception=BadRequest("message to edit not found"))
    ctx = _make_ctx(bot)
    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=123),
        effective_message=SimpleNamespace(chat_id=123),
    )

    markup = InlineKeyboardMarkup([[InlineKeyboardButton("ok", callback_data="noop")]])
    result = asyncio.run(profile_handlers.profile_update_or_send(update, ctx, "hello", markup))

    assert bot.send_calls, "send_message should be used as fallback"
    assert result is not None
    assert ctx.chat_data.get("profile_msg_id") == result.message_id


def test_profile_no_empty_screens():
    ctx = _make_ctx()

    text_topup, _ = profile_handlers.render_profile_view(ctx, "topup", {"payment_urls": {}})
    assert "Пополнение — в разработке" in text_topup

    text_invite, _ = profile_handlers.render_profile_view(ctx, "invite", {"invite_link": None})
    assert "Ссылка станет доступна позже" in text_invite

    text_history, _ = profile_handlers.render_profile_view(ctx, "history", {"entries": []})
    assert "История операций пока пуста" in text_history


def test_profile_back_returns_root(monkeypatch):
    prepared = {"snapshot": SimpleNamespace(display="5💎", warning=None, value=5), "snapshot_target": 5, "referral_url": None, "chat_id": 42}

    async def fake_prepare(update, ctx):
        return prepared

    render_calls: list[str] = []

    def fake_render(ctx, view, data=None):
        render_calls.append(view)
        return "root screen", InlineKeyboardMarkup([[InlineKeyboardButton("⬅️", callback_data="profile:back")]])

    async def fake_update(update, ctx, text, markup, *, parse_mode=profile_handlers.ParseMode.HTML):
        return SimpleNamespace(message_id=555)

    monkeypatch.setattr(profile_handlers, "_prepare_root_payload", fake_prepare)
    monkeypatch.setattr(profile_handlers, "render_profile_view", fake_render)
    monkeypatch.setattr(profile_handlers, "profile_update_or_send", fake_update)

    async def scenario():
        ctx = _make_ctx()
        update = _make_callback_update("back")
        await profile_handlers.handle_profile_view(update, ctx, "back")
        return ctx

    ctx = asyncio.run(scenario())

    assert render_calls == ["root"]
    assert ctx.chat_data.get("profile_last_view") == "root"
    assert ctx.chat_data.get("profile_render_state") == {
        "snapshot_target": prepared["snapshot_target"],
        "chat_id": prepared["chat_id"],
        "referral_url": prepared["referral_url"],
    }
