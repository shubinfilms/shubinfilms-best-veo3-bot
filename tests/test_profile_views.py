import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import BadRequest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import handlers.profile as profile_handlers
from helpers.telegram_html import sanitize_profile_html, strip_telegram_html, tg_html_safe


class DummyBot:
    def __init__(
        self,
        *,
        edit_exception: Exception | list[Exception | None] | None = None,
    ):
        self.edit_calls: list[dict] = []
        self.send_calls: list[dict] = []
        if isinstance(edit_exception, list):
            self._edit_exceptions = list(edit_exception)
            self._edit_exception = None
        else:
            self._edit_exceptions = []
            self._edit_exception = edit_exception
        self._next_message_id = 1000

    async def edit_message_text(self, **kwargs):  # type: ignore[override]
        self.edit_calls.append(kwargs)
        if self._edit_exceptions:
            exc = self._edit_exceptions.pop(0)
            if exc is not None:
                raise exc
        elif self._edit_exception is not None:
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


def test_open_profile_send_fallback():
    bot = DummyBot(edit_exception=BadRequest("message to edit not found"))
    ctx = _make_ctx(bot)
    ctx.chat_data[profile_handlers.PROFILE_MSG_ID] = 77
    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=123, type="private"),
        effective_message=SimpleNamespace(chat_id=123),
    )

    markup = InlineKeyboardMarkup([[InlineKeyboardButton("ok", callback_data="noop")]])
    result = asyncio.run(
        profile_handlers.profile_update_or_send(update, ctx, "hello<br/>world", markup)
    )

    assert bot.send_calls, "send_message should be used as fallback"
    assert result is not None
    assert ctx.chat_data.get("profile_msg_id") == result.message_id
    assert ctx.chat_data.get("profile_last_msg_id") == result.message_id
    assert len(bot.edit_calls) == 1
    assert len(bot.send_calls) == 1
    sent_kwargs = bot.send_calls[0]["kwargs"]
    assert "<br/>" not in sent_kwargs["text"]


def test_profile_no_empty_screens():
    ctx = _make_ctx()

    text_topup, _ = profile_handlers.render_profile_view(ctx, "topup", {"payment_urls": {}})
    assert "Telegram Stars — Скоро" in text_topup

    text_invite, _ = profile_handlers.render_profile_view(ctx, "invite", {"invite_link": None})
    assert "Ссылка станет доступна позже" in text_invite

    text_history, _ = profile_handlers.render_profile_view(ctx, "history", {"entries": []})
    assert "История операций пока пуста" in text_history


def test_inner_buttons_back(monkeypatch):
    prepared = {
        "snapshot": SimpleNamespace(display="5💎", warning=None, value=5),
        "snapshot_target": 5,
        "referral_url": None,
        "chat_id": 42,
    }

    async def fake_prepare(update, ctx):
        return prepared

    render_calls: list[str] = []

    def fake_render(ctx, view, data=None):
        render_calls.append(view)
        return f"view:{view}", InlineKeyboardMarkup(
            [[InlineKeyboardButton("⬅️ Назад", callback_data="profile:back")]]
        )

    async def fake_update(update, ctx, text, markup, *, parse_mode=profile_handlers.ParseMode.HTML):
        return SimpleNamespace(message_id=len(render_calls))

    monkeypatch.setattr(profile_handlers, "_prepare_root_payload", fake_prepare)
    monkeypatch.setattr(profile_handlers, "render_profile_view", fake_render)
    monkeypatch.setattr(profile_handlers, "profile_update_or_send", fake_update)

    async def scenario():
        ctx = _make_ctx()
        for view in ("topup", "history", "invite"):
            await profile_handlers.handle_profile_view(_make_callback_update(view), ctx, view)
            await profile_handlers.handle_profile_view(_make_callback_update("back"), ctx, "back")
        return ctx

    ctx = asyncio.run(scenario())

    assert render_calls == [
        "topup",
        "root",
        "history",
        "root",
        "invite",
        "root",
    ]
    assert ctx.chat_data.get("profile_last_view") == "root"


def test_html_br_sanitized():
    raw = "<b>Hello</b><br/>world<br> & friends"
    filtered = tg_html_safe(raw)
    assert "<br/>" not in filtered
    sanitized, mode = sanitize_profile_html(filtered)
    assert "<br/>" not in sanitized
    assert mode == profile_handlers.ParseMode.HTML
    plain = strip_telegram_html(sanitized)
    assert "Hello\nworld" in plain


def test_profile_open_from_menu_and_quick(monkeypatch):
    calls: list[tuple[str, bool]] = []

    async def fake_open_profile_card(update, ctx, *, source: str, suppress_nav: bool = True):
        calls.append((source, suppress_nav))
        return profile_handlers.OpenedProfile(msg_id=777, reused=False)

    monkeypatch.setattr(profile_handlers, "open_profile_card", fake_open_profile_card)

    async def scenario():
        ctx = _make_ctx()
        update = SimpleNamespace(
            effective_chat=SimpleNamespace(id=55),
            effective_message=SimpleNamespace(chat_id=55),
            effective_user=SimpleNamespace(id=99),
        )
        await profile_handlers.open_profile(update, ctx, source="menu", suppress_nav=True)
        ctx.chat_data[profile_handlers.PROFILE_OPEN_AT] = 0
        await profile_handlers.open_profile(update, ctx, source="quick", suppress_nav=True)

    asyncio.run(scenario())

    assert calls == [("menu", True), ("quick", True)]


def test_profile_invite_without_bot_name(monkeypatch):
    ctx = _make_ctx()
    ctx.bot.username = "InviteBot"
    monkeypatch.setattr(profile_handlers, "_bot_name", lambda: None)

    captured: dict[str, str] = {}

    async def fake_update(update, inner_ctx, text, markup, *, parse_mode=profile_handlers.ParseMode.HTML):
        captured["text"] = text
        captured["parse_mode"] = parse_mode
        return SimpleNamespace(message_id=321)

    monkeypatch.setattr(profile_handlers, "profile_update_or_send", fake_update)

    update = _make_callback_update("invite")
    asyncio.run(profile_handlers.handle_profile_view(update, ctx, "invite"))

    assert "https://t.me/InviteBot?start=ref_20" in captured["text"]
    assert captured["parse_mode"] == profile_handlers.ParseMode.HTML


def test_profile_topup_without_url():
    ctx = _make_ctx()
    text_topup, _ = profile_handlers.render_profile_view(ctx, "topup", {"payment_urls": {}})
    assert "Telegram Stars — Скоро" in text_topup
    assert "<br" not in text_topup


def test_profile_history_empty():
    ctx = _make_ctx()
    text_history, _ = profile_handlers.render_profile_view(ctx, "history", {"entries": []})
    assert "История операций пока пуста" in text_history
    assert "<br" not in text_history


def test_profile_edit_fallbacks():
    bot = DummyBot(edit_exception=[BadRequest("can't parse entities")])
    ctx = _make_ctx(bot)
    ctx.chat_data[profile_handlers.PROFILE_MSG_ID] = 42
    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=123, type="private"),
        effective_message=SimpleNamespace(chat_id=123),
    )

    markup = InlineKeyboardMarkup([[InlineKeyboardButton("ok", callback_data="noop")]])
    result = asyncio.run(profile_handlers.profile_update_or_send(update, ctx, "hello", markup))

    assert result is not None
    assert len(bot.edit_calls) == 1
    assert len(bot.send_calls) == 1
