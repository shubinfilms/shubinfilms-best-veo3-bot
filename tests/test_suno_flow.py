from __future__ import annotations

import asyncio
import importlib
import os
from types import SimpleNamespace
from pathlib import Path
import sys

from telegram import InlineKeyboardMarkup
from telegram.error import BadRequest

os.environ.setdefault("SUNO_API_BASE", "https://example.com")
os.environ.setdefault("SUNO_API_TOKEN", "token")
os.environ.setdefault("SUNO_CALLBACK_URL", "https://callback.example")
os.environ.setdefault("SUNO_CALLBACK_SECRET", "secret")
os.environ.setdefault("TELEGRAM_TOKEN", "dummy-token")
os.environ.setdefault("KIE_API_KEY", "test-key")
os.environ.setdefault("KIE_BASE_URL", "https://example.com")
os.environ.setdefault("DATABASE_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("LEDGER_BACKEND", "memory")
os.environ.setdefault("LOG_JSON", "false")
os.environ.setdefault("LOG_LEVEL", "WARNING")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from telegram_utils import SafeEditResult, safe_edit
from ui_helpers import refresh_suno_card, render_suno_card
from texts import SUNO_START_READY_MESSAGE
from utils.suno_state import (
    SunoState,
    build_generation_payload,
    clear_style,
    clear_title,
    load,
    save,
    set_lyrics,
    set_style,
    set_title,
)

import utils.api_client as api_client_utils

bot_module = importlib.import_module("bot")


class FakeBot:
    def __init__(self) -> None:
        self.sent: list[dict[str, object]] = []
        self.edited: list[dict[str, object]] = []
        self._next_message_id = 100
        self._edit_errors: list[Exception] = []

    def queue_edit_error(self, exc: Exception) -> None:
        self._edit_errors.append(exc)

    async def send_message(self, **kwargs):  # type: ignore[override]
        self.sent.append(kwargs)
        message_id = self._next_message_id
        self._next_message_id += 1
        return SimpleNamespace(message_id=message_id)

    async def edit_message_text(self, **kwargs):  # type: ignore[override]
        if self._edit_errors:
            raise self._edit_errors.pop(0)
        self.edited.append(kwargs)
        message_id = kwargs.get("message_id")
        return SimpleNamespace(message_id=message_id)


class FakeMessage:
    def __init__(self, chat_id: int, text: str) -> None:
        self.chat_id = chat_id
        self.text = text
        self.replies: list[dict[str, object]] = []
        self._next_message_id = 900

    async def reply_text(self, text: str, **kwargs):  # type: ignore[override]
        self._next_message_id += 1
        self.replies.append({"text": text, "kwargs": kwargs})
        return SimpleNamespace(message_id=self._next_message_id)


class MiniRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    def setex(self, key: str, ttl: int, value: str) -> bool:
        self.store[key] = value
        return True

    def get(self, key: str):
        return self.store.get(key)

    def delete(self, key: str) -> None:
        self.store.pop(key, None)

    def set(self, key: str, value: str, nx: bool = False, ex: int | None = None):
        if nx and key in self.store:
            return False
        self.store[key] = value
        return True


def _setup_suno_context() -> tuple[SimpleNamespace, dict[str, object], FakeBot, int]:
    bot = FakeBot()
    ctx = SimpleNamespace(bot=bot, user_data={})
    state_dict = bot_module.state(ctx)
    chat_id = 777

    asyncio.run(refresh_suno_card(ctx, chat_id=chat_id, state_dict=state_dict, price=bot_module.PRICE_SUNO))
    return ctx, state_dict, bot, chat_id


def _prepare_suno_params(
    bot,
    ctx: SimpleNamespace,
    *,
    title: str = "",
    style: str = "",
    lyrics: str = "",
    mode: str = "instrumental",
    lyrics_source=None,
):
    suno_state = bot.load_suno_state(ctx)
    suno_state.mode = mode  # type: ignore[assignment]
    bot.set_suno_title(suno_state, title)
    bot.set_suno_style(suno_state, style)
    if lyrics:
        bot.set_suno_lyrics(suno_state, lyrics)
    else:
        bot.clear_suno_lyrics(suno_state)
    if lyrics_source is None:
        if lyrics and mode == "lyrics":
            lyrics_source = bot.LyricsSource.USER
        else:
            lyrics_source = bot.LyricsSource.AI
    bot.set_suno_lyrics_source(suno_state, lyrics_source)
    bot.save_suno_state(ctx, suno_state)
    state_dict = bot.state(ctx)
    params = bot._suno_collect_params(state_dict, suno_state)
    ctx.user_data["suno_state"] = suno_state.to_dict()
    return params


def _render(state: SunoState, *, price: int = 30, balance: int | None = None):
    text, markup, ready = render_suno_card(
        state,
        price=price,
        balance=balance,
        generating=False,
        waiting_enqueue=False,
    )
    return text, markup, ready


def test_render_includes_escaped_fields() -> None:
    state = SunoState(mode="lyrics")
    set_title(state, "  Test <Track>  ")
    set_style(state, "Dream pop <b>lush</b>")
    set_lyrics(state, "Line one\nLine two")
    text, _, _ = _render(state)
    assert "🏷️ Название: <i>Test</i>" in text
    assert "<Track" not in text
    assert "🎹 Стиль: <i>Dream pop lush</i>" in text
    assert "📜 Текст: <i>Line one" in text


def test_render_shows_dash_for_missing_values() -> None:
    state = SunoState()
    text, _, _ = _render(state)
    assert "🏷️ Название: <i>—</i>" in text
    assert "🎹 Стиль: <i>—</i>" in text
    assert "📜" not in text

    state_lyrics = SunoState(mode="lyrics")
    text_lyrics, _, _ = _render(state_lyrics)
    assert "📜 Текст: <i>—</i>" in text_lyrics


def test_render_has_no_br_tags() -> None:
    state = SunoState()
    set_style(state, "Calm\nAmbient")
    text, _, _ = _render(state)
    assert "<br" not in text.lower()


def test_lyrics_preview_and_payload() -> None:
    lines = ["   First verse  ", "Second line", "Third"]
    lyrics = "\n".join(lines)
    state = SunoState(mode="lyrics")
    set_lyrics(state, lyrics)
    text, _, _ = _render(state)
    assert "📜 Текст: <i>First verse" in text
    payload = build_generation_payload(state, model="V5", lang="ru")
    assert payload["lyrics"] == "First verse\nSecond line\nThird"


def test_state_persistence_allows_cancel_flow() -> None:
    ctx = SimpleNamespace(user_data={})
    state = load(ctx)
    set_title(state, "Keep")
    save(ctx, state)
    reloaded = load(ctx)
    assert reloaded.title == "Keep"
    clear_title(reloaded)
    clear_style(reloaded)
    save(ctx, reloaded)
    assert load(ctx).title is None


def test_refresh_updates_title_and_message_state() -> None:
    bot = FakeBot()
    ctx = SimpleNamespace(bot=bot, user_data={})
    state_dict = {
        "suno_card": {"msg_id": None, "last_text_hash": None, "last_markup_hash": None},
        "msg_ids": {},
    }

    suno_state = load(ctx)
    save(ctx, suno_state)

    async def scenario() -> None:
        await refresh_suno_card(ctx, chat_id=123, state_dict=state_dict, price=30)
        assert bot.sent, "initial card should be sent"

        set_title(suno_state, "Test")
        save(ctx, suno_state)
        await refresh_suno_card(ctx, chat_id=123, state_dict=state_dict, price=30)

    asyncio.run(scenario())

    assert bot.edited, "card should be edited when title changes"
    edited_payload = bot.edited[-1]
    assert "Название: <i>Test</i>" in edited_payload["text"]
    assert state_dict["suno_card"]["msg_id"] == state_dict["last_ui_msg_id_suno"]
    assert load(ctx).title == "Test"


def test_start_message_sent_when_ready() -> None:
    bot = FakeBot()
    ctx = SimpleNamespace(bot=bot, user_data={})
    state_dict = {
        "suno_card": {"msg_id": None, "last_text_hash": None, "last_markup_hash": None},
        "msg_ids": {},
    }

    suno_state = load(ctx)
    set_title(suno_state, "Готовый трек")
    set_style(suno_state, "ambient, chill")
    save(ctx, suno_state)

    asyncio.run(refresh_suno_card(ctx, chat_id=321, state_dict=state_dict, price=30))

    texts = [call.get("text") for call in bot.sent]
    assert SUNO_START_READY_MESSAGE in texts, "start message should be sent when ready"
    last_markup = bot.sent[-1].get("reply_markup")
    assert isinstance(last_markup, InlineKeyboardMarkup)
    assert last_markup.inline_keyboard[0][0].callback_data == "suno:start"


def test_refresh_skips_duplicate_payload() -> None:
    bot = FakeBot()
    ctx = SimpleNamespace(bot=bot, user_data={})
    state_dict = {
        "suno_card": {"msg_id": None, "last_text_hash": None, "last_markup_hash": None},
        "msg_ids": {},
    }

    suno_state = load(ctx)
    set_style(suno_state, "Chillwave")
    save(ctx, suno_state)
    async def scenario() -> None:
        await refresh_suno_card(ctx, chat_id=55, state_dict=state_dict, price=30)
        first_edit_calls = len(bot.edited)
        await refresh_suno_card(ctx, chat_id=55, state_dict=state_dict, price=30)
        assert len(bot.edited) == first_edit_calls, "no extra edit expected"

    asyncio.run(scenario())

    first_edit_calls = len(bot.edited)
    asyncio.run(refresh_suno_card(ctx, chat_id=55, state_dict=state_dict, price=30))
    assert len(bot.edited) == first_edit_calls, "no extra edit expected"


def test_safe_edit_message_not_found_sends_new() -> None:
    bot = FakeBot()
    state: dict[str, object] = {"msg_id": 42}
    bot.queue_edit_error(BadRequest("message to edit not found"))

    async def scenario() -> SafeEditResult:
        return await safe_edit(
            bot,
            chat_id=99,
            message_id=42,
            text="hello",
            reply_markup=InlineKeyboardMarkup([]),
            state=state,
        )

    result = asyncio.run(scenario())

    assert isinstance(result, SafeEditResult)
    assert result.status == "resent"
    assert bot.sent, "a new message should be sent"
    assert state["msg_id"] == result.message_id


def test_safe_edit_skips_same_payload() -> None:
    bot = FakeBot()
    state: dict[str, object] = {}

    async def first_call() -> SafeEditResult:
        return await safe_edit(
            bot,
            chat_id=1,
            message_id=None,
            text="payload",
            reply_markup=InlineKeyboardMarkup([]),
            state=state,
        )

    first = asyncio.run(first_call())
    assert first.status == "sent"

    async def second_call() -> SafeEditResult:
        return await safe_edit(
            bot,
            chat_id=1,
            message_id=first.message_id,
            text="payload",
            reply_markup=InlineKeyboardMarkup([]),
            state=state,
        )

    second = asyncio.run(second_call())

    assert second.status == "skipped"
    assert not bot.edited, "no edit should be performed when payload is unchanged"


def test_title_inserts_into_card() -> None:
    ctx, state_dict, fake_bot, chat_id = _setup_suno_context()
    state_dict["suno_waiting_state"] = bot_module.WAIT_SUNO_TITLE
    msg = FakeMessage(chat_id, "Новая песня ✨")

    asyncio.run(
        bot_module._handle_suno_waiting_input(
            ctx,
            chat_id,
            msg,
            state_dict,
            bot_module.WAIT_SUNO_TITLE,
            user_id=123,
        )
    )

    assert load(ctx).title == "Новая песня ✨"
    assert state_dict["suno_waiting_state"] == bot_module.IDLE_SUNO
    assert msg.replies[0]["text"] == "✅ Принято"
    assert fake_bot.edited, "card should be edited after title update"
    assert "Название: <i>Новая песня ✨</i>" in fake_bot.edited[-1]["text"]


def test_style_inserts_into_card() -> None:
    ctx, state_dict, fake_bot, chat_id = _setup_suno_context()
    state_dict["suno_waiting_state"] = bot_module.WAIT_SUNO_STYLE
    msg = FakeMessage(chat_id, "Спокойный синтвейв — ночь\nГитары 🎸")

    asyncio.run(
        bot_module._handle_suno_waiting_input(
            ctx,
            chat_id,
            msg,
            state_dict,
            bot_module.WAIT_SUNO_STYLE,
            user_id=321,
        )
    )

    saved = load(ctx)
    assert saved.style == "Спокойный синтвейв — ночь\nГитары 🎸"
    assert state_dict["suno_waiting_state"] == bot_module.IDLE_SUNO
    assert msg.replies[0]["text"] == "✅ Принято"
    assert fake_bot.edited and "Стиль: <i>Спокойный синтвейв — ночь" in fake_bot.edited[-1]["text"]


def test_lyrics_inserts_into_card() -> None:
    ctx, state_dict, fake_bot, chat_id = _setup_suno_context()
    suno_state = load(ctx)
    suno_state.mode = "lyrics"
    save(ctx, suno_state)
    state_dict["suno_state"] = suno_state.to_dict()
    asyncio.run(refresh_suno_card(ctx, chat_id=chat_id, state_dict=state_dict, price=bot_module.PRICE_SUNO))

    state_dict["suno_waiting_state"] = bot_module.WAIT_SUNO_LYRICS
    lyrics_text = "Первая строка\nВторая 🎤\n\nТретья"  # includes blank line
    msg = FakeMessage(chat_id, lyrics_text)

    asyncio.run(
        bot_module._handle_suno_waiting_input(
            ctx,
            chat_id,
            msg,
            state_dict,
            bot_module.WAIT_SUNO_LYRICS,
            user_id=77,
        )
    )

    saved = load(ctx)
    assert saved.lyrics == "Первая строка\nВторая 🎤\n\nТретья"
    assert msg.replies[0]["text"] == "✅ Принято"
    assert fake_bot.edited and "Текст: <i>Первая строка" in fake_bot.edited[-1]["text"]


def test_not_modified_guard() -> None:
    ctx, state_dict, fake_bot, chat_id = _setup_suno_context()
    state_dict["suno_waiting_state"] = bot_module.WAIT_SUNO_TITLE
    msg = FakeMessage(chat_id, "Тестовая мелодия")
    asyncio.run(
        bot_module._handle_suno_waiting_input(
            ctx,
            chat_id,
            msg,
            state_dict,
            bot_module.WAIT_SUNO_TITLE,
            user_id=501,
        )
    )

    edits_before = len(fake_bot.edited)
    state_dict["suno_waiting_state"] = bot_module.WAIT_SUNO_TITLE
    repeat_msg = FakeMessage(chat_id, "Тестовая мелодия")
    asyncio.run(
        bot_module._handle_suno_waiting_input(
            ctx,
            chat_id,
            repeat_msg,
            state_dict,
            bot_module.WAIT_SUNO_TITLE,
            user_id=501,
        )
    )

    assert len(fake_bot.edited) == edits_before
    assert repeat_msg.replies[-1]["text"].endswith("(без изменений)")


def test_cancel_and_clear() -> None:
    ctx, state_dict, fake_bot, chat_id = _setup_suno_context()

    # Set initial style
    state_dict["suno_waiting_state"] = bot_module.WAIT_SUNO_STYLE
    set_msg = FakeMessage(chat_id, "Dream pop")
    asyncio.run(
        bot_module._handle_suno_waiting_input(
            ctx,
            chat_id,
            set_msg,
            state_dict,
            bot_module.WAIT_SUNO_STYLE,
            user_id=808,
        )
    )

    # Clear style using "-"
    state_dict["suno_waiting_state"] = bot_module.WAIT_SUNO_STYLE
    clear_msg = FakeMessage(chat_id, "-")
    asyncio.run(
        bot_module._handle_suno_waiting_input(
            ctx,
            chat_id,
            clear_msg,
            state_dict,
            bot_module.WAIT_SUNO_STYLE,
            user_id=808,
        )
    )
    assert load(ctx).style is None
    assert clear_msg.replies[0]["text"] == "✅ Принято"

    # Restore style
    state_dict["suno_waiting_state"] = bot_module.WAIT_SUNO_STYLE
    restore_msg = FakeMessage(chat_id, "Dream pop")
    asyncio.run(
        bot_module._handle_suno_waiting_input(
            ctx,
            chat_id,
            restore_msg,
            state_dict,
            bot_module.WAIT_SUNO_STYLE,
            user_id=808,
        )
    )

    # Cancel editing
    state_dict["suno_waiting_state"] = bot_module.WAIT_SUNO_STYLE
    cancel_msg = FakeMessage(chat_id, "/cancel")
    asyncio.run(
        bot_module._handle_suno_waiting_input(
            ctx,
            chat_id,
            cancel_msg,
            state_dict,
            bot_module.WAIT_SUNO_STYLE,
            user_id=808,
        )
    )

    assert load(ctx).style == "Dream pop"
    assert cancel_msg.replies[-1]["text"] == "✏️ Изменение отменено."
    assert state_dict["suno_waiting_state"] == bot_module.IDLE_SUNO


def test_prompt_includes_preview() -> None:
    ctx, _, _, _ = _setup_suno_context()
    suno_state = load(ctx)
    set_title(suno_state, "Мелодия ветра")
    set_style(suno_state, "Лёгкий джаз")
    save(ctx, suno_state)

    prompt_text = bot_module._suno_prompt_text("title", suno_state)
    assert 'Сейчас: “Мелодия ветра”' in prompt_text


def test_suno_card_resend_on_missing() -> None:
    ctx, state_dict, fake_bot, chat_id = _setup_suno_context()
    state_dict["suno_waiting_state"] = bot_module.WAIT_SUNO_TITLE
    msg = FakeMessage(chat_id, "Первое название")
    asyncio.run(
        bot_module._handle_suno_waiting_input(
            ctx,
            chat_id,
            msg,
            state_dict,
            bot_module.WAIT_SUNO_TITLE,
            user_id=999,
        )
    )

    initial_sent = len(fake_bot.sent)
    fake_bot.queue_edit_error(BadRequest("message to edit not found"))
    card_state = state_dict.get("suno_card")
    if isinstance(card_state, dict):
        card_state["msg_id"] = 999
    state_dict["last_ui_msg_id_suno"] = 999
    state_dict["suno_waiting_state"] = bot_module.WAIT_SUNO_TITLE
    msg2 = FakeMessage(chat_id, "Второе название")
    asyncio.run(
        bot_module._handle_suno_waiting_input(
            ctx,
            chat_id,
            msg2,
            state_dict,
            bot_module.WAIT_SUNO_TITLE,
            user_id=999,
        )
    )

    assert len(fake_bot.sent) == initial_sent + 1
    assert isinstance(state_dict["suno_card"]["msg_id"], int)
    assert state_dict["suno_card"]["msg_id"] != 999
    assert load(ctx).title == "Второе название"


def test_suno_card_resend_on_not_modified() -> None:
    ctx, state_dict, fake_bot, chat_id = _setup_suno_context()
    state_dict["suno_waiting_state"] = bot_module.WAIT_SUNO_TITLE
    initial_msg = FakeMessage(chat_id, "Начальный трек")
    asyncio.run(
        bot_module._handle_suno_waiting_input(
            ctx,
            chat_id,
            initial_msg,
            state_dict,
            bot_module.WAIT_SUNO_TITLE,
            user_id=1001,
        )
    )

    initial_sent = len(fake_bot.sent)
    fake_bot.queue_edit_error(BadRequest("Message is not modified"))
    state_dict["suno_waiting_state"] = bot_module.WAIT_SUNO_TITLE
    msg2 = FakeMessage(chat_id, "Новый трек")
    asyncio.run(
        bot_module._handle_suno_waiting_input(
            ctx,
            chat_id,
            msg2,
            state_dict,
            bot_module.WAIT_SUNO_TITLE,
            user_id=1001,
        )
    )

    if len(fake_bot.sent) == initial_sent:
        assert fake_bot.edited, "expected edit attempt when no resend occurred"
    else:
        assert len(fake_bot.sent) == initial_sent + 1
        assert fake_bot.sent[-1]["text"].startswith("🎵") or "Новый трек" in fake_bot.sent[-1]["text"]
    assert load(ctx).title == "Новый трек"


def test_suno_enqueue_retries_then_success(monkeypatch) -> None:
    bot = bot_module
    fake_redis = MiniRedis()
    monkeypatch.setattr(bot, "rds", fake_redis)
    bot._SUNO_PENDING_MEMORY.clear()
    bot._SUNO_REFUND_PENDING_MEMORY.clear()
    bot._SUNO_REFUND_MEMORY.clear()
    bot._SUNO_REFUND_REQ_MEMORY.clear()
    bot._SUNO_COOLDOWN_MEMORY.clear()
    monkeypatch.setattr(bot, "_suno_configured", lambda: True)
    monkeypatch.setattr(bot, "ensure_user", lambda uid: None)
    bot.SUNO_PER_USER_COOLDOWN_SEC = 0
    monkeypatch.setattr(bot, "SUNO_MODE_AVAILABLE", True, raising=False)

    async def async_noop(*args, **kwargs):
        return None

    monkeypatch.setattr(bot, "refresh_suno_card", async_noop)
    monkeypatch.setattr(bot, "refresh_balance_card_if_open", async_noop)
    monkeypatch.setattr(bot, "show_balance_notification", async_noop)
    monkeypatch.setattr(bot, "_suno_update_last_debit_meta", lambda *args, **kwargs: None)
    monkeypatch.setattr(bot, "_suno_set_cooldown", lambda *args, **kwargs: None)

    debit_calls = {"count": 0}

    def fake_debit(user_id, price, reason, meta):
        debit_calls["count"] += 1
        return True, 90

    monkeypatch.setattr(bot, "debit_try", fake_debit)

    status_texts: list[str] = []

    async def fake_notify(ctx_param, chat_id_param, text, **kwargs):
        status_texts.append(text)
        return SimpleNamespace(message_id=321)

    monkeypatch.setattr(bot, "_suno_notify", fake_notify)

    edited_messages: list[str] = []

    async def fake_safe_edit_message(ctx_param, chat_id_param, message_id_param, new_text, **kwargs):
        edited_messages.append(new_text)
        return True

    monkeypatch.setattr(bot, "safe_edit_message", fake_safe_edit_message)

    refunds: list[dict[str, object]] = []

    async def fake_refund(*args, **kwargs):
        refunds.append(kwargs)

    monkeypatch.setattr(bot, "_suno_issue_refund", fake_refund)

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(bot.asyncio, "to_thread", fake_to_thread)

    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(api_client_utils.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(api_client_utils.random, "uniform", lambda a, b: 0.0)

    attempts = {"count": 0}

    class DummyTask:
        def __init__(self, task_id: str) -> None:
            self.task_id = task_id
            self.items = []
            self.callback_type = "start"
            self.msg = None
            self.code = None

    def fake_start_music(*args, **kwargs):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise bot.SunoAPIError("temporary", status=500)
        return DummyTask("task-success")

    monkeypatch.setattr(bot.SUNO_SERVICE, "start_music", fake_start_music)

    ctx = SimpleNamespace(user_data={}, bot=SimpleNamespace())
    params = _prepare_suno_params(
        bot,
        ctx,
        title="Demo",
        style="Pop",
        mode="instrumental",
    )

    async def _run() -> None:
        assert bot.SUNO_MODE_AVAILABLE is True
        await bot._launch_suno_generation(
            chat_id=111,
            ctx=ctx,
            params=params,
            user_id=42,
            reply_to=None,
            trigger="test",
        )

    asyncio.run(_run())

    assert attempts["count"] == 3
    assert debit_calls["count"] == 1
    assert status_texts and status_texts[0] == "⏳ Отправляем запрос…"
    assert edited_messages and edited_messages[-1].startswith("✅ Задача создана")
    assert not refunds


def test_suno_enqueue_all_failures(monkeypatch) -> None:
    bot = bot_module
    fake_redis = MiniRedis()
    monkeypatch.setattr(bot, "rds", fake_redis)
    bot._SUNO_PENDING_MEMORY.clear()
    bot._SUNO_REFUND_PENDING_MEMORY.clear()
    bot._SUNO_REFUND_MEMORY.clear()
    bot._SUNO_REFUND_REQ_MEMORY.clear()
    bot._SUNO_COOLDOWN_MEMORY.clear()
    monkeypatch.setattr(bot, "_suno_configured", lambda: True)
    monkeypatch.setattr(bot, "ensure_user", lambda uid: None)
    bot.SUNO_PER_USER_COOLDOWN_SEC = 0
    monkeypatch.setattr(bot, "SUNO_MODE_AVAILABLE", True, raising=False)

    async def async_noop(*args, **kwargs):
        return None

    monkeypatch.setattr(bot, "refresh_suno_card", async_noop)
    monkeypatch.setattr(bot, "refresh_balance_card_if_open", async_noop)
    monkeypatch.setattr(bot, "show_balance_notification", async_noop)
    monkeypatch.setattr(bot, "_suno_update_last_debit_meta", lambda *args, **kwargs: None)
    monkeypatch.setattr(bot, "_suno_set_cooldown", lambda *args, **kwargs: None)

    def fake_debit(user_id, price, reason, meta):
        return True, 90

    monkeypatch.setattr(bot, "debit_try", fake_debit)

    status_texts: list[str] = []

    async def fake_notify(ctx_param, chat_id_param, text, **kwargs):
        status_texts.append(text)
        return SimpleNamespace(message_id=654)

    monkeypatch.setattr(bot, "_suno_notify", fake_notify)

    edited_messages: list[str] = []

    async def fake_safe_edit_message(ctx_param, chat_id_param, message_id_param, new_text, **kwargs):
        edited_messages.append(new_text)
        return True

    monkeypatch.setattr(bot, "safe_edit_message", fake_safe_edit_message)

    refunds: list[dict[str, object]] = []

    async def fake_refund(*args, **kwargs):
        refunds.append(kwargs)

    monkeypatch.setattr(bot, "_suno_issue_refund", fake_refund)

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(bot.asyncio, "to_thread", fake_to_thread)

    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(api_client_utils.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(api_client_utils.random, "uniform", lambda a, b: 0.0)

    attempts = {"count": 0}

    def failing_start_music(*args, **kwargs):
        attempts["count"] += 1
        raise bot.SunoAPIError("boom", status=500)

    monkeypatch.setattr(bot.SUNO_SERVICE, "start_music", failing_start_music)

    ctx = SimpleNamespace(user_data={}, bot=SimpleNamespace())
    params = _prepare_suno_params(
        bot,
        ctx,
        title="Demo",
        style="Pop",
        mode="instrumental",
    )

    async def _run() -> None:
        await bot._launch_suno_generation(
            chat_id=222,
            ctx=ctx,
            params=params,
            user_id=99,
            reply_to=None,
            trigger="test",
        )

    asyncio.run(_run())

    assert attempts["count"] == bot._SUNO_ENQUEUE_MAX_ATTEMPTS
    assert sum(sleeps) <= bot._SUNO_ENQUEUE_MAX_DELAY + 1e-6
    assert status_texts and status_texts[0] == "⏳ Отправляем запрос…"
    assert edited_messages and edited_messages[-1] == "⚠️ Generation failed: boom"
    assert refunds and refunds[-1]["user_message"].startswith("⚠️ Generation failed: boom")


def test_suno_enqueue_handles_400_error(monkeypatch) -> None:
    bot = bot_module
    fake_redis = MiniRedis()
    monkeypatch.setattr(bot, "rds", fake_redis)
    bot._SUNO_PENDING_MEMORY.clear()
    bot._SUNO_REFUND_PENDING_MEMORY.clear()
    bot._SUNO_REFUND_MEMORY.clear()
    bot._SUNO_REFUND_REQ_MEMORY.clear()
    bot._SUNO_COOLDOWN_MEMORY.clear()
    monkeypatch.setattr(bot, "_suno_configured", lambda: True)
    monkeypatch.setattr(bot, "ensure_user", lambda uid: None)
    bot.SUNO_PER_USER_COOLDOWN_SEC = 0
    monkeypatch.setattr(bot, "SUNO_MODE_AVAILABLE", True, raising=False)

    async def async_noop(*args, **kwargs):
        return None

    monkeypatch.setattr(bot, "refresh_suno_card", async_noop)
    monkeypatch.setattr(bot, "refresh_balance_card_if_open", async_noop)
    monkeypatch.setattr(bot, "show_balance_notification", async_noop)
    monkeypatch.setattr(bot, "_suno_update_last_debit_meta", lambda *args, **kwargs: None)
    monkeypatch.setattr(bot, "_suno_set_cooldown", lambda *args, **kwargs: None)

    def fake_debit(user_id, price, reason, meta):
        return True, 90

    monkeypatch.setattr(bot, "debit_try", fake_debit)

    status_texts: list[str] = []

    async def fake_notify(ctx_param, chat_id_param, text, **kwargs):
        status_texts.append(text)
        return SimpleNamespace(message_id=333)

    monkeypatch.setattr(bot, "_suno_notify", fake_notify)

    edited_messages: list[str] = []

    async def fake_safe_edit_message(ctx_param, chat_id_param, message_id_param, new_text, **kwargs):
        edited_messages.append(new_text)
        return True

    monkeypatch.setattr(bot, "safe_edit_message", fake_safe_edit_message)

    refunds: list[dict[str, object]] = []

    async def fake_refund(*args, **kwargs):
        refunds.append(kwargs)

    monkeypatch.setattr(bot, "_suno_issue_refund", fake_refund)

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(bot.asyncio, "to_thread", fake_to_thread)

    def failing_start_music(*args, **kwargs):
        raise bot.SunoAPIError(
            "artist name",
            status=400,
            payload={"message": "The description contains artist name: The Weeknd"},
        )

    monkeypatch.setattr(bot.SUNO_SERVICE, "start_music", failing_start_music)

    ctx = SimpleNamespace(user_data={}, bot=SimpleNamespace())
    params = _prepare_suno_params(
        bot,
        ctx,
        title="Future",
        style="Pop",
        mode="instrumental",
    )

    async def _run() -> None:
        await bot._launch_suno_generation(
            chat_id=444,
            ctx=ctx,
            params=params,
            user_id=77,
            reply_to=None,
            trigger="test",
        )

    asyncio.run(_run())

    expected = "❗️Error: your description mentions an artist/brand. Remove the reference and try again."
    assert status_texts and status_texts[0] == "⏳ Отправляем запрос…"
    assert edited_messages and edited_messages[-1] == expected
    assert refunds and refunds[-1]["user_message"].startswith(expected)
    assert "Токены возвращены" in refunds[-1]["user_message"]


def test_suno_enqueue_dedupes_failed_req(monkeypatch) -> None:
    bot = bot_module
    bot.rds = None
    bot._SUNO_PENDING_MEMORY.clear()
    bot._SUNO_REFUND_PENDING_MEMORY.clear()
    bot._SUNO_REFUND_MEMORY.clear()
    bot._SUNO_REFUND_REQ_MEMORY.clear()
    bot._SUNO_COOLDOWN_MEMORY.clear()
    monkeypatch.setattr(bot, "_suno_configured", lambda: True)
    monkeypatch.setattr(bot, "ensure_user", lambda uid: None)
    bot.SUNO_PER_USER_COOLDOWN_SEC = 0

    async def async_noop(*args, **kwargs):
        return None

    monkeypatch.setattr(bot, "refresh_suno_card", async_noop)
    monkeypatch.setattr(bot, "refresh_balance_card_if_open", async_noop)
    monkeypatch.setattr(bot, "show_balance_notification", async_noop)
    monkeypatch.setattr(bot, "_suno_update_last_debit_meta", lambda *args, **kwargs: None)
    monkeypatch.setattr(bot, "_suno_set_cooldown", lambda *args, **kwargs: None)

    debit_calls = []

    def fake_debit(user_id, price, reason, meta):
        debit_calls.append((user_id, price))
        return True, 77

    monkeypatch.setattr(bot, "debit_try", fake_debit)

    start_calls = []

    def fake_start_music(*args, **kwargs):
        start_calls.append(args)
        raise AssertionError("start_music should not be called for deduped req")

    monkeypatch.setattr(bot.SUNO_SERVICE, "start_music", fake_start_music)

    notifications: list[str] = []

    async def fake_notify(ctx_param, chat_id_param, text, **kwargs):
        notifications.append(text)
        return SimpleNamespace(message_id=222)

    monkeypatch.setattr(bot, "_suno_notify", fake_notify)
    monkeypatch.setattr(bot, "_suno_issue_refund", async_noop)
    monkeypatch.setattr(bot, "SUNO_MODE_AVAILABLE", True, raising=False)

    ctx = SimpleNamespace(user_data={}, bot=SimpleNamespace())
    state_dict = bot.state(ctx)
    req_id = "suno:test-dedup"
    ctx.user_data.setdefault("state", state_dict)
    ctx.user_data["state"]["suno_current_req_id"] = req_id
    state_dict["suno_current_req_id"] = req_id
    bot._suno_pending_store(
        req_id,
        {
            "status": "api_error",
            "task_id": None,
            "charged": True,
        },
    )

    assert bot._suno_pending_load(req_id) is not None

    params = _prepare_suno_params(
        bot,
        ctx,
        title="Demo",
        style="Pop",
        mode="instrumental",
    )

    async def _run() -> None:
        assert bot.SUNO_MODE_AVAILABLE is True
        await bot._launch_suno_generation(
            chat_id=303,
            ctx=ctx,
            params=params,
            user_id=55,
            reply_to=None,
            trigger="retry",
        )

    asyncio.run(_run())

    assert not start_calls
    assert not notifications
    assert debit_calls == []
