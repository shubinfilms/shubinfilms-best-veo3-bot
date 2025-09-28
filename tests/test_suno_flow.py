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


def _setup_suno_context() -> tuple[SimpleNamespace, dict[str, object], FakeBot, int]:
    bot = FakeBot()
    ctx = SimpleNamespace(bot=bot, user_data={})
    state_dict = bot_module.state(ctx)
    chat_id = 777

    asyncio.run(refresh_suno_card(ctx, chat_id=chat_id, state_dict=state_dict, price=bot_module.PRICE_SUNO))
    return ctx, state_dict, bot, chat_id


def _render(state: SunoState, *, price: int = 30, balance: int | None = None):
    text, markup = render_suno_card(
        state,
        price=price,
        balance=balance,
        generating=False,
    )
    return text, markup


def test_render_includes_escaped_fields() -> None:
    state = SunoState(mode="lyrics")
    set_title(state, "  Test <Track>  ")
    set_style(state, "Dream pop <b>lush</b>")
    set_lyrics(state, "Line one\nLine two")
    text, _ = _render(state)
    assert "• Название: <i>Test</i>" in text
    assert "<Track" not in text
    assert "• Стиль: <i>Dream pop lush</i>" in text
    assert "• Текст: <i>Line one" in text


def test_render_shows_dash_for_missing_values() -> None:
    state = SunoState()
    text, _ = _render(state)
    assert "• Название: <i>—</i>" in text
    assert "• Стиль: <i>—</i>" in text
    assert "• Текст: <i>—</i>" in text


def test_render_has_no_br_tags() -> None:
    state = SunoState()
    set_style(state, "Calm\nAmbient")
    text, _ = _render(state)
    assert "<br" not in text.lower()


def test_lyrics_preview_and_payload() -> None:
    lines = ["   First verse  ", "Second line", "Third"]
    lyrics = "\n".join(lines)
    state = SunoState(mode="lyrics")
    set_lyrics(state, lyrics)
    text, _ = _render(state)
    assert "• Текст: <i>First verse" in text
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
    assert msg.replies[-1]["text"] == "✅ Название обновлено."
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
    assert msg.replies[-1]["text"] == "✅ Стиль обновлён."
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
    assert msg.replies[-1]["text"] == "✅ Текст песни обновлён."
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
    assert clear_msg.replies[-1]["text"] == "✅ Стиль очищен."

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
    assert 'Сейчас: "Мелодия ветра"' in prompt_text


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

    assert len(fake_bot.sent) == initial_sent + 1
    assert fake_bot.sent[-1]["text"].startswith("🎵") or "Новый трек" in fake_bot.sent[-1]["text"]
    assert load(ctx).title == "Новый трек"
