from __future__ import annotations

import asyncio
from types import SimpleNamespace
from pathlib import Path
import sys

from telegram import InlineKeyboardMarkup
from telegram.error import BadRequest

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
