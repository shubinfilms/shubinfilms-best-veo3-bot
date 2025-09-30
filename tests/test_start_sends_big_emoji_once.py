import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from telegram.error import BadRequest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from texts import SUNO_STARTING_MESSAGE
from utils.suno_state import (
    load as load_suno_state,
    save as save_suno_state,
    set_style as set_suno_style,
    set_title as set_suno_title,
)

from tests.suno_test_utils import FakeBot, bot_module


class FakeMessage:
    def __init__(self, chat_id: int) -> None:
        self.chat_id = chat_id
        self.replies: list[str] = []

    async def reply_text(self, text: str, **_kwargs):  # type: ignore[override]
        self.replies.append(text)
        return SimpleNamespace(message_id=400)


class FakeCallback:
    def __init__(self, chat_id: int) -> None:
        self.data = "suno:start"
        self.message = FakeMessage(chat_id)
        self._answered: list[tuple[str | None, bool]] = []

    async def answer(self, text: str | None = None, show_alert: bool = False):  # type: ignore[override]
        self._answered.append((text, show_alert))


def _prepare_ready_context(ctx, chat_id: int, user_id: int) -> int:
    state_dict = bot_module.state(ctx)
    state_dict["mode"] = "suno"
    suno_state = load_suno_state(ctx)
    suno_state.mode = "instrumental"
    set_suno_title(suno_state, "Ready Song")
    set_suno_style(suno_state, "calm focus")
    save_suno_state(ctx, suno_state)
    state_dict["suno_state"] = suno_state.to_dict()
    start_msg_id = 123
    suno_state.start_msg_id = start_msg_id
    save_suno_state(ctx, suno_state)
    state_dict["suno_state"] = suno_state.to_dict()
    state_dict["suno_start_msg_id"] = start_msg_id
    msg_ids = state_dict.setdefault("msg_ids", {})
    msg_ids["suno_start"] = start_msg_id
    return start_msg_id


def test_start_sends_big_emoji_once(monkeypatch):
    bot = FakeBot()
    ctx = SimpleNamespace(bot=bot, user_data={})
    chat_id = 999
    user_id = 555
    start_msg_id = _prepare_ready_context(ctx, chat_id, user_id)

    monkeypatch.setattr(bot_module, "START_EMOJI_STICKER_ID", "sticker-file-id")
    monkeypatch.setattr(bot_module, "START_EMOJI_FALLBACK", "🎬")

    launch_calls: list[dict[str, object]] = []

    async def fake_launch(chat_id_param, ctx_param, **kwargs):  # type: ignore[override]
        launch_calls.append({
            "chat_id": chat_id_param,
            "user_id": kwargs.get("user_id"),
            "trigger": kwargs.get("trigger"),
        })

    notify_calls: list[str] = []

    async def fake_notify(_ctx, _chat_id, text, **_kwargs):  # type: ignore[override]
        notify_calls.append(text)
        return None

    monkeypatch.setattr(bot_module, "_launch_suno_generation", fake_launch)
    monkeypatch.setattr(bot_module, "_suno_notify", fake_notify)

    update = SimpleNamespace(
        callback_query=FakeCallback(chat_id),
        effective_chat=SimpleNamespace(id=chat_id),
        effective_user=SimpleNamespace(id=user_id),
    )

    asyncio.run(bot_module.on_callback(update, ctx))

    sticker_entries = [item for item in bot.sent if item.get("_method") == "send_sticker"]
    assert len(sticker_entries) == 1
    assert sticker_entries[0]["sticker"] == "sticker-file-id"

    suno_state_after = load_suno_state(ctx)
    assert suno_state_after.start_clicked is True
    assert suno_state_after.start_emoji_msg_id == 100
    assert suno_state_after.start_msg_id is None

    assert ctx.user_data["suno_state"]["start_clicked"] is True
    assert ctx.user_data["suno_state"]["start_msg_id"] is None

    assert launch_calls and launch_calls[0]["trigger"] == "start"
    assert notify_calls, "summary notification should be sent"

    # Second click should be ignored entirely.
    second_update = SimpleNamespace(
        callback_query=FakeCallback(chat_id),
        effective_chat=SimpleNamespace(id=chat_id),
        effective_user=SimpleNamespace(id=user_id),
    )

    asyncio.run(bot_module.on_callback(second_update, ctx))

    sticker_entries_after = [item for item in bot.sent if item.get("_method") == "send_sticker"]
    assert len(sticker_entries_after) == 1
    assert len(launch_calls) == 1


def test_start_button_disabled_after_click(monkeypatch):
    bot = FakeBot()
    ctx = SimpleNamespace(bot=bot, user_data={})
    chat_id = 111
    user_id = 222
    start_msg_id = _prepare_ready_context(ctx, chat_id, user_id)

    monkeypatch.setattr(bot_module, "START_EMOJI_STICKER_ID", "sticker")
    monkeypatch.setattr(bot_module, "START_EMOJI_FALLBACK", "🎬")

    async def fake_launch(*_args, **_kwargs):  # type: ignore[override]
        return None

    async def fake_notify(*_args, **_kwargs):  # type: ignore[override]
        return None

    monkeypatch.setattr(bot_module, "_launch_suno_generation", fake_launch)
    monkeypatch.setattr(bot_module, "_suno_notify", fake_notify)

    update = SimpleNamespace(
        callback_query=FakeCallback(chat_id),
        effective_chat=SimpleNamespace(id=chat_id),
        effective_user=SimpleNamespace(id=user_id),
    )

    asyncio.run(bot_module.on_callback(update, ctx))

    assert bot.edited, "start message should be edited"
    edited_payload = bot.edited[-1]
    assert edited_payload["message_id"] == start_msg_id
    assert edited_payload["chat_id"] == chat_id
    assert edited_payload["text"] == SUNO_STARTING_MESSAGE
    assert edited_payload.get("reply_markup") is None

    state_after = load_suno_state(ctx)
    assert state_after.start_msg_id is None
    assert state_after.start_clicked is True

    state_dict = bot_module.state(ctx)
    assert state_dict.get("suno_start_msg_id") is None
    msg_ids = state_dict.get("msg_ids")
    if isinstance(msg_ids, dict):
        assert "suno_start" not in msg_ids


def test_start_sticker_fallback_to_emoji(monkeypatch):
    bot = FakeBot()
    ctx = SimpleNamespace(bot=bot, user_data={})
    chat_id = 333
    user_id = 444
    _prepare_ready_context(ctx, chat_id, user_id)

    monkeypatch.setattr(bot_module, "START_EMOJI_STICKER_ID", "broken-sticker")
    monkeypatch.setattr(bot_module, "START_EMOJI_FALLBACK", "🎬")

    async def fake_launch(*_args, **_kwargs):  # type: ignore[override]
        return None

    async def fake_notify(*_args, **_kwargs):  # type: ignore[override]
        return None

    async def failing_sticker(*_args, **_kwargs):  # type: ignore[override]
        raise BadRequest("sticker failed")

    monkeypatch.setattr(bot_module, "_launch_suno_generation", fake_launch)
    monkeypatch.setattr(bot_module, "_suno_notify", fake_notify)
    monkeypatch.setattr(bot_module, "safe_send_sticker", failing_sticker)

    update = SimpleNamespace(
        callback_query=FakeCallback(chat_id),
        effective_chat=SimpleNamespace(id=chat_id),
        effective_user=SimpleNamespace(id=user_id),
    )

    asyncio.run(bot_module.on_callback(update, ctx))

    sticker_entries = [item for item in bot.sent if item.get("_method") == "send_sticker"]
    assert not sticker_entries

    fallback_messages = [item for item in bot.sent if item.get("text") == "🎬"]
    assert len(fallback_messages) == 1

    suno_state_after = load_suno_state(ctx)
    assert suno_state_after.start_emoji_msg_id == 100
