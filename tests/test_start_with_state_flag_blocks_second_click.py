import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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

    async def reply_text(self, *_args, **_kwargs):  # type: ignore[override]
        return SimpleNamespace(message_id=403)


class FakeCallback:
    def __init__(self, chat_id: int) -> None:
        self.data = "music:start"
        self.message = FakeMessage(chat_id)
        self._answers: list[tuple[str | None, bool]] = []

    async def answer(self, text: str | None = None, show_alert: bool = False):  # type: ignore[override]
        self._answers.append((text, show_alert))


def _setup(ctx, chat_id: int) -> None:
    state_dict = bot_module.state(ctx)
    state_dict["mode"] = "suno"
    suno_state = load_suno_state(ctx)
    suno_state.mode = "instrumental"
    set_suno_title(suno_state, "Flag Song")
    set_suno_style(suno_state, "calm")
    save_suno_state(ctx, suno_state)
    state_dict["suno_state"] = suno_state.to_dict()
    start_msg_id = 909
    suno_state.start_msg_id = start_msg_id
    save_suno_state(ctx, suno_state)
    state_dict["suno_state"] = suno_state.to_dict()
    state_dict["suno_start_msg_id"] = start_msg_id
    state_dict.setdefault("msg_ids", {})["suno_start"] = start_msg_id


def test_start_with_state_flag_blocks_second_click(monkeypatch):
    bot = FakeBot()
    ctx = SimpleNamespace(bot=bot, user_data={})
    chat_id = 202
    user_id = 303
    _setup(ctx, chat_id)

    monkeypatch.setattr(bot_module, "START_EMOJI_STICKER_ID", "flag-sticker")
    monkeypatch.setattr(bot_module, "START_EMOJI_FALLBACK", "🎬")
    monkeypatch.setattr(bot_module, "redis_client", None)
    monkeypatch.setattr(bot_module, "REDIS_LOCK_ENABLED", False)

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

    stickers = [item for item in bot.sent if item.get("_method") == "send_sticker"]
    assert len(stickers) == 1

    state_after = load_suno_state(ctx)
    assert state_after.start_clicked is True

    second_update = SimpleNamespace(
        callback_query=FakeCallback(chat_id),
        effective_chat=SimpleNamespace(id=chat_id),
        effective_user=SimpleNamespace(id=user_id),
    )

    asyncio.run(bot_module.on_callback(second_update, ctx))

    stickers_after = [item for item in bot.sent if item.get("_method") == "send_sticker"]
    assert len(stickers_after) == 1
