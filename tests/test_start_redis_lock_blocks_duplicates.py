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
        return SimpleNamespace(message_id=402)


class FakeCallback:
    def __init__(self, chat_id: int) -> None:
        self.data = "music:suno:start"
        self.message = FakeMessage(chat_id)
        self._answers: list[tuple[str | None, bool]] = []

    async def answer(self, text: str | None = None, show_alert: bool = False):  # type: ignore[override]
        self._answers.append((text, show_alert))


def _ready_state(ctx, chat_id: int) -> None:
    state_dict = bot_module.state(ctx)
    state_dict["mode"] = "suno"
    suno_state = load_suno_state(ctx)
    suno_state.mode = "instrumental"
    set_suno_title(suno_state, "Redis Song")
    set_suno_style(suno_state, "lofi")
    save_suno_state(ctx, suno_state)
    state_dict["suno_state"] = suno_state.to_dict()
    start_msg_id = 777
    suno_state.start_msg_id = start_msg_id
    save_suno_state(ctx, suno_state)
    state_dict["suno_state"] = suno_state.to_dict()
    state_dict["suno_start_msg_id"] = start_msg_id
    state_dict.setdefault("msg_ids", {})["suno_start"] = start_msg_id


class FakeRedis:
    def __init__(self) -> None:
        self._stored: set[str] = set()

    def set(self, key: str, value: str, *, nx: bool = False, ex: int | None = None):  # type: ignore[override]
        if nx and key in self._stored:
            return False
        if nx:
            self._stored.add(key)
        return True


def test_start_redis_lock_blocks_duplicates(monkeypatch):
    bot = FakeBot()
    shared_redis = FakeRedis()

    ctx_one = SimpleNamespace(bot=bot, user_data={})
    ctx_two = SimpleNamespace(bot=bot, user_data={})
    chat_id = 808
    user_id = 909
    _ready_state(ctx_one, chat_id)
    _ready_state(ctx_two, chat_id)

    monkeypatch.setattr(bot_module, "START_EMOJI_STICKER_ID", "redis-sticker")
    monkeypatch.setattr(bot_module, "START_EMOJI_FALLBACK", "🎬")
    monkeypatch.setattr(bot_module, "redis_client", shared_redis)
    monkeypatch.setattr(bot_module, "REDIS_LOCK_ENABLED", True)

    async def fake_launch(chat_id_param, ctx_param, **kwargs):  # type: ignore[override]
        return None

    async def fake_notify(*_args, **_kwargs):  # type: ignore[override]
        return None

    monkeypatch.setattr(bot_module, "_launch_suno_generation", fake_launch)
    monkeypatch.setattr(bot_module, "_suno_notify", fake_notify)

    update_one = SimpleNamespace(
        callback_query=FakeCallback(chat_id),
        effective_chat=SimpleNamespace(id=chat_id),
        effective_user=SimpleNamespace(id=user_id),
    )
    update_two = SimpleNamespace(
        callback_query=FakeCallback(chat_id),
        effective_chat=SimpleNamespace(id=chat_id),
        effective_user=SimpleNamespace(id=user_id),
    )

    asyncio.run(bot_module.on_callback(update_one, ctx_one))
    asyncio.run(bot_module.on_callback(update_two, ctx_two))

    stickers = [item for item in bot.sent if item.get("_method") == "send_sticker"]
    assert len(stickers) == 1

    state_one = load_suno_state(ctx_one)
    state_two = load_suno_state(ctx_two)
    assert state_one.start_clicked is True
    assert state_two.start_clicked is False
