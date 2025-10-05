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
        return SimpleNamespace(message_id=401)


class FakeCallback:
    def __init__(self, chat_id: int) -> None:
        self.data = "music:suno:start"
        self.message = FakeMessage(chat_id)
        self._answers: list[tuple[str | None, bool]] = []

    async def answer(self, text: str | None = None, show_alert: bool = False):  # type: ignore[override]
        self._answers.append((text, show_alert))


def _prepare_state(ctx, chat_id: int, user_id: int) -> None:
    state_dict = bot_module.state(ctx)
    state_dict["mode"] = "suno"
    suno_state = load_suno_state(ctx)
    suno_state.mode = "instrumental"
    set_suno_title(suno_state, "Race Song")
    set_suno_style(suno_state, "chill")
    save_suno_state(ctx, suno_state)
    state_dict["suno_state"] = suno_state.to_dict()
    start_msg_id = 555
    suno_state.start_msg_id = start_msg_id
    save_suno_state(ctx, suno_state)
    state_dict["suno_state"] = suno_state.to_dict()
    state_dict["suno_start_msg_id"] = start_msg_id
    state_dict.setdefault("msg_ids", {})["suno_start"] = start_msg_id


def test_start_idempotent_under_race(monkeypatch):
    bot = FakeBot()
    ctx = SimpleNamespace(bot=bot, user_data={})
    chat_id = 42
    user_id = 77
    _prepare_state(ctx, chat_id, user_id)

    monkeypatch.setattr(bot_module, "START_EMOJI_STICKER_ID", "race-sticker")
    monkeypatch.setattr(bot_module, "START_EMOJI_FALLBACK", "🎬")

    launch_calls: list[dict[str, object]] = []

    async def fake_launch(chat_id_param, ctx_param, **kwargs):  # type: ignore[override]
        launch_calls.append({"chat_id": chat_id_param, "user_id": kwargs.get("user_id")})

    async def fake_notify(*_args, **_kwargs):  # type: ignore[override]
        return None

    sticker_calls = 0

    async def controlled_sticker(bot_obj, chat, sticker, **kwargs):  # type: ignore[override]
        nonlocal sticker_calls
        sticker_calls += 1
        await asyncio.sleep(0)
        return await FakeBot.send_sticker(bot_obj, chat_id=chat, sticker=sticker, **kwargs)

    monkeypatch.setattr(bot_module, "_launch_suno_generation", fake_launch)
    monkeypatch.setattr(bot_module, "_suno_notify", fake_notify)
    monkeypatch.setattr(bot_module, "safe_send_sticker", controlled_sticker)

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

    async def _run_parallel() -> None:
        await asyncio.gather(
            bot_module.on_callback(update_one, ctx),
            bot_module.on_callback(update_two, ctx),
        )

    asyncio.run(_run_parallel())

    assert sticker_calls == 1
    assert len([item for item in bot.sent if item.get("_method") == "send_sticker"]) == 1
    assert len(launch_calls) == 1

    suno_state_after = load_suno_state(ctx)
    assert suno_state_after.start_clicked is True
    assert suno_state_after.start_emoji_msg_id == 100
