import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.suno_test_utils import FakeBot, bot_module


@pytest.fixture
def ctx():
    bot = FakeBot()
    return SimpleNamespace(bot=bot, user_data={}, chat_data={}, application=SimpleNamespace(logger=bot_module.log))


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    async def fake_ensure(_update):
        return None

    monkeypatch.setattr(bot_module, "ensure_user_record", fake_ensure)

    async def fake_safe_edit_message(ctx_param, chat_id_param, message_id_param, text_param, reply_markup_param, **kwargs):
        return True

    monkeypatch.setattr(bot_module, "safe_edit_message", fake_safe_edit_message)

    yield


def _make_update(chat_id: int, user_id: int):
    message = SimpleNamespace(message_id=42, chat=SimpleNamespace(id=chat_id), chat_id=chat_id)

    async def answer():
        return None

    query = SimpleNamespace(data=None, message=message, answer=answer)
    update = SimpleNamespace(
        callback_query=query,
        effective_chat=SimpleNamespace(id=chat_id),
        effective_user=SimpleNamespace(id=user_id),
        effective_message=message,
    )
    return update, query


def test_chat_mode_cleared_when_opening_video(monkeypatch, ctx):
    bot = ctx.bot

    video_calls = []

    async def fake_start_video_menu(update, context):
        video_calls.append(update.callback_query.data)
        return None

    monkeypatch.setattr(bot_module, "start_video_menu", fake_start_video_menu)

    update, query = _make_update(chat_id=100, user_id=200)

    # Open AI modes menu
    query.data = bot_module.AI_MENU_CB
    asyncio.run(bot_module.hub_router(update, ctx))

    # Enable normal chat mode
    query.data = bot_module.AI_TO_SIMPLE_CB
    asyncio.run(bot_module.hub_router(update, ctx))

    state = bot_module.state(ctx)
    assert state.get(bot_module.STATE_CHAT_MODE) == "normal"

    # Navigate to video section
    query.data = bot_module.VIDEO_MENU_CB
    asyncio.run(bot_module.hub_router(update, ctx))

    state_after = bot_module.state(ctx)
    assert state_after.get(bot_module.STATE_CHAT_MODE) is None
    assert video_calls, "Video menu should be opened"
    assert any(entry.get("text") == "🛑 Режим диалога отключён." for entry in bot.sent)


def test_prompt_master_disabled_when_opening_profile(monkeypatch, ctx):
    bot = ctx.bot

    profile_calls = []

    async def fake_show_balance(chat_id, context, *, force_new=False):
        profile_calls.append((chat_id, force_new))
        return None

    monkeypatch.setattr(bot_module, "show_balance_card", fake_show_balance)

    update, query = _make_update(chat_id=300, user_id=400)

    query.data = bot_module.AI_MENU_CB
    asyncio.run(bot_module.hub_router(update, ctx))

    query.data = bot_module.AI_TO_PROMPTMASTER_CB
    asyncio.run(bot_module.hub_router(update, ctx))

    state = bot_module.state(ctx)
    assert state.get(bot_module.STATE_CHAT_MODE) == "prompt_master"

    query.data = bot_module.PROFILE_MENU_CB
    asyncio.run(bot_module.hub_router(update, ctx))

    state_after = bot_module.state(ctx)
    assert state_after.get(bot_module.STATE_CHAT_MODE) is None
    assert profile_calls, "Profile card should be shown"
    assert not any("Обычный чат включён" in (entry.get("text") or "") for entry in bot.sent)
    assert any(entry.get("text") == "🛑 Режим диалога отключён." for entry in bot.sent)
