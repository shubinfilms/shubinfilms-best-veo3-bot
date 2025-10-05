import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chat_service import load_ctx
from redis_utils import clear_mode_state, get_active_mode
from tests.suno_test_utils import FakeBot, bot_module


@pytest.fixture(autouse=True)
def _reset_state():
    user_id = 1000
    chat_id = 2000
    asyncio.run(clear_mode_state(user_id))
    bot_module._pm_clear_step(user_id)
    bot_module._pm_clear_buffer(user_id)
    bot_module.clear_cached_pm_prompt(chat_id)
    yield
    asyncio.run(clear_mode_state(user_id))
    bot_module._pm_clear_step(user_id)
    bot_module._pm_clear_buffer(user_id)
    bot_module.clear_cached_pm_prompt(chat_id)


@pytest.fixture
def ctx():
    bot = FakeBot()
    return SimpleNamespace(bot=bot, user_data={}, chat_data={}, application=SimpleNamespace(logger=bot_module.log))


@pytest.fixture
def message():
    chat_id = 2000
    return SimpleNamespace(message_id=77, chat=SimpleNamespace(id=chat_id), chat_id=chat_id)


def _make_update(user_id: int, chat_id: int, message: SimpleNamespace):
    return SimpleNamespace(
        effective_user=SimpleNamespace(id=user_id),
        effective_chat=SimpleNamespace(id=chat_id),
        effective_message=message,
    )


def _patch_safe_io(monkeypatch, edits, sends):
    async def fake_safe_edit_message(ctx, chat_id, message_id, text, reply_markup=None, **kwargs):
        edits.append((chat_id, message_id, text, reply_markup))
        return True

    async def fake_safe_send_text(bot, chat_id, text, **kwargs):
        sends.append((chat_id, text))
        return None

    monkeypatch.setattr(bot_module, "safe_edit_message", fake_safe_edit_message)
    monkeypatch.setattr(bot_module, "safe_send_text", fake_safe_send_text)


def test_switch_dialog_to_pm_exclusive(monkeypatch, ctx, message):
    user_id = 1000
    chat_id = message.chat_id
    edits: list = []
    sends: list = []
    _patch_safe_io(monkeypatch, edits, sends)

    pm_calls: list = []

    async def fake_prompt_master_open(update, context):
        pm_calls.append((update, context))
        return None

    monkeypatch.setattr(bot_module, "prompt_master_open", fake_prompt_master_open)

    update = _make_update(user_id, chat_id, message)

    asyncio.run(bot_module.start_mode(update, ctx, "dialog_default"))

    # simulate existing conversation + prompt master cache
    bot_module.append_ctx(user_id, "user", "hello")
    bot_module.cache_pm_prompt(chat_id, "old prompt")
    bot_module._pm_set_step(user_id, "banana")
    bot_module._pm_set_buffer(user_id, {"foo": "bar"})
    ctx.user_data["pm_state"] = {"engine": "veo"}
    ctx.chat_data["prompt_master"] = {"last_result": {"raw": "cached"}}

    asyncio.run(bot_module.start_mode(update, ctx, "prompt_master"))

    assert pm_calls, "prompt_master_open should be called"
    assert asyncio.run(get_active_mode(user_id)) == "prompt_master"
    assert load_ctx(user_id) == []
    assert bot_module._pm_get_step(user_id) is None
    assert bot_module._pm_get_buffer(user_id) is None
    assert ctx.user_data.get("pm_state") is None
    assert ctx.chat_data.get("prompt_master") is None
    assert bot_module.get_cached_pm_prompt(chat_id) is None


def test_switch_pm_to_dialog_exclusive(monkeypatch, ctx, message):
    user_id = 1000
    chat_id = message.chat_id
    edits: list = []
    sends: list = []
    _patch_safe_io(monkeypatch, edits, sends)

    async def fake_prompt_master_open(update, context):
        return None

    monkeypatch.setattr(bot_module, "prompt_master_open", fake_prompt_master_open)

    update = _make_update(user_id, chat_id, message)

    asyncio.run(bot_module.start_mode(update, ctx, "prompt_master"))

    bot_module._pm_set_step(user_id, "mj")
    bot_module._pm_set_buffer(user_id, {"bar": 1})
    ctx.user_data["pm_state"] = {"engine": "mj"}
    ctx.chat_data["prompt_master"] = {"last_result": {"raw": "value"}}
    bot_module.cache_pm_prompt(chat_id, "stale prompt")
    bot_module.append_ctx(user_id, "assistant", "old reply")

    asyncio.run(bot_module.start_mode(update, ctx, "dialog_default"))

    assert asyncio.run(get_active_mode(user_id)) == "dialog_default"
    assert load_ctx(user_id) == []
    assert bot_module._pm_get_step(user_id) is None
    assert bot_module._pm_get_buffer(user_id) is None
    assert ctx.user_data.get("pm_state") is None
    assert ctx.chat_data.get("prompt_master") is None
    assert bot_module.get_cached_pm_prompt(chat_id) is None


def test_reset_clears_mode(monkeypatch, ctx, message):
    user_id = 1000
    chat_id = message.chat_id
    sends: list = []
    edits: list = []
    _patch_safe_io(monkeypatch, edits, sends)

    bot_module._pm_set_step(user_id, "banana")
    bot_module._pm_set_buffer(user_id, {"payload": True})
    bot_module.append_ctx(user_id, "user", "question")
    ctx.user_data["pm_state"] = {"engine": "veo"}
    ctx.chat_data["prompt_master"] = {"last_result": {"raw": "value"}}

    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=user_id),
        effective_chat=SimpleNamespace(id=chat_id),
        effective_message=SimpleNamespace(chat_id=chat_id),
    )

    asyncio.run(bot_module.chat_reset_command(update, ctx))

    assert asyncio.run(get_active_mode(user_id)) is None
    assert load_ctx(user_id) == []
    assert bot_module._pm_get_step(user_id) is None
    assert bot_module._pm_get_buffer(user_id) is None
    assert ctx.user_data.get("pm_state") is None
    assert ctx.chat_data.get("prompt_master") is None
    assert any("Контекст очищен" in text for _, text in sends)
