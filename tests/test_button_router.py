import asyncio
from types import SimpleNamespace

import pytest

from tests.suno_test_utils import bot_module
from ui.buttons import BUTTONS
from ui.buttons import router as button_router
from ui.buttons.results import UIResult


@pytest.fixture(autouse=True)
def fake_idempotency(monkeypatch):
    from ui.buttons import idempotency as idemp

    state: set[tuple[object, str]] = set()

    def acquire(owner, key, ttl=5):
        if owner is None:
            return True
        token = (owner, key)
        if token in state:
            return False
        state.add(token)
        return True

    def release(owner, key):
        state.discard((owner, key))

    monkeypatch.setattr(idemp, "acquire_action_lock", acquire)
    monkeypatch.setattr(idemp, "release_action_lock", release)


@pytest.mark.parametrize(
    "button_id",
    ["profile", "kb", "photo", "music", "video", "dialog", "help", "sora2"],
)
def test_registry_contains_expected_buttons(button_id):
    assert button_id in BUTTONS


def _make_update(callback_data: str):
    async def answer(**kwargs):
        return None

    message = SimpleNamespace(chat=SimpleNamespace(id=123), chat_id=123)
    query = SimpleNamespace(
        data=callback_data,
        message=message,
        from_user=SimpleNamespace(id=777),
        answer=answer,
    )
    update = SimpleNamespace(
        callback_query=query,
        effective_chat=message.chat,
        effective_user=query.from_user,
    )
    return update


async def _fake_perform(item, *, update, ctx, query, log_click):
    key = bot_module._MENU_CHATDATA_KEYS[item]
    ctx.chat_data[key] = 100 + len(item)
    return 100 + len(item)


@pytest.fixture(autouse=True)
def patch_perform(monkeypatch):
    monkeypatch.setattr(bot_module, "_perform_menu_open", _fake_perform)

    async def fake_help(update, ctx):
        ctx.user_data["help_called"] = True

    async def fake_sora(update, ctx):
        ctx.user_data["sora_called"] = True

    monkeypatch.setattr(bot_module, "help_command_entry", fake_help)
    monkeypatch.setattr(bot_module, "sora2_open_cb", fake_sora)


def test_dispatch_profile_updates_chat_data():
    update = _make_update("menu:profile")
    ctx = SimpleNamespace(chat_data={}, user_data={}, application=SimpleNamespace(bot_data={}))

    result = asyncio.run(button_router.dispatch("profile", update=update, ctx=ctx))
    assert isinstance(result, UIResult)
    assert result.kind == "menu"
    assert result.screen == "profile"
    assert ctx.chat_data[bot_module._MENU_CHATDATA_KEYS["profile"]] == 100 + len("profile")


def test_dispatch_music_requires_paid():
    update = _make_update("menu:music")
    ctx = SimpleNamespace(chat_data={}, user_data={}, application=SimpleNamespace(bot_data={}))

    result = asyncio.run(button_router.dispatch("music", update=update, ctx=ctx))
    assert result.kind == "error"
    assert result.details["error_kind"] == "access_denied"

    ctx.user_data["is_paid"] = True
    result_paid = asyncio.run(button_router.dispatch("music", update=update, ctx=ctx))
    assert result_paid.kind == "menu"


def test_dispatch_sora2_respects_feature_flag():
    update = _make_update("sora2_open")
    ctx = SimpleNamespace(chat_data={}, user_data={"is_paid": True}, application=SimpleNamespace(bot_data={}))

    result = asyncio.run(button_router.dispatch("sora2", update=update, ctx=ctx))
    assert result.kind == "error"
    assert result.details["error_kind"] == "feature_disabled"

    ctx.application.bot_data = {"feature_flags": {"FEATURE_SORA2_ENABLED": True}}
    result_enabled = asyncio.run(button_router.dispatch("sora2", update=update, ctx=ctx))
    assert result_enabled.kind == "dialog"
