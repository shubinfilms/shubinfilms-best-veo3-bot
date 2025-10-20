import asyncio
from types import SimpleNamespace

import pytest

import bot as bot_module
from ui.buttons import BUTTONS
from ui.buttons.idempotency import _RECENT_CLICKS
from ui.buttons import idempotency as idemp
from ui.buttons.results import UIResult, acknowledge
from ui.buttons.router import ButtonRouter
from ui.buttons.types import ButtonSpec


@pytest.fixture(autouse=True)
def _reset_idempotency(monkeypatch):
    state: set[tuple[object, str]] = set()

    def acquire(owner, key, ttl=1):
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
    monkeypatch.setattr(idemp, "_DEBOUNCE_WINDOW", 0.0)
    monkeypatch.setattr(idemp, "_DEBOUNCE_MS", 0.0)
    _RECENT_CLICKS.clear()


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

    result = asyncio.run(ButtonRouter(BUTTONS).dispatch("profile", update=update, ctx=ctx))
    assert isinstance(result, UIResult)
    assert result.kind == "menu"
    assert result.screen == "profile"
    assert ctx.chat_data[bot_module._MENU_CHATDATA_KEYS["profile"]] == 100 + len("profile")


def test_dispatch_music_requires_paid():
    update = _make_update("menu:music")
    ctx = SimpleNamespace(chat_data={}, user_data={}, application=SimpleNamespace(bot_data={}))

    result = asyncio.run(ButtonRouter(BUTTONS).dispatch("music", update=update, ctx=ctx))
    assert result.kind == "error"
    assert result.details["error_kind"] == "access_denied"

    ctx.user_data["is_paid"] = True
    result_paid = asyncio.run(ButtonRouter(BUTTONS).dispatch("music", update=update, ctx=ctx))
    assert result_paid.kind == "menu"


def test_dispatch_sora2_respects_feature_flag():
    update = _make_update("sora2_open")
    ctx = SimpleNamespace(chat_data={}, user_data={"is_paid": True}, application=SimpleNamespace(bot_data={}))

    result = asyncio.run(ButtonRouter(BUTTONS).dispatch("sora2", update=update, ctx=ctx))
    assert result.kind == "error"
    assert result.details["error_kind"] == "feature_disabled"

    ctx.application.bot_data = {"feature_flags": {"FEATURE_SORA2_ENABLED": True}}
    result_enabled = asyncio.run(ButtonRouter(BUTTONS).dispatch("sora2", update=update, ctx=ctx))
    assert result_enabled.kind == "dialog"


def test_early_ack_sent_immediately(monkeypatch):
    calls: list[str] = []

    async def answer(**kwargs):
        calls.append("ack")

    async def handler(context):
        calls.append("handler")
        assert calls[0] == "ack"
        return UIResult(kind="ack", button_id="test", changed=False)

    spec = ButtonSpec(
        id="test",
        title_i18n_key="t",
        access=("all",),
        handler=handler,
        open_telemetry_event="test",
    )
    router = ButtonRouter({"test": spec})

    message = SimpleNamespace(chat=SimpleNamespace(id=1), chat_id=1)
    query = SimpleNamespace(
        data="menu:test",
        message=message,
        from_user=SimpleNamespace(id=42),
        answer=answer,
    )
    update = SimpleNamespace(
        callback_query=query,
        effective_chat=message.chat,
        effective_user=query.from_user,
    )
    ctx = SimpleNamespace(chat_data={}, user_data={}, application=SimpleNamespace(bot_data={}))

    result = asyncio.run(router.dispatch("test", update=update, ctx=ctx))
    assert result.kind == "ack"
    assert calls == ["ack", "handler"]


def test_no_double_answer_when_ack_already_sent(monkeypatch):
    calls: list[str] = []

    async def answer(**kwargs):
        calls.append("answer")

    async def handler(context):
        return UIResult(kind="noop", button_id="test", changed=False)

    spec = ButtonSpec(
        id="test",
        title_i18n_key="t",
        access=("all",),
        handler=handler,
        open_telemetry_event="test",
    )
    router = ButtonRouter({"test": spec})

    message = SimpleNamespace(chat=SimpleNamespace(id=1), chat_id=1)
    query = SimpleNamespace(
        data="menu:test",
        message=message,
        from_user=SimpleNamespace(id=42),
        answer=answer,
        answered=True,
    )
    update = SimpleNamespace(
        callback_query=query,
        effective_chat=message.chat,
        effective_user=query.from_user,
    )
    ctx = SimpleNamespace(chat_data={}, user_data={}, application=SimpleNamespace(bot_data={}))

    result = asyncio.run(router.dispatch("test", update=update, ctx=ctx))
    assert result.kind == "noop"
    assert calls == []


def test_debounce_same_button_quickly(monkeypatch):
    monkeypatch.setattr(idemp, "_DEBOUNCE_WINDOW", 0.4)
    monkeypatch.setattr(idemp, "_DEBOUNCE_MS", 400.0)
    _RECENT_CLICKS.clear()

    executions: list[str] = []

    async def handler(context):
        executions.append(context.request_id)
        return UIResult(kind="menu", button_id="test", screen="x", changed=True)

    spec = ButtonSpec(
        id="test",
        title_i18n_key="t",
        access=("all",),
        handler=handler,
        open_telemetry_event="test",
    )
    router = ButtonRouter({"test": spec})

    message = SimpleNamespace(chat=SimpleNamespace(id=1), chat_id=1)

    async def answer(**kwargs):
        return None

    def run_once() -> UIResult:
        query = SimpleNamespace(
            data="menu:test",
            message=message,
            from_user=SimpleNamespace(id=42),
            answer=answer,
        )
        update = SimpleNamespace(
            callback_query=query,
            effective_chat=message.chat,
            effective_user=query.from_user,
        )
        ctx = SimpleNamespace(chat_data={}, user_data={}, application=SimpleNamespace(bot_data={}))
        return asyncio.run(router.dispatch("test", update=update, ctx=ctx))

    first = run_once()
    second = run_once()

    assert first.kind == "menu"
    assert second == acknowledge("test", message="duplicate_click")
    assert len(executions) == 1
