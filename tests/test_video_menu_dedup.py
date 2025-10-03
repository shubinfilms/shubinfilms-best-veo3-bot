import asyncio
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def bot_module(monkeypatch):
    monkeypatch.setenv("TELEGRAM_TOKEN", "dummy-token")
    monkeypatch.setenv("SUNO_API_BASE", "https://example.com")
    monkeypatch.setenv("SUNO_API_TOKEN", "token")
    monkeypatch.setenv("LEDGER_BACKEND", "memory")
    monkeypatch.setenv("DATABASE_URL", "postgres://test")
    module = importlib.import_module("bot")
    return importlib.reload(module)


def _make_ctx(bot=None):
    return SimpleNamespace(bot=bot or SimpleNamespace(), user_data={}, application=None)


def test_start_video_mode_deduplicates_card(monkeypatch, bot_module):
    cache: dict[str, str] = {}
    send_calls: list[tuple[int | None, object | None]] = []
    edit_calls: list[tuple[int, int]] = []
    release_calls: list[str] = []

    monkeypatch.setattr(bot_module, "acquire_ttl_lock", lambda name, ttl: True)
    monkeypatch.setattr(bot_module, "release_ttl_lock", lambda name: release_calls.append(name))
    monkeypatch.setattr(bot_module, "cache_get", lambda name: cache.get(name))
    monkeypatch.setattr(bot_module, "cache_set", lambda name, value, ttl: cache.update({name: value}))

    async def fake_send(ctx, *, chat_id=None, message=None):  # type: ignore[override]
        send_calls.append((chat_id, message))
        return 777

    async def fake_edit(ctx, chat_id, message_id, **kwargs):  # type: ignore[override]
        edit_calls.append((chat_id, message_id))
        return True

    monkeypatch.setattr(bot_module, "_send_video_menu_message", fake_send)
    monkeypatch.setattr(bot_module, "safe_edit_message", fake_edit)
    monkeypatch.setattr(bot_module, "input_state", SimpleNamespace(clear=lambda *args, **kwargs: None))
    monkeypatch.setattr(bot_module, "set_mode", lambda *args, **kwargs: None)
    monkeypatch.setattr(bot_module, "clear_wait", lambda *args, **kwargs: None)

    update = SimpleNamespace(
        effective_message=SimpleNamespace(chat_id=42, reply_text=lambda *args, **kwargs: SimpleNamespace(message_id=999)),
        effective_chat=SimpleNamespace(id=42),
        effective_user=SimpleNamespace(id=9),
        callback_query=None,
    )
    ctx = _make_ctx()

    asyncio.run(bot_module.start_video_mode(update, ctx))
    assert send_calls == [(42, update.effective_message)]
    assert cache == {"video:last_msg:42": "777"}

    asyncio.run(bot_module.start_video_mode(update, ctx))
    assert send_calls == [(42, update.effective_message)]
    assert edit_calls == [(42, 777)]
    assert release_calls == ["lock:video_menu:9", "lock:video_menu:9"]


def test_start_video_mode_respects_lock(monkeypatch, bot_module):
    monkeypatch.setattr(bot_module, "acquire_ttl_lock", lambda *args, **kwargs: False)
    monkeypatch.setattr(bot_module, "input_state", SimpleNamespace(clear=lambda *args, **kwargs: None))
    monkeypatch.setattr(bot_module, "set_mode", lambda *args, **kwargs: None)
    monkeypatch.setattr(bot_module, "clear_wait", lambda *args, **kwargs: None)

    async def fake_send(*args, **kwargs):  # type: ignore[override]
        raise AssertionError("send should not be called when lock is active")

    monkeypatch.setattr(bot_module, "_send_video_menu_message", fake_send)
    monkeypatch.setattr(bot_module, "safe_edit_message", fake_send)

    update = SimpleNamespace(
        effective_message=SimpleNamespace(chat_id=7),
        effective_chat=SimpleNamespace(id=7),
        effective_user=SimpleNamespace(id=12),
        callback_query=None,
    )

    asyncio.run(bot_module.start_video_mode(update, _make_ctx()))


def test_video_menu_callback_delegates_to_start(monkeypatch, bot_module):
    calls: list[tuple[object, object]] = []

    async def fake_start(update, ctx):  # type: ignore[override]
        calls.append((update, ctx))

    monkeypatch.setattr(bot_module, "start_video_mode", fake_start)

    async def fake_ensure(update):
        return None

    monkeypatch.setattr(bot_module, "ensure_user_record", fake_ensure)
    monkeypatch.setattr(bot_module, "input_state", SimpleNamespace(clear=lambda *args, **kwargs: None))
    monkeypatch.setattr(bot_module, "set_mode", lambda *args, **kwargs: None)
    monkeypatch.setattr(bot_module, "clear_wait", lambda *args, **kwargs: None)
    monkeypatch.setattr(bot_module, "clear_wait_state", lambda *args, **kwargs: None)

    class _Query:
        def __init__(self):
            self.data = bot_module.CB_VIDEO_MENU
            self.message = SimpleNamespace(chat=SimpleNamespace(id=5))
            self.from_user = SimpleNamespace(id=3)

        async def answer(self):  # type: ignore[override]
            return None

    update = SimpleNamespace(
        callback_query=_Query(),
        effective_chat=SimpleNamespace(id=5),
        effective_user=SimpleNamespace(id=3),
    )

    asyncio.run(bot_module.video_menu_callback(update, _make_ctx()))
    assert len(calls) == 1


def test_video_mode_button_creates_card(monkeypatch, bot_module):
    ctx = SimpleNamespace(bot=SimpleNamespace(), user_data={}, application=None)
    state_dict = bot_module.state(ctx)
    state_dict["last_ui_msg_id_veo"] = None

    veo_calls: list[int] = []
    wait_calls: list[dict[str, object]] = []

    async def _fake_veo_entry(chat_id: int, ctx_param):
        veo_calls.append(chat_id)
        state_dict["last_ui_msg_id_veo"] = 42

    def _fake_activate_wait_state(*, user_id, chat_id, card_msg_id, kind, meta=None):
        wait_calls.append({"user_id": user_id, "chat_id": chat_id, "card": card_msg_id, "kind": kind, "meta": meta})

    monkeypatch.setattr(bot_module, "veo_entry", _fake_veo_entry)
    monkeypatch.setattr(bot_module, "_activate_wait_state", _fake_activate_wait_state)

    class _Query:
        def __init__(self):
            self.data = bot_module.CB_VIDEO_MODE_FAST
            self.message = SimpleNamespace(chat=SimpleNamespace(id=606))
            self.from_user = SimpleNamespace(id=808)

        async def answer(self):
            return None

    update = SimpleNamespace(
        callback_query=_Query(),
        effective_chat=SimpleNamespace(id=606),
        effective_user=SimpleNamespace(id=808),
    )

    asyncio.run(bot_module.video_menu_callback(update, ctx))

    assert state_dict["mode"] == "veo_text_fast"
    assert veo_calls == [606]
    assert wait_calls and wait_calls[0]["kind"] == bot_module.WaitKind.VEO_PROMPT
