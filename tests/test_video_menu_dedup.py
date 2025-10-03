import asyncio
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
    return SimpleNamespace(bot=bot, user_data={}, application=None)


def test_show_video_menu_uses_lock(monkeypatch, bot_module):
    calls: list[tuple[int | None, object | None]] = []
    lock_calls: list[str] = []
    cache: dict[str, str] = {}

    def fake_lock(name: str, ttl: int) -> bool:
        lock_calls.append(name)
        return len(lock_calls) == 1

    monkeypatch.setattr(bot_module, "acquire_ttl_lock", fake_lock)
    monkeypatch.setattr(bot_module, "cache_get", lambda name: cache.get(name))
    monkeypatch.setattr(bot_module, "cache_set", lambda name, value, ttl: cache.update({name: value}))

    async def fake_send(ctx, *, chat_id=None, message=None):  # type: ignore[override]
        calls.append((chat_id, message))
        return 111

    monkeypatch.setattr(bot_module, "_send_video_menu_message", fake_send)

    ctx = _make_ctx()
    asyncio.run(bot_module.show_video_menu(ctx, chat_id=42))
    asyncio.run(bot_module.show_video_menu(ctx, chat_id=42))
    asyncio.run(bot_module.show_video_menu(ctx, chat_id=42))

    assert calls == [(42, None)]
    assert cache == {"video_menu:last_menu_msg_id:42": "111"}
    assert lock_calls == ["lock:video_menu:42", "lock:video_menu:42", "lock:video_menu:42"]


def test_callback_deduplicated(monkeypatch, bot_module):
    send_calls: list[tuple[int | None, object | None]] = []
    answers: list[int] = []
    lock_state = {"count": 0}

    def fake_lock(name: str, ttl: int) -> bool:
        lock_state["count"] += 1
        return lock_state["count"] == 1

    monkeypatch.setattr(bot_module, "acquire_ttl_lock", fake_lock)
    monkeypatch.setattr(bot_module, "cache_get", lambda name: None)
    monkeypatch.setattr(bot_module, "cache_set", lambda name, value, ttl: None)

    async def fake_send(ctx, *, chat_id=None, message=None):  # type: ignore[override]
        send_calls.append((chat_id, message))
        return 321

    monkeypatch.setattr(bot_module, "_send_video_menu_message", fake_send)

    class _Query:
        def __init__(self):
            self.data = bot_module.CB_VIDEO_MENU
            self.message = SimpleNamespace(chat=SimpleNamespace(id=9))
            self.from_user = SimpleNamespace(id=1)

        async def answer(self):  # type: ignore[override]
            answers.append(1)

    ctx = _make_ctx()

    update = SimpleNamespace(
        callback_query=_Query(),
        effective_chat=SimpleNamespace(id=9),
        effective_user=SimpleNamespace(id=1),
    )
    asyncio.run(bot_module.video_menu_callback(update, ctx))

    update.callback_query = _Query()
    asyncio.run(bot_module.video_menu_callback(update, ctx))

    assert len(send_calls) == 1
    assert answers == [1, 1]


def test_command_and_callback_share_lock(monkeypatch, bot_module):
    send_calls: list[tuple[int | None, object | None]] = []
    cache: dict[str, str] = {}
    lock_sequence = iter([True, False])

    def fake_lock(name: str, ttl: int) -> bool:
        return next(lock_sequence, False)

    monkeypatch.setattr(bot_module, "acquire_ttl_lock", fake_lock)
    monkeypatch.setattr(bot_module, "cache_get", lambda name: cache.get(name))
    monkeypatch.setattr(bot_module, "cache_set", lambda name, value, ttl: cache.update({name: value}))

    async def fake_send(ctx, *, chat_id=None, message=None):  # type: ignore[override]
        send_calls.append((chat_id, message))
        return 555

    monkeypatch.setattr(bot_module, "_send_video_menu_message", fake_send)

    ctx = _make_ctx()
    message = SimpleNamespace(chat_id=7, reply_text=lambda *args, **kwargs: asyncio.sleep(0))
    update_command = SimpleNamespace(
        effective_chat=SimpleNamespace(id=7),
        effective_user=SimpleNamespace(id=2),
        effective_message=message,
    )

    asyncio.run(bot_module.video_command(update_command, ctx))

    class _Query:
        def __init__(self):
            self.data = bot_module.CB_VIDEO_MENU
            self.message = SimpleNamespace(chat=SimpleNamespace(id=7))
            self.from_user = SimpleNamespace(id=2)

        async def answer(self):  # type: ignore[override]
            return None

    update_callback = SimpleNamespace(
        callback_query=_Query(),
        effective_chat=SimpleNamespace(id=7),
        effective_user=SimpleNamespace(id=2),
    )
    asyncio.run(bot_module.video_menu_callback(update_callback, ctx))

    assert len(send_calls) == 1
    assert cache == {"video_menu:last_menu_msg_id:7": "555"}


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

