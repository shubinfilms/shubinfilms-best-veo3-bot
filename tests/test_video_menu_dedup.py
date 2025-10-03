import asyncio
import importlib
import sys
import time
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
    return SimpleNamespace(
        bot=bot or SimpleNamespace(),
        user_data={},
        application=None,
        chat_data={},
    )


def test_start_video_menu_deduplicates_card(monkeypatch, bot_module):
    locks: set[tuple[str, int]] = set()
    records: dict[tuple[str, int], tuple[int, float]] = {}
    send_calls: list[tuple[int | None, object | None]] = []
    edit_calls: list[tuple[int, int]] = []
    release_calls: list[tuple[str, int]] = []

    def fake_acquire(name: str, chat_id: int, ttl: int) -> bool:
        key = (name, chat_id)
        if key in locks:
            return False
        locks.add(key)
        return True

    def fake_release(name: str, chat_id: int) -> None:
        release_calls.append((name, chat_id))
        locks.discard((name, chat_id))

    def fake_save(name: str, chat_id: int, message_id: int, ttl: int) -> None:
        records[(name, chat_id)] = (message_id, time.time())

    def fake_get(name: str, chat_id: int, *, max_age: int | None = None):
        record = records.get((name, chat_id))
        if not record:
            return None
        if max_age is not None and (time.time() - record[1]) > max_age:
            records.pop((name, chat_id), None)
            return None
        return record

    def fake_clear(name: str, chat_id: int) -> None:
        records.pop((name, chat_id), None)

    async def fake_send(ctx, *, chat_id=None, message=None):  # type: ignore[override]
        send_calls.append((chat_id, message))
        return 777

    async def fake_edit(
        ctx,
        *,
        chat_id: int,
        message_id: int,
        **kwargs,
    ) -> str:
        edit_calls.append((chat_id, message_id))
        return "ok"

    monkeypatch.setattr(bot_module, "acquire_menu_lock", fake_acquire)
    monkeypatch.setattr(bot_module, "release_menu_lock", fake_release)
    monkeypatch.setattr(bot_module, "save_menu_message", fake_save)
    monkeypatch.setattr(bot_module, "get_menu_message", fake_get)
    monkeypatch.setattr(bot_module, "clear_menu_message", fake_clear)
    monkeypatch.setattr(bot_module, "_send_video_menu_message", fake_send)
    monkeypatch.setattr(bot_module, "_try_edit_menu_card", fake_edit)
    monkeypatch.setattr(bot_module, "input_state", SimpleNamespace(clear=lambda *args, **kwargs: None))
    monkeypatch.setattr(bot_module, "set_mode", lambda *args, **kwargs: None)
    monkeypatch.setattr(bot_module, "clear_wait", lambda *args, **kwargs: None)
    monkeypatch.setattr(bot_module, "clear_wait_state", lambda *args, **kwargs: None)

    update = SimpleNamespace(
        effective_message=SimpleNamespace(chat_id=42, reply_text=lambda *args, **kwargs: SimpleNamespace(message_id=999)),
        effective_chat=SimpleNamespace(id=42),
        effective_user=SimpleNamespace(id=9),
        callback_query=None,
    )
    ctx = _make_ctx()

    asyncio.run(bot_module.start_video_menu(update, ctx))
    assert send_calls == [(42, update.effective_message)]
    key = (bot_module._VIDEO_MENU_MESSAGE_NAME, 42)
    assert records[key][0] == 777

    asyncio.run(bot_module.start_video_menu(update, ctx))
    assert send_calls == [(42, update.effective_message)]
    assert edit_calls == [(42, 777)]
    assert release_calls == [
        (bot_module._VIDEO_MENU_LOCK_NAME, 9),
        (bot_module._VIDEO_MENU_LOCK_NAME, 9),
    ]


def test_start_video_menu_edits_when_lock_active(monkeypatch, bot_module):
    message_key = (bot_module._VIDEO_MENU_MESSAGE_NAME, 7)
    records = {message_key: (123, time.time())}
    edit_calls: list[tuple[int, int]] = []

    def fake_acquire(name: str, chat_id: int, ttl: int) -> bool:
        return False

    def fake_release(name: str, chat_id: int) -> None:
        pass

    def fake_get(name: str, chat_id: int, *, max_age: int | None = None):
        return records.get((name, chat_id))

    def fake_save(name: str, chat_id: int, message_id: int, ttl: int) -> None:
        records[(name, chat_id)] = (message_id, time.time())

    async def fake_edit(
        ctx,
        *,
        chat_id: int,
        message_id: int,
        **kwargs,
    ) -> str:
        edit_calls.append((chat_id, message_id))
        return "ok"

    async def fail_send(*args, **kwargs):  # type: ignore[override]
        raise AssertionError("send should not be called when editing is possible")

    monkeypatch.setattr(bot_module, "acquire_menu_lock", fake_acquire)
    monkeypatch.setattr(bot_module, "release_menu_lock", fake_release)
    monkeypatch.setattr(bot_module, "get_menu_message", fake_get)
    monkeypatch.setattr(bot_module, "save_menu_message", fake_save)
    monkeypatch.setattr(bot_module, "clear_menu_message", lambda *args, **kwargs: None)
    monkeypatch.setattr(bot_module, "_try_edit_menu_card", fake_edit)
    monkeypatch.setattr(bot_module, "_send_video_menu_message", fail_send)
    monkeypatch.setattr(bot_module, "input_state", SimpleNamespace(clear=lambda *args, **kwargs: None))
    monkeypatch.setattr(bot_module, "set_mode", lambda *args, **kwargs: None)
    monkeypatch.setattr(bot_module, "clear_wait", lambda *args, **kwargs: None)
    monkeypatch.setattr(bot_module, "clear_wait_state", lambda *args, **kwargs: None)

    update = SimpleNamespace(
        effective_message=SimpleNamespace(chat_id=7),
        effective_chat=SimpleNamespace(id=7),
        effective_user=SimpleNamespace(id=12),
        callback_query=None,
    )

    asyncio.run(bot_module.start_video_menu(update, _make_ctx()))
    assert edit_calls == [(7, 123)]


def test_video_menu_callback_delegates_to_start(monkeypatch, bot_module):
    calls: list[tuple[object, object]] = []

    async def fake_start(update, ctx):  # type: ignore[override]
        calls.append((update, ctx))

    monkeypatch.setattr(bot_module, "start_video_menu", fake_start)

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
