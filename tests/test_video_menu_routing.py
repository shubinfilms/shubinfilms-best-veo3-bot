import asyncio
import importlib
import sys
from contextlib import asynccontextmanager
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


async def _recording_answer(**kwargs):
    return None


def _make_ctx(answer_fn):
    return SimpleNamespace(
        bot=SimpleNamespace(answer_callback_query=answer_fn),
        user_data={},
    )


def _prep_common(monkeypatch, bot_module):
    async def fake_ensure(update):  # type: ignore[override]
        return None

    monkeypatch.setattr(bot_module, "ensure_user_record", fake_ensure)
    monkeypatch.setattr(bot_module, "save_menu_message", lambda *args, **kwargs: None)
    monkeypatch.setattr(bot_module, "clear_menu_message", lambda *args, **kwargs: None)
    monkeypatch.setattr(bot_module, "start_video_menu", lambda *args, **kwargs: None)
    monkeypatch.setattr(bot_module, "_sora2_is_enabled", lambda: True)
    state_data = {"state": {"msg_ids": {}}}
    monkeypatch.setattr(bot_module, "state", lambda ctx: state_data.setdefault("state", {"msg_ids": {}}))

    @asynccontextmanager
    async def fake_with_menu_lock(name, chat_id, ttl=0):  # type: ignore[override]
        yield

    monkeypatch.setattr(bot_module, "with_menu_lock", fake_with_menu_lock)


def _make_update(data: str):
    class _Query:
        def __init__(self, payload: str):
            self.data = payload
            self.id = "qid"
            self.message = SimpleNamespace(chat=SimpleNamespace(id=77), message_id=999)
            self.from_user = SimpleNamespace(id=55)

    return SimpleNamespace(
        callback_query=_Query(data),
        effective_chat=SimpleNamespace(id=77),
        effective_user=SimpleNamespace(id=55),
    )


def test_video_pick_veo_routes_to_modes(monkeypatch, bot_module):
    _prep_common(monkeypatch, bot_module)
    calls: list[dict[str, object]] = []

    async def fake_safe_edit_or_send_menu(*args, **kwargs):  # type: ignore[override]
        calls.append(kwargs)
        return 111

    monkeypatch.setattr(bot_module, "safe_edit_or_send_menu", fake_safe_edit_or_send_menu)
    ctx = _make_ctx(_recording_answer)

    asyncio.run(bot_module.video_menu_callback(_make_update(bot_module.CB.VIDEO_PICK_VEO), ctx))

    assert calls, "safe_edit_or_send_menu was not invoked"
    assert calls[0]["text"] == bot_module.VIDEO_VEO_MENU_TEXT
    assert calls[0]["state_key"] == bot_module.VIDEO_MENU_STATE_KEY


def test_video_pick_sora_routes_to_flow(monkeypatch, bot_module):
    _prep_common(monkeypatch, bot_module)
    clear_calls: list[int] = []
    sora_calls: list[int] = []

    async def fake_clear(chat_id, *, user_id=None, ctx=None):  # type: ignore[override]
        clear_calls.append(chat_id)

    async def fake_sora(chat_id, ctx):  # type: ignore[override]
        sora_calls.append(chat_id)

    monkeypatch.setattr(bot_module, "_clear_video_menu_state", fake_clear)
    monkeypatch.setattr(bot_module, "sora2_entry", fake_sora)
    ctx = _make_ctx(_recording_answer)

    asyncio.run(bot_module.video_menu_callback(_make_update(bot_module.CB.VIDEO_PICK_SORA2), ctx))

    assert clear_calls == [77]
    assert sora_calls == [77]


def test_video_back_returns_to_menu(monkeypatch, bot_module):
    _prep_common(monkeypatch, bot_module)
    show_calls: list[int] = []

    async def fake_clear(chat_id, *, user_id=None, ctx=None):  # type: ignore[override]
        return None

    async def fake_show(chat_id, ctx, user_id=None, replace=False):  # type: ignore[override]
        show_calls.append(chat_id)

    monkeypatch.setattr(bot_module, "_clear_video_menu_state", fake_clear)
    monkeypatch.setattr(bot_module, "show_emoji_hub_for_chat", fake_show)
    ctx = _make_ctx(_recording_answer)

    asyncio.run(bot_module.video_menu_callback(_make_update(bot_module.CB.VIDEO_MENU_BACK), ctx))

    assert show_calls == [77]
