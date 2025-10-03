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


def _make_update(bot_module):
    class _Query:
        def __init__(self):
            self.data = bot_module.CB.VIDEO_MENU
            self.id = "lock-test"
            self.message = SimpleNamespace(chat=SimpleNamespace(id=12), message_id=444)
            self.from_user = SimpleNamespace(id=34)

    return SimpleNamespace(
        callback_query=_Query(),
        effective_chat=SimpleNamespace(id=12),
        effective_user=SimpleNamespace(id=34),
    )


def test_menu_lock_suppresses_duplicate(monkeypatch, bot_module):
    answered: list[tuple[str, str, bool]] = []
    state_data = {"state": {"msg_ids": {}}}

    start_calls = 0

    async def fake_start_video_menu(update, ctx):  # type: ignore[override]
        nonlocal start_calls
        start_calls += 1
        if start_calls > 1:
            raise bot_module.MenuLocked(bot_module._VIDEO_MENU_LOCK_NAME, 12)
        return None

    async def fake_answer_callback_query(*, callback_query_id, text="", show_alert=False):
        answered.append((callback_query_id, text, show_alert))

    async def fake_ensure(update):  # type: ignore[override]
        return None

    monkeypatch.setattr(bot_module, "ensure_user_record", fake_ensure)
    monkeypatch.setattr(bot_module, "state", lambda ctx: state_data.setdefault("state", {"msg_ids": {}}))
    monkeypatch.setattr(bot_module, "save_menu_message", lambda *args, **kwargs: None)
    monkeypatch.setattr(bot_module, "clear_menu_message", lambda *args, **kwargs: None)
    monkeypatch.setattr(bot_module, "start_video_menu", fake_start_video_menu)

    ctx = SimpleNamespace(bot=SimpleNamespace(answer_callback_query=fake_answer_callback_query), user_data={})

    update = _make_update(bot_module)

    asyncio.run(bot_module.video_menu_callback(update, ctx))
    asyncio.run(bot_module.video_menu_callback(update, ctx))

    assert start_calls == 2
    assert answered[-1][1] == "Обрабатываю…"
