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


@pytest.mark.parametrize(
    "callback_data, sora_enabled",
    [
        (lambda m: m.CB.VIDEO_MENU, True),
        (lambda m: m.CB.VIDEO_PICK_VEO, True),
        (lambda m: m.CB.VIDEO_PICK_SORA2, True),
        (lambda m: m.CB.VIDEO_PICK_SORA2_DISABLED, True),
        (lambda m: m.CB.VIDEO_MENU_BACK, True),
        (lambda m: m.CB.VIDEO_MODE_VEO_FAST, True),
        (lambda m: m.CB.VIDEO_MODE_SORA_TEXT, True),
        (lambda m: "engine:veo", True),
        (lambda m: m.CB.VIDEO_PICK_SORA2, False),
    ],
)
def test_video_menu_callback_answers(monkeypatch, bot_module, callback_data, sora_enabled):
    answered: list[tuple[str, str, bool]] = []
    state_store: dict[str, dict[str, object]] = {"state": {"msg_ids": {}}}

    async def fake_answer_callback_query(*, callback_query_id, text="", show_alert=False):
        answered.append((callback_query_id, text, show_alert))

    async def fake_start_video_menu(update, ctx):  # type: ignore[override]
        return None

    async def fake_safe_edit_or_send_menu(*args, **kwargs):  # type: ignore[override]
        state_dict = kwargs.get("state_dict")
        if isinstance(state_dict, dict):
            state_dict[kwargs.get("state_key")] = 321
            msg_key = kwargs.get("msg_ids_key")
            if msg_key:
                msg_ids = state_dict.setdefault("msg_ids", {})
                if isinstance(msg_ids, dict):
                    msg_ids[msg_key] = 321
        return 321

    async def fake_clear_menu(chat_id, *, user_id=None, ctx=None):  # type: ignore[override]
        return None

    async def fake_sora2_entry(chat_id, ctx):  # type: ignore[override]
        return None

    async def fake_start_mode(*args, **kwargs):  # type: ignore[override]
        return True

    async def fake_show_hub(*args, **kwargs):  # type: ignore[override]
        return None

    @asynccontextmanager
    async def fake_with_menu_lock(name, chat_id, ttl=0):  # type: ignore[override]
        yield

    def fake_state(ctx):
        return state_store.setdefault("state", {"msg_ids": {}})

    async def fake_ensure(update):  # type: ignore[override]
        return None

    monkeypatch.setattr(bot_module, "ensure_user_record", fake_ensure)
    monkeypatch.setattr(bot_module, "state", fake_state)
    monkeypatch.setattr(bot_module, "with_menu_lock", fake_with_menu_lock)
    monkeypatch.setattr(bot_module, "safe_edit_or_send_menu", fake_safe_edit_or_send_menu)
    monkeypatch.setattr(bot_module, "save_menu_message", lambda *args, **kwargs: None)
    monkeypatch.setattr(bot_module, "clear_menu_message", lambda *args, **kwargs: None)
    monkeypatch.setattr(bot_module, "start_video_menu", fake_start_video_menu)
    monkeypatch.setattr(bot_module, "_clear_video_menu_state", fake_clear_menu)
    monkeypatch.setattr(bot_module, "sora2_entry", fake_sora2_entry)
    monkeypatch.setattr(bot_module, "_start_video_mode", fake_start_mode)
    monkeypatch.setattr(bot_module, "show_emoji_hub_for_chat", fake_show_hub)
    monkeypatch.setattr(bot_module, "_sora2_is_enabled", lambda: sora_enabled)

    ctx = SimpleNamespace(
        bot=SimpleNamespace(answer_callback_query=fake_answer_callback_query),
        user_data={},
    )

    class _Query:
        def __init__(self, data_value: str):
            self.data = data_value
            self.id = "123"
            self.message = SimpleNamespace(chat=SimpleNamespace(id=10), message_id=555)
            self.from_user = SimpleNamespace(id=22)

    data_value = callback_data(bot_module)
    update = SimpleNamespace(
        callback_query=_Query(data_value),
        effective_chat=SimpleNamespace(id=10),
        effective_user=SimpleNamespace(id=22),
    )

    asyncio.run(bot_module.video_menu_callback(update, ctx))
    assert answered, f"Callback {data_value!r} did not trigger answer"
