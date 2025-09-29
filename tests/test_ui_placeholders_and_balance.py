import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("SUNO_API_BASE", "https://example.com")
os.environ.setdefault("SUNO_API_TOKEN", "token")
os.environ.setdefault("SUNO_CALLBACK_URL", "https://callback.example")
os.environ.setdefault("SUNO_CALLBACK_SECRET", "secret")
os.environ.setdefault("TELEGRAM_TOKEN", "dummy-token")
os.environ.setdefault("KIE_API_KEY", "test-key")
os.environ.setdefault("KIE_BASE_URL", "https://example.com")
os.environ.setdefault("DATABASE_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("LEDGER_BACKEND", "memory")
os.environ.setdefault("LOG_JSON", "false")
os.environ.setdefault("LOG_LEVEL", "WARNING")

from utils.input_state import classify_wait_input
from utils.telegram_utils import should_capture_to_prompt
import bot as bot_module


class FakeBot:
    def __init__(self) -> None:
        self.sent: list[dict[str, object]] = []
        self.edited: list[dict[str, object]] = []
        self.deleted: list[tuple[int, int]] = []
        self._next_message_id = 500

    async def send_message(self, **kwargs):  # type: ignore[override]
        self._next_message_id += 1
        payload = dict(kwargs)
        payload.setdefault("message_id", self._next_message_id)
        self.sent.append(payload)
        return SimpleNamespace(message_id=self._next_message_id)

    async def edit_message_text(self, **kwargs):  # type: ignore[override]
        self.edited.append(kwargs)
        return SimpleNamespace(message_id=kwargs.get("message_id"))

    async def delete_message(self, chat_id: int, message_id: int):  # type: ignore[override]
        self.deleted.append((chat_id, message_id))


class DummyMessage:
    def __init__(self, chat_id: int) -> None:
        self.chat_id = chat_id

    async def reply_text(self, *_args, **_kwargs):  # type: ignore[override]
        return SimpleNamespace(message_id=999)


class DummyQuery:
    def __init__(self, chat_id: int, data: str) -> None:
        self.data = data
        self.message = DummyMessage(chat_id)
        self._answers: list[tuple[tuple[object, ...], dict[str, object]]] = []

    async def answer(self, *args, **kwargs):  # type: ignore[override]
        self._answers.append((args, dict(kwargs)))


def _run(coro) -> None:
    asyncio.run(coro)


def _make_update(chat_id: int, user_id: int, query_data: str | None = None):
    chat = SimpleNamespace(id=chat_id)
    user = SimpleNamespace(id=user_id)
    if query_data is None:
        return SimpleNamespace(effective_chat=chat, effective_user=user, message=None)
    query = DummyQuery(chat_id, query_data)
    return SimpleNamespace(effective_chat=chat, effective_user=user, callback_query=query, message=None)


def test_balance_button_not_captured() -> None:
    assert not should_capture_to_prompt("Balance")
    assert not should_capture_to_prompt("  💎 Баланс  ")
    assert not should_capture_to_prompt("balance")


def test_balance_button_bypasses_prompt_capture() -> None:
    should, reason = classify_wait_input("💎 Баланс")
    assert not should
    assert reason == "command_label"


def test_switching_engines_resets_prompts() -> None:
    ctx = SimpleNamespace(bot=FakeBot(), user_data={})
    update = _make_update(606, 404)

    _run(bot_module.image_command(update, ctx))
    _run(bot_module.on_callback(_make_update(606, 404, "img_engine:mj"), ctx))

    state = bot_module.state(ctx)
    state["last_prompt"] = "First"

    _run(bot_module.on_callback(_make_update(606, 404, "mj:switch_engine"), ctx))
    _run(bot_module.on_callback(_make_update(606, 404, "img_engine:banana"), ctx))

    banana_state = bot_module.state(ctx)
    assert banana_state["image_engine"] == "banana"
    assert banana_state.get("last_prompt") is None

    banana_state["last_prompt"] = "Second"
    _run(bot_module.on_callback(_make_update(606, 404, "banana:switch_engine"), ctx))
    _run(bot_module.on_callback(_make_update(606, 404, "img_engine:mj"), ctx))

    final_state = bot_module.state(ctx)
    assert final_state["image_engine"] == "mj"
    assert final_state.get("last_prompt") is None
