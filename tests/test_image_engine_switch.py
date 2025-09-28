import asyncio
from types import SimpleNamespace
import os
import sys

from pathlib import Path

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

import bot as bot_module


class FakeBot:
    def __init__(self) -> None:
        self.sent: list[dict[str, object]] = []
        self.deleted: list[tuple[int, int]] = []
        self.edits: list[dict[str, object]] = []
        self._next_id = 1000

    async def send_message(self, **kwargs):  # type: ignore[override]
        self._next_id += 1
        payload = dict(kwargs)
        payload.setdefault("message_id", self._next_id)
        self.sent.append(payload)
        return SimpleNamespace(message_id=self._next_id)

    async def edit_message_text(self, **kwargs):  # type: ignore[override]
        self.edits.append(kwargs)
        return SimpleNamespace(message_id=kwargs.get("message_id", 0))

    async def edit_message_reply_markup(self, **kwargs):  # type: ignore[override]
        self.edits.append(kwargs)
        return SimpleNamespace(message_id=kwargs.get("message_id", 0))

    async def delete_message(self, chat_id: int, message_id: int):  # type: ignore[override]
        self.deleted.append((chat_id, message_id))

    async def send_invoice(self, **kwargs):  # type: ignore[override]
        return SimpleNamespace(message_id=self._next_id)


class DummyMessage:
    def __init__(self, chat_id: int) -> None:
        self.chat_id = chat_id
        self.replies: list[str] = []

    async def reply_text(self, text: str, **_: object) -> None:  # type: ignore[override]
        self.replies.append(text)


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
    return SimpleNamespace(
        effective_chat=chat,
        effective_user=user,
        callback_query=query,
        message=None,
    )


def test_first_entry_shows_engine_selection() -> None:
    ctx = SimpleNamespace(bot=FakeBot(), user_data={})
    update = _make_update(101, 555)

    _run(bot_module.image_command(update, ctx))

    state = bot_module.state(ctx)
    assert state["mode"] == "image_engine_select"
    assert state["image_engine"] is None
    assert isinstance(state.get("last_ui_msg_id_image_engine"), int)
    assert ctx.bot.sent, "selector message not sent"
    assert "Выберите движок" in str(ctx.bot.sent[-1]["text"])


def test_persist_engine_and_open_directly_next_time() -> None:
    ctx = SimpleNamespace(bot=FakeBot(), user_data={})
    base_update = _make_update(202, 777)

    _run(bot_module.image_command(base_update, ctx))
    query_update = _make_update(202, 777, "img_engine:mj")
    _run(bot_module.on_callback(query_update, ctx))

    state = bot_module.state(ctx)
    assert state["image_engine"] == "mj"
    assert state["mode"] == "mj_txt"

    ctx.bot.sent.clear()
    _run(bot_module.image_command(base_update, ctx))
    assert state["mode"] == "mj_txt"
    assert state["image_engine"] == "mj"
    assert ctx.bot.sent, "mj card not rendered"
    assert "Midjourney" in str(ctx.bot.sent[-1]["text"])


def test_switch_engine_from_mj_to_banana_and_back() -> None:
    ctx = SimpleNamespace(bot=FakeBot(), user_data={})
    update = _make_update(303, 888)

    _run(bot_module.image_command(update, ctx))
    _run(bot_module.on_callback(_make_update(303, 888, "img_engine:mj"), ctx))
    state = bot_module.state(ctx)
    assert state["image_engine"] == "mj"

    _run(bot_module.on_callback(_make_update(303, 888, "mj:switch_engine"), ctx))
    state = bot_module.state(ctx)
    assert state["image_engine"] is None
    assert state["mode"] == "image_engine_select"

    _run(bot_module.on_callback(_make_update(303, 888, "img_engine:banana"), ctx))
    state = bot_module.state(ctx)
    assert state["image_engine"] == "banana"
    assert state["mode"] == "banana"

    _run(bot_module.on_callback(_make_update(303, 888, "banana:switch_engine"), ctx))
    state = bot_module.state(ctx)
    assert state["image_engine"] is None
    assert state["mode"] == "image_engine_select"

    _run(bot_module.on_callback(_make_update(303, 888, "img_engine:mj"), ctx))
    state = bot_module.state(ctx)
    assert state["image_engine"] == "mj"
    assert state["mode"] == "mj_txt"
