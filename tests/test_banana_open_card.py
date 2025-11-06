import asyncio
import json
from types import SimpleNamespace

from handlers import banana_async_handler
from utils.redis_client import InMemoryStore


class _FakeMessage:
    def __init__(self) -> None:
        self.text = "placeholder"
        self.edited = False

    async def edit_text(self, *args, **kwargs):
        self.edited = True


class _FakeQuery:
    def __init__(self) -> None:
        self.data = "img_engine:banana"
        self.message = _FakeMessage()

    async def answer(self, *args, **kwargs):
        return None


class _FakeBot:
    def __init__(self) -> None:
        self.sent = []

    async def send_message(self, chat_id, text, reply_markup=None):
        self.sent.append((chat_id, text, reply_markup))
        return SimpleNamespace(message_id=1)


def test_open_card_initialises_store(monkeypatch):
    store = InMemoryStore()

    monkeypatch.setattr("handlers.banana_async_handler.get_redis", lambda: store)

    update = SimpleNamespace(
        callback_query=_FakeQuery(),
        effective_user=SimpleNamespace(id=123),
        effective_chat=SimpleNamespace(id=321),
    )
    context = SimpleNamespace(bot=_FakeBot())

    async def _scenario():
        await banana_async_handler.open_card(update, context)
        return await store.get("banana:123")

    stored = asyncio.run(_scenario())
    assert stored is not None
    payload = json.loads(stored)
    assert payload["images"] == []
    assert payload["prompt"] == ""
