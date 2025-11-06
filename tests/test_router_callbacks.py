import asyncio
from types import SimpleNamespace

from ui.buttons import router


class _FakeQuery:
    def __init__(self, data: str) -> None:
        self.data = data
        self.message = None

    async def answer(self, *args, **kwargs):  # pragma: no cover - behaviour mocked
        return None


def test_route_callback_dispatches_banana(monkeypatch):
    called = {}

    async def fake_open_card(update, ctx):
        called["banana"] = (update, ctx)

    monkeypatch.setattr("handlers.banana_async_handler.open_card", fake_open_card)

    query = _FakeQuery("img_engine:banana")
    update = SimpleNamespace(callback_query=query)
    context = SimpleNamespace()

    asyncio.run(router.route_callback(update, context))

    assert "banana" in called


def test_route_callback_dispatches_profile(monkeypatch):
    called = {}

    async def fake_open_profile(update, ctx):
        called["profile"] = (update, ctx)

    monkeypatch.setattr("handlers.profile.open_profile", fake_open_profile)

    query = _FakeQuery("btn:profile")
    update = SimpleNamespace(callback_query=query)
    context = SimpleNamespace()

    asyncio.run(router.route_callback(update, context))

    assert "profile" in called
