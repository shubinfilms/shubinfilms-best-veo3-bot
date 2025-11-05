import asyncio
import pytest

import bot as bot_module
from keyboards import banana_card_kb
from ui.buttons import router as buttons_router
from types import SimpleNamespace


class _FakeMessage:
    def __init__(self) -> None:
        self.text = "placeholder"

    async def edit_text(self, *args, **kwargs):
        return None

    async def reply_text(self, *args, **kwargs):
        return None


class _FakeQuery:
    def __init__(self, data: str) -> None:
        self.data = data
        self.message = _FakeMessage()

    async def answer(self, *args, **kwargs):
        return None

    async def edit_message_reply_markup(self, *args, **kwargs):
        return None


def test_route_callback_dispatches_banana(monkeypatch):
    called = {}

    async def fake_open_card(update, ctx, payload=None):
        called["ok"] = True

    monkeypatch.setattr("handlers.banana_async_handler.open_card", fake_open_card)
    query = _FakeQuery("img_engine:banana")
    update = SimpleNamespace(callback_query=query, effective_message=None)
    context = SimpleNamespace()

    asyncio.run(buttons_router.route_callback(update, context))

    assert called.get("ok") is True


def test_banana_callback_patterns_registered():
    specs = bot_module.CALLBACK_HANDLER_SPECS
    assert any(pattern == r"^banana:start$" and callback is bot_module.banana_start_generation for pattern, callback in specs)
    assert any(pattern == r"^banana:restart$" and callback is bot_module.banana_restart_generation for pattern, callback in specs)
    assert any(pattern == r"^banana:new$" and callback is bot_module.banana_new_card for pattern, callback in specs)
    assert any(pattern == r"^banana:back_photo$" and callback is bot_module.photo_modes_open_menu for pattern, callback in specs)


def test_banana_card_keyboard_rendering():
    without_inputs = banana_card_kb(False)
    rows = without_inputs.inline_keyboard
    assert len(rows) == 1
    assert rows[0][0].text == "⬅️ Назад"
    assert rows[0][0].callback_data == "banana:back_photo"

    with_inputs = banana_card_kb(True)
    rows = with_inputs.inline_keyboard
    assert rows[0][0].callback_data == "banana:start"
    assert rows[1][0].callback_data == "banana:back_photo"
