import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import handlers.payments as payments


class _DummyCounter:
    def __init__(self):
        self.calls: list[dict] = []

    def labels(self, **labels):
        self.calls.append(labels)
        return self

    def inc(self):
        return None


def test_stars_buy_success(monkeypatch):
    sent_invoices: list[dict] = []
    refreshed: list[dict] = []

    class DummyBot:
        async def send_invoice(self, **kwargs):
            sent_invoices.append(kwargs)

        async def edit_message_reply_markup(self, **kwargs):  # pragma: no cover - not used here
            refreshed.append(kwargs)

    async def fake_open_stars_menu(ctx, **kwargs):
        refreshed.append(kwargs)

    counter = _DummyCounter()

    monkeypatch.setattr(payments, "open_stars_menu", fake_open_stars_menu)
    monkeypatch.setattr(payments, "stars_buy_total", counter)

    async def fake_query_answer(*args, **kwargs):
        return "ok"

    query = SimpleNamespace(
        data="stars:buy:100",
        message=SimpleNamespace(chat=SimpleNamespace(id=777), message_id=888),
        answer=fake_query_answer,
    )
    update = SimpleNamespace(
        callback_query=query,
        effective_chat=query.message.chat,
        effective_message=query.message,
    )
    ctx = SimpleNamespace(bot=DummyBot())

    asyncio.run(payments.stars_buy(update, ctx))

    assert sent_invoices, "invoice should be sent"
    invoice = sent_invoices[0]
    assert invoice["chat_id"] == 777
    assert counter.calls[-1] == {"amount": "100", "result": "ok"}


def test_stars_buy_invalid_amount(monkeypatch):
    counter = _DummyCounter()
    monkeypatch.setattr(payments, "stars_buy_total", counter)

    class DummyBot:
        async def send_invoice(self, **kwargs):  # pragma: no cover - should not be called
            raise AssertionError("invoice should not be sent")

    async def fake_open_stars_menu(*args, **kwargs):
        return None

    monkeypatch.setattr(payments, "open_stars_menu", fake_open_stars_menu)

    async def fake_invalid_answer(*args, **kwargs):
        return "ok"

    query = SimpleNamespace(
        data="stars:buy:999",
        message=SimpleNamespace(chat=SimpleNamespace(id=1), message_id=2),
        answer=fake_invalid_answer,
    )
    update = SimpleNamespace(callback_query=query)
    ctx = SimpleNamespace(bot=DummyBot())

    asyncio.run(payments.stars_buy(update, ctx))

    assert counter.calls and counter.calls[-1]["result"] == "invalid"
