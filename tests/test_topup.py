import asyncio
import importlib
import inspect
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("LEDGER_BACKEND", "memory")

import balance
import bot


def test_topup_menu_opens_once():
    async def run() -> None:
        ctx = SimpleNamespace(bot=AsyncMock())
        query = Mock()
        query.message = Mock(chat_id=123)
        query.edit_message_text = AsyncMock()

        await bot.show_topup_menu(ctx, 123, query=query, user_id=456)

        query.edit_message_text.assert_called_once()
        assert ctx.bot.send_message.await_count == 0

    asyncio.run(run())


def test_insufficient_tokens_shows_button(monkeypatch):
    async def run() -> None:
        monkeypatch.setattr(balance, "get_balance", lambda user_id: 0)
        ctx = SimpleNamespace(bot=AsyncMock())

        result = await balance.ensure_tokens(ctx, chat_id=111, user_id=222, need=5)

        assert result is False
        ctx.bot.send_message.assert_called_once()
        kwargs = ctx.bot.send_message.call_args.kwargs
        keyboard = kwargs["reply_markup"]
        assert keyboard.inline_keyboard[0][0].callback_data == "topup:open"

    asyncio.run(run())


def test_stars_button_label():
    keyboard = bot.topup_menu_keyboard()
    assert keyboard.inline_keyboard[0][0].text.startswith("💎")
    assert keyboard.inline_keyboard[1][0].text.startswith("💳")


@pytest.fixture
def yk_environment(monkeypatch):
    monkeypatch.setenv("YOOKASSA_SHOP_ID", "shop")
    monkeypatch.setenv("YOOKASSA_SECRET_KEY", "secret")
    monkeypatch.setenv("YOOKASSA_RETURN_URL", "https://return")
    monkeypatch.setenv("LEDGER_BACKEND", "memory")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("TELEGRAM_TOKEN", "dummy-token")

    settings = importlib.import_module("settings")
    importlib.reload(settings)

    import payments.yookassa as yk
    import payments.yookassa_storage as storage
    import payments.yookassa_callback as callback

    importlib.reload(yk)
    importlib.reload(storage)
    importlib.reload(callback)

    sent = []

    class DummyTGResponse:
        status_code = 200
        ok = True
        text = ""

    def fake_tg_post(url, json=None, timeout=None):
        sent.append((url, json))
        return DummyTGResponse()

    monkeypatch.setattr(callback, "_TELEGRAM_SESSION", SimpleNamespace(post=fake_tg_post))
    ledger_module = importlib.import_module("ledger")
    ledger_instance = ledger_module.LedgerStorage(None, backend="memory")
    callback._ledger_instance = ledger_instance

    class DummyResponse:
        status_code = 200
        text = ""

        def json(self):
            return {"id": "pay-1", "confirmation": {"confirmation_url": "https://pay"}}

        @property
        def ok(self):
            return True

    monkeypatch.setattr(yk._SESSION, "post", lambda *args, **kwargs: DummyResponse())

    return yk, storage, callback, sent, ledger_instance


def test_yk_payment_flow_success(monkeypatch, yk_environment):
    async def run() -> None:
        yk, storage, callback, sent, _ = yk_environment

        payment = yk.create_payment(1001, "pack_1")
        assert payment.payment_id == "pay-1"

        pending = storage.load_pending_payment("pay-1")
        assert pending is not None

        result = callback.process_callback({"event": "payment.succeeded", "object": {"id": "pay-1"}})
        assert result["status"] == "success"

        balance_value = callback._ledger_instance.get_balance(1001)
        assert balance_value == pending.tokens_to_add

        assert len(sent) == 2
        assert sent[0][0].endswith("/sendSticker")
        assert sent[1][0].endswith("/sendMessage")

    asyncio.run(run())


def test_yk_callback_idempotent(yk_environment):
    yk, storage, callback, sent, _ = yk_environment

    payment = yk.create_payment(2002, "pack_2")
    callback.process_callback({"event": "payment.succeeded", "object": {"id": payment.payment_id}})
    sent_length = len(sent)

    duplicate = callback.process_callback({"event": "payment.succeeded", "object": {"id": payment.payment_id}})
    assert duplicate["status"] == "duplicate"
    assert len(sent) == sent_length


def test_modes_guard():
    source = inspect.getsource(bot)
    assert "await ensure_tokens(ctx, chat_id, user_id, PRICE_SUNO)" in source
    assert "if not await ensure_tokens(ctx, chat_id, user_id, price):" in source
    assert "if not await ensure_tokens(ctx, chat_id, uid, PRICE_BANANA):" in source
    assert "if not await ensure_tokens(ctx, chat_id, uid, price):" in source
