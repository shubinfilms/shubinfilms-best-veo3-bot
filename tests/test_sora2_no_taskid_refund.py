import asyncio
import importlib
import sys
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
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://bot.example")
    monkeypatch.setenv("SORA2_API_KEY", "sora-key")
    module = importlib.import_module("bot")
    return importlib.reload(module)


class DummyBot:
    def __init__(self):
        self.sent_stickers = []

    async def send_sticker(self, chat_id, sticker_id):
        self.sent_stickers.append((chat_id, sticker_id))
        return SimpleNamespace(message_id=600)


class DummyMessage:
    def __init__(self, chat_id: int):
        self.chat = SimpleNamespace(id=chat_id)
        self.chat_id = chat_id
        self.message_id = 415
        self.reply_calls = []

    async def reply_text(self, text: str):
        self.reply_calls.append(text)
        return SimpleNamespace(message_id=700)


class DummyQuery:
    def __init__(self, data: str, message: DummyMessage, user_id: int):
        self.data = data
        self.message = message
        self.from_user = SimpleNamespace(id=user_id)
        self.answer_calls = []

    async def answer(self, text: str = "", show_alert: bool = False):
        self.answer_calls.append({"text": text, "alert": show_alert})
        return None


def test_sora2_no_taskid_refund(monkeypatch, bot_module):
    ctx = SimpleNamespace(bot=DummyBot(), user_data={}, application=None)
    state = bot_module.state(ctx)
    state["mode"] = "sora2_ttv"
    state["sora2_prompt"] = "Story"

    monkeypatch.setattr(bot_module, "ensure_user_record", lambda update: asyncio.sleep(0))
    monkeypatch.setattr(bot_module, "ensure_user", lambda user_id: None)

    async def fake_ensure_tokens(ctx_param, chat_id, user_id, price):
        return True

    monkeypatch.setattr(bot_module, "ensure_tokens", fake_ensure_tokens)

    monkeypatch.setattr(bot_module, "debit_try", lambda uid, price, reason, meta: (True, 800))

    credit_calls = []

    def fake_credit(user_id, amount, reason, meta):
        credit_calls.append({"user_id": user_id, "amount": amount, "reason": reason, "meta": meta})
        return 900

    monkeypatch.setattr(bot_module, "credit_balance", fake_credit)

    async def fake_balance_notification(ctx_param, chat_id, user_id, text):
        return None

    monkeypatch.setattr(bot_module, "show_balance_notification", fake_balance_notification)

    def fake_create_task(payload):
        return {"error": "Bad model"}

    monkeypatch.setattr(bot_module, "sora2_create_task", fake_create_task)
    monkeypatch.setattr(bot_module, "_schedule_sora2_poll", lambda *args, **kwargs: None)

    release_calls = []
    monkeypatch.setattr(bot_module, "acquire_sora2_lock", lambda user_id, ttl=60: True)
    monkeypatch.setattr(bot_module, "release_sora2_lock", lambda user_id: release_calls.append(user_id))

    message = DummyMessage(chat_id=777)
    query = DummyQuery("s2_go_t2v", message, user_id=55)
    update = SimpleNamespace(callback_query=query, effective_user=SimpleNamespace(id=55))

    asyncio.run(bot_module.sora2_start_t2v(update, ctx))

    assert bot_module.ACTIVE_TASKS.get(777) is None
    assert ctx.bot.sent_stickers == []
    assert any("Sora 2 не приняла задачу" in text for text in message.reply_calls)
    assert credit_calls, "tokens were not refunded"
    assert credit_calls[0]["amount"] == bot_module.PRICE_SORA2_TEXT
    assert release_calls == [55]

