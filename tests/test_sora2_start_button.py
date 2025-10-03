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
        self._next_id = 200

    async def send_sticker(self, chat_id, sticker_id):
        self.sent_stickers.append((chat_id, sticker_id))
        self._next_id += 1
        return SimpleNamespace(message_id=self._next_id)


class DummyMessage:
    def __init__(self, chat_id: int):
        self.chat = SimpleNamespace(id=chat_id)
        self.chat_id = chat_id
        self.message_id = 777
        self.reply_calls = []

    async def reply_text(self, text: str):
        self.reply_calls.append(text)
        return SimpleNamespace(message_id=888)


class DummyQuery:
    def __init__(self, data: str, message: DummyMessage, user_id: int):
        self.data = data
        self.message = message
        self.from_user = SimpleNamespace(id=user_id)
        self.answer_calls = []

    async def answer(self, text: str = "", show_alert: bool = False):
        self.answer_calls.append({"text": text, "alert": show_alert})
        return None


def test_sora2_start_button(monkeypatch, bot_module):
    ctx = SimpleNamespace(bot=DummyBot(), user_data={}, application=None)
    state = bot_module.state(ctx)
    state["mode"] = "sora2_ttv"
    state["sora2_prompt"] = "Make a movie"
    state["aspect"] = "9:16"

    ensure_calls = []
    async def fake_ensure_tokens(ctx_param, chat_id, user_id, price):
        ensure_calls.append((chat_id, user_id, price))
        return True

    async def fake_balance_notification(ctx_param, chat_id, user_id, text):
        return None

    monkeypatch.setattr(bot_module, "ensure_tokens", fake_ensure_tokens)
    monkeypatch.setattr(bot_module, "show_balance_notification", fake_balance_notification)
    monkeypatch.setattr(bot_module, "ensure_user_record", lambda update: asyncio.sleep(0))
    monkeypatch.setattr(bot_module, "ensure_user", lambda user_id: None)
    monkeypatch.setattr(bot_module, "debit_try", lambda uid, price, reason, meta: (True, 900))
    monkeypatch.setattr(bot_module, "credit_balance", lambda *args, **kwargs: 0)

    payloads = []

    def fake_create_task(payload):
        payloads.append(payload)
        return {"taskId": "task-123"}

    monkeypatch.setattr(bot_module, "sora2_create_task", fake_create_task)
    monkeypatch.setattr(bot_module, "_schedule_sora2_poll", lambda *args, **kwargs: None)

    saved_meta = {}

    def fake_save_task_meta(task_id, chat_id, message_id, mode, aspect, extra, ttl):
        saved_meta.update({
            "task_id": task_id,
            "chat_id": chat_id,
            "message_id": message_id,
            "mode": mode,
            "aspect": aspect,
            "extra": extra,
            "ttl": ttl,
        })

    monkeypatch.setattr(bot_module, "save_task_meta", fake_save_task_meta)

    lock_calls = []
    monkeypatch.setattr(
        bot_module,
        "acquire_sora2_lock",
        lambda user_id, ttl=60: lock_calls.append((user_id, ttl)) or True,
    )
    release_calls = []
    monkeypatch.setattr(
        bot_module,
        "release_sora2_lock",
        lambda user_id: release_calls.append(user_id),
    )

    message = DummyMessage(chat_id=555)
    query = DummyQuery("s2_go_t2v", message, user_id=42)
    update = SimpleNamespace(
        callback_query=query,
        effective_chat=SimpleNamespace(id=555),
        effective_user=SimpleNamespace(id=42),
    )

    asyncio.run(bot_module.sora2_start_t2v(update, ctx))

    assert payloads, "payload was not sent"
    payload = payloads[0]
    assert payload["model"] == "sora2-text-to-video"
    assert payload["aspect_ratio"] == "9:16"
    assert payload["input"]["prompt"] == "Make a movie"
    assert "image_urls" not in payload["input"]

    assert ctx.bot.sent_stickers == [(555, bot_module.SORA2_WAIT_STICKER_ID)]
    assert ensure_calls
    assert state.get("sora2_wait_msg_id") is not None
    assert state.get("sora2_last_task_id") == "task-123"
    assert bot_module.ACTIVE_TASKS.get(555) == "task-123"
    assert lock_calls == [(42, bot_module.SORA2_LOCK_TTL)]
    assert release_calls == [42]
    assert saved_meta.get("extra", {}).get("wait_message_id") == state.get("sora2_wait_msg_id")
    assert saved_meta.get("extra", {}).get("aspect_ratio") == "9:16"

    bot_module.ACTIVE_TASKS.clear()
