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
    monkeypatch.setenv("SORA2_ENABLED", "true")
    monkeypatch.setenv("SORA2_API_KEY", "sora-key")
    settings_module = importlib.import_module("settings")
    settings_module = importlib.reload(settings_module)
    module = importlib.import_module("bot")
    module = importlib.reload(module)
    module.clear_sora2_unavailable()
    return module


class DummyBot:
    def __init__(self):
        self.sent_stickers = []
        self._next_id = 200
        self.deleted = []
        self.sent_documents = []

    async def send_sticker(self, chat_id, sticker_id):
        self.sent_stickers.append((chat_id, sticker_id))
        self._next_id += 1
        return SimpleNamespace(message_id=self._next_id)

    async def delete_message(self, chat_id, message_id):
        self.deleted.append((chat_id, message_id))

    async def send_document(self, chat_id, document, caption=None, reply_markup=None):
        self.sent_documents.append({
            "chat_id": chat_id,
            "caption": caption,
            "reply_markup": reply_markup is not None,
        })
        return SimpleNamespace(message_id=999)


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
        return bot_module.CreateTaskResponse("task-123", {"taskId": "task-123"})

    monkeypatch.setattr(bot_module, "sora2_create_task", fake_create_task)
    monkeypatch.setattr(bot_module, "_schedule_sora2_poll", lambda *args, **kwargs: None)
    monkeypatch.setattr(bot_module, "_refresh_video_menu_ui", lambda *args, **kwargs: asyncio.sleep(0))

    async def immediate_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(bot_module.asyncio, "to_thread", immediate_to_thread)

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
    assert "task_type" not in payload
    assert payload["model"] == "sora-2-text-to-video"
    assert payload["input"]["aspect_ratio"] == "portrait"
    assert payload["input"]["prompt"] == "Make a movie"
    assert payload["input"]["quality"] == "standard"
    assert "image_urls" not in payload["input"]
    assert "resolution" not in payload
    assert "duration" not in payload
    assert "audio" not in payload
    assert payload["callBackUrl"].endswith("/sora2-callback")

    assert ctx.bot.sent_stickers == [(555, bot_module.SORA2_WAIT_STICKER_ID)]
    assert ensure_calls
    assert state.get("sora2_wait_msg_id") is not None
    assert state.get("sora2_last_task_id") == "task-123"
    assert bot_module.ACTIVE_TASKS.get(555) == "task-123"
    assert lock_calls == [(42, bot_module.SORA2_LOCK_TTL)]
    assert release_calls == []
    assert saved_meta.get("extra", {}).get("wait_message_id") == state.get("sora2_wait_msg_id")
    assert saved_meta.get("extra", {}).get("aspect_ratio") == "9:16"
    assert saved_meta.get("extra", {}).get("image_urls") == []
    extra = saved_meta.get("extra", {})
    assert extra.get("submit_raw") == {"taskId": "task-123"}
    assert extra.get("duration") == bot_module.SORA2_DEFAULT_TTV_DURATION
    assert extra.get("resolution") == bot_module.SORA2_DEFAULT_TTV_RESOLUTION
    assert extra.get("audio") is True
    assert extra.get("quality") == "standard"

    bot_module.ACTIVE_TASKS.clear()


def test_video_menu_disables_sora_on_flag(monkeypatch, bot_module):
    ctx = SimpleNamespace(bot=DummyBot(), user_data={}, application=None)
    bot_module.mark_sora2_unavailable()
    keyboard = bot_module.video_menu_kb()
    sora_button = keyboard.inline_keyboard[1][0]
    assert "скоро" in sora_button.text
    assert sora_button.callback_data == bot_module.CB_VIDEO_ENGINE_SORA2_DISABLED
    bot_module.clear_sora2_unavailable()


def test_sora2_double_click_shows_busy(monkeypatch, bot_module):
    ctx = SimpleNamespace(bot=DummyBot(), user_data={}, application=None)
    state = bot_module.state(ctx)
    state["mode"] = "sora2_ttv"
    state["sora2_prompt"] = "Short"
    bot_module.clear_sora2_unavailable()

    monkeypatch.setattr(bot_module, "ensure_user_record", lambda update: asyncio.sleep(0))
    monkeypatch.setattr(bot_module, "ensure_tokens", lambda *args, **kwargs: True)
    monkeypatch.setattr(bot_module, "acquire_sora2_lock", lambda user_id, ttl=60: False)

    message = DummyMessage(chat_id=777)
    query = DummyQuery("s2_go_t2v", message, user_id=99)
    update = SimpleNamespace(
        callback_query=query,
        effective_chat=SimpleNamespace(id=777),
        effective_user=SimpleNamespace(id=99),
    )

    asyncio.run(bot_module.sora2_start_t2v(update, ctx))

    assert query.answer_calls
    assert any("уже" in call["text"] for call in query.answer_calls)
    assert ctx.bot.sent_stickers == []
