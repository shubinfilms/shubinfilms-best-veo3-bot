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
    module = importlib.import_module("bot")
    return importlib.reload(module)


class DummyBot:
    def __init__(self, bot_module, *, edit_error=False, video_error=False):
        self._telegram_error = bot_module.TelegramError
        self.edit_error = edit_error
        self.video_error = video_error
        self.edit_calls = []
        self.delete_calls = []
        self.send_video_calls = []
        self.send_document_calls = []
        self.send_message_calls = []

    async def edit_message_media(self, chat_id, message_id, media, reply_markup):
        if self.edit_error:
            raise self._telegram_error("edit failed")
        self.edit_calls.append({
            "chat_id": chat_id,
            "message_id": message_id,
            "media": media,
            "reply_markup": reply_markup,
        })
        return None

    async def delete_message(self, chat_id, message_id):
        self.delete_calls.append((chat_id, message_id))
        return None

    async def send_video(self, chat_id, video, caption, reply_markup, supports_streaming=True):
        if self.video_error:
            raise self._telegram_error("video failed")
        self.send_video_calls.append({
            "chat_id": chat_id,
            "video": video,
            "caption": caption,
            "reply_markup": reply_markup,
            "supports_streaming": supports_streaming,
        })
        return SimpleNamespace(message_id=902)

    async def send_document(self, chat_id, document, caption, reply_markup):
        self.send_document_calls.append({
            "chat_id": chat_id,
            "document": document,
            "caption": caption,
            "reply_markup": reply_markup,
        })
        return SimpleNamespace(message_id=903)

    async def send_message(self, chat_id, text, reply_markup):
        self.send_message_calls.append({
            "chat_id": chat_id,
            "text": text,
            "reply_markup": reply_markup,
        })
        return SimpleNamespace(message_id=904)


def _setup_context(monkeypatch, bot_module, *, edit_error=False, video_error=False):
    bot = DummyBot(bot_module, edit_error=edit_error, video_error=video_error)
    ctx = SimpleNamespace(bot=bot, user_data={}, application=None)
    state = bot_module.state(ctx)
    state["sora2_generating"] = True
    release_calls = []
    monkeypatch.setattr(bot_module, "release_sora2_lock", lambda user_id: release_calls.append(user_id))
    return ctx, state, bot, release_calls


async def _passthrough_safe_send(method, *, method_name, kind, req_id=None, **kwargs):
    return await method(**kwargs)


def test_sora2_callback_replaces_sticker(monkeypatch, bot_module):
    ctx, state, bot, release_calls = _setup_context(monkeypatch, bot_module)
    wait_id = 1234
    chat_id = 999
    state["sora2_wait_msg_id"] = wait_id
    state["video_wait_message_id"] = wait_id
    bot_module.ACTIVE_TASKS[chat_id] = "task-123"

    monkeypatch.setattr(bot_module, "tg_safe_send", _passthrough_safe_send)
    update_calls = []
    monkeypatch.setattr(bot_module, "update_task_meta", lambda *args, **kwargs: update_calls.append((args, kwargs)))
    clear_calls = []
    monkeypatch.setattr(bot_module, "clear_task_meta", lambda task_id: clear_calls.append(task_id))

    result_payload = {"videoUrl": "https://cdn.example/video.mp4"}
    meta = {
        "chat_id": chat_id,
        "wait_message_id": wait_id,
        "user_id": 42,
        "price": 0,
        "service": "SORA2_TTV",
        "mode": "sora2_ttv",
    }

    asyncio.run(
        bot_module._finalize_sora2_task(
            ctx,
            "task-123",
            meta,
            "success",
            result_payload,
            "webhook",
        )
    )

    assert bot.edit_calls, "edit was not attempted"
    call = bot.edit_calls[0]
    assert call["chat_id"] == chat_id
    assert call["message_id"] == wait_id
    assert call["media"].media == "https://cdn.example/video.mp4"
    buttons = call["reply_markup"].inline_keyboard
    assert any(btn.text == "🔁 Сгенерировать ещё" for row in buttons for btn in row)
    assert not bot.delete_calls
    assert not bot.send_video_calls
    assert state.get("sora2_wait_msg_id") is None
    assert chat_id not in bot_module.ACTIVE_TASKS
    assert clear_calls == ["task-123"]
    assert release_calls == [42]


def test_sora2_callback_fallback_to_send(monkeypatch, bot_module):
    ctx, state, bot, release_calls = _setup_context(monkeypatch, bot_module, edit_error=True)
    wait_id = 2222
    chat_id = 333
    state["sora2_wait_msg_id"] = wait_id
    state["video_wait_message_id"] = wait_id
    bot_module.ACTIVE_TASKS[chat_id] = "task-456"

    monkeypatch.setattr(bot_module, "tg_safe_send", _passthrough_safe_send)
    monkeypatch.setattr(bot_module, "update_task_meta", lambda *args, **kwargs: None)
    monkeypatch.setattr(bot_module, "clear_task_meta", lambda *args, **kwargs: None)

    result_payload = {"videoUrl": "https://cdn.example/video.mp4"}
    meta = {
        "chat_id": chat_id,
        "wait_message_id": wait_id,
        "user_id": 77,
        "price": 0,
        "service": "SORA2_TTV",
        "mode": "sora2_ttv",
    }

    asyncio.run(
        bot_module._finalize_sora2_task(
            ctx,
            "task-456",
            meta,
            "success",
            result_payload,
            "webhook",
        )
    )

    assert bot.delete_calls == [(chat_id, wait_id)]
    assert bot.send_video_calls, "video should be sent when edit fails"
    buttons = bot.send_video_calls[0]["reply_markup"].inline_keyboard
    assert any(btn.text == "🔁 Сгенерировать ещё" for row in buttons for btn in row)
    assert state.get("sora2_wait_msg_id") is None
    assert release_calls == [77]
    bot_module.ACTIVE_TASKS.clear()
