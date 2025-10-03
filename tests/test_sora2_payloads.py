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
    module = importlib.import_module("bot")
    return importlib.reload(module)


class DummyBot:
    def __init__(self):
        self.sent_stickers = []
        self._next_id = 900

    async def send_sticker(self, chat_id, sticker_id):
        self.sent_stickers.append((chat_id, sticker_id))
        self._next_id += 1
        return SimpleNamespace(message_id=self._next_id)


class DummyMessage:
    def __init__(self, chat_id: int):
        self.chat = SimpleNamespace(id=chat_id)
        self.chat_id = chat_id
        self.message_id = 501
        self.reply_calls = []

    async def reply_text(self, text: str):
        self.reply_calls.append(text)
        return SimpleNamespace(message_id=700)


@pytest.fixture
def ctx(bot_module):
    return SimpleNamespace(bot=DummyBot(), user_data={}, application=None)


def _setup_common(monkeypatch, bot_module):
    monkeypatch.setattr(bot_module, "ensure_user", lambda user_id: None)
    monkeypatch.setattr(bot_module, "debit_try", lambda uid, price, reason, meta: (True, 777))
    monkeypatch.setattr(bot_module, "credit_balance", lambda *args, **kwargs: 0)

    async def immediate_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(bot_module.asyncio, "to_thread", immediate_to_thread)

    async def fake_ensure_tokens(ctx_param, chat_id, user_id, price):
        return True

    async def fake_balance_notification(ctx_param, chat_id, user_id, text):
        return None

    monkeypatch.setattr(bot_module, "ensure_tokens", fake_ensure_tokens)
    monkeypatch.setattr(bot_module, "show_balance_notification", fake_balance_notification)
    monkeypatch.setattr(bot_module, "_schedule_sora2_poll", lambda *args, **kwargs: None)

    saved_meta = {}

    def fake_save_task_meta(task_id, chat_id, message_id, mode, aspect, extra, ttl):
        saved_meta.update(
            {
                "task_id": task_id,
                "chat_id": chat_id,
                "message_id": message_id,
                "mode": mode,
                "aspect": aspect,
                "extra": extra,
                "ttl": ttl,
            }
        )

    monkeypatch.setattr(bot_module, "save_task_meta", fake_save_task_meta)

    async def fake_show_wait(ctx_param, chat_id, sticker_id):
        return 991

    monkeypatch.setattr(bot_module, "show_wait_sticker", fake_show_wait)

    return saved_meta


def test_sora2_payload_text_to_video(monkeypatch, bot_module, ctx):
    saved_meta = _setup_common(monkeypatch, bot_module)

    def fail_upload(urls):
        raise AssertionError("upload should not be called")

    monkeypatch.setattr(bot_module, "sora2_upload_image_urls", fail_upload)

    payloads = []

    def fake_create_task(payload):
        payloads.append(payload)
        return bot_module.CreateTaskResponse("ttv-1", {"taskId": "ttv-1"})

    monkeypatch.setattr(bot_module, "sora2_create_task", fake_create_task)
    monkeypatch.setattr(bot_module, "_refresh_video_menu_ui", lambda *args, **kwargs: asyncio.sleep(0))

    state = bot_module.state(ctx)
    state["mode"] = "sora2_ttv"
    message = DummyMessage(chat_id=321)

    asyncio.run(
        bot_module._start_sora2_generation(
            ctx,
            chat_id=321,
            user_id=11,
            message=message,
            mode="sora2_ttv",
            prompt="Epic scene",
            image_urls=[],
            aspect_ratio="4:5",
        )
    )

    assert payloads, "payload not built"
    payload = payloads[0]
    assert "task_type" not in payload
    assert payload["model"] == "sora-2-text-to-video"
    assert payload["input"]["aspect_ratio"] == "portrait"
    assert payload["input"]["prompt"] == "Epic scene"
    assert payload["input"]["quality"] == "standard"
    assert "image_urls" not in payload["input"]
    extra = saved_meta.get("extra", {})
    assert extra.get("image_urls") == []
    assert extra.get("submit_raw") == {"taskId": "ttv-1"}
    assert extra.get("duration") == bot_module.SORA2_DEFAULT_TTV_DURATION
    assert extra.get("resolution") == bot_module.SORA2_DEFAULT_TTV_RESOLUTION
    assert extra.get("audio") is True
    assert extra.get("quality") == "standard"
    bot_module.ACTIVE_TASKS.clear()


def test_sora2_payload_image_to_video(monkeypatch, bot_module, ctx):
    saved_meta = _setup_common(monkeypatch, bot_module)

    upload_calls = []

    def fake_upload(urls):
        upload_calls.append(list(urls))
        return [f"https://uploads.example/{index}" for index in range(len(urls))]

    monkeypatch.setattr(bot_module, "sora2_upload_image_urls", fake_upload)

    payloads = []

    def fake_create_task(payload):
        payloads.append(payload)
        return bot_module.CreateTaskResponse("itv-42", {"taskId": "itv-42"})

    monkeypatch.setattr(bot_module, "sora2_create_task", fake_create_task)

    state = bot_module.state(ctx)
    state["mode"] = "sora2_itv"
    message = DummyMessage(chat_id=654)

    raw_images = [
        " https://img.one/a.png ",
        "https://img.one/a.png",
        "https://img.two/b.png",
        "ftp://invalid/link",
    ]

    asyncio.run(
        bot_module._start_sora2_generation(
            ctx,
            chat_id=654,
            user_id=22,
            message=message,
            mode="sora2_itv",
            prompt="",
            image_urls=raw_images,
            aspect_ratio="16:9",
        )
    )

    assert upload_calls == [["https://img.one/a.png", "https://img.two/b.png"]]
    assert payloads, "payload not built"
    payload = payloads[0]
    assert "task_type" not in payload
    assert payload["model"] == "sora-2-image-to-video"
    assert payload["input"]["aspect_ratio"] == "landscape"
    assert payload["input"].get("prompt", "") == ""
    assert payload["input"]["quality"] == "standard"
    assert payload["input"]["image_urls"] == ["https://uploads.example/0", "https://uploads.example/1"]
    assert "resolution" not in payload
    assert "duration" not in payload
    assert "audio" not in payload
    extra = saved_meta.get("extra", {})
    assert extra.get("image_urls") == payload["input"]["image_urls"]
    assert extra.get("submit_raw") == {"taskId": "itv-42"}
    assert extra.get("duration") == bot_module.SORA2_DEFAULT_ITV_DURATION
    assert extra.get("resolution") == bot_module.SORA2_DEFAULT_ITV_RESOLUTION
    assert extra.get("audio") is None
    assert extra.get("quality") == "standard"
    bot_module.ACTIVE_TASKS.clear()

