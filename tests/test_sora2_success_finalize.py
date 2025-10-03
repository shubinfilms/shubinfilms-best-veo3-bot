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
        self.deleted = []
        self.sent_docs = []

    async def delete_message(self, chat_id, message_id):
        self.deleted.append((chat_id, message_id))

    async def send_document(self, chat_id, document, caption=None, reply_markup=None):
        self.sent_docs.append({
            "chat_id": chat_id,
            "caption": caption,
            "reply_markup": reply_markup,
        })
        return SimpleNamespace(message_id=1234)


def test_sora2_success_releases_lock(tmp_path, monkeypatch, bot_module):
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"data")

    async def fake_download(result):
        return [str(video_path)]

    monkeypatch.setattr(bot_module, "_download_sora2_assets", fake_download)

    released = []
    monkeypatch.setattr(bot_module, "release_sora2_lock", lambda user_id: released.append(user_id))

    ctx = SimpleNamespace(bot=DummyBot(), user_data={}, application=None)
    bot_module.ACTIVE_TASKS[999] = "task-1"
    state = bot_module.state(ctx)
    state["sora2_last_task_id"] = "task-1"
    state["sora2_generating"] = True
    state["video_wait_message_id"] = 333
    state["sora2_wait_msg_id"] = 333

    meta = {
        "user_id": 42,
        "price": bot_module.PRICE_SORA2_TEXT,
        "service": "SORA2_TTV",
        "mode": "sora2_ttv",
        "wait_message_id": 333,
        "chat_id": 999,
        "duration": bot_module.SORA2_DEFAULT_TTV_DURATION,
        "resolution": bot_module.SORA2_DEFAULT_TTV_RESOLUTION,
    }

    result_payload = {"video_url": "https://example.com/result.mp4"}

    asyncio.run(
        bot_module._finalize_sora2_task(
            ctx,
            task_id="task-1",
            meta=meta,
            status="success",
            result_payload=result_payload,
            source="poll",
        )
    )

    assert released == [42]
    assert ctx.bot.deleted == [(999, 333)]
    assert ctx.bot.sent_docs
    sent_doc = ctx.bot.sent_docs[0]
    assert sent_doc["chat_id"] == 999
    expected_caption = (
        f"Sora 2 • Text-to-Video • {bot_module.SORA2_DEFAULT_TTV_DURATION}s "
        f"• {bot_module.SORA2_DEFAULT_TTV_RESOLUTION.replace('x', '×')}"
    )
    assert sent_doc["caption"] == expected_caption
    assert bot_module.ACTIVE_TASKS.get(999) is None
