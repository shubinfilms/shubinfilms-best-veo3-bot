import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from helpers.progress import PROGRESS_STORAGE_KEY, send_progress_message


class DummyJob:
    def __init__(self, callback, data):
        self.callback = callback
        self.data = data
        self.removed = False

    def schedule_removal(self) -> None:
        self.removed = True


class DummyJobQueue:
    def __init__(self) -> None:
        self.jobs: list[DummyJob] = []

    def run_once(self, callback, when, data=None):  # type: ignore[override]
        job = DummyJob(callback, data or {})
        self.jobs.append(job)
        return job


class DummyBot:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.edits: list[dict] = []
        self.deleted: list[dict] = []

    async def send_message(self, **kwargs):  # type: ignore[override]
        self.sent.append(kwargs)
        return SimpleNamespace(message_id=len(self.sent))

    async def edit_message_text(self, **kwargs):  # type: ignore[override]
        self.edits.append(kwargs)

    async def delete_message(self, **kwargs):  # type: ignore[override]
        self.deleted.append(kwargs)


def _make_context() -> SimpleNamespace:
    bot = DummyBot()
    job_queue = DummyJobQueue()
    return SimpleNamespace(bot=bot, chat_data={}, job_queue=job_queue)


def test_progress_flow_success_path() -> None:
    ctx = _make_context()
    progress = {
        "chat_id": 100,
        "user_id": 200,
        "mode": "veo_animate",
        "reply_to_message_id": 55,
        "success": False,
    }
    ctx.chat_data[PROGRESS_STORAGE_KEY] = progress

    asyncio.run(send_progress_message(ctx, "start"))

    assert ctx.bot.sent == [
        {
            "chat_id": 100,
            "text": "🎬 Отправляю задачу в обработку…",
            "disable_notification": True,
            "reply_to_message_id": 55,
            "allow_sending_without_reply": True,
        }
    ]
    assert "message_id" in progress
    assert len(ctx.job_queue.jobs) == 1

    progress["job_id"] = "job-123"
    asyncio.run(send_progress_message(ctx, "render"))

    assert ctx.bot.edits and ctx.bot.edits[0]["text"].startswith("💫 Видео")
    assert progress["phase"] == "render"

    progress["success"] = True
    asyncio.run(send_progress_message(ctx, "finish"))

    assert ctx.bot.edits[-1]["text"] == "✅ Готово!"
    assert PROGRESS_STORAGE_KEY not in ctx.chat_data
    job = ctx.job_queue.jobs[0]
    assert job.removed is True


def test_progress_flow_failure_deletes_message() -> None:
    ctx = _make_context()
    progress = {
        "chat_id": 1,
        "user_id": 2,
        "mode": "veo_fast",
        "reply_to_message_id": 7,
        "success": False,
    }
    ctx.chat_data[PROGRESS_STORAGE_KEY] = progress

    asyncio.run(send_progress_message(ctx, "start"))
    asyncio.run(send_progress_message(ctx, "finish"))

    assert ctx.bot.deleted == [{"chat_id": 1, "message_id": 1}]
    assert PROGRESS_STORAGE_KEY not in ctx.chat_data
