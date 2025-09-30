from types import SimpleNamespace
import asyncio
import os
import sys
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("LEDGER_BACKEND", "memory")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bot
from suno.service import RecordInfoPollResult


def test_suno_poll_success(monkeypatch):
    monkeypatch.setattr(bot, "SUNO_POLL_FIRST_DELAY", 0.0, raising=False)
    monkeypatch.setattr(bot, "SUNO_POLL_BACKOFF_SERIES", [0.0], raising=False)
    monkeypatch.setattr(bot, "SUNO_POLL_TIMEOUT", 1.0, raising=False)

    poll_sequence = iter(
        [
            RecordInfoPollResult(
                state="pending",
                status_code=404,
                payload={"data": {"status": "QUEUED"}},
                message="pending",
            ),
            RecordInfoPollResult(
                state="ready",
                status_code=200,
                payload={
                    "code": 200,
                    "data": {
                        "taskId": "task-123",
                        "status": "SUCCESS",
                        "response": {
                            "tracks": [
                                {
                                    "id": "track-1",
                                    "audioUrl": "https://cdn.example/track.mp3",
                                    "imageUrl": "https://cdn.example/cover.jpg",
                                    "duration": 42,
                                }
                            ]
                        },
                    },
                },
            ),
        ]
    )

    def fake_poll(task_id: str, *, user_id=None):
        try:
            return next(poll_sequence)
        except StopIteration:
            return RecordInfoPollResult(
                state="pending",
                status_code=404,
                payload={"data": {"status": "PROCESSING"}},
                message="pending",
            )

    monkeypatch.setattr(bot.SUNO_SERVICE, "poll_record_info_once", fake_poll, raising=False)
    monkeypatch.setattr(bot.SUNO_SERVICE, "_recently_delivered", lambda *_: False, raising=False)
    monkeypatch.setattr(bot.SUNO_SERVICE, "get_task_record", lambda *_: {}, raising=False)

    deliveries: list[tuple] = []

    def fake_handle(task, req_id=None, delivery_via="webhook"):
        deliveries.append((task, req_id, delivery_via))

    monkeypatch.setattr(bot.SUNO_SERVICE, "handle_callback", fake_handle, raising=False)

    notifications: list[str] = []

    async def fake_notify(ctx, chat_id, text, **kwargs):
        notifications.append(text)
        return None

    monkeypatch.setattr(bot, "_suno_notify", fake_notify, raising=False)

    refunds: list[dict[str, str]] = []

    async def fake_issue_refund(ctx, chat_id, user_id, *, base_meta, task_id, error_text, reason, **kwargs):
        refunds.append({"reason": reason, "error": error_text})
        return None

    monkeypatch.setattr(bot, "_suno_issue_refund", fake_issue_refund, raising=False)

    async def fake_refresh(*args, **kwargs):
        return None

    monkeypatch.setattr(bot, "refresh_suno_card", fake_refresh, raising=False)
    monkeypatch.setattr(bot, "refresh_balance_card_if_open", fake_refresh, raising=False)
    monkeypatch.setattr(bot, "_reset_suno_card_cache", lambda _: None, raising=False)

    ctx = SimpleNamespace(bot=SimpleNamespace(), user_data={})
    params = {"title": "Demo", "style": "Chill", "lyrics": "", "instrumental": True}
    meta = {"req_id": "req-123"}

    asyncio.run(
        bot._poll_suno_and_send(
            chat_id=111,
            ctx=ctx,
            user_id=7,
            task_id="task-123",
            params=params,
            meta=meta,
            req_id="req-123",
            reply_to=None,
        )
    )

    assert deliveries, "callback delivery should be invoked"
    delivered_task, delivered_req, via = deliveries[0]
    assert delivered_req == "req-123"
    assert via == "poll"
    assert delivered_task.items and delivered_task.items[0].audio_url.endswith(".mp3")
    assert refunds == []
    assert notifications == []
