import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from suno.service import RecordInfoPollResult

os.environ.setdefault("DATABASE_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("LEDGER_BACKEND", "memory")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bot  # noqa: E402


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_reconcile_delivers_late_result(monkeypatch):
    monkeypatch.setattr(bot, "SUNO_POLL_FIRST_DELAY", 0.0, raising=False)
    monkeypatch.setattr(bot, "SUNO_POLL_BACKOFF_SERIES", [0.0], raising=False)
    monkeypatch.setattr(bot, "SUNO_POLL_TIMEOUT", 0.0, raising=False)

    pending_result = RecordInfoPollResult(
        state="pending",
        status_code=404,
        payload={"data": {}},
        message="pending",
    )

    monkeypatch.setattr(
        bot.SUNO_SERVICE,
        "poll_record_info_once",
        lambda *args, **kwargs: pending_result,
        raising=False,
    )
    monkeypatch.setattr(bot.SUNO_SERVICE, "_recently_delivered", lambda *_: False, raising=False)
    monkeypatch.setattr(bot.SUNO_SERVICE, "get_task_record", lambda *_: {}, raising=False)

    refunds: list[str] = []

    async def fake_issue_refund(*args, **kwargs):
        refunds.append(kwargs.get("reason", ""))

    async def async_noop(*args, **kwargs):
        return None

    monkeypatch.setattr(bot, "_suno_issue_refund", fake_issue_refund, raising=False)
    monkeypatch.setattr(bot, "_suno_notify", async_noop, raising=False)
    monkeypatch.setattr(bot, "refresh_suno_card", async_noop, raising=False)
    monkeypatch.setattr(bot, "refresh_balance_card_if_open", async_noop, raising=False)
    monkeypatch.setattr(bot, "_reset_suno_card_cache", lambda *_: None, raising=False)

    ctx = SimpleNamespace(bot=SimpleNamespace(), user_data={})

    await bot._poll_suno_and_send(
        chat_id=555,
        ctx=ctx,
        user_id=42,
        task_id="task-timeout",
        params={"title": "Song", "style": "Pop", "lyrics": "", "instrumental": True},
        meta={"req_id": "req-timeout"},
        req_id="req-timeout",
        reply_to=None,
    )

    assert not refunds

    ready_payload = {
        "data": {
            "response": {
                "tracks": [
                    {
                        "audioUrl": "https://example.com/track.mp3",
                        "imageUrl": "https://example.com/cover.jpg",
                        "title": "Song",
                    }
                ]
            }
        }
    }

    ready_result = RecordInfoPollResult(
        state="ready",
        status_code=200,
        payload=ready_payload,
        message=None,
    )

    monkeypatch.setattr(
        bot.SUNO_SERVICE,
        "poll_record_info_once",
        lambda *args, **kwargs: ready_result,
        raising=False,
    )

    now_iso = datetime.now(timezone.utc).isoformat()
    record = {
        "task_id": "task-timeout",
        "status": "started",
        "tracks": [],
        "req_id": "req-timeout",
        "created_at": now_iso,
        "updated_at": now_iso,
        "user_id": 42,
    }

    monkeypatch.setattr(
        bot.SUNO_SERVICE,
        "list_last_tasks",
        lambda limit=5: [record],
        raising=False,
    )
    monkeypatch.setattr(bot.SUNO_SERVICE, "get_request_id", lambda *_: "req-timeout", raising=False)
    monkeypatch.setattr(bot.SUNO_SERVICE, "_recently_delivered", lambda *_: False, raising=False)

    delivered: list[dict[str, object]] = []

    def fake_handle_callback(task, req_id=None, *, delivery_via="webhook"):
        delivered.append({
            "task_id": task.task_id,
            "via": delivery_via,
            "tracks": len(getattr(task, "items", [])),
        })

    monkeypatch.setattr(bot.SUNO_SERVICE, "handle_callback", fake_handle_callback, raising=False)
    bot._SUNO_RECONCILE_ATTEMPTS.clear()

    await bot._suno_reconcile_once()

    assert len(delivered) == 1
    assert delivered[0]["via"] == "reconcile"
    assert not refunds
