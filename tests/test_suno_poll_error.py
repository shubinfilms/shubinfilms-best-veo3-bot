from types import SimpleNamespace
import asyncio
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("DATABASE_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("LEDGER_BACKEND", "memory")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bot
from suno.service import RecordInfoPollResult


def test_suno_poll_error_refunds(monkeypatch):
    monkeypatch.setattr(bot, "SUNO_POLL_FIRST_DELAY", 0.0, raising=False)
    monkeypatch.setattr(bot, "SUNO_POLL_BACKOFF_SERIES", [0.0], raising=False)
    monkeypatch.setattr(bot, "SUNO_POLL_TIMEOUT", 1.0, raising=False)

    def fake_poll(task_id: str, *, user_id=None):
        return RecordInfoPollResult(
            state="hard_error",
            status_code=400,
            payload={"data": {"status": "ERROR"}},
            message="Invalid request",
        )

    monkeypatch.setattr(bot.SUNO_SERVICE, "poll_record_info_once", fake_poll, raising=False)
    monkeypatch.setattr(bot.SUNO_SERVICE, "_recently_delivered", lambda *_: False, raising=False)
    monkeypatch.setattr(bot.SUNO_SERVICE, "get_task_record", lambda *_: {}, raising=False)

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

    asyncio.run(
        bot._poll_suno_and_send(
            chat_id=333,
            ctx=ctx,
            user_id=5,
            task_id="task-error",
            params={"title": "Demo", "style": "Pop", "lyrics": "", "instrumental": True},
            meta={"req_id": "req-error"},
            req_id="req-error",
            reply_to=None,
        )
    )

    assert refunds, "hard error should trigger refund"
    assert refunds[0]["reason"] == "suno:refund:status_err"
    assert "Invalid request" in refunds[0]["error"]
    assert notifications, "user should be notified about the error"
