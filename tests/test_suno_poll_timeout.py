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


def test_suno_poll_timeout(monkeypatch):
    monkeypatch.setattr(bot, "SUNO_POLL_FIRST_DELAY", 0.0, raising=False)
    monkeypatch.setattr(bot, "SUNO_POLL_BACKOFF_SERIES", [0.05], raising=False)
    monkeypatch.setattr(bot, "SUNO_POLL_TIMEOUT", 0.12, raising=False)

    def fake_poll(task_id: str, *, user_id=None):
        return RecordInfoPollResult(
            state="pending",
            status_code=404,
            payload={"data": {"status": "PROCESSING"}},
            message="pending",
        )

    monkeypatch.setattr(bot.SUNO_SERVICE, "poll_record_info_once", fake_poll, raising=False)
    monkeypatch.setattr(bot.SUNO_SERVICE, "_recently_delivered", lambda *_: False, raising=False)
    monkeypatch.setattr(bot.SUNO_SERVICE, "get_task_record", lambda *_: {}, raising=False)

    refunds: list[dict[str, str]] = []

    async def fake_issue_refund(ctx, chat_id, user_id, *, base_meta, task_id, error_text, reason, **kwargs):
        refunds.append({"reason": reason, "error": error_text})
        return None

    monkeypatch.setattr(bot, "_suno_issue_refund", fake_issue_refund, raising=False)

    async def fake_notify(ctx, chat_id, text, **kwargs):
        return None

    monkeypatch.setattr(bot, "_suno_notify", fake_notify, raising=False)

    async def fake_refresh(*args, **kwargs):
        return None

    monkeypatch.setattr(bot, "refresh_suno_card", fake_refresh, raising=False)
    monkeypatch.setattr(bot, "refresh_balance_card_if_open", fake_refresh, raising=False)
    monkeypatch.setattr(bot, "_reset_suno_card_cache", lambda _: None, raising=False)

    ctx = SimpleNamespace(bot=SimpleNamespace(), user_data={})

    asyncio.run(
        bot._poll_suno_and_send(
            chat_id=222,
            ctx=ctx,
            user_id=9,
            task_id="task-timeout",
            params={"title": "Demo", "style": "Ambient", "lyrics": "", "instrumental": True},
            meta={"req_id": "req-timeout"},
            req_id="req-timeout",
            reply_to=None,
        )
    )

    assert refunds, "timeout should trigger a refund"
    assert refunds[0]["reason"] == "suno:refund:timeout"
    assert "не ответил" in refunds[0]["error"]
