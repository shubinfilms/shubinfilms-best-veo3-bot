import asyncio
from typing import Any

from telegram.error import BadRequest

from db import postgres as db_postgres
from scripts import migrate_from_redis
from telegram_utils import safe_edit_long_message


def test_users_schema_idempotent(monkeypatch):
    executed: list[str] = []

    class _Connection:
        def execute(self, statement: Any) -> None:
            executed.append(str(statement))

    class _Begin:
        def __enter__(self) -> _Connection:
            return _Connection()

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    class _Engine:
        def begin(self) -> _Begin:
            return _Begin()

    monkeypatch.setattr(db_postgres, "_ensure_engine", lambda: _Engine())

    db_postgres._ensure_tables_once()
    db_postgres._ensure_tables_once()

    user_statements = [
        stmt for stmt in executed if "CREATE TABLE IF NOT EXISTS users" in stmt
    ]
    assert len(user_statements) == 2
    assert any("referral_earned_total" in stmt for stmt in executed)
    assert any("DO $$" in stmt for stmt in executed)


def test_execute_users_import(monkeypatch):
    captured: list[list[dict[str, Any]]] = []

    async def _fake_execute(engine, stmt, rows, stage):  # noqa: ANN001
        captured.append(list(rows))
        return len(rows)

    monkeypatch.setattr(migrate_from_redis, "_execute_with_retry", _fake_execute)

    sample_rows = [
        {
            "id": 7377603294,
            "username": "ghavcsh",
            "referrer_id": None,
            "joined_at": None,
            "referral_earned_total": None,
        },
        {
            "id": 5254733138,
            "username": "NioSkittle",
            "referrer_id": 123,
            "joined_at": "2024-10-01T10:00:00+00:00",
            "referral_earned_total": "5",
        },
        {
            "id": 878622103,
            "username": None,
            "referrer_id": None,
            "joined_at": None,
            "referral_earned_total": None,
        },
    ]

    first = asyncio.run(migrate_from_redis._execute_users(object(), sample_rows))
    second = asyncio.run(migrate_from_redis._execute_users(object(), sample_rows))

    assert first == len(sample_rows)
    assert second == len(sample_rows)
    assert captured
    flattened = [item for batch in captured for item in batch]
    ids = {row["id"] for row in flattened}
    assert ids == {row["id"] for row in sample_rows}


def test_safe_edit_long_message_document_fallback():
    class _Message:
        chat_id = 42

        async def edit_text(self, text: str) -> None:  # noqa: D401
            raise BadRequest("Message is too long")

    class _Bot:
        def __init__(self) -> None:
            self.sent_document = None
            self.sent_message = None

        async def send_message(self, chat_id: int, text: str):
            self.sent_message = (chat_id, text)
            return {"ok": True}

        async def send_document(self, chat_id: int, document, caption: str | None = None):
            document.seek(0)
            payload = document.read()
            self.sent_document = (chat_id, payload, caption, getattr(document, "name", None))
            return {"ok": True}

    bot = _Bot()
    message = _Message()
    long_text = "x" * 5000

    result = asyncio.run(
        safe_edit_long_message(
            bot=bot,
            message=message,
            chat_id=None,
            text=long_text,
            caption="Migration summary",
            filename="redis_summary.txt",
        )
    )

    assert bot.sent_document is not None
    chat_id, payload, caption, name = bot.sent_document
    assert chat_id == message.chat_id
    assert payload == long_text.encode("utf-8")
    assert caption == "Migration summary"
    assert name == "redis_summary.txt"
    assert result == {"ok": True}

