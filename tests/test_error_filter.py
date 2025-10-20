import asyncio
from types import SimpleNamespace

from telegram.error import BadRequest

import bot


def test_old_query_error_is_suppressed(monkeypatch):
    sent: list[tuple[int, str]] = []

    async def fake_send(chat_id, text):
        sent.append((chat_id, text))

    update = SimpleNamespace(effective_chat=SimpleNamespace(id=42))
    context = SimpleNamespace(
        error=BadRequest("Query is too old and response timeout expired or query id is invalid"),
        bot=SimpleNamespace(send_message=fake_send),
    )

    asyncio.run(bot.error_handler(update, context))

    assert sent == []
