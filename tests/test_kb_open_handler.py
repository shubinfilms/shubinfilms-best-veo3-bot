import asyncio
from types import SimpleNamespace

from telegram.error import BadRequest

import handlers.knowledge_base as kb_module
from telegram_utils import safe_answer


def test_safe_answer_handles_old_query():
    class _Query:
        answered = False

        async def answer(self, **kwargs):
            raise BadRequest("Query is too old and response timeout expired or query id is invalid")

    query = _Query()
    result = asyncio.run(safe_answer(query))

    assert result == "late"
    assert query.answered is False


def test_kb_open_handler_suppresses_old_query(monkeypatch):
    calls: list[str] = []

    async def fake_render(update, context, origin):
        calls.append(origin)

    monkeypatch.setattr(kb_module, "_kb_render_or_send", fake_render)

    async def failing_answer(**kwargs):
        raise BadRequest("Query is too old and response timeout expired or query id is invalid")

    message = SimpleNamespace(chat=SimpleNamespace(id=10), chat_id=10, message_id=55)
    query = SimpleNamespace(
        data="kb_open",
        message=message,
        from_user=SimpleNamespace(id=77),
        answer=failing_answer,
    )
    update = SimpleNamespace(
        callback_query=query,
        effective_chat=message.chat,
        effective_user=query.from_user,
    )
    ctx = SimpleNamespace(bot=SimpleNamespace(), chat_data={}, application=SimpleNamespace(bot_data={}))

    asyncio.run(kb_module.kb_open_handler(update, ctx))

    assert calls == ["callback"]
