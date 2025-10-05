import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.suno_test_utils import FakeBot, bot_module
from texts import TXT_KNOWLEDGE_INTRO


@pytest.fixture
def ctx():
    bot = FakeBot()
    return SimpleNamespace(bot=bot, user_data={}, chat_data={}, application=SimpleNamespace(logger=bot_module.log))


@pytest.fixture(autouse=True)
def _setup(monkeypatch):
    async def fake_ensure(_update):
        return None

    monkeypatch.setattr(bot_module, "ensure_user_record", fake_ensure)
    yield


def _make_update(chat_id: int, user_id: int):
    message = SimpleNamespace(message_id=77, chat=SimpleNamespace(id=chat_id), chat_id=chat_id)

    async def answer():
        return None

    query = SimpleNamespace(data=None, message=message, answer=answer)
    update = SimpleNamespace(
        callback_query=query,
        effective_chat=SimpleNamespace(id=chat_id),
        effective_user=SimpleNamespace(id=user_id),
        effective_message=message,
    )
    return update, query


def test_kb_templates_to_video(monkeypatch, ctx):
    bot = ctx.bot
    edit_calls = []
    video_calls = []

    async def fake_safe_edit_message(ctx_param, chat_id_param, message_id_param, text_param, reply_markup_param, **kwargs):
        edit_calls.append(text_param)
        return True

    async def fake_start_video_menu(update, context):
        video_calls.append(True)
        return None

    monkeypatch.setattr(bot_module, "safe_edit_message", fake_safe_edit_message)
    monkeypatch.setattr(bot_module, "start_video_menu", fake_start_video_menu)

    update, query = _make_update(chat_id=555, user_id=777)

    query.data = bot_module.KNOWLEDGE_MENU_CB
    asyncio.run(bot_module.hub_router(update, ctx))

    meaningful = [entry for entry in bot.sent if (entry.get("text") or "").strip()]
    assert any(entry.get("text") == TXT_KNOWLEDGE_INTRO for entry in meaningful)

    query.data = "kb_templates"
    asyncio.run(bot_module.hub_router(update, ctx))
    assert edit_calls and "✨ Готовые шаблоны" in edit_calls[-1]

    query.data = "tpl_video"
    asyncio.run(bot_module.hub_router(update, ctx))
    assert video_calls, "Video menu should be triggered from templates"
