import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from telegram import ReplyKeyboardMarkup

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.suno_test_utils import FakeBot, bot_module


@pytest.fixture
def ctx():
    bot = FakeBot()
    return SimpleNamespace(bot=bot, user_data={}, chat_data={}, application=SimpleNamespace(logger=bot_module.log))


@pytest.fixture(autouse=True)
def _patch_ensure(monkeypatch):
    async def fake_ensure(_update):
        return None

    monkeypatch.setattr(bot_module, "ensure_user_record", fake_ensure)

    async def fake_safe_edit_message(ctx_param, chat_id_param, message_id_param, text_param, reply_markup_param, **kwargs):
        return True

    monkeypatch.setattr(bot_module, "safe_edit_message", fake_safe_edit_message)
    yield


def _make_update(chat_id: int, user_id: int):
    message = SimpleNamespace(message_id=11, chat=SimpleNamespace(id=chat_id), chat_id=chat_id)

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


def test_quick_keyboard_sent_once(monkeypatch, ctx):
    bot = ctx.bot

    async def fake_start_video_menu(update, context):
        return None

    monkeypatch.setattr(bot_module, "start_video_menu", fake_start_video_menu)

    command_update = SimpleNamespace(
        callback_query=None,
        effective_chat=SimpleNamespace(id=999),
        effective_user=SimpleNamespace(id=888),
        effective_message=SimpleNamespace(message_id=21, chat_id=999),
        message=SimpleNamespace(message_id=21, chat_id=999),
    )

    asyncio.run(bot_module.handle_menu(command_update, ctx))

    update, query = _make_update(chat_id=999, user_id=888)

    query.data = bot_module.VIDEO_MENU_CB
    asyncio.run(bot_module.hub_router(update, ctx))

    reply_keyboards = [
        entry for entry in bot.sent if isinstance(entry.get("reply_markup"), ReplyKeyboardMarkup)
    ]
    assert len(reply_keyboards) == 0
