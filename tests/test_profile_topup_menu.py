import sys
from pathlib import Path
from types import SimpleNamespace

import asyncio

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.suno_test_utils import FakeBot, bot_module
from texts import (
    TXT_PAY_CARD,
    TXT_PAY_CRYPTO,
    TXT_PAY_STARS,
    TXT_TOPUP_CHOOSE,
    TXT_TOPUP_ENTRY,
)


def _make_update(callback_data: str, *, chat_id: int = 123, message_id: int = 456):
    async def _answer():
        return None

    message = SimpleNamespace(chat_id=chat_id, message_id=message_id)
    query = SimpleNamespace(data=callback_data, message=message, answer=_answer)
    update = SimpleNamespace(
        callback_query=query,
        effective_chat=SimpleNamespace(id=chat_id),
        effective_user=None,
    )
    return update, query


def test_profile_menu_has_topup_entry():
    markup = bot_module.balance_menu_kb()
    buttons = [button.text for row in markup.inline_keyboard for button in row]
    assert TXT_TOPUP_ENTRY in buttons


def test_profile_topup_open_shows_methods(monkeypatch):
    bot = FakeBot()
    ctx = SimpleNamespace(bot=bot, user_data={})
    update, query = _make_update(bot_module.CB_PROFILE_TOPUP)

    captured = {}

    async def fake_safe_edit_message(ctx_param, chat_id_param, message_id_param, text_param, reply_markup_param, **kwargs):
        captured["chat_id"] = chat_id_param
        captured["message_id"] = message_id_param
        captured["text"] = text_param
        captured["markup"] = reply_markup_param
        return True

    monkeypatch.setattr(bot_module, "safe_edit_message", fake_safe_edit_message)

    handled = asyncio.run(bot_module.handle_topup_callback(update, ctx, bot_module.CB_PROFILE_TOPUP))
    assert handled is True
    assert captured["text"] == TXT_TOPUP_CHOOSE

    markup = captured["markup"]
    rows = markup.inline_keyboard
    texts = [button.text for row in rows for button in row]
    assert TXT_PAY_STARS in texts
    assert TXT_PAY_CARD in texts
    assert TXT_PAY_CRYPTO in texts
