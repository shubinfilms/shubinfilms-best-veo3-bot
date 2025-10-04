import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.suno_test_utils import FakeBot, bot_module
from texts import TXT_AI_DIALOG_CHOOSE, TXT_AI_DIALOG_NORMAL, TXT_AI_DIALOG_PM, TXT_KB_AI_DIALOG


def test_ai_dialog_submenu_render(monkeypatch):
    bot = FakeBot()
    ctx = SimpleNamespace(bot=bot, user_data={}, chat_data={})

    async def fake_ensure(_update):
        return None

    monkeypatch.setattr(bot_module, "ensure_user_record", fake_ensure)

    captured = {}

    async def fake_safe_edit_message(ctx_param, chat_id_param, message_id_param, text_param, reply_markup_param, **kwargs):
        captured["chat_id"] = chat_id_param
        captured["message_id"] = message_id_param
        captured["text"] = text_param
        captured["markup"] = reply_markup_param
        return True

    monkeypatch.setattr(bot_module, "safe_edit_message", fake_safe_edit_message)

    message = SimpleNamespace(chat_id=321, message_id=654)

    async def fake_answer():
        return None

    query = SimpleNamespace(
        data=bot_module.CB_MAIN_AI_DIALOG,
        message=message,
        answer=fake_answer,
    )
    update = SimpleNamespace(
        callback_query=query,
        effective_chat=SimpleNamespace(id=321),
        effective_user=None,
    )

    asyncio.run(bot_module.hub_router(update, ctx))

    assert captured["text"] == f"{TXT_KB_AI_DIALOG}\n{TXT_AI_DIALOG_CHOOSE}"
    markup = captured["markup"]
    rows = markup.inline_keyboard
    assert rows[0][0].text == TXT_AI_DIALOG_NORMAL
    assert rows[1][0].text == TXT_AI_DIALOG_PM
