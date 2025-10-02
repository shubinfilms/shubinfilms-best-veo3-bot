import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace


from telegram import ReplyKeyboardMarkup


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from tests.suno_test_utils import FakeBot, bot_module


def _build_update(chat_id: int = 123, user_id: int = 555):
    message = SimpleNamespace(message_id=42, chat_id=chat_id)
    chat = SimpleNamespace(id=chat_id)
    user = SimpleNamespace(id=user_id)
    return SimpleNamespace(
        callback_query=None,
        effective_chat=chat,
        effective_user=user,
        effective_message=message,
        message=message,
    )


def test_main_menu_keyboard_layout():
    markup = bot_module.main_menu_kb()
    assert isinstance(markup, ReplyKeyboardMarkup)

    rows = markup.keyboard
    labels = [[button.text for button in row] for row in rows]

    assert labels == [
        [bot_module.MENU_BTN_VIDEO],
        [bot_module.MENU_BTN_IMAGE],
        [bot_module.MENU_BTN_SUNO],
        [bot_module.MENU_BTN_PM],
        [bot_module.MENU_BTN_CHAT],
        [bot_module.MENU_BTN_BALANCE],
    ]
    flattened = [text for row in labels for text in row]
    assert bot_module.MENU_BTN_SUPPORT not in flattened


def test_menu_command_always_sends_welcome_block(monkeypatch):
    bot = FakeBot()
    ctx = SimpleNamespace(bot=bot, user_data={})

    async def fake_ensure(update):
        return None

    monkeypatch.setattr(bot_module, "ensure_user_record", fake_ensure)
    monkeypatch.setattr(bot_module, "set_mode", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(bot_module, "clear_wait", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(bot_module, "_safe_get_balance", lambda _uid: 123)

    update = _build_update()

    asyncio.run(bot_module.handle_menu(update, ctx))
    first_hub_id = ctx.user_data.get("hub_msg_id")

    assert isinstance(first_hub_id, int)
    assert bot.sent
    assert bot.sent[0]["text"].startswith("👋 Добро пожаловать!")

    asyncio.run(bot_module.handle_menu(update, ctx))
    second_hub_id = ctx.user_data.get("hub_msg_id")

    assert isinstance(second_hub_id, int)
    assert second_hub_id != first_hub_id

    welcome_texts = [entry["text"] for entry in bot.sent]
    assert len(welcome_texts) == 2
    assert all(text.startswith("👋 Добро пожаловать!") for text in welcome_texts)

    assert bot.deleted
    deleted_ids = {payload.get("message_id") for payload in bot.deleted}
    assert first_hub_id in deleted_ids
