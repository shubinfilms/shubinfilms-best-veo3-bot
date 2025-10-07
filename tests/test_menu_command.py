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
        ["👤 Профиль", "📚 База знаний"],
        ["📸 Режим фото", "🎧 Режим музыки"],
        ["📹 Режим видео", "🧠 Диалог с ИИ"],
    ]


def test_video_menu_keyboard_options():
    markup = bot_module.video_menu_kb()
    rows = markup.inline_keyboard

    assert rows[0][0].text == "🎥 VEO"
    assert rows[0][0].callback_data == bot_module.CB_VIDEO_ENGINE_VEO

    assert "Sora2" in rows[1][0].text
    assert rows[1][0].callback_data in (
        bot_module.CB_VIDEO_ENGINE_SORA2,
        bot_module.CB_VIDEO_ENGINE_SORA2_DISABLED,
    )

    assert rows[2][0].text == "⬅️ Назад"
    assert rows[2][0].callback_data == bot_module.CB_VIDEO_BACK


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
    assert bot.sent[0]["text"].startswith("<b>📋 Главное меню</b>")

    asyncio.run(bot_module.handle_menu(update, ctx))
    second_hub_id = ctx.user_data.get("hub_msg_id")

    assert isinstance(second_hub_id, int)
    assert second_hub_id != first_hub_id

    welcome_texts = [entry["text"] for entry in bot.sent if entry.get("text", "").strip()]
    assert len(welcome_texts) == 2
    assert all(text.startswith("<b>📋 Главное меню</b>") for text in welcome_texts)

    assert bot.deleted
    deleted_ids = {payload.get("message_id") for payload in bot.deleted}
    assert first_hub_id in deleted_ids


def test_callback_handler_patterns_unique():
    patterns = [pattern if pattern is not None else "<default>" for pattern, _ in bot_module.CALLBACK_HANDLER_SPECS]
    assert len(patterns) == len(set(patterns))
