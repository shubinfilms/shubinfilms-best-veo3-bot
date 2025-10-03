import asyncio
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def bot_module(monkeypatch):
    monkeypatch.setenv("TELEGRAM_TOKEN", "test-token")
    monkeypatch.setenv("SUNO_API_BASE", "https://example.com")
    monkeypatch.setenv("SUNO_API_TOKEN", "dummy-token")
    monkeypatch.setenv("LEDGER_BACKEND", "memory")
    monkeypatch.setenv("DATABASE_URL", "postgres://test")
    module = importlib.import_module("bot")
    module = importlib.reload(module)
    module.mj_log.disabled = True
    return module


def test_mj_ui_buttons(monkeypatch, bot_module):
    root_keyboard = bot_module.mj_upscale_root_keyboard("grid")
    texts = [[button.text for button in row] for row in root_keyboard.inline_keyboard]
    assert texts == [
        ["✨ Улучшить качество"],
        ["🔁 Сгенерировать ещё"],
        ["🏠 Назад в меню"],
    ]

    select_keyboard = bot_module.mj_upscale_select_keyboard("grid", count=4)
    select_texts = [row[0].text for row in select_keyboard.inline_keyboard[:-1]]
    assert select_texts == [
        "Первая фотография",
        "Вторая фотография",
        "Третья фотография",
        "Четвёртая фотография",
    ]

    monkeypatch.setattr(
        bot_module,
        "_download_mj_image_bytes",
        lambda url, index: (b"x" * 2048, f"midjourney_{index:02d}.jpeg", "image/jpeg", url),
    )
    monkeypatch.setattr(bot_module, "set_mj_gallery", lambda *args: None)

    async def fake_send_document(chat_id, document, **kwargs):
        return SimpleNamespace(message_id=index_counter.pop(0))

    sent_messages = []

    async def fake_send_message(chat_id, text, reply_markup=None):
        sent_messages.append((text, reply_markup))
        return SimpleNamespace(message_id=500)

    index_counter = [11, 12, 13, 14]
    bot = SimpleNamespace(send_document=fake_send_document, send_message=fake_send_message)
    ctx = SimpleNamespace(bot=bot, user_data={})

    state = bot_module.state(ctx)
    state["mj_locale"] = "ru"

    asyncio.run(
        bot_module._deliver_mj_grid_documents(
            ctx,
            chat_id=1,
            user_id=2,
            grid_id="grid",
            urls=[f"https://cdn/{i}.jpeg" for i in range(4)],
            prompt="p",
        )
    )

    assert sent_messages
    text, markup = sent_messages[0]
    assert text == "Галерея сгенерирована."
    assert markup is not None
    markup_texts = [[button.text for button in row] for row in markup.inline_keyboard]
    assert markup_texts[0][0] == "✨ Улучшить качество"
    assert markup_texts[1][0] == "🔁 Сгенерировать ещё"
    assert markup_texts[2][0] == "🏠 Назад в меню"
