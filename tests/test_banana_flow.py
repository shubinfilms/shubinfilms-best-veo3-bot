import asyncio
import importlib
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest
from telegram import InlineKeyboardMarkup

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def bot_module(monkeypatch):
    monkeypatch.setenv("TELEGRAM_TOKEN", "dummy-token")
    monkeypatch.setenv("SUNO_API_BASE", "https://example.com")
    monkeypatch.setenv("SUNO_API_TOKEN", "test-token")
    monkeypatch.setenv("LEDGER_BACKEND", "memory")
    monkeypatch.setenv("DATABASE_URL", "postgres://test")
    module = importlib.import_module("bot")
    return importlib.reload(module)


class _FakeBot:
    def __init__(self):
        self.photo_calls = []
        self.document_calls = []
        self.messages = []

    async def send_message(self, chat_id, text, **kwargs):
        payload = {"chat_id": chat_id, "text": text, **kwargs}
        self.messages.append(payload)
        return SimpleNamespace(message_id=len(self.messages))

    async def send_photo(self, **kwargs):
        self.photo_calls.append(kwargs)
        return SimpleNamespace(message_id=len(self.photo_calls))

    async def send_document(self, **kwargs):
        self.document_calls.append(kwargs)
        return SimpleNamespace(message_id=len(self.document_calls))


def test_banana_generate_flow(monkeypatch, tmp_path, bot_module):
    chat_id = 1
    user_id = 2
    fake_bot = _FakeBot()
    ctx = SimpleNamespace(bot=fake_bot, user_data={}, application=None)

    monkeypatch.setattr(bot_module, "create_banana_task", lambda *a, **k: "task-1")
    monkeypatch.setattr(bot_module, "wait_for_banana_result", lambda *a, **k: ["https://cdn.example.com/banana.png"])
    monkeypatch.setattr(bot_module, "_download_binary", lambda url: (b"image-bytes", "image/png"))

    def _save_bytes(data, suffix=".png"):
        path = tmp_path / f"banana{suffix}"
        path.write_bytes(data)
        return path

    monkeypatch.setattr(bot_module, "save_bytes_to_temp", _save_bytes)

    menu_calls = []

    async def _fake_show_menu(*args, **kwargs):
        menu_calls.append(True)

    monkeypatch.setattr(bot_module, "show_main_menu", _fake_show_menu)

    asyncio.run(
        bot_module._banana_run_and_send(  # type: ignore[attr-defined]
            chat_id,
            ctx,
            src_urls=["https://telegram.org/file.jpg"],
            prompt="make it sunny",
            price=5,
            user_id=user_id,
        )
    )

    assert len(fake_bot.photo_calls) == 1
    assert len(fake_bot.document_calls) == 1
    assert menu_calls == []
    assert len(fake_bot.messages) == 1
    first_message = fake_bot.messages[0]
    assert first_message["text"].startswith("🍌 Задача Banana")
    photo_call = fake_bot.photo_calls[0]
    markup = photo_call.get("reply_markup")
    assert isinstance(markup, InlineKeyboardMarkup)
    buttons = markup.inline_keyboard
    assert buttons and buttons[0][0].text == "🔁 Сгенерировать ещё"
    assert buttons[0][0].callback_data == "banana_regenerate_fresh"
