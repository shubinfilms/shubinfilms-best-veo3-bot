import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def bot_module(monkeypatch):
    monkeypatch.setenv("TELEGRAM_TOKEN", "dummy-token")
    monkeypatch.setenv("SUNO_API_BASE", "https://example.com")
    monkeypatch.setenv("SUNO_API_TOKEN", "test-token")
    monkeypatch.setenv("LEDGER_BACKEND", "memory")
    monkeypatch.setenv("DATABASE_URL", "postgres://user:pass@localhost:5432/testdb")
    module = importlib.import_module("bot")
    return importlib.reload(module)


def test_banana_card_compact_summary(bot_module):
    state = {"banana_images": [{"url": "https://example.com"}], "last_prompt": "", "banana_balance": 15}

    text = bot_module.banana_card_text(state)

    assert "Примеры запросов" not in text
    assert "📸 Фото: 1/4" in text
    assert "Промпт: нет" in text
    assert "banana_helper_line" not in state


def test_banana_card_shows_prompt(bot_module):
    state = {"banana_images": [], "last_prompt": "замени фон на студийный", "banana_balance": 575}

    text = bot_module.banana_card_text(state)

    assert "Промпт: есть" in text


def test_banana_keyboard_layout(bot_module):
    keyboard = bot_module.banana_kb()
    rows = keyboard.inline_keyboard

    assert rows[0][0].text == "✨ Готовые шаблоны"
    assert rows[1][0].text == "⚙️ Движок"
    assert rows[1][1].text == "⬅️ Назад"
