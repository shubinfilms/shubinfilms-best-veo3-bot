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
    monkeypatch.setenv("DATABASE_URL", "postgres://test")
    module = importlib.import_module("bot")
    return importlib.reload(module)


def test_banana_card_shows_helper_line(bot_module):
    state = {"banana_images": [{"url": "https://example.com"}], "last_prompt": "", "banana_balance": 15}

    text = bot_module.banana_card_text(state)

    assert "Примеры запросов" not in text
    assert bot_module.BANANA_HELPER_LINE in text
    assert state.get("banana_helper_line") == bot_module.BANANA_HELPER_LINE
