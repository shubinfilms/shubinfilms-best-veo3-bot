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
    monkeypatch.setenv("SUNO_API_TOKEN", "token")
    monkeypatch.setenv("LEDGER_BACKEND", "memory")
    monkeypatch.setenv("DATABASE_URL", "postgres://test")
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://bot.example")
    monkeypatch.setenv("SORA2_ENABLED", "true")
    module = importlib.import_module("bot")
    return importlib.reload(module)


def test_video_menu_single_card(bot_module):
    markup = bot_module.video_menu_kb()
    rows = markup.inline_keyboard

    assert len(rows) == 3

    engine_titles = [row[0].text for row in rows[:2]]
    assert "VEO" in engine_titles[0]
    assert any("Sora2" in title for title in engine_titles)

    veo_row = rows[0]
    assert veo_row[0].callback_data == bot_module.CB.VIDEO_PICK_VEO

    sora_row = rows[1]
    assert sora_row[0].callback_data in {
        bot_module.CB.VIDEO_PICK_SORA2,
        bot_module.CB.VIDEO_PICK_SORA2_DISABLED,
    }

    back_row = rows[-1]
    assert len(back_row) == 1
    assert back_row[0].callback_data == bot_module.CB.VIDEO_MENU_BACK
