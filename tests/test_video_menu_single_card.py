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
    module = importlib.import_module("bot")
    return importlib.reload(module)


def test_video_menu_single_card(bot_module):
    markup = bot_module.video_menu_kb()
    rows = markup.inline_keyboard

    # 5 generation options + 1 back button
    assert len(rows) == 6

    titles = [row[0].text for row in rows[:5]]
    assert any("Veo Fast" in title for title in titles)
    assert any("Veo Quality" in title for title in titles)
    assert any("Оживить изображение (Veo)" in title for title in titles)
    assert any("Sora 2 (Text-to-Video)" in title for title in titles)
    assert any("Sora 2 (Image-to-Video)" in title for title in titles)

    back_row = rows[-1]
    assert len(back_row) == 1
    assert back_row[0].callback_data == bot_module.CB_VIDEO_BACK
