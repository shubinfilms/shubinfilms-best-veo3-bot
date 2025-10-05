import sys
from pathlib import Path

from telegram import InlineKeyboardMarkup

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from telegram_utils import build_hub_keyboard
from texts import (
    TXT_KB_AI_DIALOG,
    TXT_KB_KNOWLEDGE,
    TXT_KB_MUSIC,
    TXT_KB_PHOTO,
    TXT_KB_PROFILE,
    TXT_KB_VIDEO,
)


def test_main_menu_profile_first_layout():
    markup = build_hub_keyboard()
    assert isinstance(markup, InlineKeyboardMarkup)

    rows = markup.inline_keyboard
    assert len(rows) == 3
    assert [button.text for button in rows[0]] == [TXT_KB_PROFILE, TXT_KB_KNOWLEDGE]
    assert [button.text for button in rows[1]] == [TXT_KB_PHOTO, TXT_KB_MUSIC]
    assert [button.text for button in rows[2]] == [TXT_KB_VIDEO, TXT_KB_AI_DIALOG]
