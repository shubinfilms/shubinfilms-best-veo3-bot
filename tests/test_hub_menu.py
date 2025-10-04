from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from keyboards import (
    CB_MAIN_AI_DIALOG,
    CB_MAIN_KNOWLEDGE,
    CB_MAIN_MUSIC,
    CB_MAIN_PHOTO,
    CB_MAIN_PROFILE,
    CB_MAIN_VIDEO,
)
from telegram_utils import build_hub_keyboard, build_hub_text
from texts import (
    TXT_KB_AI_DIALOG,
    TXT_KB_KNOWLEDGE,
    TXT_KB_MUSIC,
    TXT_KB_PHOTO,
    TXT_KB_PROFILE,
    TXT_KB_VIDEO,
    TXT_MENU_TITLE,
)


def test_build_hub_keyboard_layout():
    keyboard = build_hub_keyboard()
    rows = keyboard.inline_keyboard

    assert [len(row) for row in rows] == [1, 1, 2, 2]

    expected = [
        (TXT_KB_PROFILE, CB_MAIN_PROFILE),
        (TXT_KB_KNOWLEDGE, CB_MAIN_KNOWLEDGE),
        (TXT_KB_PHOTO, CB_MAIN_PHOTO),
        (TXT_KB_MUSIC, CB_MAIN_MUSIC),
        (TXT_KB_VIDEO, CB_MAIN_VIDEO),
        (TXT_KB_AI_DIALOG, CB_MAIN_AI_DIALOG),
    ]

    actual = [(button.text, button.callback_data) for row in rows for button in row]

    assert actual == expected


def test_build_hub_text_contains_balance_and_link():
    text = build_hub_text(123)

    assert text.startswith("👋 Добро пожаловать!")
    assert TXT_MENU_TITLE in text
    assert "💎 Ваш баланс: 123" in text

    link_marker = "[канал с промптами]("
    assert link_marker in text
    start = text.index(link_marker) + len(link_marker)
    end = text.index(")", start)
    url = text[start:end]

    assert url
    assert url.strip() != ""
