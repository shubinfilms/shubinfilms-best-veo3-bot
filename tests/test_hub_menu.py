from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from telegram_utils import build_hub_keyboard, build_hub_text


def test_build_hub_keyboard_layout():
    keyboard = build_hub_keyboard()
    rows = keyboard.inline_keyboard

    assert len(rows) == 2
    assert all(len(row) == 3 for row in rows)

    expected = [
        ("🎬", "hub:video"),
        ("🎨", "hub:image"),
        ("🎵", "hub:music"),
        ("🧠", "hub:prompt"),
        ("💬", "hub:chat"),
        ("💎", "hub:balance"),
    ]

    actual = [(button.text, button.callback_data) for row in rows for button in row]

    assert actual == expected


def test_build_hub_text_contains_balance_and_link():
    text = build_hub_text(123)

    assert text.startswith("👋 Добро пожаловать!")
    assert "💎 Ваш баланс: 123" in text

    link_marker = "[канал с промптами]("
    assert link_marker in text
    start = text.index(link_marker) + len(link_marker)
    end = text.index(")", start)
    url = text[start:end]

    assert url
    assert url.strip() != ""
