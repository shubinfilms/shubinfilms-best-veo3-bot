from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from telegram_utils import build_hub_keyboard, build_hub_text


def test_build_hub_keyboard_layout():
    keyboard = build_hub_keyboard()
    rows = keyboard.inline_keyboard

    expected = [
        [("🧠 Prompt-Master", "hub:prompt")],
        [
            ("🎬 Генерация видео", "hub:video"),
            ("🎨 Генерация изображений", "hub:image"),
        ],
        [
            ("🎵 Генерация музыки", "hub:music"),
            ("💬 Обычный чат", "hub:chat"),
        ],
        [
            ("💎 Баланс", "hub:balance"),
            ("🌐 Язык", "hub:lang"),
        ],
        [("🆘 Поддержка", "hub:help")],
    ]

    actual = [[(button.text, button.callback_data) for button in row] for row in rows]

    assert actual == expected


def test_build_hub_text_contains_balance_and_link():
    text = build_hub_text(123)

    assert text == "<b>🏠 Главное меню</b>\nВыберите нужный раздел:"
