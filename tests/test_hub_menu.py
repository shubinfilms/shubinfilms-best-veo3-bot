from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from telegram_utils import build_hub_keyboard, build_hub_text


def test_build_hub_keyboard_layout():
    keyboard = build_hub_keyboard()
    rows = keyboard.inline_keyboard

    assert [len(row) for row in rows] == [2, 2, 2]

    texts = [button.text for row in rows for button in row]
    callbacks = [button.callback_data for row in rows for button in row]

    assert texts == [
        "👤 Профиль",
        "📚 База знаний",
        "📸 Режим фото",
        "🎧 Режим музыки",
        "📹 Режим видео",
        "🧠 Диалог с ИИ",
    ]
    assert callbacks == [
        "hub:open:profile",
        "hub:open:kb",
        "hub:open:photo",
        "hub:open:music",
        "hub:open:video",
        "hub:open:dialog",
    ]


def test_build_hub_text_contains_balance_and_link():
    text = build_hub_text(123)

    assert text == "<b>📋 Главное меню</b>\n<i>Выберите раздел:</i>"
