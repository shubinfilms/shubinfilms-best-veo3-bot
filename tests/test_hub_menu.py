from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from telegram_utils import build_hub_keyboard, build_hub_text


def test_build_hub_keyboard_layout():
    keyboard = build_hub_keyboard()
    rows = keyboard.inline_keyboard

    assert [len(row) for row in rows] == [1, 1, 2, 2]

    texts = [button.text for row in rows for button in row]
    callbacks = [button.callback_data for row in rows for button in row]

    assert texts == [
        "👥 Профиль",
        "📚 База знаний",
        "📸 Режим фото",
        "🎧 Режим музыки",
        "📹 Режим видео",
        "🧠 Диалог с ИИ",
    ]
    assert callbacks == [
        "profile:menu",
        "kb:menu",
        "image:menu",
        "music:menu",
        "video:menu",
        "ai:menu",
    ]


def test_build_hub_text_contains_balance_and_link():
    text = build_hub_text(123)

    assert text.startswith("👋 Добро пожаловать!")
    assert "💎 Ваш баланс: 123💎" in text
    assert "🧾 Больше идей и примеров — [канал с промптами](" in text
    assert text.strip().endswith("Выберите, что хотите сделать:")

    link_marker = "[канал с промптами]("
    start = text.index(link_marker) + len(link_marker)
    end = text.index(")", start)
    url = text[start:end]

    assert url
    assert url.strip() != ""
