import os
import sys
from pathlib import Path

from keyboards import kb_home_menu, menu_pay_unified

sys.path.append(str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("DATABASE_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("LEDGER_BACKEND", "memory")


def test_kb_home_menu_layout():
    markup = kb_home_menu()
    rows = markup.inline_keyboard

    assert len(rows) == 4
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
        "home:profile",
        "home:kb",
        "home:photo",
        "home:music",
        "home:video",
        "home:chat",
    ]


def test_menu_pay_unified_layout():
    markup = menu_pay_unified()
    rows = markup.inline_keyboard

    assert len(rows) == 4

    texts = [button.text for row in rows for button in row]
    assert texts == [
        "⭐️ Телеграм Stars",
        "💳 Оплата картой",
        "🔐 Crypto",
        "⬅️ Назад",
    ]
