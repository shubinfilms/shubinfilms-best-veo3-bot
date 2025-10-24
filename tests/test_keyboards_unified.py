import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from keyboards import kb_home_menu, menu_pay_unified
from ui.buttons.registry import btn_data

os.environ.setdefault("DATABASE_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("LEDGER_BACKEND", "memory")


def test_kb_home_menu_layout():
    markup = kb_home_menu()
    rows = markup.inline_keyboard

    assert len(rows) == 3
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
        btn_data("profile"),
        "kb_open",
        "menu:photo",
        "menu:music",
        "menu:video",
        "menu:dialog",
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
