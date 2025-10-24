from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from keyboards import main_menu_buttons  # noqa: E402
from ui.buttons import registry as buttons_registry  # noqa: E402


def test_profile_button_in_registry_and_keyboard_match():
    assert "profile" in buttons_registry.REGISTRY
    expected = buttons_registry.btn_data("profile")
    assert any(button.callback_data == expected for button in main_menu_buttons())
