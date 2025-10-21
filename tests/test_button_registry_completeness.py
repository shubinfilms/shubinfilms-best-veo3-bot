import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.buttons import registry as buttons_registry  # noqa: E402


def test_all_keyboard_actions_in_registry():
    buttons_registry.assert_registry_matches_keyboard()


def test_home_button_callbacks_match_ids():
    mappings = buttons_registry.iter_home_button_mappings()
    assert mappings, "expected home button mappings"
    for button_id, callback in mappings.items():
        assert button_id in buttons_registry.BUTTONS
        assert button_id in callback
        assert isinstance(callback, str)
