import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.suno_state import SunoState, set_style as set_suno_style

from tests.suno_test_utils import bot_module


def test_no_style_preview_guard(monkeypatch):
    state = SunoState(mode="instrumental")
    set_suno_style(state, "dream pop with long description" * 3)

    monkeypatch.delattr(bot_module, "suno_style_preview", raising=False)

    summary = bot_module._suno_summary_text(state)
    assert "dream pop" in summary
