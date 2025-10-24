import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ui.buttons.idempotency as idempotency


def test_profile_debounce_window(monkeypatch):
    idempotency._RECENT_CLICKS.clear()
    times = iter([0.0, 0.1, 0.31])

    def fake_now():
        return next(times)

    monkeypatch.setattr(idempotency, "_now", fake_now)

    assert idempotency.should_process(42, "btn:profile|src=video") is True
    assert idempotency.should_process(42, "btn:profile|src=hub") is False
    assert idempotency.should_process(42, "btn:profile") is True
