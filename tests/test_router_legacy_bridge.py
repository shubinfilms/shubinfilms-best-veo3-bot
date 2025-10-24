import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ui.buttons.router as router


class _DummyCounter:
    def __init__(self):
        self.calls: list[dict] = []

    def labels(self, **labels):
        self.calls.append(labels)
        return self

    def inc(self):
        return None


@pytest.mark.parametrize(
    "raw,expected_source",
    [
        ("profile", "button"),
        ("profile.open", "button"),
        ("open:profile", "menu"),
        ("quick:profile", "quick"),
    ],
)
def test_legacy_payloads_route_to_profile(raw, expected_source, monkeypatch):
    ack_calls: list[int] = []

    async def fake_safe_answer(query, cache_time=0):
        ack_calls.append(cache_time)
        return "ok"

    dispatched: list[tuple] = []

    async def fake_dispatch(action, update, ctx, *, raw_data, payload=None, legacy=False):
        dispatched.append((action, raw_data, payload, legacy))

    legacy_counter = _DummyCounter()

    monkeypatch.setattr(router, "safe_answer", fake_safe_answer)
    monkeypatch.setattr(router, "dispatch_via_registry", fake_dispatch)
    monkeypatch.setattr(router, "ui_callback_legacy_forwarded_total", legacy_counter)

    query = SimpleNamespace(data=raw)
    update = SimpleNamespace(
        callback_query=query,
        effective_chat=SimpleNamespace(id=1),
        effective_user=SimpleNamespace(id=2),
    )
    ctx = SimpleNamespace()

    asyncio.run(router.on_callback(update, ctx))

    assert ack_calls, "callback should be acknowledged"
    assert dispatched and dispatched[0][0] == "profile"
    assert dispatched[0][1] == "btn:profile"
    assert dispatched[0][3] is True
    payload = dispatched[0][2] or {}
    assert payload.get("legacy") is True
    assert payload.get("source") == expected_source
    assert legacy_counter.calls and legacy_counter.calls[0]["target"] == "profile"
