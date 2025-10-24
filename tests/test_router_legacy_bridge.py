import asyncio
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ui.buttons.router as router


@pytest.mark.parametrize(
    "raw,expected_action,expected_source",
    [
        ("profile", "open", "button"),
        ("profile.open", "open", "button"),
        ("open:profile", "open", "menu"),
        ("quick:profile", "open", "quick"),
        ("profile:invite", "invite", "button"),
        ("profile:promo", "promo", "button"),
    ],
)
def test_legacy_profile_payloads_bridge_to_namespace(raw, expected_action, expected_source, monkeypatch):
    ack_calls: list[int] = []
    dispatched: list[router.NormalizedCallback] = []

    async def fake_safe_answer(query, cache_time=0):
        ack_calls.append(cache_time)
        return "ok"

    async def fake_namespace_dispatch(normalized, update, ctx):
        dispatched.append(normalized)

    monkeypatch.setattr(router, "safe_answer", fake_safe_answer)
    monkeypatch.setattr(router, "dispatch_via_registry", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("registry should not be used")))
    monkeypatch.setattr(router, "_dispatch_namespace_callback", fake_namespace_dispatch)
    monkeypatch.setattr(router, "_legacy_bridge_enabled", lambda: True)

    query = SimpleNamespace(data=raw)
    update = SimpleNamespace(
        callback_query=query,
        effective_chat=SimpleNamespace(id=1),
        effective_user=SimpleNamespace(id=2),
    )
    ctx = SimpleNamespace()

    asyncio.run(router.on_callback(update, ctx))

    assert ack_calls, "callback should be acknowledged"
    assert dispatched, "namespace dispatch should be used"
    normalized = dispatched[0]
    assert normalized.namespace == "profile"
    assert normalized.action == expected_action
    assert normalized.legacy is True
    payload = dict(normalized.payload)
    assert payload.get("legacy") is True
    assert payload.get("source") == expected_source


def test_stars_buy_routes_to_payments(monkeypatch):
    ack_calls: list[int] = []
    dispatched: list[router.NormalizedCallback] = []

    async def fake_safe_answer(query, cache_time=0):
        ack_calls.append(cache_time)
        return "ok"

    async def fake_namespace_dispatch(normalized, update, ctx):
        dispatched.append(normalized)

    monkeypatch.setattr(router, "safe_answer", fake_safe_answer)
    monkeypatch.setattr(router, "dispatch_via_registry", lambda *args, **kwargs: None)
    monkeypatch.setattr(router, "_dispatch_namespace_callback", fake_namespace_dispatch)
    monkeypatch.setattr(router, "_legacy_bridge_enabled", lambda: True)

    query = SimpleNamespace(data="stars:buy:50")
    update = SimpleNamespace(
        callback_query=query,
        effective_chat=SimpleNamespace(id=10),
        effective_user=SimpleNamespace(id=20),
    )
    ctx = SimpleNamespace()

    asyncio.run(router.on_callback(update, ctx))

    assert ack_calls
    assert dispatched, "payments namespace should be dispatched"
    normalized = dispatched[0]
    assert normalized.namespace == "payments"
    assert normalized.action == "stars_buy"
    assert dict(normalized.payload).get("amount") == 50
