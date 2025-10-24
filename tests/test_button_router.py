from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import handlers.profile as profile_handlers
import ui.buttons.idempotency as idempotency
import ui.buttons.router as router
from ui.buttons.registry import btn_data


@pytest.fixture(autouse=True)
def _reset_idempotency(monkeypatch):
    idempotency._RECENT_CLICKS.clear()
    monkeypatch.setattr(idempotency, "_DEBOUNCE_WINDOW", 0.5)
    monkeypatch.setattr(idempotency, "_DEBOUNCE_MS", 500.0)


def _make_update(data: str, *, user_id: int = 42) -> SimpleNamespace:
    message = SimpleNamespace(chat=SimpleNamespace(id=1001), message_id=555)
    query = SimpleNamespace(data=data, message=message, from_user=SimpleNamespace(id=user_id))
    return SimpleNamespace(
        callback_query=query,
        effective_chat=message.chat,
        effective_user=query.from_user,
    )


def test_profile_click_routed(monkeypatch):
    calls: list[str] = []

    async def fake_safe_answer(query, **kwargs):
        calls.append("ack")
        return "ok"

    async def fake_open(update, ctx, payload=None, *, suppress_nav=True, reuse=True):
        calls.append("handler")
        assert payload is None
        assert suppress_nav is True
        assert reuse is True

    monkeypatch.setattr(router, "safe_answer", fake_safe_answer)
    monkeypatch.setattr(profile_handlers, "open", fake_open)

    update = _make_update(btn_data("profile"))
    ctx = SimpleNamespace(application=SimpleNamespace(bot_data={}))

    async def scenario():
        await router.on_callback(update, ctx)

    asyncio.run(scenario())

    assert calls == ["ack", "handler"]


def test_unmatched_callback_logged_not_crash(monkeypatch, caplog):
    ack_calls: list[str] = []

    async def fake_safe_answer(query, **kwargs):
        ack_calls.append("ack")
        return "ok"

    monkeypatch.setattr(router, "safe_answer", fake_safe_answer)

    update = _make_update("btn:unknown")
    ctx = SimpleNamespace(application=SimpleNamespace(bot_data={}))

    with caplog.at_level("INFO"):
        asyncio.run(router.on_callback(update, ctx))

    assert ack_calls == ["ack"]
    assert any("ui.callback.unmatched" in record.getMessage() for record in caplog.records)


def test_profile_click_coalesced(monkeypatch):
    metrics_calls: list[tuple[str, dict[str, str]]] = []
    handler_calls: list[str] = []

    async def fake_safe_answer(query, **kwargs):
        return "ok"

    async def fake_open(update, ctx, payload=None, *, suppress_nav=True, reuse=True):
        handler_calls.append("handler")

    def fake_metrics_inc(name: str, *, tags, value: float = 1.0, service: str = "bot"):
        metrics_calls.append((name, dict(tags)))

    monkeypatch.setattr(router, "safe_answer", fake_safe_answer)
    monkeypatch.setattr(profile_handlers, "open", fake_open)
    monkeypatch.setattr(router, "metrics_inc", fake_metrics_inc)

    update = _make_update(btn_data("profile"))
    ctx = SimpleNamespace(application=SimpleNamespace(bot_data={}))

    async def scenario():
        await router.on_callback(update, ctx)
        second_update = _make_update(btn_data("profile"))
        await router.on_callback(second_update, ctx)

    asyncio.run(scenario())

    assert handler_calls == ["handler"]
    assert ("ui_callback_total", {"action": "profile", "result": "coalesced"}) in metrics_calls
