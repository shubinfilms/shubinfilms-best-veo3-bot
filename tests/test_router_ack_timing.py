import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ui.buttons.router as router


def test_ack_precedes_dispatch(monkeypatch):
    order: list[str] = []

    async def fake_safe_answer(query, cache_time=0):
        order.append("ack")
        return "ok"

    async def fake_dispatch(action, update, ctx, *, raw_data, payload=None, legacy=False):
        order.append("dispatch")

    monkeypatch.setattr(router, "safe_answer", fake_safe_answer)
    monkeypatch.setattr(router, "dispatch_via_registry", fake_dispatch)

    query = SimpleNamespace(data="btn:profile", _ui_button_handled=False)
    update = SimpleNamespace(
        callback_query=query,
        effective_chat=SimpleNamespace(id=1),
        effective_user=SimpleNamespace(id=2),
    )
    ctx = SimpleNamespace()

    asyncio.run(router.on_callback(update, ctx))

    assert order[:2] == ["ack", "dispatch"]
