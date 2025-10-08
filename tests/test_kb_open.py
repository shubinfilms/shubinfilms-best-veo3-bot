from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.suno_test_utils import FakeBot  # noqa: E402
import handlers.knowledge_base as kb  # noqa: E402


def test_kb_send_then_edit(monkeypatch):
    ctx = SimpleNamespace(
        bot=FakeBot(),
        chat_data={},
        user_data={},
        application=SimpleNamespace(bot_data={}),
    )

    state_store: dict[str, object] = {}
    fallbacks: list[object] = []
    sent_ids = iter([200, 200])

    async def fake_send_menu(**kwargs):
        fallbacks.append(kwargs.get("fallback_message_id"))
        return next(sent_ids)

    kb.configure(send_menu=fake_send_menu, state_getter=lambda _ctx: state_store)

    first = asyncio.run(kb.open_root(ctx, 123))
    second = asyncio.run(kb.open_root(ctx, 123))

    assert first == 200
    assert second == 200
    assert fallbacks[0] is None
    assert fallbacks[1] == 200
    assert ctx.chat_data.get("kb_msg_id") == 200
