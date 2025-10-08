from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.suno_test_utils import FakeBot, bot_module  # noqa: E402


def test_nav_event_suppresses_notice():
    bot = FakeBot()
    ctx = SimpleNamespace(
        bot=bot,
        chat_data={"nav_event": True},
        user_data={},
        application=SimpleNamespace(bot_data={}),
    )

    asyncio.run(bot_module.reset_user_state(ctx, chat_id=123, notify_chat_off=True))

    assert ctx.chat_data.get("nav_event") is None
    assert not any(entry.get("text") == "🛑 Режим диалога отключён." for entry in bot.sent)
