import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bot as bot_module
import handlers.profile as profile_handlers
from tests.test_profile_views import DummyBot, _make_ctx


def test_profile_button_always_rerenders(monkeypatch):
    ctx = _make_ctx(DummyBot())
    ctx.chat_data[profile_handlers.PROFILE_MSG_ID] = 31345

    force_clear_calls: list[tuple[int, str]] = []

    async def fake_force_clear(user_id, *, reason="profile_open"):
        force_clear_calls.append((int(user_id), reason))

    monkeypatch.setattr(profile_handlers, "force_clear_user_state", fake_force_clear)

    open_calls: list[dict[str, object]] = []

    async def fake_open_profile_card(
        update,
        ctx,
        *,
        edit: bool = True,
        suppress_nav: bool = True,
        force_new: bool = False,
    ) -> int:
        open_calls.append(
            {"edit": edit, "force_new": force_new, "suppress_nav": suppress_nav}
        )
        return 777

    monkeypatch.setattr(bot_module, "open_profile_card", fake_open_profile_card)

    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=100),
        effective_message=SimpleNamespace(chat_id=100),
        effective_user=SimpleNamespace(id=555),
    )

    async def scenario() -> None:
        await profile_handlers.open(update, ctx, payload={"source": "button"})

    asyncio.run(scenario())

    assert force_clear_calls == [(555, "profile_open")]
    assert open_calls == [{"edit": False, "force_new": True, "suppress_nav": False}]
    assert ctx.chat_data.get(profile_handlers.PROFILE_MSG_ID) == 777
