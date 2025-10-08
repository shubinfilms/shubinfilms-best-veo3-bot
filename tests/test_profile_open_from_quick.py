from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.suno_test_utils import FakeBot, bot_module  # noqa: E402
import handlers.profile as profile_handlers  # noqa: E402


def _build_update(chat_id: int, user_id: int):
    message = SimpleNamespace(
        text="Профиль",
        chat_id=chat_id,
        chat=SimpleNamespace(id=chat_id),
        replies=[],
    )

    async def reply_text(text, **_kwargs):
        message.replies.append(text)

    message.reply_text = reply_text

    update = SimpleNamespace(
        message=message,
        effective_message=message,
        effective_chat=message.chat,
        effective_user=SimpleNamespace(id=user_id),
    )
    return update


def test_quick_button_reuses_message(monkeypatch):
    bot = FakeBot()
    ctx = SimpleNamespace(
        bot=bot,
        chat_data={},
        user_data={},
        application=SimpleNamespace(bot_data={}),
    )
    ctx._user_id_and_data = (777, {})

    async def fake_disable(*_args, **_kwargs):
        return None

    async def fake_ensure(_update):
        return None

    calls: list[dict[str, object]] = []

    async def fake_core_open(update, context, *, suppress_nav, edit, force_new):
        calls.append({"edit": edit, "force_new": force_new, "suppress": suppress_nav})
        context.chat_data["profile_msg_id"] = 555
        return 555

    monkeypatch.setattr(bot_module, "disable_chat_mode", fake_disable)
    monkeypatch.setattr(bot_module, "ensure_user_record", fake_ensure)
    monkeypatch.setattr(bot_module, "open_profile_card", fake_core_open)
    time_values = [100.0, 101.0, 102.0, 103.0]
    times = iter(time_values)

    def fake_monotonic():
        try:
            return next(times)
        except StopIteration:
            return time_values[-1]

    monkeypatch.setattr(profile_handlers.time, "monotonic", fake_monotonic)

    update = _build_update(chat_id=100, user_id=777)

    asyncio.run(bot_module.on_text(update, ctx))
    asyncio.run(bot_module.on_text(update, ctx))

    assert len(calls) == 2
    assert calls[0]["edit"] is False
    assert calls[1]["edit"] is True
    assert ctx.chat_data.get("profile_msg_id") == 555
    assert not any(entry.get("text") == "🛑 Режим диалога отключён." for entry in bot.sent)
