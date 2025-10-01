import asyncio
import os
import sys
from types import SimpleNamespace

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("LEDGER_BACKEND", "memory")

from tests.suno_test_utils import FakeBot, bot_module  # noqa: E402
import telegram_utils  # noqa: E402


def _make_update(user_id: int, chat_id: int, text: str) -> SimpleNamespace:
    message = SimpleNamespace(text=text, caption=None, chat_id=chat_id)
    return SimpleNamespace(
        effective_user=SimpleNamespace(id=user_id),
        effective_chat=SimpleNamespace(id=chat_id),
        message=message,
        effective_message=message,
        callback_query=None,
    )


def test_menu_command_clears_wait_flags(monkeypatch):
    ctx = SimpleNamespace(
        bot=FakeBot(),
        bot_data={"redis": object(), "redis_prefix": "test-prefix"},
        user_data={},
        chat_data={},
        args=[],
    )

    clear_calls: list[tuple[object, int, str | None]] = []

    async def fake_clear(redis_client, user_id, *, prefix=None):
        clear_calls.append((redis_client, user_id, prefix))
        return 1

    monkeypatch.setattr(telegram_utils, "clear_all_waits", fake_clear)

    handler = telegram_utils.with_state_reset(bot_module.on_menu)
    update = _make_update(user_id=111, chat_id=777, text="/menu")

    asyncio.run(handler(update, ctx))

    assert clear_calls == [(ctx.bot_data["redis"], 111, "test-prefix")]
    sent_menu = [payload for payload in ctx.bot.sent if isinstance(payload, dict)]
    assert sent_menu, "menu should be rendered after reset"
