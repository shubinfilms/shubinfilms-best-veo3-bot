import asyncio
import os
import sys
from typing import Optional
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

    reset_calls: list[tuple[object, str, int, Optional[int]]] = []

    fake_redis = object()

    def fake_get_redis() -> object:
        return fake_redis

    def fake_get_prefix() -> str:
        return "test-prefix"

    def fake_reset(redis_client, prefix, user_id, chat_id=None):
        reset_calls.append((redis_client, prefix, user_id, chat_id))
        return 1

    monkeypatch.setattr(telegram_utils, "get_redis", fake_get_redis)
    monkeypatch.setattr(telegram_utils, "get_prefix", fake_get_prefix)
    monkeypatch.setattr(telegram_utils, "reset_user_state", fake_reset)

    handler = telegram_utils.with_state_reset(bot_module.on_menu)
    update = _make_update(user_id=111, chat_id=777, text="/menu")

    asyncio.run(handler(update, ctx))

    assert reset_calls == [(fake_redis, "test-prefix", 111, 777)]
    sent_menu = [payload for payload in ctx.bot.sent if isinstance(payload, dict)]
    assert sent_menu, "menu should be rendered after reset"
