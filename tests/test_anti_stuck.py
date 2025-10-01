import asyncio
import fnmatch
import os
import sys
from types import SimpleNamespace

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("LEDGER_BACKEND", "memory")

from redis_utils import cleanup_stale_waits
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

    clear_calls: list[tuple[object, int, str]] = []

    async def fake_reset(redis_client, prefix: str, user_id: int):
        clear_calls.append((redis_client, prefix, user_id))
        return 1

    monkeypatch.setattr(telegram_utils, "redis_reset_user_state", fake_reset)

    handler = telegram_utils.with_state_reset(bot_module.on_menu)
    update = _make_update(user_id=111, chat_id=777, text="/menu")

    asyncio.run(handler(update, ctx))

    assert clear_calls == [(ctx.bot_data["redis"], "test-prefix", 111)]
    sent_menu = [payload for payload in ctx.bot.sent if isinstance(payload, dict)]
    assert sent_menu, "menu should be rendered after reset"


class _CleanupRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.ttl_map: dict[str, float] = {}
        self.deleted: list[str] = []

    def iscan(self, match: str):
        async def _generator():
            for key in list(self.store.keys()):
                if fnmatch.fnmatch(key, match):
                    yield key
        return _generator()

    def scan_iter(self, pattern: str):
        for key in list(self.store.keys()):
            if fnmatch.fnmatch(key, pattern):
                yield key

    async def delete(self, *keys: str) -> int:
        removed = 0
        for key in keys:
            if key in self.store:
                removed += 1
                self.store.pop(key, None)
                self.ttl_map.pop(key, None)
                self.deleted.append(key)
        return removed

    async def ttl(self, key: str) -> float:
        if key not in self.store:
            return -2
        return self.ttl_map.get(key, -1)


def test_cleanup_stale_waits_removes_stuck_entries():
    fake_redis = _CleanupRedis()
    fake_redis.store = {
        "pref:wait:100:prompt": "1",
        "pref:wait:session:100:prompt": "1",
        "pref:wait:200:prompt": "1",
    }
    fake_redis.ttl_map = {
        "pref:wait:100:prompt": -1,  # no expiry — must be removed
        "pref:wait:session:100:prompt": 5000,  # excessive ttl — must be removed
        "pref:wait:200:prompt": 300,  # healthy
    }

    deleted = asyncio.run(cleanup_stale_waits(fake_redis, "pref", max_ttl_seconds=900))

    assert deleted == 2
    assert fake_redis.deleted == [
        "pref:wait:100:prompt",
        "pref:wait:session:100:prompt",
    ]
    assert "pref:wait:200:prompt" in fake_redis.store
