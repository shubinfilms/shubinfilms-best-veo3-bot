import asyncio
import fnmatch
import os
import sys
from types import SimpleNamespace

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from redis_utils import reset_user_state as redis_reset_user_state
from tests.suno_test_utils import FakeBot, bot_module


class FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}
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
                self.deleted.append(key)
        return removed


def _make_update(user_id: int = 99, chat_id: int = 77) -> SimpleNamespace:
    message = SimpleNamespace(text="/menu", caption=None, chat_id=chat_id)
    return SimpleNamespace(
        effective_user=SimpleNamespace(id=user_id),
        effective_chat=SimpleNamespace(id=chat_id),
        message=message,
        effective_message=message,
        callback_query=None,
    )


def test_menu_command_clears_wait_keys():
    fake_redis = FakeRedis()
    fake_redis.store = {
        "test:wait:99:prompt": "1",
        "test:wait:chat:99:prompt": "1",
        "test:fsm:99:state": "1",
        "test:signup_bonus:99": "1",
        "test:wait:42:other": "1",
    }

    ctx = SimpleNamespace(
        bot=FakeBot(),
        user_data={},
        args=[],
        bot_data={"redis": fake_redis, "redis_prefix": "test"},
    )

    update = _make_update()

    asyncio.run(bot_module.on_menu(update, ctx))

    assert "test:wait:42:other" in fake_redis.store
    assert set(fake_redis.deleted) == {
        "test:wait:99:prompt",
        "test:wait:chat:99:prompt",
        "test:fsm:99:state",
        "test:signup_bonus:99",
    }


def test_reset_user_state_returns_deleted_count():
    fake_redis = FakeRedis()
    fake_redis.store = {
        "acme:wait:100:prompt": "1",
        "acme:wait:input:100:text": "1",
        "acme:fsm:100:state": "1",
        "acme:signup_bonus:100": "1",
        "acme:wait:other": "1",
    }

    deleted = asyncio.run(redis_reset_user_state(fake_redis, "acme", 100))

    assert deleted == 4
    assert "acme:wait:other" in fake_redis.store
    assert "acme:wait:100:prompt" not in fake_redis.store
    assert "acme:signup_bonus:100" not in fake_redis.store
