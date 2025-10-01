import asyncio
import fnmatch
from types import SimpleNamespace

from tests.suno_test_utils import FakeBot, bot_module


class FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.deleted: list[str] = []

    def scan_iter(self, pattern: str):
        for key in list(self.store.keys()):
            if fnmatch.fnmatch(key, pattern):
                yield key

    def delete(self, *keys: str) -> int:
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
        "test:fsm:99:state": "1",
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

    assert "test:wait:99:prompt" not in fake_redis.store
    assert "test:fsm:99:state" not in fake_redis.store
    assert "test:wait:42:other" in fake_redis.store
    assert set(fake_redis.deleted) == {"test:wait:99:prompt", "test:fsm:99:state"}
