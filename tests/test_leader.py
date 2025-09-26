import importlib
import json
import logging
import time
from pathlib import Path

import pytest


class FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.ttl: dict[str, int] = {}
        self.history: list[dict[str, object]] = []
        self.last_stolen = False

    def set(self, key, value, nx=False, px=None, xx=False):  # type: ignore[override]
        if nx and key in self.store:
            return False
        if xx and key not in self.store:
            return False
        if px is not None:
            self.ttl[key] = int(px)
        self.store[key] = str(value)
        self.history.append({"key": key, "value": self.store[key], "nx": nx, "xx": xx, "px": px})
        return True

    def get(self, key):  # type: ignore[override]
        return self.store.get(key)

    def delete(self, key):  # type: ignore[override]
        self.store.pop(key, None)
        self.ttl.pop(key, None)
        return 1

    def eval(self, script, numkeys, *keys_and_args):  # type: ignore[override]
        key = keys_and_args[0]
        payload = keys_and_args[1]
        ttl_ms = int(keys_and_args[2])
        now_ms = int(keys_and_args[3])
        stale_ms = int(keys_and_args[4])
        current = self.store.get(key)
        if current is None:
            self.set(key, payload, px=ttl_ms)
            return 1
        try:
            data = json.loads(current)
            ts = int(data.get("ts", 0))
        except Exception:
            ts = 0
        if now_ms - ts > stale_ms:
            self.set(key, payload, px=ttl_ms)
            self.last_stolen = True
            return 2
        return 0


@pytest.fixture
def fake_redis(monkeypatch):
    fake = FakeRedis()
    import redis

    monkeypatch.setattr(redis.Redis, "from_url", classmethod(lambda cls, url, decode_responses=True: fake))
    return fake


def reload_bot(monkeypatch, **env):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    monkeypatch.setenv("LEDGER_BACKEND", "memory")
    monkeypatch.setenv("DATABASE_URL", "")
    monkeypatch.setenv("POSTGRES_DSN", "")
    monkeypatch.setenv("SUNO_API_BASE", "https://example.com")
    monkeypatch.setenv("SUNO_API_TOKEN", "token")
    monkeypatch.setenv("SUNO_CALLBACK_URL", "https://callback")
    monkeypatch.setenv("SUNO_CALLBACK_SECRET", "secret")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    module = importlib.import_module("bot")
    return importlib.reload(module)


def stop_leader(bot_module):
    ctx = getattr(bot_module, "_leader_context", None)
    if ctx is not None:
        ctx.stop()
        bot_module._leader_context = None


def test_leader_acquired_and_heartbeat(monkeypatch, fake_redis):
    bot = reload_bot(
        monkeypatch,
        REDIS_URL="redis://test",
        TELEGRAM_TOKEN="test-token",
        BOT_SINGLETON_DISABLED="false",
        BOT_LEADER_HEARTBEAT_INTERVAL_SEC="0.05",
        ENV_NAME="stage",
    )

    bot.acquire_singleton_lock()
    time.sleep(0.2)

    ctx = bot._leader_context
    assert ctx is not None

    key = bot._build_leader_key()
    assert key in fake_redis.store

    values = [json.loads(entry["value"]) for entry in fake_redis.history]
    assert len(values) >= 2
    assert all("ts" in item for item in values)
    assert values[-1]["ts"] >= values[0]["ts"]

    stop_leader(bot)


def test_stale_leader_is_stolen(monkeypatch, fake_redis, caplog):
    bot = reload_bot(
        monkeypatch,
        REDIS_URL="redis://test",
        TELEGRAM_TOKEN="test-token",
        BOT_SINGLETON_DISABLED="false",
        BOT_LEADER_HEARTBEAT_INTERVAL_SEC="0.05",
        ENV_NAME="prod",
    )

    key = bot._build_leader_key()
    fake_redis.store[key] = json.dumps({"owner": "old", "ts": int(time.time() * 1000) - 60_000})

    caplog.set_level(logging.INFO)
    bot.acquire_singleton_lock()
    time.sleep(0.1)

    assert fake_redis.last_stolen is True
    assert "leader: stale leader stolen" in caplog.text
    assert bot._leader_context is not None

    stop_leader(bot)


def test_singleton_disabled(monkeypatch, fake_redis, caplog):
    bot = reload_bot(
        monkeypatch,
        REDIS_URL="redis://test",
        TELEGRAM_TOKEN="test-token",
        BOT_SINGLETON_DISABLED="true",
        ENV_NAME="prod",
    )

    caplog.set_level(logging.WARNING)
    bot.acquire_singleton_lock()

    assert not fake_redis.history
    assert bot._leader_context is None
    assert "BOT_SINGLETON_DISABLED" in caplog.text
