import asyncio
import importlib
import os
import time

import pytest
from telegram.error import RetryAfter

os.environ.setdefault("SUNO_API_BASE", "https://example.com")
os.environ.setdefault("SUNO_API_TOKEN", "token")
os.environ.setdefault("SUNO_CALLBACK_URL", "https://callback")
os.environ.setdefault("SUNO_CALLBACK_SECRET", "secret")

from telegram_utils import safe_send

os.environ.setdefault("TELEGRAM_TOKEN", "dummy")


def test_safe_send_retries(monkeypatch):
    calls = {"count": 0}

    class Dummy:
        async def method(self, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise RetryAfter(1)
            return "ok"

    dummy = Dummy()

    async def _instant_sleep(delay):
        return None

    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)
    monkeypatch.setattr("telegram_utils.asyncio.sleep", _instant_sleep)
    result = asyncio.run(
        safe_send(
            dummy.method,
            method_name="sendMessage",
            kind="message",
            req_id="req-test",
            chat_id=123,
            text="hi",
        )
    )
    assert result == "ok"
    assert calls["count"] == 2


def _import_bot(monkeypatch):
    module = importlib.import_module("bot")
    module._SUNO_REFUND_MEMORY.clear()
    module._SUNO_COOLDOWN_MEMORY.clear()
    return module


def test_refund_idempotent(monkeypatch):
    bot = _import_bot(monkeypatch)

    class FakeRedis:
        def __init__(self):
            self.store = set()

        def set(self, key, value, nx=False, ex=None):
            if nx and key in self.store:
                return False
            self.store.add(key)
            return True

    fake = FakeRedis()
    monkeypatch.setattr(bot, "rds", fake)
    assert bot._suno_acquire_refund("task-one") is True
    assert bot._suno_acquire_refund("task-one") is False


def test_cooldown_blocks(monkeypatch):
    bot = _import_bot(monkeypatch)
    bot.SUNO_PER_USER_COOLDOWN_SEC = 5

    class FakeRedis:
        def __init__(self):
            self.ttl_values = {}

        def setex(self, key, ttl, value):
            self.ttl_values[key] = ttl

        def ttl(self, key):
            return self.ttl_values.get(key, -2)

    fake = FakeRedis()
    monkeypatch.setattr(bot, "rds", fake)
    bot._suno_set_cooldown(42)
    remaining = bot._suno_cooldown_remaining(42)
    assert remaining == 5

    monkeypatch.setattr(bot, "rds", None)
    bot._SUNO_COOLDOWN_MEMORY[84] = time.time() + 3
    remaining_local = bot._suno_cooldown_remaining(84)
    assert remaining_local >= 2
