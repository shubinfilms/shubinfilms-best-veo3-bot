import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import chat_service
from chat_service import append_ctx, clear_ctx, estimate_tokens, load_ctx


@pytest.fixture(autouse=True)
def reset_chat_state(monkeypatch):
    chat_service._memory["ctx"].clear()
    chat_service._memory["mode"].clear()
    chat_service._memory["rate"].clear()
    monkeypatch.setattr(chat_service, "_get_redis", lambda: None)
    yield


def test_estimate_tokens_various_lengths():
    assert estimate_tokens("") == 1
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcdefgh") == 2
    assert estimate_tokens("a" * 25) == 6


def test_load_ctx_limits_pairs_and_tokens():
    user_id = 101
    for idx in range(3):
        append_ctx(user_id, "user", f"u{idx}")
        append_ctx(user_id, "assistant", f"a{idx}")

    limited_tokens = load_ctx(user_id, max_tokens=2)
    assert [item["role"] for item in limited_tokens] == ["user", "assistant"]
    assert limited_tokens[-1]["content"] == "a2"

    limited_pairs = load_ctx(user_id, max_pairs=1, max_tokens=100)
    assert [item["role"] for item in limited_pairs] == ["user", "assistant"]
    assert limited_pairs[0]["content"] == "u2"


def test_append_and_clear_with_redis(monkeypatch):
    class FakeRedis:
        def __init__(self) -> None:
            self.store: dict[str, list[str]] = {}
            self.values: dict[str, str] = {}

        def rpush(self, key: str, value: str) -> None:
            self.store.setdefault(key, []).append(value)

        def expire(self, key: str, ttl: int) -> None:  # pragma: no cover - no-op
            return None

        def lrange(self, key: str, start: int, end: int) -> list[str]:
            items = self.store.get(key, [])
            if end == -1:
                end = len(items) - 1
            return items[start : end + 1]

        def delete(self, key: str) -> None:
            self.store.pop(key, None)

        def setex(self, key: str, ttl: int, value: str) -> None:
            self.values[key] = value

        def get(self, key: str) -> Any:
            return self.values.get(key)

    fake = FakeRedis()
    monkeypatch.setattr(chat_service, "_get_redis", lambda: fake)

    user_id = 7
    append_ctx(user_id, "user", "hello")
    append_ctx(user_id, "assistant", "world")

    history = load_ctx(user_id)
    assert len(history) == 2
    assert history[0]["content"] == "hello"

    clear_ctx(user_id)
    assert fake.store == {}
    assert load_ctx(user_id) == []
