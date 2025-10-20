import fnmatch

import bot as bot_module


class FakeRedis:
    def __init__(self, ttl_map: dict[str, int]) -> None:
        self._ttl_map = dict(ttl_map)
        self.scan_history: list[tuple[int, str, int]] = []
        self.unlink_history: list[list[str]] = []

    def scan(self, cursor: int = 0, match: str | None = None, count: int = 10):
        pattern = match or "*"
        keys = sorted(key for key in self._ttl_map if fnmatch.fnmatch(key, pattern))
        start = max(0, int(cursor))
        end = min(start + int(count), len(keys))
        chunk = keys[start:end]
        next_cursor = 0 if end >= len(keys) else end
        self.scan_history.append((cursor, pattern, count))
        return next_cursor, chunk

    def ttl(self, key: str) -> int:
        return int(self._ttl_map.get(key, -2))

    def unlink(self, *keys: str) -> int:
        removed = 0
        for key in keys:
            if key in self._ttl_map:
                removed += 1
                del self._ttl_map[key]
        if keys:
            self.unlink_history.append(list(keys))
        return removed


def test_redis_cleanup_scan_removes_stale_keys() -> None:
    ttl_map = {
        "app:lock:sora2:1": -1,
        "app:lock:sora2:2": 30,
        "app:user:lock:5:act:profile": -1,
        "app:wait-input:42": -1,
        "app:other:keep": -1,
    }
    client = FakeRedis(ttl_map)
    stats = bot_module._redis_cleanup_scan(
        client,
        patterns=("app:lock:*", "app:user:lock:*", "app:wait-*"),
        scan_count=2,
        unlink_batch=1,
    )
    assert stats.deleted == 3
    assert stats.scanned >= 3
    assert stats.patterns["app:lock:*"] == 1
    assert stats.patterns["app:user:lock:*"] == 1
    assert stats.patterns["app:wait-*"] == 1
    assert "app:lock:sora2:2" in client._ttl_map
    assert "app:other:keep" in client._ttl_map
    # Each stale key was unlinked independently due to batch size = 1
    assert len(client.unlink_history) == 3


def test_redis_cleanup_scan_handles_empty_patterns() -> None:
    client = FakeRedis({"app:lock:one": -1})
    stats = bot_module._redis_cleanup_scan(client, patterns=("",), scan_count=10, unlink_batch=10)
    assert stats.deleted == 0
    assert client._ttl_map["app:lock:one"] == -1
