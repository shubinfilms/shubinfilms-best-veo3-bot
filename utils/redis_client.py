"""Utility helpers for accessing Redis with a safe in-memory fallback."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from threading import Lock
from typing import Any, Optional

try:  # pragma: no cover - optional dependency in some environments
    import redis.asyncio as redis_asyncio
except Exception:  # pragma: no cover - gracefully degrade when redis is missing
    redis_asyncio = None  # type: ignore

try:  # pragma: no cover - optional settings module is always available in runtime
    import settings as app_settings
except Exception:  # pragma: no cover - tests may stub settings lazily
    app_settings = None  # type: ignore

log = logging.getLogger("utils.redis_client")


def _resolve_url() -> Optional[str]:
    env_value = os.getenv("REDIS_URL")
    if env_value:
        return env_value
    if app_settings is not None:
        try:
            url = getattr(app_settings, "REDIS_URL", None)
            if url:
                return str(url)
        except Exception:  # pragma: no cover - defensive guard for dynamic settings
            return None
    return None


def _resolve_prefix() -> str:
    if app_settings is not None:
        try:
            prefix = getattr(app_settings, "REDIS_PREFIX", "")
            if prefix:
                return str(prefix)
        except Exception:  # pragma: no cover - defensive
            return ""
    return ""


def _ttl_seconds(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric <= 0:
        return 0.0
    return numeric


def _ttl_from_kwargs(*, ex: Any = None, px: Any = None) -> Optional[float]:
    px_value = _ttl_seconds(px)
    if px_value is not None:
        return max(px_value / 1000.0, 0.0)
    ex_value = _ttl_seconds(ex)
    if ex_value is not None:
        return max(ex_value, 0.0)
    return None


def _now() -> float:
    return time.monotonic()


class InMemoryStore:
    """Simple async-friendly key/value store used when Redis is unavailable."""

    __slots__ = ("_data", "_lock")

    def __init__(self) -> None:
        self._data: dict[str, tuple[Optional[float], str]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[str]:
        async with self._lock:
            record = self._data.get(key)
            if record is None:
                return None
            expires_at, value = record
            if expires_at is not None and expires_at <= _now():
                self._data.pop(key, None)
                return None
            return value

    async def set(
        self,
        key: str,
        value: Any,
        *,
        ex: Any = None,
        px: Any = None,
        nx: bool = False,
    ) -> bool:
        ttl = _ttl_from_kwargs(ex=ex, px=px)
        expires_at = None if ttl is None else (_now() + max(ttl, 0.0))
        serialized = "" if value is None else str(value)
        async with self._lock:
            record = self._data.get(key)
            if record is not None:
                current_expiry, _ = record
                if current_expiry is not None and current_expiry <= _now():
                    record = None
                    self._data.pop(key, None)
            if nx and record is not None:
                return False
            self._data[key] = (expires_at, serialized)
        return True

    async def setex(self, key: str, ttl: Any, value: Any) -> bool:
        return await self.set(key, value, ex=ttl)

    async def delete(self, key: str) -> int:
        async with self._lock:
            existed = key in self._data
            self._data.pop(key, None)
        return 1 if existed else 0

    async def expire(self, key: str, ttl: Any) -> bool:
        ttl_seconds = _ttl_seconds(ttl)
        if ttl_seconds is None:
            return False
        async with self._lock:
            record = self._data.get(key)
            if record is None:
                return False
            expires_at = _now() + max(ttl_seconds, 0.0)
            self._data[key] = (expires_at, record[1])
            return True

    async def exists(self, key: str) -> bool:
        async with self._lock:
            record = self._data.get(key)
            if record is None:
                return False
            expires_at, _ = record
            if expires_at is not None and expires_at <= _now():
                self._data.pop(key, None)
                return False
            return True


@dataclass(slots=True)
class RedisHandle:
    client: Any
    memory: bool = False


_lock = Lock()
_cached: Optional[RedisHandle] = None
_fallback_logged = False


def _log_memory_fallback() -> None:
    global _fallback_logged
    if not _fallback_logged:
        _fallback_logged = True
        log.info("redis.fallback=memory", extra={"prefix": _resolve_prefix()})


def get_redis() -> Any:
    """Return the configured Redis client or a safe in-memory fallback."""

    global _cached
    if _cached is not None:
        return _cached.client

    with _lock:
        if _cached is not None:
            return _cached.client

        url = _resolve_url()
        if not url or str(url).strip().lower().startswith("memory://"):
            store = InMemoryStore()
            _cached = RedisHandle(client=store, memory=True)
            _log_memory_fallback()
            return store

        if redis_asyncio is None:
            store = InMemoryStore()
            _cached = RedisHandle(client=store, memory=True)
            _log_memory_fallback()
            return store

        try:
            client = redis_asyncio.from_url(
                url,
                encoding="utf-8",
                decode_responses=True,
                health_check_interval=30,
                socket_keepalive=True,
            )
        except Exception as exc:  # pragma: no cover - network/config issues
            log.warning("redis.connect_failed url=%s err=%s", url, exc)
            store = InMemoryStore()
            _cached = RedisHandle(client=store, memory=True)
            _log_memory_fallback()
            return store

        _cached = RedisHandle(client=client, memory=False)
        return client


__all__ = ["get_redis", "InMemoryStore"]

