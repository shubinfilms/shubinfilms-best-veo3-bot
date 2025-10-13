"""Asynchronous Redis-backed session and card state management.

This module provides a thin wrapper around ``redis.asyncio`` to store
per-chat session dictionaries together with lightweight locking helpers.
It is intentionally dependency-free and degrades gracefully to an in-memory
fallback when Redis is not configured (for test environments).

Usage::

    from state import state

    async with state.lock(chat_id):
        session = await state.load(chat_id)
        session["mode"] = "banana"
        await state.save(chat_id, session)

Locks are implemented with ``SET NX PX`` and therefore automatically expire
after the configured timeout.  Long-running operations should be performed
outside of the critical section to minimise contention.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Optional

try:  # pragma: no cover - optional dependency in some environments
    import redis.asyncio as redis_asyncio
except Exception:  # pragma: no cover - redis is optional for tests
    redis_asyncio = None

from settings import REDIS_PREFIX, REDIS_URL

_REDIS_URL = REDIS_URL
_MEMORY_ONLY = bool(_REDIS_URL and _REDIS_URL.startswith("memory://"))
_STATE_KEY_TMPL = f"{REDIS_PREFIX}:state:{{chat_id}}"
_CARD_KEY_TMPL = f"{REDIS_PREFIX}:card:{{chat_id}}:{{module}}"
_LOCK_KEY_TMPL = f"{REDIS_PREFIX}:lock:state:{{chat_id}}"


class StateLockTimeout(RuntimeError):
    """Raised when a state lock cannot be acquired within the timeout."""


@dataclass(slots=True)
class CardInfo:
    message_id: Optional[int] = None
    updated_at: float = 0.0


def _default_session() -> dict[str, Any]:
    """Return the default empty session payload."""

    return {
        "mode": None,
        "last_panel": None,
        "msg_ids": {},
    }


class _MemoryBackend:
    """Simple in-memory backend used when Redis is unavailable."""

    def __init__(self) -> None:
        self._state: dict[int, dict[str, Any]] = {}
        self._cards: dict[tuple[int, str], CardInfo] = {}
        self._locks: dict[int, asyncio.Lock] = {}

    @asynccontextmanager
    async def lock(self, chat_id: int) -> AsyncIterator[None]:
        lock = self._locks.setdefault(chat_id, asyncio.Lock())
        await lock.acquire()
        try:
            yield
        finally:
            lock.release()

    async def load(self, chat_id: int) -> dict[str, Any]:
        return json.loads(json.dumps(self._state.get(chat_id) or _default_session()))

    async def save(self, chat_id: int, payload: dict[str, Any]) -> None:
        self._state[chat_id] = json.loads(json.dumps(payload))

    async def get_card(self, chat_id: int, module: str) -> CardInfo:
        return self._cards.get((chat_id, module), CardInfo())

    async def set_card(self, chat_id: int, module: str, card: CardInfo) -> None:
        self._cards[(chat_id, module)] = card

    async def clear_card(self, chat_id: int, module: str) -> None:
        self._cards.pop((chat_id, module), None)


class StateManager:
    """Redis-based session manager with cooperative locking."""

    def __init__(self) -> None:
        if _REDIS_URL and not _MEMORY_ONLY and redis_asyncio is not None:
            self._redis = redis_asyncio.from_url(
                _REDIS_URL,
                decode_responses=True,
                encoding="utf-8",
                health_check_interval=30,
                socket_keepalive=True,
            )
        else:  # pragma: no cover - exercised implicitly in tests
            self._redis = None
        self._memory = _MemoryBackend()

    # ------------------------------------------------------------------
    # Lock helpers
    # ------------------------------------------------------------------
    @asynccontextmanager
    async def lock(self, chat_id: int, *, ttl_ms: int = 1000, timeout: float = 1.0) -> AsyncIterator[None]:
        """Acquire a short-lived Redis lock for the given ``chat_id``.

        The lock uses ``SET NX PX`` semantics.  If Redis is unavailable the
        in-memory fallback lock is used instead.  ``timeout`` controls how
        long the coroutine waits before raising :class:`StateLockTimeout`.
        """

        if self._redis is None:
            async with self._memory.lock(chat_id):
                yield
                return

        lock_key = _LOCK_KEY_TMPL.format(chat_id=int(chat_id))
        token = uuid.uuid4().hex
        start = time.monotonic()
        acquired = False
        while not acquired:
            acquired = await self._redis.set(lock_key, token, nx=True, px=max(1, ttl_ms))
            if acquired:
                break
            if (time.monotonic() - start) >= timeout:
                raise StateLockTimeout(f"lock busy for chat {chat_id}")
            await asyncio.sleep(0.05)

        try:
            yield
        finally:
            lua = """
            if redis.call('get', KEYS[1]) == ARGV[1] then
                return redis.call('del', KEYS[1])
            end
            return 0
            """
            try:
                await self._redis.eval(lua, 1, lock_key, token)
            except Exception:  # pragma: no cover - network issues
                pass

    # ------------------------------------------------------------------
    # Session helpers
    # ------------------------------------------------------------------
    async def load(self, chat_id: int) -> dict[str, Any]:
        """Load the session payload for ``chat_id``."""

        if self._redis is None:
            return await self._memory.load(chat_id)

        raw = await self._redis.get(_STATE_KEY_TMPL.format(chat_id=int(chat_id)))
        if not raw:
            return _default_session()
        try:
            payload = json.loads(raw)
        except Exception:  # pragma: no cover - corrupt data
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        for key, value in _default_session().items():
            payload.setdefault(key, json.loads(json.dumps(value)))
        return payload

    async def save(self, chat_id: int, payload: dict[str, Any]) -> None:
        """Persist the payload for ``chat_id``."""

        if self._redis is None:
            await self._memory.save(chat_id, payload)
            return
        await self._redis.set(
            _STATE_KEY_TMPL.format(chat_id=int(chat_id)),
            json.dumps(payload, ensure_ascii=False),
        )

    # ------------------------------------------------------------------
    # Card helpers
    # ------------------------------------------------------------------
    async def get_card(self, chat_id: int, module: str) -> CardInfo:
        module_key = module.strip().lower()
        if not module_key:
            return CardInfo()

        if self._redis is None:
            return await self._memory.get_card(chat_id, module_key)

        raw = await self._redis.get(
            _CARD_KEY_TMPL.format(chat_id=int(chat_id), module=module_key)
        )
        if not raw:
            return CardInfo()
        try:
            payload = json.loads(raw)
        except Exception:  # pragma: no cover - corrupt data
            return CardInfo()
        message_id = payload.get("message_id")
        updated_at = float(payload.get("updated_at", 0))
        if isinstance(message_id, bool):  # guard against Redis bool encoding
            message_id = None
        if isinstance(message_id, str) and message_id.isdigit():
            message_id = int(message_id)
        if not isinstance(message_id, int):
            message_id = None
        return CardInfo(message_id=message_id, updated_at=updated_at)

    async def set_card(self, chat_id: int, module: str, message_id: int) -> CardInfo:
        info = CardInfo(message_id=message_id, updated_at=time.time())
        module_key = module.strip().lower()
        if self._redis is None:
            await self._memory.set_card(chat_id, module_key, info)
            return info

        await self._redis.set(
            _CARD_KEY_TMPL.format(chat_id=int(chat_id), module=module_key),
            json.dumps({"message_id": message_id, "updated_at": info.updated_at}),
        )
        return info

    async def clear_card(self, chat_id: int, module: str) -> None:
        module_key = module.strip().lower()
        if self._redis is None:
            await self._memory.clear_card(chat_id, module_key)
            return
        await self._redis.delete(
            _CARD_KEY_TMPL.format(chat_id=int(chat_id), module=module_key)
        )


# Public singleton used across the codebase
state = StateManager()


__all__ = [
    "CardInfo",
    "StateLockTimeout",
    "StateManager",
    "state",
]
