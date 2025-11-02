"""State management helpers for the Banana image workflow."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

try:  # pragma: no cover - optional dependency
    from redis_utils import rds as _redis_client
except Exception:  # pragma: no cover - redis helpers may be unavailable in tests
    _redis_client = None  # type: ignore[assignment]

try:  # pragma: no cover - optional settings import
    from settings import REDIS_PREFIX as _REDIS_PREFIX
except Exception:  # pragma: no cover - default fallback when settings missing
    _REDIS_PREFIX = "bot"

_KEY_TEMPLATE = f"{_REDIS_PREFIX}:banana:state:{{user_id}}"
_TTL_SECONDS = 24 * 60 * 60
_MEMORY_FALLBACK: Dict[int, Dict[str, Any]] = {}
_MEMORY_LOCK = asyncio.Lock()


@dataclass
class BananaState:
    """Represents the current Banana card state for a user."""

    images: List[str]
    prompt: Optional[str]
    last_result_id: Optional[str] = None

    @property
    def ready(self) -> bool:
        return bool(self.prompt) and len(self.images) >= 1


def _resolve_client(candidate: Any) -> Any:
    if candidate is not None:
        return candidate
    return _redis_client


async def _redis_get(redis: Any, key: str) -> Optional[str]:
    client = _resolve_client(redis)
    if client is None:
        return None
    getter = getattr(client, "get", None)
    if getter is None:
        return None
    if asyncio.iscoroutinefunction(getter):  # pragma: no cover - aioredis path
        return await getter(key)
    return await asyncio.to_thread(getter, key)


async def _redis_set(redis: Any, key: str, value: str, *, ex: int) -> None:
    client = _resolve_client(redis)
    if client is None:
        return
    setter = getattr(client, "set", None)
    if setter is None:
        return
    if asyncio.iscoroutinefunction(setter):  # pragma: no cover - aioredis path
        await setter(key, value, ex=ex)
        return
    await asyncio.to_thread(setter, key, value, ex)


async def _redis_delete(redis: Any, key: str) -> None:
    client = _resolve_client(redis)
    if client is None:
        return
    deleter = getattr(client, "delete", None)
    if deleter is None:
        return
    if asyncio.iscoroutinefunction(deleter):  # pragma: no cover - aioredis path
        await deleter(key)
        return
    await asyncio.to_thread(deleter, key)


async def load(redis: Any, user_id: int) -> BananaState:
    """Load the Banana state for ``user_id`` from Redis or fallback memory."""

    key = _KEY_TEMPLATE.format(user_id=int(user_id))
    raw = await _redis_get(redis, key)
    if raw:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            images = [str(x) for x in data.get("images") or []]
            prompt = data.get("prompt")
            last_result_id = data.get("last_result_id")
            if prompt is not None:
                prompt = str(prompt)
            if last_result_id is not None:
                last_result_id = str(last_result_id)
            return BananaState(images=images, prompt=prompt, last_result_id=last_result_id)

    async with _MEMORY_LOCK:
        snapshot = _MEMORY_FALLBACK.get(int(user_id))
        if snapshot:
            return BananaState(**snapshot)
    return BananaState(images=[], prompt=None)


async def save(redis: Any, user_id: int, state: BananaState) -> None:
    """Persist ``state`` for ``user_id`` in Redis with a TTL."""

    payload = json.dumps(asdict(state))
    key = _KEY_TEMPLATE.format(user_id=int(user_id))
    await _redis_set(redis, key, payload, ex=_TTL_SECONDS)
    async with _MEMORY_LOCK:
        _MEMORY_FALLBACK[int(user_id)] = asdict(state)


async def clear(redis: Any, user_id: int) -> None:
    """Remove stored state for ``user_id`` from Redis and memory."""

    key = _KEY_TEMPLATE.format(user_id=int(user_id))
    await _redis_delete(redis, key)
    async with _MEMORY_LOCK:
        _MEMORY_FALLBACK.pop(int(user_id), None)


__all__ = ["BananaState", "load", "save", "clear"]

