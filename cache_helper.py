"""Async cache helpers backed by Redis."""

from __future__ import annotations

import json
from typing import Any, Awaitable, Callable, Mapping, MutableMapping, Optional

RedisLike = Any

__all__ = ["get_or_set_cache"]


async def get_or_set_cache(
    key: str,
    coro: Awaitable[Mapping[str, Any]] | Callable[[], Awaitable[Mapping[str, Any]]],
    *,
    redis: Optional[RedisLike] = None,
    ttl: int = 3600,
) -> MutableMapping[str, Any]:
    """Return cached payload for ``key`` or store the computed value."""

    if redis is None:
        if callable(coro):
            result = await coro()
        else:
            result = await coro
        return {**result, "_cached": False}

    try:
        raw = await redis.get(key)
    except AttributeError:
        raw = None
    if raw:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="ignore")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, Mapping):
            return {**payload, "_cached": True}

    if callable(coro):
        result = await coro()
    else:
        result = await coro

    try:
        serialized = json.dumps(result)
    except (TypeError, ValueError):
        serialized = None
    if serialized is not None:
        try:
            await redis.set(key, serialized, ex=max(int(ttl), 1))
        except AttributeError:
            try:
                redis.set(key, serialized, ex=max(int(ttl), 1))
            except Exception:
                pass

    return {**result, "_cached": False}
