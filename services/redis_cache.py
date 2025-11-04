"""Simple Redis-backed caching helpers for generation results."""

from __future__ import annotations

import json
from typing import Any, Mapping, MutableMapping, Optional

__all__ = ["configure", "get_cached_result", "set_cached_result"]

_redis_client: Optional[Any] = None


def configure(client: Any) -> None:
    """Configure a default Redis client for cache helpers."""

    global _redis_client
    _redis_client = client


def _coerce_client(candidate: Any = None) -> Optional[Any]:
    if candidate is not None:
        return candidate
    return _redis_client


async def get_cached_result(key: str, *, redis: Any = None) -> Optional[MutableMapping[str, Any]]:
    """Return a cached payload for ``key`` if available."""

    client = _coerce_client(redis)
    if client is None:
        return None
    try:
        raw = await client.get(key)
    except AttributeError:  # pragma: no cover - non-async client
        raw = None
    if not raw:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="ignore")
    if not isinstance(raw, str):
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if isinstance(payload, MutableMapping):
        return payload
    if isinstance(payload, Mapping):
        return dict(payload)
    return None


async def set_cached_result(
    key: str,
    value: Mapping[str, Any],
    ttl: int = 3600,
    *,
    redis: Any = None,
) -> None:
    """Store ``value`` under ``key`` for ``ttl`` seconds."""

    client = _coerce_client(redis)
    if client is None:
        return
    try:
        payload = json.dumps(value)
    except (TypeError, ValueError):  # pragma: no cover - serialization guard
        return
    expires = max(int(ttl), 1)
    try:
        await client.set(key, payload, ex=expires)
    except AttributeError:  # pragma: no cover - non-async client fallback
        try:
            client.set(key, payload, ex=expires)
        except Exception:
            return

