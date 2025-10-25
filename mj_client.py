"""Utilities for interacting with Midjourney status endpoints."""
from __future__ import annotations

import logging
import time
from typing import Iterable, Sequence

from redis_utils import rds
from settings import REDIS_PREFIX

_log = logging.getLogger(__name__)

_STATUS_CACHE_KEY = f"{REDIS_PREFIX}:mj:status:path"
_STATUS_CACHE_TTL = 10 * 60
_memory_cache: tuple[str, float] | None = None


def _now() -> float:
    return time.monotonic()


def _normalize_path(path: str | None) -> str | None:
    if path is None:
        return None
    normalized = path.strip()
    return normalized or None


def _deduplicate(paths: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for path in paths:
        normalized = _normalize_path(path)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _load_cached_path() -> str | None:
    global _memory_cache
    now = _now()
    if _memory_cache is not None:
        cached_path, expires_at = _memory_cache
        if expires_at > now:
            return cached_path
        _memory_cache = None

    if rds is None:
        return None

    try:
        raw = rds.get(_STATUS_CACHE_KEY)
    except Exception:  # pragma: no cover - defensive guard
        _log.debug("mj.status.cache.redis_get_failed", exc_info=True)
        return None

    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="ignore")
    if isinstance(raw, str):
        normalized = _normalize_path(raw)
        if normalized:
            _memory_cache = (normalized, now + 60.0)
            return normalized
    return None


def build_status_candidates(base_paths: Sequence[str]) -> list[str]:
    """Return ordered MJ status endpoints honouring cached preference."""

    candidates = _deduplicate(base_paths)
    if not candidates:
        return []

    cached = _load_cached_path()
    if cached and cached in candidates:
        return [cached] + [p for p in candidates if p != cached]
    return candidates


def remember_status_path(path: str, *, ttl: int = _STATUS_CACHE_TTL) -> None:
    """Persist the working status endpoint for subsequent requests."""

    normalized = _normalize_path(path)
    if not normalized:
        return

    expires_at = _now() + max(1, ttl)
    global _memory_cache
    _memory_cache = (normalized, expires_at)

    if rds is None:
        return

    try:
        rds.setex(_STATUS_CACHE_KEY, max(1, int(ttl)), normalized)
    except Exception:  # pragma: no cover - defensive guard
        _log.debug("mj.status.cache.redis_set_failed", exc_info=True)


def clear_status_cache() -> None:
    """Drop cached MJ status endpoint preference."""

    global _memory_cache
    _memory_cache = None

    if rds is None:
        return
    try:
        rds.delete(_STATUS_CACHE_KEY)
    except Exception:  # pragma: no cover - defensive guard
        _log.debug("mj.status.cache.redis_del_failed", exc_info=True)


__all__ = [
    "build_status_candidates",
    "remember_status_path",
    "clear_status_cache",
]
