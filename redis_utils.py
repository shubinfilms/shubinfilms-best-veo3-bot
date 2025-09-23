import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

try:  # pragma: no cover - optional import for type checking only
    from typing import TYPE_CHECKING
except ImportError:  # pragma: no cover - Python < 3.8 safeguard
    TYPE_CHECKING = False

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from telegram import User as TelegramUser
else:  # pragma: no cover - fallback alias to avoid runtime dependency
    TelegramUser = Any

import redis

_logger = logging.getLogger("redis-utils")

_redis_url = os.getenv("REDIS_URL")
_r = redis.from_url(_redis_url) if _redis_url else None
rds = _r
_PFX = os.getenv("REDIS_PREFIX", "veo3")
_TTL = 24 * 60 * 60

_USERS_SET_KEY = f"{_PFX}:users"


def _user_profile_key(user_id: int) -> str:
    return f"{_PFX}:user:{user_id}"

_memory_store: Dict[str, Tuple[float, str]] = {}
_memory_lock = Lock()

if not _redis_url:
    _logger.warning(
        "REDIS_URL is not configured; falling back to in-memory task-meta store"
    )


def task_key(task_id: str) -> str:
    return f"{_PFX}:task:{task_id}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _memory_set(key: str, value: str) -> None:
    expires_at = time.time() + _TTL
    with _memory_lock:
        _memory_store[key] = (expires_at, value)


def _memory_get(key: str) -> Optional[str]:
    now = time.time()
    with _memory_lock:
        entry = _memory_store.get(key)
        if not entry:
            return None
        expires_at, value = entry
        if expires_at <= now:
            _memory_store.pop(key, None)
            return None
        return value


def _memory_delete(key: str) -> None:
    with _memory_lock:
        _memory_store.pop(key, None)


def save_task_meta(
    task_id: str,
    chat_id: int,
    message_id: int,
    mode: str,
    aspect: str,
) -> None:
    doc: Dict[str, Any] = {
        "chat_id": int(chat_id),
        "message_id": int(message_id),
        "mode": mode,
        "aspect": aspect,
        "created_at": _now_iso(),
    }
    payload = json.dumps(doc, ensure_ascii=False)
    key = task_key(task_id)
    if _r:
        _r.setex(key, _TTL, payload)
    else:
        _memory_set(key, payload)


def load_task_meta(task_id: str) -> Optional[Dict[str, Any]]:
    key = task_key(task_id)
    raw: Optional[str]
    if _r:
        redis_raw = _r.get(key)
        raw = None if redis_raw is None else redis_raw.decode("utf-8") if isinstance(redis_raw, bytes) else str(redis_raw)
    else:
        raw = _memory_get(key)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        _logger.exception("Failed to decode task-meta for %s", task_id)
        return None


def clear_task_meta(task_id: str) -> None:
    key = task_key(task_id)
    if _r:
        _r.delete(key)
    else:
        _memory_delete(key)


async def add_user(redis_conn: Optional["redis.Redis"], user: "TelegramUser") -> bool:
    if user is None:
        return False
    user_id = getattr(user, "id", None)
    if not user_id:
        return False

    if not redis_conn:
        _logger.warning("add_user: Redis client is not configured; skipping user %s", user_id)
        return False

    profile = {
        "id": str(int(user_id)),
        "username": getattr(user, "username", "") or "",
        "first_name": getattr(user, "first_name", "") or "",
        "last_name": getattr(user, "last_name", "") or "",
        "is_premium": "1" if getattr(user, "is_premium", False) else "0",
        "is_bot": "1" if getattr(user, "is_bot", False) else "0",
        "language_code": getattr(user, "language_code", "") or "",
        "last_seen_ts": _now_iso(),
    }

    def _store() -> bool:
        try:
            pipe = redis_conn.pipeline()
            pipe.sadd(_USERS_SET_KEY, int(user_id))
            pipe.hset(_user_profile_key(int(user_id)), mapping=profile)
            pipe.execute()
            return True
        except Exception as exc:  # pragma: no cover - network failure path
            _logger.warning("Failed to add user %s to Redis: %s", user_id, exc)
            return False

    return await asyncio.to_thread(_store)


async def get_users_count(redis_conn: Optional["redis.Redis"]) -> Optional[int]:
    if not redis_conn:
        _logger.warning("get_users_count: Redis client is not configured")
        return None

    def _call() -> Optional[int]:
        try:
            return int(redis_conn.scard(_USERS_SET_KEY))
        except Exception as exc:  # pragma: no cover - network failure path
            _logger.warning("Failed to fetch users count: %s", exc)
            return None

    return await asyncio.to_thread(_call)


async def get_all_user_ids(redis_conn: Optional["redis.Redis"]) -> List[int]:
    if not redis_conn:
        _logger.warning("get_all_user_ids: Redis client is not configured")
        return []

    def _call() -> List[int]:
        try:
            raw_ids = redis_conn.smembers(_USERS_SET_KEY)
        except Exception as exc:  # pragma: no cover - network failure path
            _logger.warning("Failed to load users set: %s", exc)
            return []
        result: List[int] = []
        for item in raw_ids:
            try:
                if isinstance(item, bytes):
                    item = item.decode("utf-8")
                result.append(int(item))
            except (TypeError, ValueError):
                continue
        return result

    return await asyncio.to_thread(_call)


async def remove_user(redis_conn: Optional["redis.Redis"], user_id: int) -> None:
    if not redis_conn:
        _logger.warning("remove_user: Redis client is not configured; user_id=%s", user_id)
        return

    def _call() -> None:
        try:
            pipe = redis_conn.pipeline()
            pipe.srem(_USERS_SET_KEY, int(user_id))
            pipe.delete(_user_profile_key(int(user_id)))
            pipe.execute()
        except Exception as exc:  # pragma: no cover - network failure path
            _logger.warning("Failed to remove user %s from Redis: %s", user_id, exc)

    await asyncio.to_thread(_call)


async def user_exists(redis_conn: Optional["redis.Redis"], user_id: int) -> bool:
    if not redis_conn:
        return False

    def _call() -> bool:
        try:
            return bool(redis_conn.exists(_user_profile_key(int(user_id))))
        except Exception as exc:  # pragma: no cover - network failure path
            _logger.warning("Failed to check user %s existence: %s", user_id, exc)
            return False

    return await asyncio.to_thread(_call)
