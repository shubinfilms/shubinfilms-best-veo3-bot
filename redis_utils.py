import json
import logging
import os
import time
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, Optional, Tuple

import redis

_logger = logging.getLogger("redis-utils")

_redis_url = os.getenv("REDIS_URL")
_r = redis.from_url(_redis_url) if _redis_url else None
_PFX = os.getenv("REDIS_PREFIX", "veo3")
_TTL = 24 * 60 * 60

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
