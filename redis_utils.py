import json
import logging
import os
import time
from typing import Any, Dict, Optional

import redis

_logger = logging.getLogger("redis-utils")

_redis_url = os.getenv("REDIS_URL")
_r = redis.from_url(_redis_url) if _redis_url else None
_PFX = os.getenv("REDIS_PREFIX", "veo3")
_TTL = int(os.getenv("TASK_TTL_SECONDS", "86400"))


def task_key(task_id: str) -> str:
    return f"{_PFX}:task:{task_id}"


def save_task_meta(task_id: str, chat_id: int, message_id: int, mode: str, user_id: int) -> None:
    if not _r:
        _logger.warning("Redis URL is not configured; skipping task-meta save for %s", task_id)
        return
    doc: Dict[str, Any] = {
        "chat_id": chat_id,
        "message_id": message_id,
        "mode": mode,
        "user_id": user_id,
        "created_at": int(time.time()),
    }
    _r.setex(task_key(task_id), _TTL, json.dumps(doc, ensure_ascii=False))


def load_task_meta(task_id: str) -> Optional[Dict[str, Any]]:
    if not _r:
        _logger.warning("Redis URL is not configured; cannot load task-meta for %s", task_id)
        return None
    raw = _r.get(task_key(task_id))
    return None if raw is None else json.loads(raw)


def clear_task_meta(task_id: str) -> None:
    if not _r:
        return
    _r.delete(task_key(task_id))
