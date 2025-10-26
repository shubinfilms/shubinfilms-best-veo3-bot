from __future__ import annotations

import random
import string
import threading
import time
from dataclasses import dataclass
from typing import Dict

from settings import REDIS_PREFIX
from redis_utils import rds as redis_client

_TASK_KEY_TMPL = f"{REDIS_PREFIX}:sum:task:{{}}"
_MEMORY_TASKS: Dict[str, dict] = {}
_MEMORY_LOCK = threading.Lock()


@dataclass(slots=True)
class SummaryStatus:
    success: bool
    result: str | None = None
    error: str | None = None


def _random_task_id() -> str:
    return "tsk_" + "".join(random.choices(string.ascii_lowercase + string.digits, k=12))


def _store_task(task_id: str, payload: dict) -> None:
    client = redis_client
    if client is not None:
        try:
            client.setex(_TASK_KEY_TMPL.format(task_id), 900, str(payload))
            return
        except Exception:
            pass
    with _MEMORY_LOCK:
        _MEMORY_TASKS[task_id] = {"payload": payload, "expires_at": time.time() + 900}


def _load_task(task_id: str) -> dict | None:
    client = redis_client
    if client is not None:
        try:
            raw = client.get(_TASK_KEY_TMPL.format(task_id))
        except Exception:
            raw = None
        if raw:
            return {"payload": {"result": str(raw)}}
    with _MEMORY_LOCK:
        entry = _MEMORY_TASKS.get(task_id)
        if not entry:
            return None
        expires_at = entry.get("expires_at", 0)
        if expires_at and expires_at < time.time():
            _MEMORY_TASKS.pop(task_id, None)
            return None
        return dict(entry)


def validate_summary_request(request: dict) -> None:
    if not isinstance(request, dict):
        raise ValueError("request must be a dict")
    text = str(request.get("input") or "").strip()
    if len(text) < 5:
        raise ValueError("input must contain at least 5 characters")
    if len(text) > 10000:
        raise ValueError("input is too long")


def _summarize_text(text: str) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= 400:
        return normalized
    return normalized[:397].rstrip() + "…"


def create(request: dict) -> str:
    validate_summary_request(request)
    task_id = _random_task_id()
    result = _summarize_text(str(request.get("input") or ""))
    _store_task(task_id, {"result": result, "status": "ready"})
    return task_id


def status(task_id: str) -> SummaryStatus:
    if not task_id:
        return SummaryStatus(success=False, error="taskId is empty")
    entry = _load_task(task_id)
    if not entry:
        return SummaryStatus(success=False, error="task not found")
    payload = entry.get("payload") if isinstance(entry, dict) else None
    if not isinstance(payload, dict):
        return SummaryStatus(success=False, error="invalid payload")
    result = str(payload.get("result") or "").strip()
    if not result:
        return SummaryStatus(success=False, error="no result yet")
    return SummaryStatus(success=True, result=result)
