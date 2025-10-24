from __future__ import annotations

import logging
import os
import threading
import time
from typing import Callable, Optional

from redis_utils import acquire_action_lock, release_action_lock

from .results import UIResult, acknowledge
from .types import ButtonContext, ButtonHandler
from runtime_metrics import increment_ui_callback_counter


log = logging.getLogger(__name__)

_DEFAULT_TTL = max(int(os.getenv("UI_BUTTONS_LOCK_TTL", "1") or "1"), 1)
_DEBOUNCE_MS = float(os.getenv("UI_BUTTONS_DEBOUNCE_WINDOW_MS", "500") or "500")
_DEBOUNCE_MS = max(_DEBOUNCE_MS, 0.0)
_DEBOUNCE_WINDOW = _DEBOUNCE_MS / 1000.0 if _DEBOUNCE_MS else 0.0
_DEBOUNCE_LOCK = threading.Lock()
_RECENT_CLICKS: dict[tuple[int, str], float] = {}


def _now() -> float:
    return time.monotonic()


def should_process(user_id: Optional[int], callback_data: Optional[str]) -> bool:
    if user_id is None or not callback_data or _DEBOUNCE_WINDOW <= 0:
        return True

    normalized = callback_data.strip()
    if not normalized:
        return True

    key = (int(user_id), normalized)
    current = _now()
    with _DEBOUNCE_LOCK:
        expires = _RECENT_CLICKS.get(key, 0.0)
        if expires and expires > current:
            log.debug(
                "ui.click.coalesced",
                extra={
                    "user_id": int(user_id),
                    "callback_data": normalized,
                },
            )
            return False
        _RECENT_CLICKS[key] = current + _DEBOUNCE_WINDOW

        # Garbage collect stale entries lazily.
        cutoff = current - (_DEBOUNCE_WINDOW * 4)
        if cutoff > 0:
            stale = [item for item, exp in _RECENT_CLICKS.items() if exp <= cutoff]
            for item in stale:
                _RECENT_CLICKS.pop(item, None)
    return True


def with_idempotency(handler: ButtonHandler, *, ttl: int = _DEFAULT_TTL) -> ButtonHandler:
    """Wrap ``handler`` with a Redis-backed idempotency guard."""

    async def wrapped(context: ButtonContext) -> UIResult:
        owner = context.chat_id or context.user_id
        if owner is None:
            return await handler(context)

        lock_key = f"ui:btn:{context.spec.id}:{owner}"
        acquired = acquire_action_lock(owner, lock_key, ttl=ttl)
        if not acquired:
            increment_ui_callback_counter("dedup", "redis_blocked")
            return acknowledge(context.spec.id, message="duplicate_click")

        try:
            return await handler(context)
        finally:
            release_action_lock(owner, lock_key)

    return wrapped
