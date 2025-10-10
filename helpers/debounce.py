from __future__ import annotations

import time
from threading import RLock
from typing import Hashable

__all__ = ["debounce"]


_last_clicks: dict[tuple[Hashable, Hashable], float] = {}
_lock = RLock()


def debounce(user_id: Hashable, action: Hashable, *, delay: float = 1.0) -> bool:
    """Return ``True`` if the action should be processed.

    A simple in-memory debounce helper that filters out duplicate callback
    invocations that arrive within ``delay`` seconds for the same user and
    action key. ``user_id`` and ``action`` can be any hashable values; ``None``
    is treated as a shared bucket.
    """

    now = time.monotonic()
    key = (user_id, action)
    with _lock:
        last = _last_clicks.get(key)
        if last is not None and now - last < delay:
            return False
        _last_clicks[key] = now
    return True
