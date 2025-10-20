from __future__ import annotations

from typing import Callable

from redis_utils import acquire_action_lock, release_action_lock

from .results import UIResult, acknowledge
from .types import ButtonContext, ButtonHandler


_DEFAULT_TTL = 5


def with_idempotency(handler: ButtonHandler, *, ttl: int = _DEFAULT_TTL) -> ButtonHandler:
    """Wrap ``handler`` with a Redis-backed idempotency guard."""

    async def wrapped(context: ButtonContext) -> UIResult:
        owner = context.chat_id or context.user_id
        if owner is None:
            return await handler(context)

        lock_key = f"ui:btn:{context.spec.id}:{owner}"
        acquired = acquire_action_lock(owner, lock_key, ttl=ttl)
        if not acquired:
            return acknowledge(context.spec.id, message="duplicate_click")

        try:
            return await handler(context)
        finally:
            release_action_lock(owner, lock_key)

    return wrapped
