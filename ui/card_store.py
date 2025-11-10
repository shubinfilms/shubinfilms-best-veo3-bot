"""Helpers for storing per-user UI card message identifiers."""

from __future__ import annotations

from typing import Optional

_CARD_KEY_TEMPLATE = "ui:card:{namespace}:{user_id}"
_CARD_TTL_SECONDS = 24 * 60 * 60


def _card_key(namespace: str, user_id: int) -> str:
    normalized_ns = (namespace or "").strip().lower()
    return _CARD_KEY_TEMPLATE.format(namespace=normalized_ns, user_id=int(user_id))


async def store_card_message_id(redis, namespace: str, user_id: int, message_id: int) -> None:
    await redis.set(
        _card_key(namespace, user_id),
        int(message_id),
        ex=_CARD_TTL_SECONDS,
    )


async def load_card_message_id(redis, namespace: str, user_id: int) -> Optional[int]:
    raw = await redis.get(_card_key(namespace, user_id))
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


async def clear_card_message_id(redis, namespace: str, user_id: int) -> None:
    await redis.delete(_card_key(namespace, user_id))


__all__ = [
    "store_card_message_id",
    "load_card_message_id",
    "clear_card_message_id",
]
