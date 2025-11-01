"""Per-user Banana card state backed by Redis."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import List, Optional


@dataclass(slots=True)
class BananaState:
    """Store Banana inputs awaiting generation."""

    images: List[str]
    prompt: Optional[str]
    last_result_id: Optional[str] = None

    @property
    def ready(self) -> bool:
        """Return ``True`` when both prompt and at least one image are present."""

        return bool(self.prompt) and len(self.images) >= 1


_KEY_TMPL = "banana:state:{user_id}"
_TTL_SECONDS = 60 * 60 * 24


def _key_for(user_id: int) -> str:
    return _KEY_TMPL.format(user_id=int(user_id))


async def load(redis, user_id: int) -> BananaState:
    """Load Banana state for ``user_id`` from Redis."""

    raw = await redis.get(_key_for(user_id))
    if not raw:
        return BananaState(images=[], prompt=None)
    data = json.loads(raw)
    return BananaState(**data)


async def save(redis, user_id: int, state: BananaState) -> None:
    """Persist ``state`` for ``user_id`` with a one-day TTL."""

    await redis.set(_key_for(user_id), json.dumps(asdict(state)), ex=_TTL_SECONDS)


async def clear(redis, user_id: int) -> None:
    """Remove Banana state for ``user_id`` from Redis."""

    await redis.delete(_key_for(user_id))


__all__ = ["BananaState", "load", "save", "clear"]
