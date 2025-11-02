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

    def _normalized_prompt(self) -> Optional[str]:
        if self.prompt is None:
            return None
        stripped = self.prompt.strip()
        return stripped or None

    @property
    def ready(self) -> bool:
        """Return ``True`` when at least one input is present."""

        return self.can_start()

    def has_photos_or_prompt(self) -> bool:
        """Return ``True`` when the card has photos or text prompt."""

        return bool(self.images) or bool(self._normalized_prompt())

    def can_start(self) -> bool:
        """Determine whether generation can be started."""

        return self.has_photos_or_prompt()

    def reset(self) -> None:
        """Clear inputs while keeping the instance reusable."""

        self.images.clear()
        self.prompt = None
        self.last_result_id = None


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


async def ensure(redis, user_id: int) -> BananaState:
    """Ensure there is a Banana state stored for ``user_id``."""

    state = await load(redis, user_id)
    if not state.has_photos_or_prompt() and state.last_result_id is None:
        await save(redis, user_id, state)
    return state


async def get(redis, user_id: int) -> BananaState:
    """Alias of :func:`load` for compatibility with higher-level handlers."""

    return await load(redis, user_id)


__all__ = ["BananaState", "load", "save", "clear", "ensure", "get"]
