"""Per-user Banana card state backed by Redis."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class BananaState:
    """Store Banana inputs awaiting generation."""

    photos: List[str]
    prompt: Optional[str]
    last_job_id: Optional[str] = None
    last_payload: Optional[Dict[str, Any]] = None
    last_result_msg_id: Optional[int] = None
    last_result_id: Optional[str] = None  # legacy alias for stored message id

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

        return bool(self.photos) or bool(self._normalized_prompt())

    def can_start(self) -> bool:
        """Determine whether generation can be started."""

        return self.has_photos_or_prompt()

    def reset(self) -> None:
        """Clear inputs while keeping the instance reusable."""

        self.photos.clear()
        self.prompt = None
        self.last_job_id = None
        self.last_payload = None
        self.last_result_msg_id = None
        self.last_result_id = None

    # ---- Legacy aliases ----

    @property
    def images(self) -> List[str]:  # pragma: no cover - backwards compatibility
        return self.photos

    @images.setter
    def images(self, value: List[str]) -> None:  # pragma: no cover - backwards compatibility
        self.photos = value


_KEY_TMPL = "banana:state:{user_id}"
_TTL_SECONDS = 60 * 60 * 24


def _key_for(user_id: int) -> str:
    return _KEY_TMPL.format(user_id=int(user_id))


async def load(redis, user_id: int) -> BananaState:
    """Load Banana state for ``user_id`` from Redis."""

    raw = await redis.get(_key_for(user_id))
    if not raw:
        return BananaState(photos=[], prompt=None)
    data = json.loads(raw)
    if "photos" not in data:
        photos = data.get("images") or []
        data["photos"] = photos
    data.pop("images", None)
    data.setdefault("prompt", None)
    data.setdefault("last_job_id", None)
    data.setdefault("last_payload", None)
    data.setdefault("last_result_msg_id", None)
    data.setdefault("last_result_id", None)
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
