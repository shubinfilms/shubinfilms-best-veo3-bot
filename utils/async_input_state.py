"""Asynchronous per-user input state helpers backed by Redis."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from settings import REDIS_PREFIX
from utils.redis_client import get_redis

_LOG = logging.getLogger("async-input-state")

_KEY_TMPL = f"{REDIS_PREFIX}:input-state:{{user_id}}"
_TTL_SECONDS = 60 * 60  # 1 hour


def _key_for(user_id: int) -> str:
    return _KEY_TMPL.format(user_id=int(user_id))


class _AsyncInputState:
    """Store lightweight per-user input flags in Redis."""

    async def get(self, user_id: int) -> Dict[str, Any]:
        key = _key_for(user_id)
        client = get_redis()
        try:
            raw = await client.get(key)
        except Exception:  # pragma: no cover - best effort
            _LOG.debug("input_state.get_failed", exc_info=True, extra={"user_id": int(user_id)})
            return {}
        if not raw:
            return {}
        try:
            data = json.loads(raw)
        except Exception:  # pragma: no cover - defensive
            return {}
        if not isinstance(data, dict):
            return {}
        return data

    async def set(self, user_id: int, **fields: Any) -> None:
        if not fields:
            return
        current = await self.get(user_id)
        for key, value in fields.items():
            if value is None:
                current.pop(key, None)
            else:
                current[key] = value
        payload = json.dumps(current, ensure_ascii=False)
        client = get_redis()
        try:
            await client.set(_key_for(user_id), payload, ex=_TTL_SECONDS)
        except Exception:  # pragma: no cover - best effort persistence
            _LOG.debug("input_state.set_failed", exc_info=True, extra={"user_id": int(user_id)})

    async def clear(self, user_id: int, *, reason: str = "manual") -> None:
        client = get_redis()
        try:
            await client.delete(_key_for(user_id))
        except Exception:  # pragma: no cover - best effort
            _LOG.debug(
                "input_state.clear_failed", exc_info=True, extra={"user_id": int(user_id), "reason": reason}
            )

    async def is_mode(self, user_id: int, mode: str) -> bool:
        data = await self.get(user_id)
        value = data.get("mode")
        return isinstance(value, str) and value == mode

    async def pop(self, user_id: int, key: str) -> Optional[Any]:
        data = await self.get(user_id)
        if key not in data:
            return None
        value = data.pop(key)
        payload = json.dumps(data, ensure_ascii=False)
        client = get_redis()
        try:
            await client.set(_key_for(user_id), payload, ex=_TTL_SECONDS)
        except Exception:  # pragma: no cover - best effort persistence
            _LOG.debug("input_state.pop_failed", exc_info=True, extra={"user_id": int(user_id), "key": key})
        return value


input_state = _AsyncInputState()

__all__ = ["input_state"]
