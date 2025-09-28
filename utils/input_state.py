from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional

from settings import REDIS_PREFIX

try:  # pragma: no cover - optional redis backend
    from redis_utils import rds as _redis
except Exception:  # pragma: no cover - fall back to None if redis unavailable
    _redis = None

_logger = logging.getLogger("input-state")

_KEY_TMPL = f"{REDIS_PREFIX}:wait-input:{{user_id}}"


class WaitKind(str, Enum):
    VEO_PROMPT = "veo_prompt"
    BANANA_PROMPT = "banana_prompt"
    SUNO_TITLE = "suno_title"
    SUNO_STYLE = "suno_style"
    SUNO_LYRICS = "suno_lyrics"
    MJ_PROMPT = "mj_prompt"


@dataclass
class WaitInputState:
    kind: WaitKind
    card_msg_id: int
    chat_id: int
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        meta_payload: Dict[str, Any]
        if isinstance(self.meta, dict):
            try:
                meta_payload = {str(k): v for k, v in self.meta.items()}
            except Exception:
                meta_payload = {str(k): str(v) for k, v in self.meta.items()}
        else:
            meta_payload = {}
        return {
            "kind": self.kind.value,
            "card_msg_id": int(self.card_msg_id),
            "chat_id": int(self.chat_id),
            "meta": meta_payload,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> Optional["WaitInputState"]:
        try:
            raw_kind = data.get("kind")  # type: ignore[attr-defined]
            if not isinstance(raw_kind, str):
                return None
            kind = WaitKind(raw_kind)
            raw_msg_id = data.get("card_msg_id")
            raw_chat = data.get("chat_id")
            if raw_msg_id is None or raw_chat is None:
                return None
            card_msg_id = int(raw_msg_id)
            chat_id = int(raw_chat)
            meta_raw = data.get("meta")
            meta: Dict[str, Any]
            if isinstance(meta_raw, Mapping):
                meta = dict(meta_raw)
            else:
                meta = {}
        except (ValueError, KeyError):
            return None
        return cls(kind=kind, card_msg_id=card_msg_id, chat_id=chat_id, meta=meta)


_memory_store: Dict[int, Dict[str, Any]] = {}


def _key(user_id: int) -> str:
    return _KEY_TMPL.format(user_id=int(user_id))


def set_wait_state(user_id: int, state: WaitInputState) -> None:
    payload = json.dumps(state.to_dict(), ensure_ascii=False)
    storage_key = _key(user_id)
    if _redis:
        try:
            _redis.set(storage_key, payload, ex=24 * 60 * 60)
        except Exception:  # pragma: no cover - redis connectivity issues
            _logger.exception("Failed to save wait-state to redis", extra={"user_id": user_id})
            _memory_store[int(user_id)] = state.to_dict()
    else:
        _memory_store[int(user_id)] = state.to_dict()
    _logger.info(
        "WAIT_SET kind=%s user_id=%s card=%s",
        state.kind.value,
        user_id,
        state.card_msg_id,
    )


def get_wait_state(user_id: int) -> Optional[WaitInputState]:
    storage_key = _key(user_id)
    doc: Optional[Dict[str, Any]] = None
    if _redis:
        try:
            raw = _redis.get(storage_key)
        except Exception:  # pragma: no cover - redis connectivity issues
            _logger.exception("Failed to load wait-state from redis", extra={"user_id": user_id})
            raw = None
        if raw:
            try:
                text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
                doc = json.loads(text)
            except Exception:  # pragma: no cover - invalid payload
                _logger.exception("Failed to parse wait-state payload", extra={"user_id": user_id})
                doc = None
    if doc is None:
        doc = _memory_store.get(int(user_id))
    if not isinstance(doc, Mapping):
        return None
    state = WaitInputState.from_mapping(doc)
    return state


def clear_wait_state(user_id: int) -> None:
    storage_key = _key(user_id)
    if _redis:
        try:
            _redis.delete(storage_key)
        except Exception:  # pragma: no cover - redis connectivity issues
            _logger.exception("Failed to clear wait-state in redis", extra={"user_id": user_id})
    _memory_store.pop(int(user_id), None)
    _logger.info("WAIT_CLEAR user_id=%s", user_id)


def refresh_card_pointer(user_id: int, new_message_id: int) -> None:
    state = get_wait_state(user_id)
    if not state:
        return
    updated = WaitInputState(
        kind=state.kind,
        card_msg_id=int(new_message_id),
        chat_id=state.chat_id,
        meta=dict(state.meta),
    )
    set_wait_state(user_id, updated)

