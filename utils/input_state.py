from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Tuple, Literal

import time

from settings import REDIS_PREFIX

from utils.telegram_utils import should_capture_to_prompt

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
    SORA2_PROMPT = "sora2_prompt"
    PROMO_CODE = "promo_code"


def classify_wait_input(text: Optional[str]) -> Tuple[bool, Optional[str]]:
    """Return whether ``text`` should be captured together with ignore reason."""

    if text is None:
        return False, "none"
    stripped = text.strip()
    stripped_for_match = stripped
    if "<" in stripped_for_match and ">" in stripped_for_match:
        cleaned = re.sub(r"<[^>]*>", " ", stripped_for_match)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if cleaned:
            stripped_for_match = cleaned
    if not stripped:
        return False, "empty"
    if not should_capture_to_prompt(stripped_for_match):
        return False, "command_label"
    return True, None


@dataclass
class WaitInputState:
    kind: WaitKind
    card_msg_id: int
    chat_id: int
    meta: Dict[str, Any] = field(default_factory=dict)
    expires_at: Optional[float] = None

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
            "expires_at": self.expires_at,
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
            raw_expires = data.get("expires_at")
            expires_at: Optional[float]
            if isinstance(raw_expires, (int, float)):
                expires_at = float(raw_expires)
            else:
                expires_at = None
        except (ValueError, KeyError):
            return None
        return cls(
            kind=kind,
            card_msg_id=card_msg_id,
            chat_id=chat_id,
            meta=meta,
            expires_at=expires_at,
        )

    def is_expired(self, *, now: Optional[float] = None) -> bool:
        if self.expires_at is None:
            return False
        ts = time.time() if now is None else now
        return ts >= self.expires_at


_memory_store: Dict[int, Dict[str, Any]] = {}


def _key(user_id: int) -> str:
    return _KEY_TMPL.format(user_id=int(user_id))


_DEFAULT_TTL_SECONDS = 90


def _with_new_expiry(state: WaitInputState, ttl_seconds: int) -> WaitInputState:
    expires_at = time.time() + max(ttl_seconds, 1)
    return WaitInputState(
        kind=state.kind,
        card_msg_id=state.card_msg_id,
        chat_id=state.chat_id,
        meta=dict(state.meta),
        expires_at=expires_at,
    )


def set_wait_state(user_id: int, state: WaitInputState, *, ttl_seconds: int = _DEFAULT_TTL_SECONDS) -> None:
    state_with_expiry = _with_new_expiry(state, ttl_seconds)
    payload = json.dumps(state_with_expiry.to_dict(), ensure_ascii=False)
    storage_key = _key(user_id)
    if _redis:
        try:
            _redis.set(storage_key, payload, ex=24 * 60 * 60)
        except Exception:  # pragma: no cover - redis connectivity issues
            _logger.exception("Failed to save wait-state to redis", extra={"user_id": user_id})
            _memory_store[int(user_id)] = state_with_expiry.to_dict()
    else:
        _memory_store[int(user_id)] = state_with_expiry.to_dict()
    _logger.info(
        "WAIT_SET kind=%s user_id=%s card=%s",
        state_with_expiry.kind.value,
        user_id,
        state_with_expiry.card_msg_id,
    )


def _load_wait_state(user_id: int) -> Optional[WaitInputState]:
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


def get_wait_state(user_id: int, *, now: Optional[float] = None) -> Optional[WaitInputState]:
    state = _load_wait_state(user_id)
    if not state:
        return None
    current_ts = time.time() if now is None else now
    if state.expires_at and state.expires_at <= current_ts:
        clear_wait_state(user_id, reason="expired")
        return None
    return state


def clear_wait_state(user_id: int, *, reason: str = "manual") -> None:
    storage_key = _key(user_id)
    if _redis:
        try:
            _redis.delete(storage_key)
        except Exception:  # pragma: no cover - redis connectivity issues
            _logger.exception("Failed to clear wait-state in redis", extra={"user_id": user_id})
    _memory_store.pop(int(user_id), None)
    _logger.info("WAIT_CLEAR user_id=%s reason=%s", user_id, reason)


def refresh_card_pointer(user_id: int, new_message_id: int) -> None:
    state = get_wait_state(user_id)
    if not state:
        return
    updated = WaitInputState(
        kind=state.kind,
        card_msg_id=int(new_message_id),
        chat_id=state.chat_id,
        meta=dict(state.meta),
        expires_at=state.expires_at,
    )
    set_wait_state(user_id, updated)


def set_wait(
    user_id: int,
    kind: Literal[
        "veo_prompt",
        "mj_prompt",
        "suno_title",
        "suno_style",
        "suno_lyrics",
        "banana_prompt",
        "sora2_prompt",
    ],
    card_msg_id: Optional[int],
    *,
    chat_id: Optional[int],
    meta: Optional[Mapping[str, Any]] = None,
) -> None:
    if chat_id is None:
        raise ValueError("chat_id is required to set wait state")

    payload_meta: Dict[str, Any]
    if isinstance(meta, Mapping):
        payload_meta = dict(meta)
    else:
        payload_meta = {}

    wait_state = WaitInputState(
        kind=WaitKind(kind),
        card_msg_id=int(card_msg_id or 0),
        chat_id=int(chat_id),
        meta=payload_meta,
    )
    set_wait_state(user_id, wait_state)


def clear_wait(user_id: int, *, reason: str = "manual") -> None:
    clear_wait_state(user_id, reason=reason)


def get_wait(user_id: int) -> Optional[WaitInputState]:
    return get_wait_state(user_id)


def is_waiting(user_id: int) -> bool:
    return get_wait_state(user_id) is not None


def touch_wait(user_id: int, *, ttl_seconds: int = _DEFAULT_TTL_SECONDS) -> Optional[WaitInputState]:
    state = get_wait_state(user_id)
    if not state:
        return None
    set_wait_state(user_id, state, ttl_seconds=ttl_seconds)
    return get_wait_state(user_id)


def has_wait(user_id: int) -> bool:
    return is_waiting(user_id)


WAIT_TTL = 180  # сек


@dataclass
class WaitState:
    kind: str
    card_msg_id: int
    data: Dict[str, Any]
    expires_at: float

    def touch(self) -> None:
        self.expires_at = time.time() + WAIT_TTL


class InputRegistry:
    """Простейший in-memory реестр ожидания ввода по chat_id."""

    def __init__(self) -> None:
        self._by_chat: Dict[int, WaitState] = {}
        self._log = logging.getLogger("input-registry")

    def _purge_if_expired(self, chat_id: int, state: Optional[WaitState]) -> Optional[WaitState]:
        if not state:
            return None
        if state.expires_at > time.time():
            return state
        self._by_chat.pop(chat_id, None)
        self._log.info("registry.expired", extra={"chat_id": chat_id})
        return None

    def get(self, chat_id: int) -> Optional[WaitState]:
        stored = self._by_chat.get(int(chat_id))
        return self._purge_if_expired(int(chat_id), stored)

    def set(self, chat_id: int, state: WaitState, *, reason: str = "set") -> None:
        state.touch()
        self._by_chat[int(chat_id)] = state
        self._log.info("registry.set", extra={"chat_id": int(chat_id), "kind": state.kind, "reason": reason})

    def touch(self, chat_id: int, *, reason: str = "refresh") -> Optional[WaitState]:
        state = self.get(chat_id)
        if not state:
            return None
        state.touch()
        self._by_chat[int(chat_id)] = state
        self._log.info("registry.touch", extra={"chat_id": int(chat_id), "kind": state.kind, "reason": reason})
        return state

    def clear(self, chat_id: int, *, reason: str = "manual") -> None:
        removed = self._by_chat.pop(int(chat_id), None)
        if removed:
            self._log.info("registry.clear", extra={"chat_id": int(chat_id), "kind": removed.kind, "reason": reason})


input_state = InputRegistry()

