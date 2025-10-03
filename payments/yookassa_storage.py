"""Persistence helpers for YooKassa payments."""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

from redis_utils import rds
from settings import REDIS_PREFIX

log = logging.getLogger("payments.yookassa.storage")

_PENDING_KEY_TMPL = f"{REDIS_PREFIX}:yk:pending:{{}}"
_LOCK_KEY_TMPL = f"{REDIS_PREFIX}:yk:lock:{{}}"
_PENDING_TTL = 24 * 60 * 60

_memory_pending: Dict[str, tuple[float, Dict[str, Any]]] = {}
_memory_locks: Dict[str, float] = {}
_memory_lock = threading.Lock()


@dataclass(slots=True)
class PendingPayment:
    payment_id: str
    user_id: int
    pack_id: str
    amount: str
    currency: str
    tokens_to_add: int
    status: str = "PENDING"
    idempotence_key: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_json(self) -> str:
        payload = asdict(self)
        return json.dumps(payload, ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "PendingPayment":
        data = json.loads(raw)
        return cls(
            payment_id=str(data.get("payment_id")),
            user_id=int(data.get("user_id")),
            pack_id=str(data.get("pack_id")),
            amount=str(data.get("amount")),
            currency=str(data.get("currency")),
            tokens_to_add=int(data.get("tokens_to_add")),
            status=str(data.get("status", "PENDING")),
            idempotence_key=data.get("idempotence_key"),
            created_at=float(data.get("created_at", time.time())),
            updated_at=float(data.get("updated_at", time.time())),
        )

    def with_status(self, status: str) -> "PendingPayment":
        self.status = status
        self.updated_at = time.time()
        return self


def _pending_key(payment_id: str) -> str:
    return _PENDING_KEY_TMPL.format(payment_id)


def _lock_key(payment_id: str) -> str:
    return _LOCK_KEY_TMPL.format(payment_id)


def save_pending_payment(payment: PendingPayment) -> None:
    payload = payment.to_json()
    key = _pending_key(payment.payment_id)
    if rds is not None:
        try:
            rds.setex(key, _PENDING_TTL, payload)
            return
        except Exception as exc:  # pragma: no cover - Redis failure fallback
            log.warning("yookassa.pending.redis_save_failed", extra={"meta": {"key": key, "err": str(exc)}})
    with _memory_lock:
        _memory_pending[key] = (time.time() + _PENDING_TTL, json.loads(payload))


def load_pending_payment(payment_id: str) -> Optional[PendingPayment]:
    key = _pending_key(payment_id)
    if rds is not None:
        try:
            raw = rds.get(key)
        except Exception as exc:  # pragma: no cover - Redis failure fallback
            log.warning("yookassa.pending.redis_get_failed", extra={"meta": {"key": key, "err": str(exc)}})
        else:
            if raw:
                try:
                    return PendingPayment.from_json(raw.decode("utf-8"))
                except Exception:  # pragma: no cover - invalid data
                    log.exception("yookassa.pending.redis_payload_invalid", extra={"meta": {"key": key}})

    with _memory_lock:
        entry = _memory_pending.get(key)
        if not entry:
            return None
        expires_at, payload = entry
        if expires_at < time.time():
            _memory_pending.pop(key, None)
            return None
        return PendingPayment.from_json(json.dumps(payload, ensure_ascii=False))


def update_pending_status(payment_id: str, status: str) -> Optional[PendingPayment]:
    record = load_pending_payment(payment_id)
    if not record:
        return None
    record.with_status(status)
    save_pending_payment(record)
    return record


def acquire_payment_lock(payment_id: str, ttl: int = 30) -> bool:
    key = _lock_key(payment_id)
    now = time.time()
    if rds is not None:
        try:
            stored = rds.set(key, str(now), nx=True, ex=max(ttl, 1))
            if stored:
                return True
        except Exception as exc:  # pragma: no cover - Redis failure fallback
            log.warning("yookassa.lock.redis_error", extra={"meta": {"key": key, "err": str(exc)}})
    with _memory_lock:
        expires_at = _memory_locks.get(key)
        if expires_at and expires_at > now:
            return False
        _memory_locks[key] = now + max(ttl, 1)
    return True


def release_payment_lock(payment_id: str) -> None:
    key = _lock_key(payment_id)
    if rds is not None:
        try:
            rds.delete(key)
        except Exception as exc:  # pragma: no cover - Redis failure fallback
            log.warning("yookassa.lock.redis_release_failed", extra={"meta": {"key": key, "err": str(exc)}})
    with _memory_lock:
        _memory_locks.pop(key, None)


__all__ = [
    "PendingPayment",
    "save_pending_payment",
    "load_pending_payment",
    "update_pending_status",
    "acquire_payment_lock",
    "release_payment_lock",
]
