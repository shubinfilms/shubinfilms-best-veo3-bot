"""Persistent Suno job tracking with Postgres fallback."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
import logging
import threading
from typing import Any, Dict, Iterable, List, Mapping, Optional

try:  # SQLAlchemy is optional in tests
    from sqlalchemy import text
    from sqlalchemy.exc import SQLAlchemyError
except Exception:  # pragma: no cover - fallback for environments without SQLAlchemy
    text = None  # type: ignore

    class SQLAlchemyError(Exception):  # type: ignore
        pass

from db import postgres as db_postgres

log = logging.getLogger("suno.jobs_store")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SunoJobState(str, Enum):
    """Finite states for the Suno job lifecycle."""

    ENQUEUED = "ENQUEUED"
    PENDING = "PENDING"
    READY = "READY"
    DELIVERED = "DELIVERED"
    FAILED = "FAILED"
    FAILED_FINAL = "FAILED_FINAL"
    TIMEOUT = "TIMEOUT"
    RECONCILING = "RECONCILING"


class RefundState(str, Enum):
    ESCROW = "ESCROW"
    CAPTURED = "CAPTURED"
    REFUNDED = "REFUNDED"


@dataclass
class SunoJobRecord:
    """In-memory representation of a persisted Suno job."""

    job_id: str
    user_id: Optional[int] = None
    chat_id: Optional[int] = None
    request_id: Optional[str] = None
    title: Optional[str] = None
    state: SunoJobState = SunoJobState.ENQUEUED
    refund_state: RefundState = RefundState.ESCROW
    last_error: Optional[str] = None
    ledger_txn_id: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)
    delivered_at: Optional[datetime] = None
    webhook_seen_at: Optional[datetime] = None
    last_polled_at: Optional[datetime] = None
    retries: int = 0
    delivery_key: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.job_id,
            "user_id": self.user_id,
            "chat_id": self.chat_id,
            "request_id": self.request_id,
            "title": self.title,
            "state": self.state.value,
            "refund_state": self.refund_state.value,
            "last_error": self.last_error,
            "ledger_txn_id": self.ledger_txn_id,
            "payload": self.payload or {},
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "delivered_at": self.delivered_at.isoformat() if self.delivered_at else None,
            "webhook_seen_at": self.webhook_seen_at.isoformat() if self.webhook_seen_at else None,
            "last_polled_at": self.last_polled_at.isoformat() if self.last_polled_at else None,
            "retries": self.retries,
            "delivery_key": self.delivery_key,
        }


class SunoJobStore:
    """Persist Suno job state in Postgres with an in-memory fallback."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._memory: Dict[str, SunoJobRecord] = {}
        self._engine = self._try_get_engine()

    # ------------------------------------------------------------------
    #   Engine helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _try_get_engine():
        try:
            return db_postgres.get_engine()
        except Exception:  # pragma: no cover - optional in tests
            return None

    def _with_lock(self, fn, *args, **kwargs):
        with self._lock:
            return fn(*args, **kwargs)

    def _persist(self, sql: str, params: Mapping[str, Any]) -> None:
        engine = self._engine
        if engine is None or text is None:
            return
        try:
            with engine.begin() as conn:
                conn.execute(text(sql), params)
        except SQLAlchemyError as exc:  # pragma: no cover - best effort logging
            log.warning("suno.jobs_store.sql_failed", extra={"error": str(exc)})

    def _load_memory(self, job_id: str) -> Optional[SunoJobRecord]:
        return self._memory.get(job_id)

    def _save_memory(self, record: SunoJobRecord) -> None:
        self._memory[record.job_id] = record

    def _get_or_create(self, job_id: str) -> SunoJobRecord:
        record = self._memory.get(job_id)
        if record is None:
            record = SunoJobRecord(job_id=job_id)
            self._memory[job_id] = record
        return record

    # ------------------------------------------------------------------
    #   Public API
    # ------------------------------------------------------------------
    def record_enqueue(
        self,
        job_id: str,
        *,
        user_id: Optional[int],
        chat_id: Optional[int],
        request_id: Optional[str],
        title: Optional[str],
        payload: Optional[Mapping[str, Any]] = None,
    ) -> SunoJobRecord:
        def _update() -> SunoJobRecord:
            record = self._get_or_create(job_id)
            now = _utcnow()
            record.user_id = user_id
            record.chat_id = chat_id
            record.request_id = request_id
            record.title = title
            record.payload = dict(payload or {})
            record.state = SunoJobState.ENQUEUED
            record.refund_state = RefundState.ESCROW
            record.last_error = None
            record.created_at = record.created_at or now
            record.updated_at = now
            self._save_memory(record)
            return record

        record = self._with_lock(_update)
        self._persist(
            """
            INSERT INTO suno_jobs (id, user_id, chat_id, request_id, title, state, refund_state, payload)
            VALUES (:id, :user_id, :chat_id, :request_id, :title, :state, :refund_state, :payload)
            ON CONFLICT (id) DO UPDATE
               SET user_id = EXCLUDED.user_id,
                   chat_id = EXCLUDED.chat_id,
                   request_id = EXCLUDED.request_id,
                   title = EXCLUDED.title,
                   state = EXCLUDED.state,
                   refund_state = EXCLUDED.refund_state,
                   payload = EXCLUDED.payload,
                   updated_at = NOW()
            """,
            {
                "id": job_id,
                "user_id": user_id,
                "chat_id": chat_id,
                "request_id": request_id,
                "title": title,
                "state": record.state.value,
                "refund_state": record.refund_state.value,
                "payload": json.dumps(record.payload or {}, ensure_ascii=False),
            },
        )
        return record

    def mark_pending(
        self,
        job_id: str,
        *,
        payload: Optional[Mapping[str, Any]] = None,
        status: Optional[str] = None,
    ) -> None:
        def _update() -> SunoJobRecord:
            record = self._get_or_create(job_id)
            record.state = SunoJobState.PENDING
            record.updated_at = _utcnow()
            record.last_polled_at = record.updated_at
            if payload is not None:
                record.payload = dict(payload)
            if status:
                record.payload = dict(record.payload or {})
                record.payload["status_label"] = status
            self._save_memory(record)
            return record

        record = self._with_lock(_update)
        self._persist(
            """
            UPDATE suno_jobs
               SET state = :state,
                   payload = COALESCE(:payload, payload),
                   last_polled_at = NOW(),
                   updated_at = NOW()
             WHERE id = :id
            """,
            {
                "id": job_id,
                "state": record.state.value,
                "payload": json.dumps(record.payload or {}, ensure_ascii=False)
                if payload is not None or status
                else None,
            },
        )

    def mark_ready(
        self,
        job_id: str,
        *,
        payload: Optional[Mapping[str, Any]],
    ) -> None:
        def _update() -> SunoJobRecord:
            record = self._get_or_create(job_id)
            record.state = SunoJobState.READY
            record.refund_state = RefundState.CAPTURED
            record.payload = dict(payload or {})
            record.updated_at = _utcnow()
            record.last_polled_at = record.updated_at
            self._save_memory(record)
            return record

        record = self._with_lock(_update)
        self._persist(
            """
            UPDATE suno_jobs
               SET state = :state,
                   refund_state = :refund_state,
                   payload = :payload,
                   last_polled_at = NOW(),
                   updated_at = NOW()
             WHERE id = :id
            """,
            {
                "id": job_id,
                "state": record.state.value,
                "refund_state": record.refund_state.value,
                "payload": json.dumps(record.payload or {}, ensure_ascii=False),
            },
        )

    def mark_failed(
        self,
        job_id: str,
        *,
        message: Optional[str],
        payload: Optional[Mapping[str, Any]] = None,
        final: bool = False,
        refunded: bool = False,
    ) -> None:
        state = SunoJobState.FAILED_FINAL if final else SunoJobState.FAILED
        refund_state = RefundState.REFUNDED if refunded else RefundState.ESCROW

        def _update() -> SunoJobRecord:
            record = self._get_or_create(job_id)
            record.state = state
            record.last_error = message
            if payload is not None:
                record.payload = dict(payload)
            record.updated_at = _utcnow()
            record.refund_state = refund_state
            self._save_memory(record)
            return record

        record = self._with_lock(_update)
        self._persist(
            """
            UPDATE suno_jobs
               SET state = :state,
                   last_error = :last_error,
                   refund_state = :refund_state,
                   payload = COALESCE(:payload, payload),
                   updated_at = NOW()
             WHERE id = :id
            """,
            {
                "id": job_id,
                "state": record.state.value,
                "last_error": message,
                "refund_state": record.refund_state.value,
                "payload": json.dumps(record.payload or {}, ensure_ascii=False)
                if payload is not None
                else None,
            },
        )

    def mark_timeout(self, job_id: str) -> None:
        def _update() -> SunoJobRecord:
            record = self._get_or_create(job_id)
            record.state = SunoJobState.TIMEOUT
            record.updated_at = _utcnow()
            self._save_memory(record)
            return record

        record = self._with_lock(_update)
        self._persist(
            """
            UPDATE suno_jobs
               SET state = :state,
                   updated_at = NOW()
             WHERE id = :id
            """,
            {"id": job_id, "state": record.state.value},
        )

    def mark_reconciling(self, job_id: str) -> None:
        def _update() -> SunoJobRecord:
            record = self._get_or_create(job_id)
            record.state = SunoJobState.RECONCILING
            record.retries += 1
            record.updated_at = _utcnow()
            self._save_memory(record)
            return record

        record = self._with_lock(_update)
        self._persist(
            """
            UPDATE suno_jobs
               SET state = :state,
                   retries = retries + 1,
                   updated_at = NOW()
             WHERE id = :id
            """,
            {"id": job_id, "state": record.state.value},
        )

    def mark_delivered(self, job_id: str, *, delivery_key: Optional[str] = None) -> None:
        def _update() -> SunoJobRecord:
            record = self._get_or_create(job_id)
            record.state = SunoJobState.DELIVERED
            record.delivered_at = _utcnow()
            record.updated_at = record.delivered_at
            record.refund_state = RefundState.CAPTURED
            if delivery_key:
                record.delivery_key = delivery_key
            self._save_memory(record)
            return record

        record = self._with_lock(_update)
        self._persist(
            """
            UPDATE suno_jobs
               SET state = :state,
                   delivered_at = NOW(),
                   refund_state = :refund_state,
                   delivery_key = COALESCE(:delivery_key, delivery_key),
                   updated_at = NOW()
             WHERE id = :id
            """,
            {
                "id": job_id,
                "state": record.state.value,
                "refund_state": record.refund_state.value,
                "delivery_key": delivery_key,
            },
        )

    def mark_webhook_seen(self, job_id: str) -> None:
        def _update() -> SunoJobRecord:
            record = self._get_or_create(job_id)
            record.webhook_seen_at = _utcnow()
            self._save_memory(record)
            return record

        self._with_lock(_update)
        self._persist(
            """
            UPDATE suno_jobs
               SET webhook_seen_at = NOW(),
                   updated_at = NOW()
             WHERE id = :id
            """,
            {"id": job_id},
        )

    def should_deliver(self, job_id: str, *, delivery_key: str) -> bool:
        def _check() -> bool:
            record = self._get_or_create(job_id)
            if record.delivery_key and record.delivery_key == delivery_key:
                return False
            if record.state == SunoJobState.DELIVERED:
                return False
            return True

        return self._with_lock(_check)

    def get(self, job_id: str) -> Optional[SunoJobRecord]:
        return self._with_lock(lambda: self._load_memory(job_id))

    def iter_reconcilable(
        self,
        *,
        since: datetime,
        limit: int,
        states: Iterable[SunoJobState] = (
            SunoJobState.TIMEOUT,
            SunoJobState.PENDING,
            SunoJobState.RECONCILING,
            SunoJobState.FAILED,
        ),
    ) -> List[SunoJobRecord]:
        def _collect() -> List[SunoJobRecord]:
            target_states = {state for state in states}
            result: List[SunoJobRecord] = []
            for record in self._memory.values():
                if record.state not in target_states:
                    continue
                if record.updated_at < since:
                    continue
                result.append(record)
            result.sort(key=lambda rec: rec.updated_at, reverse=True)
            return result[:limit]

        return self._with_lock(_collect)


jobs_store = SunoJobStore()

__all__ = ["SunoJobState", "RefundState", "SunoJobRecord", "SunoJobStore", "jobs_store"]
