# -*- coding: utf-8 -*-
"""Persistent ledger storage for user balances."""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional

try:  # psycopg is optional when using the in-memory backend
    import psycopg
    from psycopg.errors import UniqueViolation
    from psycopg_pool import ConnectionPool
except ImportError:  # pragma: no cover - optional dependency handling
    psycopg = None  # type: ignore

    class UniqueViolation(Exception):  # type: ignore[override]
        pass

    ConnectionPool = None  # type: ignore

log = logging.getLogger(__name__)


@dataclass
class LedgerOpResult:
    """Result of a balance operation."""

    applied: bool
    balance: int
    op_id: str
    reason: str
    old_balance: int
    duplicate: bool = False


@dataclass
class BalanceRecalcResult:
    """Result of recalculating a balance from the ledger."""

    previous: int
    calculated: int
    updated: bool


class InsufficientBalance(RuntimeError):
    """Raised when a debit would make the balance negative."""

    def __init__(self, balance: int, required: int):
        super().__init__(f"insufficient balance: have {balance}, need {required}")
        self.balance = balance
        self.required = required


class _LedgerHelpers:
    """Utility helpers shared by ledger backends."""

    @staticmethod
    def _json_meta(meta: Optional[Dict[str, Any]]) -> Optional[str]:
        if not meta:
            return None
        return json.dumps(meta, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _log_operation(
        op_type: str,
        user_id: int,
        op_id: str,
        amount: int,
        reason: str,
        old_balance: int,
        new_balance: int,
        meta: Optional[Dict[str, Any]],
    ) -> None:
        try:
            meta_repr = json.dumps(meta or {}, ensure_ascii=False, sort_keys=True)
        except Exception:
            meta_repr = "{}"
        log.info(
            "ledger %s user=%s op_id=%s amount=%s reason=%s old=%s new=%s meta=%s",
            op_type,
            user_id,
            op_id,
            amount,
            reason,
            old_balance,
            new_balance,
            meta_repr,
        )


class _PostgresLedgerStorage(_LedgerHelpers):
    """Ledger-backed balance storage with atomic operations."""

    def __init__(self, dsn: str):
        if not dsn:
            raise RuntimeError("DATABASE_URL is required for ledger storage")
        if psycopg is None or ConnectionPool is None:
            raise RuntimeError(
                "Postgres ledger backend requires psycopg and psycopg_pool to be installed"
            )
        self.dsn = dsn
        self.pool = ConnectionPool(conninfo=dsn, max_size=10, kwargs={"autocommit": False})
        self.pool.wait()
        self._prepare()

    # ------------------------------------------------------------------
    #   Internal helpers
    # ------------------------------------------------------------------
    def _prepare(self) -> None:
        ddl_users = """
        CREATE TABLE IF NOT EXISTS users (
            id BIGINT PRIMARY KEY,
            balance BIGINT NOT NULL DEFAULT 0 CHECK (balance >= 0),
            signup_bonus_granted BOOLEAN NOT NULL DEFAULT FALSE,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """

        ddl_ledger = """
        CREATE TABLE IF NOT EXISTS ledger (
            id BIGSERIAL PRIMARY KEY,
            user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            type TEXT NOT NULL CHECK (type IN ('credit','debit')),
            amount BIGINT NOT NULL CHECK (amount >= 0),
            reason TEXT NOT NULL,
            op_id TEXT NOT NULL UNIQUE,
            meta JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """

        ddl_ledger_idx = """
        CREATE INDEX IF NOT EXISTS idx_ledger_user_created_at
            ON ledger(user_id, created_at DESC)
        """

        ddl_promo = """
        CREATE TABLE IF NOT EXISTS promo_usages (
            user_id BIGINT NOT NULL,
            promo_id TEXT NOT NULL,
            used_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (user_id, promo_id),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """

        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(ddl_users)
                cur.execute(ddl_ledger)
                cur.execute(ddl_ledger_idx)
                cur.execute(ddl_promo)
            conn.commit()

        try:
            with self.pool.connection() as conn:
                old_autocommit = conn.autocommit
                conn.autocommit = True
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS idx_ledger_opid_unique
                              ON ledger (op_id)
                            """
                        )
                finally:
                    conn.autocommit = old_autocommit
        except Exception:  # pragma: no cover - defensive logging
            log.exception("failed to ensure idx_ledger_opid_unique")

    @staticmethod
    def _ensure_user(cur: psycopg.Cursor[Any], uid: int) -> None:
        cur.execute("INSERT INTO users (id) VALUES (%s) ON CONFLICT DO NOTHING", (uid,))

    # ------------------------------------------------------------------
    #   Public API
    # ------------------------------------------------------------------
    def ping(self) -> bool:
        try:
            with self.pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
            return True
        except Exception:
            log.exception("ledger ping failed")
            return False

    def get_promo_owner(self, promo_code: str) -> Optional[int]:
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT user_id FROM promo_usages WHERE promo_id=%s ORDER BY used_at ASC LIMIT 1",
                    (promo_code,),
                )
                row = cur.fetchone()
                return int(row[0]) if row else None

    def get_balance(self, uid: int) -> int:
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                self._ensure_user(cur, uid)
                cur.execute("SELECT balance FROM users WHERE id=%s", (uid,))
                row = cur.fetchone()
                return int(row[0]) if row else 0

    def credit(
        self,
        uid: int,
        amount: int,
        reason: str,
        op_id: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> LedgerOpResult:
        amount = int(amount)
        if amount <= 0:
            balance = self.get_balance(uid)
            return LedgerOpResult(False, balance, op_id, reason, balance, duplicate=True)

        meta_json = self._json_meta(meta)

        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                self._ensure_user(cur, uid)
                cur.execute("SELECT balance FROM users WHERE id=%s FOR UPDATE", (uid,))
                row = cur.fetchone()
                old_balance = int(row[0]) if row else 0

                cur.execute("SELECT 1 FROM ledger WHERE op_id=%s", (op_id,))
                if cur.fetchone():
                    return LedgerOpResult(False, old_balance, op_id, reason, old_balance, duplicate=True)

                cur.execute(
                    """
                    UPDATE users
                       SET balance = balance + %s,
                           updated_at = now()
                     WHERE id = %s
                     RETURNING balance
                    """,
                    (amount, uid),
                )
                new_balance_row = cur.fetchone()
                new_balance = int(new_balance_row[0]) if new_balance_row else old_balance

                cur.execute(
                    """
                    INSERT INTO ledger (user_id, type, amount, reason, op_id, meta)
                    VALUES (%s, 'credit', %s, %s, %s, COALESCE(%s::jsonb, '{}'::jsonb))
                    """,
                    (uid, amount, reason, op_id, meta_json),
                )

        self._log_operation("credit", uid, op_id, amount, reason, old_balance, new_balance, meta)
        return LedgerOpResult(True, new_balance, op_id, reason, old_balance)

    def debit(
        self,
        uid: int,
        amount: int,
        reason: str,
        op_id: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> LedgerOpResult:
        amount = int(amount)
        if amount <= 0:
            balance = self.get_balance(uid)
            return LedgerOpResult(False, balance, op_id, reason, balance, duplicate=True)

        meta_json = self._json_meta(meta)

        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                self._ensure_user(cur, uid)
                cur.execute("SELECT balance FROM users WHERE id=%s FOR UPDATE", (uid,))
                row = cur.fetchone()
                old_balance = int(row[0]) if row else 0

                cur.execute("SELECT 1 FROM ledger WHERE op_id=%s", (op_id,))
                if cur.fetchone():
                    return LedgerOpResult(False, old_balance, op_id, reason, old_balance, duplicate=True)

                if old_balance < amount:
                    raise InsufficientBalance(old_balance, amount)

                cur.execute(
                    """
                    UPDATE users
                       SET balance = balance - %s,
                           updated_at = now()
                     WHERE id = %s
                     RETURNING balance
                    """,
                    (amount, uid),
                )
                new_balance_row = cur.fetchone()
                new_balance = int(new_balance_row[0]) if new_balance_row else old_balance

                cur.execute(
                    """
                    INSERT INTO ledger (user_id, type, amount, reason, op_id, meta)
                    VALUES (%s, 'debit', %s, %s, %s, COALESCE(%s::jsonb, '{}'::jsonb))
                    """,
                    (uid, amount, reason, op_id, meta_json),
                )

        self._log_operation("debit", uid, op_id, amount, reason, old_balance, new_balance, meta)
        return LedgerOpResult(True, new_balance, op_id, reason, old_balance)

    def grant_signup_bonus(
        self, uid: int, amount: int, meta: Optional[Dict[str, Any]] = None
    ) -> LedgerOpResult:
        amount = int(amount)
        op_id = f"signup_bonus:{uid}"
        meta_json = self._json_meta(meta)

        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                self._ensure_user(cur, uid)
                cur.execute(
                    "SELECT balance, signup_bonus_granted FROM users WHERE id=%s FOR UPDATE",
                    (uid,),
                )
                row = cur.fetchone()
                if row:
                    old_balance = int(row[0])
                    already = bool(row[1])
                else:
                    old_balance = 0
                    already = False

                if already:
                    return LedgerOpResult(False, old_balance, op_id, "signup_bonus", old_balance, duplicate=True)

                if amount <= 0:
                    cur.execute(
                        "UPDATE users SET signup_bonus_granted = TRUE WHERE id=%s",
                        (uid,),
                    )
                    return LedgerOpResult(False, old_balance, op_id, "signup_bonus", old_balance, duplicate=True)

                cur.execute(
                    """
                    UPDATE users
                       SET balance = balance + %s,
                           signup_bonus_granted = TRUE,
                           updated_at = now()
                     WHERE id = %s
                     RETURNING balance
                    """,
                    (amount, uid),
                )
                new_balance_row = cur.fetchone()
                new_balance = int(new_balance_row[0]) if new_balance_row else old_balance

                try:
                    cur.execute(
                        """
                        INSERT INTO ledger (user_id, type, amount, reason, op_id, meta)
                        VALUES (%s, 'credit', %s, %s, %s, COALESCE(%s::jsonb, '{}'::jsonb))
                        """,
                        (uid, amount, "signup_bonus", op_id, meta_json),
                    )
                except UniqueViolation:
                    cur.execute(
                        "UPDATE users SET signup_bonus_granted = TRUE WHERE id=%s",
                        (uid,),
                    )
                    conn.rollback()
                    return LedgerOpResult(
                        False,
                        old_balance,
                        op_id,
                        "signup_bonus",
                        old_balance,
                        duplicate=True,
                    )

        self._log_operation(
            "credit", uid, op_id, amount, "signup_bonus", old_balance, new_balance, meta
        )
        return LedgerOpResult(True, new_balance, op_id, "signup_bonus", old_balance)

    def apply_promo(
        self,
        uid: int,
        promo_code: str,
        amount: int,
        meta: Optional[Dict[str, Any]] = None,
    ) -> LedgerOpResult:
        op_id = f"promo:{promo_code}:{uid}"
        meta = dict(meta or {})
        meta.setdefault("promo_code", promo_code)
        meta_json = self._json_meta(meta)

        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                self._ensure_user(cur, uid)
                cur.execute("SELECT balance FROM users WHERE id=%s FOR UPDATE", (uid,))
                row = cur.fetchone()
                old_balance = int(row[0]) if row else 0

                cur.execute("SELECT user_id FROM promo_usages WHERE promo_id=%s ORDER BY used_at ASC LIMIT 1", (promo_code,))
                owner_row = cur.fetchone()
                if owner_row:
                    owner = int(owner_row[0])
                    if owner != uid:
                        return LedgerOpResult(False, old_balance, op_id, "promo", old_balance, duplicate=True)

                cur.execute("SELECT 1 FROM ledger WHERE op_id=%s", (op_id,))
                if cur.fetchone():
                    return LedgerOpResult(False, old_balance, op_id, "promo", old_balance, duplicate=True)

                cur.execute(
                    """
                    INSERT INTO promo_usages (user_id, promo_id)
                    VALUES (%s, %s)
                    ON CONFLICT DO NOTHING
                    RETURNING promo_id
                    """,
                    (uid, promo_code),
                )
                inserted = cur.fetchone()
                if not inserted:
                    return LedgerOpResult(False, old_balance, op_id, "promo", old_balance, duplicate=True)

                amount = int(amount)
                if amount <= 0:
                    return LedgerOpResult(False, old_balance, op_id, "promo", old_balance, duplicate=True)

                cur.execute(
                    """
                    UPDATE users
                       SET balance = balance + %s,
                           updated_at = now()
                     WHERE id = %s
                     RETURNING balance
                    """,
                    (amount, uid),
                )
                new_balance_row = cur.fetchone()
                new_balance = int(new_balance_row[0]) if new_balance_row else old_balance

                cur.execute(
                    """
                    INSERT INTO ledger (user_id, type, amount, reason, op_id, meta)
                    VALUES (%s, 'credit', %s, %s, %s, COALESCE(%s::jsonb, '{}'::jsonb))
                    """,
                    (uid, amount, "promo", op_id, meta_json),
                )

        self._log_operation("credit", uid, op_id, amount, "promo", old_balance, new_balance, meta)
        return LedgerOpResult(True, new_balance, op_id, "promo", old_balance)

    def rename_operation(
        self,
        old_op_id: str,
        new_op_id: str,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if old_op_id == new_op_id:
            return True
        meta_json = self._json_meta(extra_meta)
        try:
            with self.pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE ledger
                           SET op_id = %s,
                               meta = COALESCE(meta, '{}'::jsonb) || COALESCE(%s::jsonb, '{}'::jsonb)
                         WHERE op_id = %s
                        """,
                        (new_op_id, meta_json, old_op_id),
                    )
                    updated = cur.rowcount
            if updated:
                log.info("ledger rename op_id %s -> %s", old_op_id, new_op_id)
            return bool(updated)
        except UniqueViolation:
            log.warning("ledger rename conflict for %s -> %s", old_op_id, new_op_id)
            return False

    def recalc_user_balance(self, uid: int) -> BalanceRecalcResult:
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                self._ensure_user(cur, uid)
                cur.execute("SELECT balance FROM users WHERE id=%s", (uid,))
                row = cur.fetchone()
                previous = int(row[0]) if row else 0

                cur.execute(
                    """
                    SELECT COALESCE(SUM(CASE WHEN type='credit' THEN amount ELSE -amount END), 0)
                      FROM ledger
                     WHERE user_id = %s
                    """,
                    (uid,),
                )
                calc_row = cur.fetchone()
                calculated = int(calc_row[0]) if calc_row else 0

                updated = calculated != previous
                if updated:
                    cur.execute(
                        "UPDATE users SET balance = %s, updated_at = now() WHERE id = %s",
                        (calculated, uid),
                    )
        if updated:
            log.info(
                "ledger recalc user=%s previous=%s calculated=%s", uid, previous, calculated
            )
        return BalanceRecalcResult(previous=previous, calculated=calculated, updated=updated)


class _MemoryLedgerStorage(_LedgerHelpers):
    """In-memory ledger implementation for environments without Postgres."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._users: Dict[int, Dict[str, Any]] = {}
        self._ledger: Dict[str, Dict[str, Any]] = {}
        self._ledger_order: list[str] = []
        self._promo_first_owner: Dict[str, int] = {}
        self._promo_usages: set[tuple[int, str]] = set()

    # ------------------------------------------------------------------
    #   Internal helpers
    # ------------------------------------------------------------------
    def _ensure_user(self, uid: int) -> Dict[str, Any]:
        return self._users.setdefault(
            uid,
            {
                "balance": 0,
                "signup_bonus_granted": False,
            },
        )

    # ------------------------------------------------------------------
    #   Public API
    # ------------------------------------------------------------------
    def ping(self) -> bool:
        return True

    def get_promo_owner(self, promo_code: str) -> Optional[int]:
        with self._lock:
            owner = self._promo_first_owner.get(promo_code)
            return int(owner) if owner is not None else None

    def get_balance(self, uid: int) -> int:
        with self._lock:
            user = self._ensure_user(uid)
            return int(user["balance"])

    def credit(
        self,
        uid: int,
        amount: int,
        reason: str,
        op_id: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> LedgerOpResult:
        amount = int(amount)
        with self._lock:
            user = self._ensure_user(uid)
            old_balance = int(user["balance"])

            if amount <= 0:
                return LedgerOpResult(False, old_balance, op_id, reason, old_balance, duplicate=True)

            if op_id in self._ledger:
                return LedgerOpResult(False, old_balance, op_id, reason, old_balance, duplicate=True)

            new_balance = old_balance + amount
            user["balance"] = new_balance

            self._ledger[op_id] = {
                "user_id": uid,
                "type": "credit",
                "amount": amount,
                "reason": reason,
                "meta": meta or {},
            }
            self._ledger_order.append(op_id)

        self._log_operation("credit", uid, op_id, amount, reason, old_balance, new_balance, meta)
        return LedgerOpResult(True, new_balance, op_id, reason, old_balance)

    def debit(
        self,
        uid: int,
        amount: int,
        reason: str,
        op_id: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> LedgerOpResult:
        amount = int(amount)
        with self._lock:
            user = self._ensure_user(uid)
            old_balance = int(user["balance"])

            if amount <= 0:
                return LedgerOpResult(False, old_balance, op_id, reason, old_balance, duplicate=True)

            if op_id in self._ledger:
                return LedgerOpResult(False, old_balance, op_id, reason, old_balance, duplicate=True)

            if old_balance < amount:
                raise InsufficientBalance(old_balance, amount)

            new_balance = old_balance - amount
            user["balance"] = new_balance

            self._ledger[op_id] = {
                "user_id": uid,
                "type": "debit",
                "amount": amount,
                "reason": reason,
                "meta": meta or {},
            }
            self._ledger_order.append(op_id)

        self._log_operation("debit", uid, op_id, amount, reason, old_balance, new_balance, meta)
        return LedgerOpResult(True, new_balance, op_id, reason, old_balance)

    def grant_signup_bonus(
        self, uid: int, amount: int, meta: Optional[Dict[str, Any]] = None
    ) -> LedgerOpResult:
        amount = int(amount)
        op_id = f"signup_bonus:{uid}"
        with self._lock:
            user = self._ensure_user(uid)
            old_balance = int(user["balance"])

            if user.get("signup_bonus_granted"):
                return LedgerOpResult(False, old_balance, op_id, "signup_bonus", old_balance, duplicate=True)

            if op_id in self._ledger:
                user["signup_bonus_granted"] = True
                return LedgerOpResult(False, old_balance, op_id, "signup_bonus", old_balance, duplicate=True)

            if amount <= 0:
                user["signup_bonus_granted"] = True
                return LedgerOpResult(False, old_balance, op_id, "signup_bonus", old_balance, duplicate=True)

            new_balance = old_balance + amount
            user["balance"] = new_balance
            user["signup_bonus_granted"] = True

            self._ledger[op_id] = {
                "user_id": uid,
                "type": "credit",
                "amount": amount,
                "reason": "signup_bonus",
                "meta": meta or {},
            }
            self._ledger_order.append(op_id)

        self._log_operation(
            "credit", uid, op_id, amount, "signup_bonus", old_balance, new_balance, meta
        )
        return LedgerOpResult(True, new_balance, op_id, "signup_bonus", old_balance)

    def apply_promo(
        self,
        uid: int,
        promo_code: str,
        amount: int,
        meta: Optional[Dict[str, Any]] = None,
    ) -> LedgerOpResult:
        op_id = f"promo:{promo_code}:{uid}"
        meta = dict(meta or {})
        meta.setdefault("promo_code", promo_code)
        amount = int(amount)

        with self._lock:
            user = self._ensure_user(uid)
            old_balance = int(user["balance"])

            owner = self._promo_first_owner.get(promo_code)
            if owner is not None and owner != uid:
                return LedgerOpResult(False, old_balance, op_id, "promo", old_balance, duplicate=True)

            if (uid, promo_code) in self._promo_usages:
                return LedgerOpResult(False, old_balance, op_id, "promo", old_balance, duplicate=True)

            if op_id in self._ledger or amount <= 0:
                return LedgerOpResult(False, old_balance, op_id, "promo", old_balance, duplicate=True)

            self._promo_usages.add((uid, promo_code))
            self._promo_first_owner.setdefault(promo_code, uid)

            new_balance = old_balance + amount
            user["balance"] = new_balance

            self._ledger[op_id] = {
                "user_id": uid,
                "type": "credit",
                "amount": amount,
                "reason": "promo",
                "meta": meta,
            }
            self._ledger_order.append(op_id)

        self._log_operation("credit", uid, op_id, amount, "promo", old_balance, new_balance, meta)
        return LedgerOpResult(True, new_balance, op_id, "promo", old_balance)

    def rename_operation(
        self,
        old_op_id: str,
        new_op_id: str,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if old_op_id == new_op_id:
            return True

        with self._lock:
            if new_op_id in self._ledger:
                log.warning("ledger rename conflict for %s -> %s", old_op_id, new_op_id)
                return False

            entry = self._ledger.get(old_op_id)
            if not entry:
                return False

            entry = dict(entry)
            entry["op_id"] = new_op_id
            if extra_meta:
                current_meta = dict(entry.get("meta") or {})
                current_meta.update(extra_meta)
                entry["meta"] = current_meta

            index = None
            try:
                index = self._ledger_order.index(old_op_id)
            except ValueError:
                pass

            del self._ledger[old_op_id]
            self._ledger[new_op_id] = entry
            if index is not None:
                self._ledger_order[index] = new_op_id

        log.info("ledger rename op_id %s -> %s", old_op_id, new_op_id)
        return True

    def recalc_user_balance(self, uid: int) -> BalanceRecalcResult:
        with self._lock:
            user = self._ensure_user(uid)
            previous = int(user["balance"])
            calculated = 0
            for entry in self._ledger.values():
                if entry.get("user_id") != uid:
                    continue
                if entry.get("type") == "credit":
                    calculated += int(entry.get("amount", 0))
                else:
                    calculated -= int(entry.get("amount", 0))

            updated = calculated != previous
            if updated:
                user["balance"] = calculated

        if updated:
            log.info(
                "ledger recalc user=%s previous=%s calculated=%s", uid, previous, calculated
            )
        return BalanceRecalcResult(previous=previous, calculated=calculated, updated=updated)


class LedgerStorage:
    """Facade that selects the appropriate ledger backend."""

    def __init__(self, dsn: Optional[str], *, backend: str = "postgres") -> None:
        backend = (backend or "postgres").lower()
        self.backend = backend

        if backend == "memory":
            self._impl = _MemoryLedgerStorage()
        else:
            if not dsn:
                raise RuntimeError(
                    "DATABASE_URL (or POSTGRES_DSN) must be set for persistent ledger storage"
                )
            self._impl = _PostgresLedgerStorage(dsn)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._impl, name)
