# -*- coding: utf-8 -*-
"""Persistent ledger storage for user balances."""

from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlsplit, urlunsplit

import psycopg
from psycopg.errors import UniqueViolation
from psycopg_pool import ConnectionPool

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


class _BaseLedgerBackend:
    """Common helpers shared by concrete ledger backends."""

    backend_name = "unknown"

    def __init__(self) -> None:
        self.safe_dsn: Optional[str] = None

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

    @staticmethod
    def _merge_meta(existing: Optional[str], extra: Optional[Dict[str, Any]]) -> Optional[str]:
        if not extra:
            return existing
        try:
            current = json.loads(existing) if existing else {}
            if not isinstance(current, dict):
                current = {}
        except Exception:
            current = {}
        merged = dict(current)
        merged.update(extra)
        return json.dumps(merged, ensure_ascii=False, sort_keys=True)


def _sanitize_postgres_dsn(dsn: str) -> str:
    """Remove password information from a PostgreSQL DSN for logging."""

    try:
        split = urlsplit(dsn)
    except Exception:
        return dsn
    netloc = split.netloc
    if "@" in netloc:
        userinfo, host = netloc.rsplit("@", 1)
        if ":" in userinfo:
            username = userinfo.split(":", 1)[0]
        else:
            username = userinfo
        netloc = f"{username}@{host}" if username else host
    return urlunsplit((split.scheme, netloc, split.path, split.query, split.fragment))


def _resolve_sqlite_target(dsn: str, split: Optional[Any] = None) -> tuple[str, str]:
    """Return the database path and a safe representation for logging."""

    if split is None:
        split = urlsplit(dsn)
    scheme = (split.scheme or "").lower()
    base_scheme = scheme.split("+", 1)[0] if scheme else ""
    if base_scheme and base_scheme not in {"sqlite", "file"}:
        raise ValueError(f"Unsupported SQLite DSN scheme: {dsn}")

    if split.query or split.fragment:
        log.warning("Ignoring SQLite DSN modifiers for %s", dsn)

    netloc = split.netloc
    path = split.path or ""

    if base_scheme.startswith("file") and not path and netloc:
        path = netloc
        netloc = ""

    if netloc:
        path = f"/{netloc}{path}"

    if path.startswith("//"):
        path = path[1:]

    if not path:
        raise ValueError("SQLite DSN must include a database path")

    path_obj = Path(path).expanduser()
    if not path_obj.is_absolute():
        path_obj = Path.cwd() / path_obj

    path_obj.parent.mkdir(parents=True, exist_ok=True)
    return str(path_obj), str(path_obj)


class _MemoryLedger(_BaseLedgerBackend):
    """In-memory ledger implementation used when persistence is unavailable."""

    backend_name = "memory"

    def __init__(self) -> None:
        super().__init__()
        self._users: Dict[int, Dict[str, Any]] = {}
        self._operations: Dict[str, Dict[str, Any]] = {}
        self._user_ops: Dict[int, list[Dict[str, Any]]] = {}
        self._promo_first_owner: Dict[str, int] = {}
        self._promo_pairs: set[tuple[int, str]] = set()
        self._lock = threading.RLock()

    def _ensure_user(self, uid: int) -> Dict[str, Any]:
        user = self._users.get(uid)
        if user is None:
            user = {"balance": 0, "signup_bonus_granted": False}
            self._users[uid] = user
            self._user_ops.setdefault(uid, [])
        return user

    def ping(self) -> bool:  # pragma: no cover - trivial
        return True

    def get_promo_owner(self, promo_code: str) -> Optional[int]:
        with self._lock:
            owner = self._promo_first_owner.get(promo_code)
            return int(owner) if owner is not None else None

    def get_balance(self, uid: int) -> int:
        with self._lock:
            user = self._ensure_user(uid)
            return int(user.get("balance", 0))

    def credit(
        self,
        uid: int,
        amount: int,
        reason: str,
        op_id: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> LedgerOpResult:
        amount = int(amount)
        meta_dict = dict(meta or {})
        with self._lock:
            user = self._ensure_user(uid)
            old_balance = int(user.get("balance", 0))
            if amount <= 0:
                return LedgerOpResult(False, old_balance, op_id, reason, old_balance, duplicate=True)
            if op_id in self._operations:
                return LedgerOpResult(False, old_balance, op_id, reason, old_balance, duplicate=True)

            new_balance = old_balance + amount
            user["balance"] = new_balance

            entry = {
                "user_id": uid,
                "type": "credit",
                "amount": amount,
                "reason": reason,
                "op_id": op_id,
                "meta": meta_dict,
            }
            self._operations[op_id] = entry
            self._user_ops.setdefault(uid, []).append(entry)

        self._log_operation("credit", uid, op_id, amount, reason, old_balance, new_balance, meta_dict)
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
        meta_dict = dict(meta or {})
        with self._lock:
            user = self._ensure_user(uid)
            old_balance = int(user.get("balance", 0))
            if amount <= 0:
                return LedgerOpResult(False, old_balance, op_id, reason, old_balance, duplicate=True)
            if op_id in self._operations:
                return LedgerOpResult(False, old_balance, op_id, reason, old_balance, duplicate=True)
            if old_balance < amount:
                raise InsufficientBalance(old_balance, amount)

            new_balance = old_balance - amount
            user["balance"] = new_balance

            entry = {
                "user_id": uid,
                "type": "debit",
                "amount": amount,
                "reason": reason,
                "op_id": op_id,
                "meta": meta_dict,
            }
            self._operations[op_id] = entry
            self._user_ops.setdefault(uid, []).append(entry)

        self._log_operation("debit", uid, op_id, amount, reason, old_balance, new_balance, meta_dict)
        return LedgerOpResult(True, new_balance, op_id, reason, old_balance)

    def grant_signup_bonus(
        self, uid: int, amount: int, meta: Optional[Dict[str, Any]] = None
    ) -> LedgerOpResult:
        amount = int(amount)
        meta_dict = dict(meta or {})
        op_id = f"signup:{uid}"
        with self._lock:
            user = self._ensure_user(uid)
            old_balance = int(user.get("balance", 0))
            if user.get("signup_bonus_granted"):
                return LedgerOpResult(False, old_balance, op_id, "signup_bonus", old_balance, duplicate=True)
            if op_id in self._operations or amount <= 0:
                user["signup_bonus_granted"] = True
                return LedgerOpResult(False, old_balance, op_id, "signup_bonus", old_balance, duplicate=True)

            new_balance = old_balance + amount
            user["balance"] = new_balance
            user["signup_bonus_granted"] = True

            entry = {
                "user_id": uid,
                "type": "credit",
                "amount": amount,
                "reason": "signup_bonus",
                "op_id": op_id,
                "meta": meta_dict,
            }
            self._operations[op_id] = entry
            self._user_ops.setdefault(uid, []).append(entry)

        self._log_operation("credit", uid, op_id, amount, "signup_bonus", old_balance, new_balance, meta_dict)
        return LedgerOpResult(True, new_balance, op_id, "signup_bonus", old_balance)

    def apply_promo(
        self,
        uid: int,
        promo_code: str,
        amount: int,
        meta: Optional[Dict[str, Any]] = None,
    ) -> LedgerOpResult:
        amount = int(amount)
        meta_dict = dict(meta or {})
        meta_dict.setdefault("promo_code", promo_code)
        op_id = f"promo:{promo_code}:{uid}"
        with self._lock:
            user = self._ensure_user(uid)
            old_balance = int(user.get("balance", 0))
            owner = self._promo_first_owner.get(promo_code)
            if owner and owner != uid:
                return LedgerOpResult(False, old_balance, op_id, "promo", old_balance, duplicate=True)
            if op_id in self._operations:
                return LedgerOpResult(False, old_balance, op_id, "promo", old_balance, duplicate=True)
            if (uid, promo_code) in self._promo_pairs:
                return LedgerOpResult(False, old_balance, op_id, "promo", old_balance, duplicate=True)

            if owner is None:
                self._promo_first_owner[promo_code] = uid
            self._promo_pairs.add((uid, promo_code))

            if amount <= 0:
                return LedgerOpResult(False, old_balance, op_id, "promo", old_balance, duplicate=True)

            new_balance = old_balance + amount
            user["balance"] = new_balance

            entry = {
                "user_id": uid,
                "type": "credit",
                "amount": amount,
                "reason": "promo",
                "op_id": op_id,
                "meta": meta_dict,
            }
            self._operations[op_id] = entry
            self._user_ops.setdefault(uid, []).append(entry)

        self._log_operation("credit", uid, op_id, amount, "promo", old_balance, new_balance, meta_dict)
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
            entry = self._operations.get(old_op_id)
            if not entry or new_op_id in self._operations:
                return False
            if extra_meta:
                meta = dict(entry.get("meta") or {})
                meta.update(extra_meta)
                entry["meta"] = meta
            entry["op_id"] = new_op_id
            self._operations[new_op_id] = entry
            del self._operations[old_op_id]
        log.info("ledger rename op_id %s -> %s", old_op_id, new_op_id)
        return True

    def recalc_user_balance(self, uid: int) -> BalanceRecalcResult:
        with self._lock:
            user = self._ensure_user(uid)
            previous = int(user.get("balance", 0))
            calculated = 0
            for entry in self._user_ops.get(uid, []):
                if entry.get("type") == "credit":
                    calculated += int(entry.get("amount", 0))
                else:
                    calculated -= int(entry.get("amount", 0))
            updated = calculated != previous
            if updated:
                user["balance"] = calculated
        if updated:
            log.info("ledger recalc user=%s previous=%s calculated=%s", uid, previous, calculated)
        return BalanceRecalcResult(previous=previous, calculated=calculated, updated=updated)


class _PostgresLedger(_BaseLedgerBackend):
    """PostgreSQL-backed ledger implementation."""

    backend_name = "postgres"

    def __init__(self, dsn: str) -> None:
        super().__init__()
        if not dsn:
            raise RuntimeError("DATABASE_URL is required for ledger storage")
        self.dsn = dsn
        self.safe_dsn = _sanitize_postgres_dsn(dsn)
        self.pool = ConnectionPool(conninfo=dsn, max_size=10, kwargs={"autocommit": False})
        self.pool.wait()
        self._prepare()

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

    @staticmethod
    def _ensure_user(cur: psycopg.Cursor[Any], uid: int) -> None:
        cur.execute("INSERT INTO users (id) VALUES (%s) ON CONFLICT DO NOTHING", (uid,))

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
        op_id = f"signup:{uid}"
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

                cur.execute(
                    """
                    SELECT 1 FROM ledger WHERE op_id=%s
                    """,
                    (op_id,),
                )
                if cur.fetchone():
                    cur.execute(
                        "UPDATE users SET signup_bonus_granted = TRUE WHERE id=%s",
                        (uid,),
                    )
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

                cur.execute(
                    """
                    INSERT INTO ledger (user_id, type, amount, reason, op_id, meta)
                    VALUES (%s, 'credit', %s, %s, %s, COALESCE(%s::jsonb, '{}'::jsonb))
                    """,
                    (uid, amount, "signup_bonus", op_id, meta_json),
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


class _SQLiteLedger(_BaseLedgerBackend):
    """SQLite-backed ledger implementation."""

    backend_name = "sqlite"

    def __init__(self, database: str, safe_dsn: str) -> None:
        super().__init__()
        self._database = database
        self.safe_dsn = safe_dsn
        self._lock = threading.RLock()
        self._prepare()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self._database,
            timeout=30,
            isolation_level=None,
            check_same_thread=False,
            detect_types=sqlite3.PARSE_DECLTYPES,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=60000")
        return conn

    def _prepare(self) -> None:
        if self._database != ":memory:":
            try:
                Path(self._database).expanduser().parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
        conn = self._connect()
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    balance INTEGER NOT NULL DEFAULT 0 CHECK (balance >= 0),
                    signup_bonus_granted INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ledger (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    type TEXT NOT NULL CHECK (type IN ('credit','debit')),
                    amount INTEGER NOT NULL CHECK (amount >= 0),
                    reason TEXT NOT NULL,
                    op_id TEXT NOT NULL UNIQUE,
                    meta TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_ledger_user_created_at
                    ON ledger(user_id, created_at DESC)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS promo_usages (
                    user_id INTEGER NOT NULL,
                    promo_id TEXT NOT NULL,
                    used_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, promo_id),
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
                """
            )
        finally:
            conn.close()

    @contextlib.contextmanager
    def _transaction(self):
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                cur = conn.cursor()
                try:
                    yield cur
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
                finally:
                    cur.close()
            finally:
                conn.close()

    @contextlib.contextmanager
    def _read_cursor(self):
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.cursor()
                try:
                    yield cur
                finally:
                    cur.close()
            finally:
                conn.close()

    @staticmethod
    def _ensure_user(cur: sqlite3.Cursor, uid: int) -> None:
        cur.execute("INSERT OR IGNORE INTO users (id) VALUES (?)", (uid,))

    def ping(self) -> bool:
        try:
            with self._read_cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
            return True
        except Exception:
            log.exception("ledger ping failed")
            return False

    def get_promo_owner(self, promo_code: str) -> Optional[int]:
        with self._read_cursor() as cur:
            cur.execute(
                "SELECT user_id FROM promo_usages WHERE promo_id=? ORDER BY used_at ASC LIMIT 1",
                (promo_code,),
            )
            row = cur.fetchone()
            return int(row["user_id"]) if row else None

    def get_balance(self, uid: int) -> int:
        with self._transaction() as cur:
            self._ensure_user(cur, uid)
            cur.execute("SELECT balance FROM users WHERE id=?", (uid,))
            row = cur.fetchone()
            return int(row["balance"]) if row else 0

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
        with self._transaction() as cur:
            self._ensure_user(cur, uid)
            cur.execute("SELECT balance FROM users WHERE id=?", (uid,))
            row = cur.fetchone()
            old_balance = int(row["balance"]) if row else 0

            cur.execute("SELECT 1 FROM ledger WHERE op_id=?", (op_id,))
            if cur.fetchone():
                return LedgerOpResult(False, old_balance, op_id, reason, old_balance, duplicate=True)

            cur.execute(
                "UPDATE users SET balance = balance + ?, updated_at = CURRENT_TIMESTAMP WHERE id=?",
                (amount, uid),
            )
            cur.execute(
                "INSERT INTO ledger (user_id, type, amount, reason, op_id, meta) VALUES (?, 'credit', ?, ?, ?, ?)",
                (uid, amount, reason, op_id, meta_json),
            )
            cur.execute("SELECT balance FROM users WHERE id=?", (uid,))
            new_balance_row = cur.fetchone()
            new_balance = int(new_balance_row["balance"]) if new_balance_row else old_balance
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
        with self._transaction() as cur:
            self._ensure_user(cur, uid)
            cur.execute("SELECT balance FROM users WHERE id=?", (uid,))
            row = cur.fetchone()
            old_balance = int(row["balance"]) if row else 0

            cur.execute("SELECT 1 FROM ledger WHERE op_id=?", (op_id,))
            if cur.fetchone():
                return LedgerOpResult(False, old_balance, op_id, reason, old_balance, duplicate=True)

            if old_balance < amount:
                raise InsufficientBalance(old_balance, amount)

            cur.execute(
                "UPDATE users SET balance = balance - ?, updated_at = CURRENT_TIMESTAMP WHERE id=?",
                (amount, uid),
            )
            cur.execute(
                "INSERT INTO ledger (user_id, type, amount, reason, op_id, meta) VALUES (?, 'debit', ?, ?, ?, ?)",
                (uid, amount, reason, op_id, meta_json),
            )
            cur.execute("SELECT balance FROM users WHERE id=?", (uid,))
            new_balance_row = cur.fetchone()
            new_balance = int(new_balance_row["balance"]) if new_balance_row else old_balance
        self._log_operation("debit", uid, op_id, amount, reason, old_balance, new_balance, meta)
        return LedgerOpResult(True, new_balance, op_id, reason, old_balance)

    def grant_signup_bonus(
        self, uid: int, amount: int, meta: Optional[Dict[str, Any]] = None
    ) -> LedgerOpResult:
        amount = int(amount)
        op_id = f"signup:{uid}"
        meta_json = self._json_meta(meta)
        with self._transaction() as cur:
            self._ensure_user(cur, uid)
            cur.execute(
                "SELECT balance, signup_bonus_granted FROM users WHERE id=?",
                (uid,),
            )
            row = cur.fetchone()
            if row:
                old_balance = int(row["balance"])
                already = bool(row["signup_bonus_granted"])
            else:
                old_balance = 0
                already = False

            if already:
                return LedgerOpResult(False, old_balance, op_id, "signup_bonus", old_balance, duplicate=True)

            cur.execute("SELECT 1 FROM ledger WHERE op_id=?", (op_id,))
            if cur.fetchone() or amount <= 0:
                cur.execute(
                    "UPDATE users SET signup_bonus_granted = 1 WHERE id=?",
                    (uid,),
                )
                return LedgerOpResult(False, old_balance, op_id, "signup_bonus", old_balance, duplicate=True)

            cur.execute(
                """
                UPDATE users
                   SET balance = balance + ?,
                       signup_bonus_granted = 1,
                       updated_at = CURRENT_TIMESTAMP
                 WHERE id=?
                """,
                (amount, uid),
            )
            cur.execute(
                "INSERT INTO ledger (user_id, type, amount, reason, op_id, meta) VALUES (?, 'credit', ?, 'signup_bonus', ?, ?)",
                (uid, amount, op_id, meta_json),
            )
            cur.execute("SELECT balance FROM users WHERE id=?", (uid,))
            new_balance_row = cur.fetchone()
            new_balance = int(new_balance_row["balance"]) if new_balance_row else old_balance
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
        amount = int(amount)
        op_id = f"promo:{promo_code}:{uid}"
        meta_dict = dict(meta or {})
        meta_dict.setdefault("promo_code", promo_code)
        meta_json = self._json_meta(meta_dict)
        with self._transaction() as cur:
            self._ensure_user(cur, uid)
            cur.execute("SELECT balance FROM users WHERE id=?", (uid,))
            row = cur.fetchone()
            old_balance = int(row["balance"]) if row else 0

            cur.execute(
                "SELECT user_id FROM promo_usages WHERE promo_id=? ORDER BY used_at ASC LIMIT 1",
                (promo_code,),
            )
            owner_row = cur.fetchone()
            if owner_row:
                owner = int(owner_row["user_id"])
                if owner != uid:
                    return LedgerOpResult(False, old_balance, op_id, "promo", old_balance, duplicate=True)

            cur.execute("SELECT 1 FROM ledger WHERE op_id=?", (op_id,))
            if cur.fetchone():
                return LedgerOpResult(False, old_balance, op_id, "promo", old_balance, duplicate=True)

            cur.execute(
                "INSERT OR IGNORE INTO promo_usages (user_id, promo_id) VALUES (?, ?)",
                (uid, promo_code),
            )
            if cur.rowcount == 0:
                return LedgerOpResult(False, old_balance, op_id, "promo", old_balance, duplicate=True)

            if amount <= 0:
                return LedgerOpResult(False, old_balance, op_id, "promo", old_balance, duplicate=True)

            cur.execute(
                "UPDATE users SET balance = balance + ?, updated_at = CURRENT_TIMESTAMP WHERE id=?",
                (amount, uid),
            )
            cur.execute(
                "INSERT INTO ledger (user_id, type, amount, reason, op_id, meta) VALUES (?, 'credit', ?, 'promo', ?, ?)",
                (uid, amount, op_id, meta_json),
            )
            cur.execute("SELECT balance FROM users WHERE id=?", (uid,))
            new_balance_row = cur.fetchone()
            new_balance = int(new_balance_row["balance"]) if new_balance_row else old_balance
        self._log_operation("credit", uid, op_id, amount, "promo", old_balance, new_balance, meta_dict)
        return LedgerOpResult(True, new_balance, op_id, "promo", old_balance)

    def rename_operation(
        self,
        old_op_id: str,
        new_op_id: str,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if old_op_id == new_op_id:
            return True
        try:
            with self._transaction() as cur:
                cur.execute("SELECT meta FROM ledger WHERE op_id=?", (old_op_id,))
                row = cur.fetchone()
                if not row:
                    return False
                meta_value = row["meta"] if row else None
                merged_meta = self._merge_meta(meta_value, extra_meta)
                if merged_meta is not None:
                    cur.execute(
                        "UPDATE ledger SET op_id=?, meta=? WHERE op_id=?",
                        (new_op_id, merged_meta, old_op_id),
                    )
                else:
                    cur.execute(
                        "UPDATE ledger SET op_id=? WHERE op_id=?",
                        (new_op_id, old_op_id),
                    )
                updated = cur.rowcount
        except sqlite3.IntegrityError:
            log.warning("ledger rename conflict for %s -> %s", old_op_id, new_op_id)
            return False
        if updated:
            log.info("ledger rename op_id %s -> %s", old_op_id, new_op_id)
            return True
        return False

    def recalc_user_balance(self, uid: int) -> BalanceRecalcResult:
        with self._transaction() as cur:
            self._ensure_user(cur, uid)
            cur.execute("SELECT balance FROM users WHERE id=?", (uid,))
            row = cur.fetchone()
            previous = int(row["balance"]) if row else 0

            cur.execute(
                """
                SELECT COALESCE(SUM(CASE WHEN type='credit' THEN amount ELSE -amount END), 0) AS total
                  FROM ledger
                 WHERE user_id = ?
                """,
                (uid,),
            )
            calc_row = cur.fetchone()
            total = calc_row["total"] if calc_row and calc_row["total"] is not None else 0
            calculated = int(total)
            updated = calculated != previous
            if updated:
                cur.execute(
                    "UPDATE users SET balance = ?, updated_at = CURRENT_TIMESTAMP WHERE id=?",
                    (calculated, uid),
                )
        if updated:
            log.info("ledger recalc user=%s previous=%s calculated=%s", uid, previous, calculated)
        return BalanceRecalcResult(previous=previous, calculated=calculated, updated=updated)


class LedgerStorage:
    """Facade that selects the appropriate ledger backend."""

    def __init__(self, dsn: Optional[str]) -> None:
        self.dsn = dsn
        self._impl = self._create_backend(dsn)
        self.backend_name = self._impl.backend_name
        self.safe_dsn = self._impl.safe_dsn
        self.pool = getattr(self._impl, "pool", None)

    @staticmethod
    def _create_backend(dsn: Optional[str]) -> _BaseLedgerBackend:
        if dsn is None:
            log.warning(
                "DATABASE_URL/POSTGRES_DSN not provided. Using in-memory ledger backend."
            )
            return _MemoryLedger()

        normalized = dsn.strip()
        if not normalized:
            log.warning(
                "DATABASE_URL/POSTGRES_DSN empty. Using in-memory ledger backend."
            )
            return _MemoryLedger()

        split = urlsplit(normalized)
        scheme = (split.scheme or "").lower()
        base_scheme = scheme.split("+", 1)[0] if scheme else ""

        if base_scheme in {"postgres", "postgresql"}:
            return _PostgresLedger(normalized)

        if base_scheme in {"sqlite", "file"}:
            if split.path in {":memory:", "/:memory:"} and not split.netloc:
                log.warning(
                    "SQLite :memory: DSN detected. Falling back to in-memory ledger backend."
                )
                return _MemoryLedger()
            database, safe = _resolve_sqlite_target(normalized, split)
            return _SQLiteLedger(database, safe)

        if not scheme:
            try:
                database, safe = _resolve_sqlite_target(f"sqlite:///{normalized}")
                return _SQLiteLedger(database, safe)
            except Exception:
                pass

        raise RuntimeError(f"Unsupported DATABASE_URL/POSTGRES_DSN value: {dsn}")

    # Public API delegated to the concrete backend
    def ping(self) -> bool:
        return self._impl.ping()

    def get_promo_owner(self, promo_code: str) -> Optional[int]:
        return self._impl.get_promo_owner(promo_code)

    def get_balance(self, uid: int) -> int:
        return self._impl.get_balance(uid)

    def credit(
        self,
        uid: int,
        amount: int,
        reason: str,
        op_id: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> LedgerOpResult:
        return self._impl.credit(uid, amount, reason, op_id, meta)

    def debit(
        self,
        uid: int,
        amount: int,
        reason: str,
        op_id: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> LedgerOpResult:
        return self._impl.debit(uid, amount, reason, op_id, meta)

    def grant_signup_bonus(
        self, uid: int, amount: int, meta: Optional[Dict[str, Any]] = None
    ) -> LedgerOpResult:
        return self._impl.grant_signup_bonus(uid, amount, meta)

    def apply_promo(
        self,
        uid: int,
        promo_code: str,
        amount: int,
        meta: Optional[Dict[str, Any]] = None,
    ) -> LedgerOpResult:
        return self._impl.apply_promo(uid, promo_code, amount, meta)

    def rename_operation(
        self,
        old_op_id: str,
        new_op_id: str,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> bool:
        return self._impl.rename_operation(old_op_id, new_op_id, extra_meta)

    def recalc_user_balance(self, uid: int) -> BalanceRecalcResult:
        return self._impl.recalc_user_balance(uid)


__all__ = [
    "LedgerOpResult",
    "BalanceRecalcResult",
    "InsufficientBalance",
    "LedgerStorage",
]
