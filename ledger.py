# -*- coding: utf-8 -*-
"""Persistent ledger storage for user balances."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, TypeVar

from core.db.postgres import normalize_dsn

try:  # psycopg is optional when using the in-memory backend
    import psycopg
    from psycopg.errors import UniqueViolation
    from psycopg_pool import ConnectionPool
except ImportError:  # pragma: no cover - optional dependency handling
    psycopg = None  # type: ignore
    ConnectionPool = None  # type: ignore

    class UniqueViolation(Exception):  # type: ignore[override]
        pass

log = logging.getLogger(__name__)

T = TypeVar("T")


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
        self._dsn = normalize_dsn(dsn)
        self.log = log
        self._pool: Optional[ConnectionPool] = None
        self._pool_lock = threading.RLock()
        self._default_retries = 3

    # ------------------------------------------------------------------
    #   Lifecycle helpers
    # ------------------------------------------------------------------
    def _read_pool_setting(self, name: str, default: int, *, minimum: int = 1) -> int:
        raw = os.getenv(name)
        if raw is None:
            return max(default, minimum)
        try:
            value = int(raw)
        except (TypeError, ValueError):
            self.log.warning(
                "ledger.pool.env_invalid", extra={"name": name, "value": raw}
            )
            return max(default, minimum)
        if value < minimum:
            return minimum
        return value

    def start(self) -> None:
        if psycopg is None or ConnectionPool is None:
            raise RuntimeError("psycopg not available")
        min_size = self._read_pool_setting("PG_POOL_MIN", 1)
        max_size = self._read_pool_setting("PG_POOL_MAX", 10, minimum=min_size)
        max_idle = self._read_pool_setting("PG_POOL_MAX_IDLE", 30, minimum=0)
        with self._pool_lock:
            if self._pool is not None:
                return
            pool = ConnectionPool(
                conninfo=self._dsn,
                min_size=min_size,
                max_size=max_size,
                max_idle=max_idle,
                timeout=10,
                open=False,
            )
            try:
                pool.open()
                with pool.connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1")
                self._pool = pool
            except Exception:
                try:
                    pool.close()
                    pool.wait(timeout=5)
                except Exception:
                    pass
                raise
        self.log.info("Postgres connected and validated")
        self.log.info(
            "ledger.pool.started",
            extra={"min_size": min_size, "max_size": max_size, "max_idle": max_idle},
        )
        self._prepare()

    def stop(self) -> None:
        with self._pool_lock:
            pool = self._pool
            self._pool = None
        if not pool:
            return
        try:
            pool.close()
            pool.wait(timeout=5)
        except Exception as exc:  # pragma: no cover - defensive
            self.log.warning("ledger.pool.close_failed", extra={"error": str(exc)})
        else:
            self.log.info("ledger.pool.closed")

    # ------------------------------------------------------------------
    #   Internal helpers
    # ------------------------------------------------------------------
    def _ensure_pool(self) -> ConnectionPool:
        if self._pool is None:
            self.start()
        if self._pool is None:
            raise RuntimeError("Postgres connection pool is not available")
        return self._pool

    def _with_connection(
        self,
        fn: Callable[[psycopg.Connection], T],
        *,
        op: str,
        retries: Optional[int] = None,
        **ctx: Any,
    ) -> T:
        attempts = max(int(retries or self._default_retries), 1)
        last_exc: Optional[BaseException] = None
        for attempt in range(1, attempts + 1):
            try:
                pool = self._ensure_pool()
                with pool.connection() as conn:
                    result = fn(conn)
                if attempt > 1:
                    self.log.info(
                        "ledger.db.retry_ok",
                        extra={"op": op, "attempt": attempt, **ctx},
                    )
                return result
            except psycopg.OperationalError as exc:
                last_exc = exc
                if attempt == attempts:
                    break
                self.log.warning(
                    "ledger.db.retry",
                    extra={"op": op, "attempt": attempt, "error": str(exc), **ctx},
                )
                time.sleep(min(0.2 * attempt, 1.0))
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"ledger operation {op} failed without exception")

    def _prepare(self) -> None:
        ddl_users = """
        CREATE TABLE IF NOT EXISTS users (
            id BIGINT PRIMARY KEY,
            username TEXT,
            referrer_id BIGINT,
            joined_at TIMESTAMPTZ DEFAULT now(),
            updated_at TIMESTAMPTZ DEFAULT now(),
            referral_earned_total BIGINT NOT NULL DEFAULT 0
        )
        """

        ddl_balances = """
        CREATE TABLE IF NOT EXISTS balances (
            user_id BIGINT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
            tokens BIGINT NOT NULL DEFAULT 0 CHECK (tokens >= 0),
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

        ddl_transactions = """
        CREATE TABLE IF NOT EXISTS transactions (
            id BIGSERIAL PRIMARY KEY,
            user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            type TEXT NOT NULL,
            amount BIGINT NOT NULL,
            reason TEXT,
            key TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """

        ddl_transactions_idx = """
        CREATE INDEX IF NOT EXISTS idx_transactions_user_id
            ON transactions(user_id)
        """

        ddl_transactions_migrations = [
            """
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1 FROM information_schema.columns
                     WHERE table_name = 'transactions' AND column_name = 'direction'
                ) AND NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                     WHERE table_name = 'transactions' AND column_name = 'type'
                ) THEN
                    ALTER TABLE transactions RENAME COLUMN direction TO type;
                END IF;
            END;
            $$;
            """,
            "ALTER TABLE transactions ADD COLUMN IF NOT EXISTS type TEXT",
            "UPDATE transactions SET type = COALESCE(type, 'credit') WHERE type IS NULL",
            "ALTER TABLE transactions ALTER COLUMN type SET NOT NULL",
            "ALTER TABLE transactions DROP CONSTRAINT IF EXISTS transactions_direction_check",
            "ALTER TABLE transactions DROP CONSTRAINT IF EXISTS transactions_amount_check",
            "ALTER TABLE transactions ALTER COLUMN amount TYPE BIGINT",
            "ALTER TABLE transactions ALTER COLUMN amount SET NOT NULL",
            "ALTER TABLE transactions ADD COLUMN IF NOT EXISTS reason TEXT",
            "ALTER TABLE transactions ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT now()",
            "ALTER TABLE transactions ALTER COLUMN created_at SET DEFAULT now()",
            "ALTER TABLE transactions ALTER COLUMN created_at SET NOT NULL",
            "ALTER TABLE transactions ADD COLUMN IF NOT EXISTS key TEXT",
            "ALTER TABLE transactions ADD CONSTRAINT IF NOT EXISTS transactions_user_type_key_key UNIQUE (user_id, type, key)",
            """
            UPDATE transactions
               SET amount = CASE
                    WHEN type = 'debit' AND amount > 0 THEN -amount
                    ELSE amount
                END
             WHERE type = 'debit' AND amount > 0
            """,
        ]

        ddl_referrals = """
        CREATE TABLE IF NOT EXISTS referrals (
            id BIGSERIAL PRIMARY KEY,
            referrer_id BIGINT REFERENCES users(id) ON DELETE CASCADE,
            referred_id BIGINT REFERENCES users(id) ON DELETE CASCADE,
            created_at TIMESTAMPTZ DEFAULT now(),
            earned_tokens BIGINT NOT NULL DEFAULT 0,
            UNIQUE (referrer_id, referred_id)
        )
        """

        ddl_audit = """
        CREATE TABLE IF NOT EXISTS audit_log (
            id BIGSERIAL PRIMARY KEY,
            user_id BIGINT,
            actor_id BIGINT,
            action TEXT,
            amount BIGINT,
            meta JSONB NOT NULL DEFAULT '{}'::jsonb,
            ts TIMESTAMPTZ DEFAULT now()
        )
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

        def operation(conn: psycopg.Connection) -> None:
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.execute(ddl_users)
                    cur.execute(ddl_balances)
                    cur.execute(ddl_ledger)
                    cur.execute(ddl_ledger_idx)
                    cur.execute(ddl_transactions)
                    cur.execute(ddl_transactions_idx)
                    for statement in ddl_transactions_migrations:
                        cur.execute(statement)
                    cur.execute(ddl_referrals)
                    cur.execute(ddl_audit)
                    cur.execute(ddl_promo)

        self._with_connection(operation, op="prepare")

    @staticmethod
    def _ensure_user(
        cur: psycopg.Cursor[Any],
        uid: int,
        *,
        username: Optional[str] = None,
        referrer_id: Optional[int] = None,
    ) -> None:
        cur.execute(
            """
            INSERT INTO users (id, username, referrer_id)
            VALUES (%s, %s, %s)
            ON CONFLICT (id) DO UPDATE
               SET username = COALESCE(EXCLUDED.username, users.username),
                   referrer_id = COALESCE(users.referrer_id, EXCLUDED.referrer_id),
                   updated_at = now()
            """,
            (uid, username, referrer_id),
        )
        cur.execute(
            "INSERT INTO balances (user_id) VALUES (%s) ON CONFLICT DO NOTHING",
            (uid,),
        )

    @staticmethod
    def _record_transaction(
        cur: psycopg.Cursor[Any],
        uid: int,
        tx_type: str,
        amount: int,
        reason: str,
        *,
        key: Optional[str] = None,
    ) -> None:
        amount_val = int(amount)
        if amount_val == 0:
            return
        tx_type_norm = (tx_type or "").strip().lower() or "adjustment"
        reason_text = reason or tx_type_norm
        if tx_type_norm == "debit":
            signed_amount = -abs(amount_val)
        elif tx_type_norm == "credit":
            signed_amount = abs(amount_val)
        else:
            signed_amount = amount_val
        cur.execute(
            """
            INSERT INTO transactions (user_id, type, amount, reason, key)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (user_id, type, key) DO NOTHING
            """,
            (uid, tx_type_norm, signed_amount, reason_text, key),
        )

    def diag_slow(self) -> list[Dict[str, Any]]:
        diag_user_id = -1
        probes: list[tuple[str, str, Callable[[], tuple[Any, ...]], bool]] = [
            ("select_1", "SELECT 1", lambda: (), True),
            ("count_users", "SELECT COUNT(*) FROM users", lambda: (), True),
            (
                "touch_user",
                """
                INSERT INTO users (id, username, updated_at)
                VALUES (%s, %s, now())
                ON CONFLICT (id) DO UPDATE
                    SET username = EXCLUDED.username,
                        updated_at = now()
                """,
                lambda: (diag_user_id, "diag_slow"),
                False,
            ),
        ]

        results: list[Dict[str, Any]] = []
        for name, sql, params_factory, fetch in probes:
            attempts = {"value": 0}
            started = time.perf_counter()

            def runner(conn: psycopg.Connection) -> Any:
                attempts["value"] += 1
                with conn.transaction():
                    with conn.cursor() as cur:
                        params = params_factory()
                        cur.execute(sql, params)
                        if fetch:
                            return cur.fetchone()
                        return None

            try:
                row = self._with_connection(runner, op="diag", query=name)
                duration = time.perf_counter() - started
                entry: Dict[str, Any] = {
                    "query": name,
                    "duration": duration,
                    "attempts": attempts["value"],
                    "retries": max(attempts["value"] - 1, 0),
                }
                if fetch and row is not None:
                    entry["result"] = row[0] if isinstance(row, tuple) else row
                results.append(entry)
            except Exception as exc:  # pragma: no cover - diagnostics only
                duration = time.perf_counter() - started
                results.append(
                    {
                        "query": name,
                        "duration": duration,
                        "attempts": attempts["value"],
                        "retries": max(attempts["value"] - 1, 0),
                        "error": str(exc),
                    }
                )
        return results

    # ------------------------------------------------------------------
    #   Public API
    # ------------------------------------------------------------------
    def ping(self) -> bool:
        def operation(conn: psycopg.Connection) -> None:
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()

        try:
            self._with_connection(operation, op="ping")
            return True
        except Exception:
            log.exception("ledger ping failed")
            return False

    def get_promo_owner(self, promo_code: str) -> Optional[int]:
        def operation(conn: psycopg.Connection) -> Optional[int]:
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT user_id FROM promo_usages WHERE promo_id=%s ORDER BY used_at ASC LIMIT 1",
                        (promo_code,),
                    )
                    row = cur.fetchone()
                    return int(row[0]) if row else None

        return self._with_connection(operation, op="get_promo_owner", promo=promo_code)

    def get_balance(self, uid: int) -> int:
        def operation(conn: psycopg.Connection) -> int:
            with conn.transaction():
                with conn.cursor() as cur:
                    self._ensure_user(cur, uid)
                    cur.execute("SELECT tokens FROM balances WHERE user_id=%s", (uid,))
                    row = cur.fetchone()
                    return int(row[0]) if row else 0

        return self._with_connection(operation, op="get_balance", uid=uid)

    def ensure_user(
        self,
        uid: int,
        *,
        username: Optional[str] = None,
        referrer_id: Optional[int] = None,
    ) -> None:
        def operation(conn: psycopg.Connection) -> None:
            with conn.transaction():
                with conn.cursor() as cur:
                    self._ensure_user(cur, uid, username=username, referrer_id=referrer_id)

        context: Dict[str, Any] = {"uid": uid}
        if username:
            context["username"] = username
        if referrer_id is not None:
            context["referrer_id"] = referrer_id
        self._with_connection(operation, op="ensure_user", **context)

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

        def operation(conn: psycopg.Connection) -> LedgerOpResult:
            with conn.transaction():
                with conn.cursor() as cur:
                    self._ensure_user(cur, uid)
                    cur.execute(
                        "SELECT tokens FROM balances WHERE user_id=%s FOR UPDATE",
                        (uid,),
                    )
                    row = cur.fetchone()
                    old_balance = int(row[0]) if row else 0

                    cur.execute("SELECT 1 FROM ledger WHERE op_id=%s", (op_id,))
                    if cur.fetchone():
                        return LedgerOpResult(
                            False, old_balance, op_id, reason, old_balance, duplicate=True
                        )

                    cur.execute(
                        """
                        UPDATE balances
                           SET tokens = tokens + %s,
                               updated_at = now()
                         WHERE user_id = %s
                         RETURNING tokens
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
                    self._record_transaction(cur, uid, "credit", amount, reason)

                    return LedgerOpResult(True, new_balance, op_id, reason, old_balance)

        result = self._with_connection(
            operation,
            op="credit",
            uid=uid,
            op_id=op_id,
            amount=amount,
        )
        if result.applied:
            self._log_operation(
                "credit", uid, op_id, amount, reason, result.old_balance, result.balance, meta
            )
        return result

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

        def operation(conn: psycopg.Connection) -> LedgerOpResult:
            with conn.transaction():
                with conn.cursor() as cur:
                    self._ensure_user(cur, uid)
                    cur.execute(
                        "SELECT tokens FROM balances WHERE user_id=%s FOR UPDATE",
                        (uid,),
                    )
                    row = cur.fetchone()
                    old_balance = int(row[0]) if row else 0

                    cur.execute("SELECT 1 FROM ledger WHERE op_id=%s", (op_id,))
                    if cur.fetchone():
                        return LedgerOpResult(
                            False, old_balance, op_id, reason, old_balance, duplicate=True
                        )

                    if old_balance < amount:
                        raise InsufficientBalance(old_balance, amount)

                    cur.execute(
                        """
                        UPDATE balances
                           SET tokens = tokens - %s,
                               updated_at = now()
                         WHERE user_id = %s
                         RETURNING tokens
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
                    self._record_transaction(cur, uid, "debit", amount, reason)

                    return LedgerOpResult(True, new_balance, op_id, reason, old_balance)

        result = self._with_connection(
            operation,
            op="debit",
            uid=uid,
            op_id=op_id,
            amount=amount,
        )
        if result.applied:
            self._log_operation(
                "debit", uid, op_id, -amount, reason, result.old_balance, result.balance, meta
            )
        return result

    def grant_signup_bonus(
        self, uid: int, amount: int, meta: Optional[Dict[str, Any]] = None
    ) -> LedgerOpResult:
        amount = int(amount)
        op_id = f"signup:{uid}"
        meta_json = self._json_meta(meta)
        
        def operation(conn: psycopg.Connection) -> LedgerOpResult:
            with conn.transaction():
                with conn.cursor() as cur:
                    self._ensure_user(cur, uid)
                    cur.execute(
                        "SELECT tokens, signup_bonus_granted FROM balances WHERE user_id=%s FOR UPDATE",
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
                        return LedgerOpResult(
                            False,
                            old_balance,
                            op_id,
                            "signup_bonus",
                            old_balance,
                            duplicate=True,
                        )

                    cur.execute("SELECT 1 FROM ledger WHERE op_id=%s", (op_id,))
                    if cur.fetchone():
                        cur.execute(
                            "UPDATE balances SET signup_bonus_granted = TRUE WHERE user_id=%s",
                            (uid,),
                        )
                        return LedgerOpResult(
                            False,
                            old_balance,
                            op_id,
                            "signup_bonus",
                            old_balance,
                            duplicate=True,
                        )

                    if amount <= 0:
                        cur.execute(
                            "UPDATE balances SET signup_bonus_granted = TRUE WHERE user_id=%s",
                            (uid,),
                        )
                        return LedgerOpResult(
                            False,
                            old_balance,
                            op_id,
                            "signup_bonus",
                            old_balance,
                            duplicate=True,
                        )

                    cur.execute(
                        """
                        UPDATE balances
                           SET tokens = tokens + %s,
                               signup_bonus_granted = TRUE,
                               updated_at = now()
                         WHERE user_id = %s
                         RETURNING tokens
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
                    self._record_transaction(
                        cur, uid, "credit", amount, "signup_bonus", key="signup"
                    )

                    return LedgerOpResult(True, new_balance, op_id, "signup_bonus", old_balance)

        result = self._with_connection(
            operation,
            op="grant_signup_bonus",
            uid=uid,
            op_id=op_id,
            amount=amount,
        )
        if result.applied:
            self._log_operation(
                "credit", uid, op_id, amount, "signup_bonus", result.old_balance, result.balance, meta
            )
        return result

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
        amount_value = int(amount)

        def operation(conn: psycopg.Connection) -> LedgerOpResult:
            with conn.transaction():
                with conn.cursor() as cur:
                    self._ensure_user(cur, uid)
                    cur.execute(
                        "SELECT tokens FROM balances WHERE user_id=%s FOR UPDATE",
                        (uid,),
                    )
                    row = cur.fetchone()
                    old_balance = int(row[0]) if row else 0

                    cur.execute(
                        "SELECT user_id FROM promo_usages WHERE promo_id=%s ORDER BY used_at ASC LIMIT 1",
                        (promo_code,),
                    )
                    owner_row = cur.fetchone()
                    if owner_row:
                        owner = int(owner_row[0])
                        if owner != uid:
                            return LedgerOpResult(
                                False,
                                old_balance,
                                op_id,
                                "promo",
                                old_balance,
                                duplicate=True,
                            )

                    cur.execute("SELECT 1 FROM ledger WHERE op_id=%s", (op_id,))
                    if cur.fetchone():
                        return LedgerOpResult(
                            False, old_balance, op_id, "promo", old_balance, duplicate=True
                        )

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
                        return LedgerOpResult(
                            False, old_balance, op_id, "promo", old_balance, duplicate=True
                        )

                    if amount_value <= 0:
                        return LedgerOpResult(
                            False, old_balance, op_id, "promo", old_balance, duplicate=True
                        )

                    cur.execute(
                        """
                        UPDATE balances
                           SET tokens = tokens + %s,
                               updated_at = now()
                         WHERE user_id = %s
                         RETURNING tokens
                        """,
                        (amount_value, uid),
                    )
                    new_balance_row = cur.fetchone()
                    new_balance = int(new_balance_row[0]) if new_balance_row else old_balance

                    cur.execute(
                        """
                        INSERT INTO ledger (user_id, type, amount, reason, op_id, meta)
                        VALUES (%s, 'credit', %s, %s, %s, COALESCE(%s::jsonb, '{}'::jsonb))
                        """,
                        (uid, amount_value, "promo", op_id, meta_json),
                    )
                    self._record_transaction(
                        cur,
                        uid,
                        "credit",
                        amount_value,
                        "promo",
                        key=f"promo:{promo_code}",
                    )

                    return LedgerOpResult(True, new_balance, op_id, "promo", old_balance)

        result = self._with_connection(
            operation,
            op="apply_promo",
            uid=uid,
            op_id=op_id,
            promo=promo_code,
            amount=amount_value,
        )
        if result.applied:
            self._log_operation(
                "credit", uid, op_id, amount_value, "promo", result.old_balance, result.balance, meta
            )
        return result

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
            def operation(conn: psycopg.Connection) -> int:
                with conn.transaction():
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
                        return cur.rowcount

            updated = self._with_connection(
                operation,
                op="rename_operation",
                old_op_id=old_op_id,
                new_op_id=new_op_id,
            )
            if updated:
                log.info("ledger rename op_id %s -> %s", old_op_id, new_op_id)
            return bool(updated)
        except UniqueViolation:
            log.warning("ledger rename conflict for %s -> %s", old_op_id, new_op_id)
            return False

    def recalc_user_balance(self, uid: int) -> BalanceRecalcResult:
        def operation(conn: psycopg.Connection) -> BalanceRecalcResult:
            with conn.transaction():
                with conn.cursor() as cur:
                    self._ensure_user(cur, uid)
                    cur.execute("SELECT tokens FROM balances WHERE user_id=%s", (uid,))
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
                            "UPDATE balances SET tokens = %s, updated_at = now() WHERE user_id = %s",
                            (calculated, uid),
                        )

                    return BalanceRecalcResult(
                        previous=previous, calculated=calculated, updated=updated
                    )

        result = self._with_connection(operation, op="recalc_user_balance", uid=uid)
        if result.updated:
            log.info(
                "ledger recalc user=%s previous=%s calculated=%s", uid, result.previous, result.calculated
            )
        return result


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

    def start(self) -> None:  # pragma: no cover - memory backend is eager
        return

    def stop(self) -> None:  # pragma: no cover - memory backend is eager
        return

    def get_promo_owner(self, promo_code: str) -> Optional[int]:
        with self._lock:
            owner = self._promo_first_owner.get(promo_code)
            return int(owner) if owner is not None else None

    def get_balance(self, uid: int) -> int:
        with self._lock:
            user = self._ensure_user(uid)
            return int(user["balance"])

    def ensure_user(
        self,
        uid: int,
        *,
        username: Optional[str] = None,
        referrer_id: Optional[int] = None,
    ) -> None:
        with self._lock:
            self._ensure_user(uid)

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
        op_id = f"signup:{uid}"
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
            forbid_memory = os.getenv("FORBID_MEMORY_DB", "").lower() in {"1", "true", "yes", "on"}
            if forbid_memory:
                raise RuntimeError("Memory ledger backend is disabled by configuration")
            self._impl = _MemoryLedgerStorage()
            self._started = True
        else:
            if not dsn:
                raise RuntimeError(
                    "DATABASE_URL (or POSTGRES_DSN) must be set for persistent ledger storage"
                )
            self._impl = _PostgresLedgerStorage(dsn)
            self._started = False

    def start(self) -> None:
        if hasattr(self._impl, "start"):
            self._impl.start()
        self._started = True

    def stop(self) -> None:
        if hasattr(self._impl, "stop"):
            try:
                self._impl.stop()
            finally:
                self._started = False
        else:
            self._started = False

    def __getattr__(self, name: str) -> Any:
        if not getattr(self, "_started", False) and hasattr(self._impl, "start"):
            self.start()
        return getattr(self._impl, name)
