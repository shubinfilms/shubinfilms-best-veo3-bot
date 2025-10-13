"""Utility helpers for working with the Neon PostgreSQL database."""

from __future__ import annotations

import contextlib
import json
import logging
import random
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, URL, make_url
from sqlalchemy.exc import OperationalError, SQLAlchemyError

try:  # psycopg is optional for some environments
    from psycopg.errors import OperationalError as PsycopgOperationalError
except Exception:  # pragma: no cover - optional dependency missing
    PsycopgOperationalError = None  # type: ignore


log = logging.getLogger("db.postgres")

_DSN: str = ""
_ENGINE: Optional[Engine] = None
ENGINE: Optional[Engine] = None
_LOCK = threading.Lock()
_SQLA_PREFIX_DETECTED = False
_LAST_NORMALIZE_PREFIX = False

_RETRYABLE_MESSAGES = (
    "SSL connection has been closed unexpectedly",
    "server closed the connection unexpectedly",
    "connection already closed",
    "timeout expired",
    "timed out",
)

_USERS_DDL = """
CREATE TABLE IF NOT EXISTS users (
    id BIGINT PRIMARY KEY,
    username TEXT,
    referrer_id BIGINT,
    joined_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    referral_earned_total BIGINT NOT NULL DEFAULT 0
);
"""

_BALANCES_DDL = """
CREATE TABLE IF NOT EXISTS balances (
    user_id BIGINT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    tokens BIGINT NOT NULL DEFAULT 0 CHECK (tokens >= 0),
    signup_bonus_granted BOOLEAN NOT NULL DEFAULT FALSE,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

_REFERRALS_DDL = """
CREATE TABLE IF NOT EXISTS referrals (
    id BIGSERIAL PRIMARY KEY,
    referrer_id BIGINT REFERENCES users(id) ON DELETE CASCADE,
    referred_id BIGINT REFERENCES users(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    earned_tokens BIGINT NOT NULL DEFAULT 0,
    UNIQUE (referrer_id, referred_id)
);
"""

_AUDIT_DDL = """
CREATE TABLE IF NOT EXISTS audit_log (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT,
    actor_id BIGINT,
    action TEXT,
    amount BIGINT,
    meta JSONB NOT NULL DEFAULT '{}'::jsonb,
    ts TIMESTAMPTZ DEFAULT NOW()
);
"""

_TRANSACTIONS_DDL = """
CREATE TABLE IF NOT EXISTS transactions (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    direction TEXT NOT NULL CHECK (direction IN ('credit', 'debit', 'adjustment')),
    amount BIGINT NOT NULL CHECK (amount >= 0),
    balance_after BIGINT CHECK (balance_after >= 0),
    reason TEXT,
    actor_id BIGINT,
    meta JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

_TRANSACTIONS_INDEX_DDL = """
CREATE INDEX IF NOT EXISTS idx_transactions_user_created_at
    ON transactions(user_id, created_at DESC);
"""


def _render_postgres_url(url: URL) -> str:
    rendered = url.render_as_string(hide_password=False)
    if rendered.startswith("postgresql://"):
        return "postgres://" + rendered[len("postgresql://") :]
    return rendered


def mask_dsn(dsn: str) -> str:
    """Return a DSN safe for logging with the password redacted."""

    if not dsn:
        return ""
    try:
        parsed = make_url(dsn)
    except Exception:  # pragma: no cover - defensive
        return "***"
    if parsed.password is None:
        return _render_postgres_url(parsed)
    masked = parsed.set(password="***")
    return _render_postgres_url(masked)


def normalize_dsn(raw: str) -> str:
    """Normalise any PostgreSQL DSN into the canonical ``postgres://`` form."""

    global _LAST_NORMALIZE_PREFIX

    candidate = (raw or "").strip()
    if not candidate:
        raise ValueError(
            "❌ DSN invalid: строка подключения пуста. "
            "Используйте стандартный URL вида: "
            "postgres://user:pass@host:port/db?sslmode=require"
        )

    try:
        parsed: URL = make_url(candidate)
    except Exception as exc:  # pragma: no cover - invalid DSN formatting
        raise ValueError(
            "❌ DSN invalid: не удалось разобрать URL. "
            "Используйте стандартный URL вида: "
            "postgres://user:pass@host:port/db?sslmode=require"
        ) from exc

    driver = (parsed.drivername or "").lower()
    _LAST_NORMALIZE_PREFIX = False
    if driver.startswith("postgresql+"):
        _LAST_NORMALIZE_PREFIX = True
        driver = "postgresql"
    elif driver in {"postgresql", "postgres"}:
        driver = "postgresql"
    else:
        raise ValueError(
            "❌ DSN invalid: обнаружена неподдерживаемая схема. "
            "Используйте URL вида: "
            "postgres://user:pass@host:port/db?sslmode=require"
        )

    if not parsed.host:
        raise ValueError(
            "❌ DSN invalid: отсутствует хост. "
            "Пример: postgres://user:pass@host:5432/db?sslmode=require"
        )
    if not parsed.database:
        raise ValueError(
            "❌ DSN invalid: отсутствует имя базы данных. "
            "Пример: postgres://user:pass@host:5432/db?sslmode=require"
        )

    query_items: "OrderedDict[str, str]" = OrderedDict(parsed.query.items())
    lowered = {key.lower(): key for key in query_items}
    if "sslmode" not in lowered:
        query_items["sslmode"] = "require"
    else:
        actual = lowered["sslmode"]
        if not (query_items.get(actual) or "").strip():
            query_items[actual] = "require"

    normalized = parsed.set(drivername="postgresql", query=query_items)
    return _render_postgres_url(normalized)


def _iter_exception_chain(exc: BaseException) -> Iterable[BaseException]:
    current: Optional[BaseException] = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        yield current
        seen.add(id(current))
        current = getattr(current, "__cause__", None) or getattr(current, "__context__", None)


def _is_retryable_error(exc: BaseException) -> bool:
    for candidate in _iter_exception_chain(exc):
        if isinstance(candidate, (OperationalError, ConnectionResetError, TimeoutError)):
            return True
        if PsycopgOperationalError is not None and isinstance(candidate, PsycopgOperationalError):
            return True
        message = str(candidate)
        if any(token in message for token in _RETRYABLE_MESSAGES):
            return True
    return False


def _test_connection(engine: Engine) -> None:
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))


def connect_with_retry(dsn: str, attempts: int = 6, backoff: float = 1.5) -> Engine:
    """Create a SQLAlchemy engine with retry logic and health check."""

    if attempts <= 0:
        raise ValueError("attempts must be a positive integer")
    if backoff <= 1.0:
        backoff = 1.5

    psycopg_url = make_url(dsn).set(drivername="postgresql+psycopg")
    engine_url = psycopg_url.render_as_string(hide_password=False)

    last_error: Optional[BaseException] = None
    for attempt in range(1, attempts + 1):
        log.info("postgres.connect_attempt | attempt=%s/%s", attempt, attempts)
        engine: Optional[Engine] = None
        try:
            engine = create_engine(
                engine_url,
                future=True,
                pool_pre_ping=True,
                pool_size=5,
                max_overflow=5,
                pool_timeout=15,
            )
            _test_connection(engine)
            log.info("postgres.connected | driver=postgresql+psycopg")
            return engine
        except Exception as exc:
            last_error = exc
            if engine is not None:
                with contextlib.suppress(Exception):  # pragma: no cover - best effort
                    engine.dispose()

            retryable = _is_retryable_error(exc)
            if not retryable or attempt == attempts:
                event = "postgres.connect_fail"
                log.error("%s | attempt=%s/%s err=%s", event, attempt, attempts, exc, exc_info=True)
                if retryable:
                    error_class = exc.__class__.__name__
                    message = (
                        f"⚠️ PostgreSQL: не удалось подключиться за {attempts} попыток "
                        f"({error_class}). Проверьте, что Neon compute «включён», и что "
                        "firewall/SSL не блокируют соединение."
                    )
                    raise RuntimeError(message) from exc
                raise

            sleep_base = backoff ** (attempt - 1)
            jitter = random.uniform(0.9, 1.1)
            sleep_for = sleep_base * jitter
            log.warning(
                "postgres.connect_retry | attempt=%s/%s err=%s wait=%.2fs",
                attempt,
                attempts,
                exc,
                sleep_for,
            )
            time.sleep(sleep_for)

    raise RuntimeError(
        f"⚠️ PostgreSQL: не удалось подключиться за {attempts} попыток (timeout)."
    ) from last_error


def configure_engine(dsn: str) -> Engine:
    """Normalise the DSN, establish a connection and store the engine."""

    normalized = normalize_dsn(dsn)
    log.info("postgres.configure_engine | dsn=%s", mask_dsn(normalized))

    engine = connect_with_retry(normalized)

    global _DSN, _ENGINE, ENGINE, _SQLA_PREFIX_DETECTED
    with _LOCK:
        if _ENGINE is not None and _ENGINE is not engine:
            with contextlib.suppress(Exception):  # pragma: no cover - best effort
                _ENGINE.dispose()
        _DSN = normalized
        _ENGINE = engine
        ENGINE = engine
        _SQLA_PREFIX_DETECTED = bool(_LAST_NORMALIZE_PREFIX)

    log.info("postgres.initialized")
    return engine


def configure(dsn: str) -> None:
    """Backward compatible wrapper for legacy callers."""

    configure_engine(dsn)


def _ensure_engine() -> Engine:
    global _ENGINE, ENGINE
    if _ENGINE is not None:
        return _ENGINE
    if not _DSN:
        raise RuntimeError("PostgreSQL DSN not configured")
    with _LOCK:
        if _ENGINE is None:
            _ENGINE = connect_with_retry(_DSN)
            ENGINE = _ENGINE
    return _ENGINE


def ensure_tables() -> None:
    """Create required tables if they do not exist."""

    engine = _ensure_engine()
    with engine.begin() as conn:
        conn.execute(text(_USERS_DDL))
        conn.execute(
            text(
                """
                ALTER TABLE users
                  ADD COLUMN IF NOT EXISTS username TEXT,
                  ADD COLUMN IF NOT EXISTS referrer_id BIGINT
                """
            )
        )
        conn.execute(text(_BALANCES_DDL))
        conn.execute(text(_REFERRALS_DDL))
        conn.execute(text(_AUDIT_DDL))
        conn.execute(text(_TRANSACTIONS_DDL))
        conn.execute(text(_TRANSACTIONS_INDEX_DDL))


def get_engine() -> Engine:
    """Expose the configured SQLAlchemy engine."""

    return _ensure_engine()


def _short_version(raw: Any) -> str:
    text_value = str(raw or "").strip()
    if not text_value:
        return "unknown"
    primary = text_value.splitlines()[0]
    primary = primary.split(" on ")[0].split(",")[0].strip()
    return primary or "unknown"


def _current_sslmode() -> str:
    if not _DSN:
        return ""
    try:
        parsed = make_url(_DSN)
    except Exception:  # pragma: no cover - defensive
        return ""
    value = parsed.query.get("sslmode")
    return str(value) if value is not None else ""


def db_overview() -> Dict[str, Any]:
    """Collect statistics about the database and connection pool."""

    engine = _ensure_engine()
    tables = ["users", "balances", "referrals", "transactions"]
    overview: Dict[str, Any] = {}

    try:
        with engine.connect() as conn:
            version_row = conn.execute(text("SELECT version()"))
            version = _short_version(version_row.scalar())
            current_db = conn.execute(text("SELECT current_database()"))
            db_name = str(current_db.scalar() or "")
            current_schema = conn.execute(text("SELECT current_schema()"))
            schema_name = str(current_schema.scalar() or "")

            table_metrics: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
            for table in tables:
                try:
                    result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    count_value = int(result.scalar() or 0)
                except SQLAlchemyError:
                    count_value = 0
                table_metrics[table] = {"rows": count_value, "size_bytes": 0}

            size_stmt = text(
                """
                SELECT c.relname AS name,
                       pg_total_relation_size(c.oid) AS size
                  FROM pg_class AS c
                  JOIN pg_namespace AS n ON n.oid = c.relnamespace
                 WHERE n.nspname = current_schema() AND c.relname = ANY(:tables)
                """
            )
            size_rows = conn.execute(size_stmt, {"tables": tables}).mappings().all()
            size_map = {
                str(row.get("name")): int(row.get("size") or 0) for row in size_rows
            }
            for name, size_val in size_map.items():
                if name in table_metrics:
                    table_metrics[name]["size_bytes"] = size_val

    except Exception as exc:
        log.error("postgres.check.fail | err=%s", exc, exc_info=True)
        raise

    pool = getattr(engine, "pool", None)
    in_use = 0
    available = 0
    if pool is not None:
        checkout = getattr(pool, "checkedout", None)
        if callable(checkout):
            with contextlib.suppress(Exception):
                value = checkout()
                if isinstance(value, int):
                    in_use = max(value, 0)
        checkin = getattr(pool, "checkedin", None)
        if callable(checkin):
            with contextlib.suppress(Exception):
                value = checkin()
                if isinstance(value, int):
                    available = max(value, 0)
        if available == 0:
            size_fn = getattr(pool, "size", None)
            if callable(size_fn):
                with contextlib.suppress(Exception):
                    size_val = size_fn()
                    if isinstance(size_val, int):
                        available = max(size_val - in_use, 0)

    tables_payload = OrderedDict()
    for table, metrics in table_metrics.items():
        size_bytes = int(metrics.get("size_bytes", 0))
        tables_payload[table] = {
            "rows": int(metrics.get("rows", 0)),
            "size_bytes": size_bytes,
            "size_mb": round(size_bytes / (1024 * 1024), 2) if size_bytes else 0.0,
        }

    overview.update(
        {
            "server_version": version,
            "database": db_name,
            "schema": schema_name,
            "tables": tables_payload,
            "pool": {
                "in_use": in_use,
                "available": available,
            },
            "sslmode": _current_sslmode() or "",
            "sqlalchemy_prefix": _SQLA_PREFIX_DETECTED,
            "dsn": mask_dsn(_DSN),
        }
    )

    log.info(
        "postgres.check.ok | db=%s schema=%s in_use=%s available=%s",
        db_name,
        schema_name,
        in_use,
        available,
    )
    return overview


def get_database_overview() -> Dict[str, Any]:
    """Backward compatible wrapper returning the database overview."""

    return db_overview()


def check_health() -> OrderedDict[str, int]:
    """Return row counts for key tables to assess DB health."""

    snapshot = db_overview()
    counts: "OrderedDict[str, int]" = OrderedDict()
    for table, metrics in snapshot.get("tables", {}).items():
        counts[table] = int(metrics.get("rows", 0))
    return counts


def sqlalchemy_prefix_detected() -> bool:
    """Return whether the last configured DSN had a SQLAlchemy driver prefix."""

    return _SQLA_PREFIX_DETECTED


def current_dsn() -> str:
    """Return the currently configured normalised DSN."""

    return _DSN


def export_balances_snapshot() -> Dict[str, Any]:
    """Return a JSON-serialisable snapshot of user balances."""

    engine = _ensure_engine()
    query = text(
        """
        SELECT b.user_id,
               b.tokens,
               b.signup_bonus_granted,
               b.updated_at,
               u.username,
               u.referrer_id,
               u.joined_at
          FROM balances AS b
     LEFT JOIN users AS u ON u.id = b.user_id
      ORDER BY b.user_id
        """
    )
    records: List[Dict[str, Any]] = []
    with engine.connect() as conn:
        for row in conn.execute(query).mappings():
            records.append(dict(row))
    generated_at = datetime.now(timezone.utc).isoformat()
    return {"generated_at": generated_at, "count": len(records), "balances": records}


def log_transaction(
    user_id: int,
    direction: str,
    amount: int,
    *,
    balance_after: Optional[int] = None,
    reason: Optional[str] = None,
    actor_id: Optional[int] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist a balance change event in the ``transactions`` table."""

    direction_norm = (direction or "").strip().lower()
    if direction_norm not in {"credit", "debit", "adjustment"}:
        raise ValueError("direction must be credit, debit or adjustment")
    amount_value = abs(int(amount))
    payload = json.dumps(meta or {}, ensure_ascii=False)
    engine = _ensure_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO transactions (user_id, direction, amount, balance_after, reason, actor_id, meta)
                VALUES (:user_id, :direction, :amount, :balance_after, :reason, :actor_id, :meta::jsonb)
                """
            ),
            {
                "user_id": int(user_id),
                "direction": direction_norm,
                "amount": amount_value,
                "balance_after": int(balance_after) if balance_after is not None else None,
                "reason": reason,
                "actor_id": int(actor_id) if actor_id is not None else None,
                "meta": payload,
            },
        )


def apply_balance_delta(
    user_id: int,
    delta: int,
    *,
    actor_id: Optional[int] = None,
    reason: str = "admin_adjust",
    note: Optional[str] = None,
) -> Dict[str, int]:
    """Atomically adjust a user's balance and return old/new values."""

    user_id = int(user_id)
    delta = int(delta)
    if delta == 0:
        raise ValueError("delta must be a non-zero integer")

    engine = _ensure_engine()
    direction = "credit" if delta > 0 else "debit"
    amount = abs(delta)
    metadata: Dict[str, Any] = {"delta": delta}
    if note:
        metadata["note"] = note

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO users (id)
                VALUES (:uid)
                ON CONFLICT (id) DO NOTHING
                """
            ),
            {"uid": user_id},
        )
        conn.execute(
            text(
                """
                INSERT INTO balances (user_id)
                VALUES (:uid)
                ON CONFLICT (user_id) DO NOTHING
                """
            ),
            {"uid": user_id},
        )
        current_raw = conn.execute(
            text("SELECT tokens FROM balances WHERE user_id = :uid FOR UPDATE"),
            {"uid": user_id},
        ).scalar()
        current_balance = int(current_raw or 0)
        new_balance = current_balance + delta
        if new_balance < 0:
            raise ValueError(
                f"insufficient balance: have {current_balance}, need {abs(delta)}"
            )
        conn.execute(
            text(
                """
                UPDATE balances
                   SET tokens = :new_balance,
                       updated_at = NOW()
                 WHERE user_id = :uid
                """
            ),
            {"new_balance": new_balance, "uid": user_id},
        )
        payload = json.dumps(metadata, ensure_ascii=False)
        conn.execute(
            text(
                """
                INSERT INTO audit_log (user_id, actor_id, action, amount, meta)
                VALUES (:user_id, :actor_id, :action, :amount, :meta::jsonb)
                """
            ),
            {
                "user_id": user_id,
                "actor_id": int(actor_id) if actor_id is not None else None,
                "action": reason,
                "amount": delta,
                "meta": payload,
            },
        )
        conn.execute(
            text(
                """
                INSERT INTO transactions (user_id, direction, amount, balance_after, reason, actor_id, meta)
                VALUES (:user_id, :direction, :amount, :balance_after, :reason, :actor_id, :meta::jsonb)
                """
            ),
            {
                "user_id": user_id,
                "direction": direction,
                "amount": amount,
                "balance_after": new_balance,
                "reason": reason,
                "actor_id": int(actor_id) if actor_id is not None else None,
                "meta": payload,
            },
        )

    return {"old_balance": current_balance, "new_balance": new_balance}


def ensure_user(user_id: int, *, username: Optional[str] = None, referrer_id: Optional[int] = None) -> None:
    """Ensure a user exists in the ``users`` and ``balances`` tables."""

    engine = _ensure_engine()
    params = {
        "id": int(user_id),
        "username": username,
        "referrer_id": referrer_id,
    }
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO users (id, username, referrer_id)
                VALUES (:id, :username, :referrer_id)
                ON CONFLICT (id) DO UPDATE
                   SET username = COALESCE(EXCLUDED.username, users.username),
                       referrer_id = COALESCE(EXCLUDED.referrer_id, users.referrer_id),
                       updated_at = NOW()
                """
            ),
            params,
        )
        conn.execute(
            text(
                """
                INSERT INTO balances (user_id)
                VALUES (:id)
                ON CONFLICT (user_id) DO NOTHING
                """
            ),
            {"id": int(user_id)},
        )


def set_referrer(user_id: int, referrer_id: int) -> bool:
    """Attach a referrer to a user if not already set.

    Returns ``True`` if the referrer was stored.
    """

    engine = _ensure_engine()
    user_id = int(user_id)
    referrer_id = int(referrer_id)
    if user_id == referrer_id or user_id <= 0 or referrer_id <= 0:
        return False
    try:
        ensure_user(referrer_id)
        ensure_user(user_id)
    except Exception as exc:
        log.warning(
            "postgres.set_referrer.ensure_user_failed | user=%s inviter=%s err=%s",
            user_id,
            referrer_id,
            exc,
        )
    with engine.begin() as conn:
        current = conn.execute(
            text("SELECT referrer_id FROM users WHERE id = :uid FOR UPDATE"),
            {"uid": user_id},
        ).scalar()
        if current:
            return False
        conn.execute(
            text("UPDATE users SET referrer_id = :ref WHERE id = :uid"),
            {"ref": referrer_id, "uid": user_id},
        )
        conn.execute(
            text(
                """
                INSERT INTO referrals (referrer_id, referred_id)
                VALUES (:referrer, :referred)
                ON CONFLICT (referrer_id, referred_id) DO NOTHING
                """
            ),
            {"referrer": referrer_id, "referred": user_id},
        )
    return True


def increment_referral_earnings(
    referrer_id: int,
    amount: int,
    *,
    referred_id: Optional[int] = None,
) -> int:
    """Increment referral earnings and return the total."""

    engine = _ensure_engine()
    referrer_id = int(referrer_id)
    amount = int(amount)
    if amount <= 0:
        return get_referral_total(referrer_id)
    try:
        ensure_user(referrer_id)
        if referred_id is not None:
            ensure_user(int(referred_id))
    except Exception as exc:
        log.warning(
            "postgres.increment_referral.ensure_user_failed | referrer=%s referred=%s err=%s",
            referrer_id,
            referred_id,
            exc,
        )
    with engine.begin() as conn:
        if referred_id is not None:
            conn.execute(
                text(
                    """
                    INSERT INTO referrals (referrer_id, referred_id, earned_tokens)
                    VALUES (:referrer, :referred, :amount)
                    ON CONFLICT (referrer_id, referred_id)
                    DO UPDATE SET earned_tokens = referrals.earned_tokens + EXCLUDED.earned_tokens
                    """
                ),
                {"referrer": referrer_id, "referred": int(referred_id), "amount": amount},
            )
        total = conn.execute(
            text(
                """
                UPDATE users
                   SET referral_earned_total = referral_earned_total + :amount,
                       updated_at = NOW()
                 WHERE id = :referrer
             RETURNING referral_earned_total
                """
            ),
            {"amount": amount, "referrer": referrer_id},
        ).scalar()
    return int(total or 0)


def get_referral_total(referrer_id: int) -> int:
    """Return the total referral earnings for ``referrer_id``."""

    engine = _ensure_engine()
    with engine.connect() as conn:
        value = conn.execute(
            text("SELECT referral_earned_total FROM users WHERE id = :uid"),
            {"uid": int(referrer_id)},
        ).scalar()
    return int(value or 0)


def get_referral_stats(referrer_id: int) -> Tuple[int, int]:
    """Return ``(count, total_earned)`` for a referrer."""

    engine = _ensure_engine()
    with engine.begin() as conn:
        count = conn.execute(
            text("SELECT COUNT(*) FROM referrals WHERE referrer_id = :uid"),
            {"uid": int(referrer_id)},
        ).scalar()
        total = conn.execute(
            text("SELECT referral_earned_total FROM users WHERE id = :uid"),
            {"uid": int(referrer_id)},
        ).scalar()
    return int(count or 0), int(total or 0)


def list_referrals(referrer_id: int) -> List[Dict[str, Optional[str]]]:
    """Return a list of referrals for the given user."""

    engine = _ensure_engine()
    rows: List[Dict[str, Optional[str]]] = []
    with engine.connect() as conn:
        result = conn.execute(
            text(
                """
                SELECT r.referred_id, u.username, u.joined_at, r.created_at, r.earned_tokens
                  FROM referrals AS r
             LEFT JOIN users AS u ON u.id = r.referred_id
                 WHERE r.referrer_id = :uid
              ORDER BY r.created_at DESC
                """
            ),
            {"uid": int(referrer_id)},
        )
        for row in result.mappings():
            rows.append(
                {
                    "user_id": row["referred_id"],
                    "username": row.get("username"),
                    "joined_at": row.get("joined_at"),
                    "created_at": row.get("created_at"),
                    "earned_tokens": row.get("earned_tokens"),
                }
            )
    return rows


def get_user_balance(user_id: int) -> Optional[int]:
    """Return the balance in tokens for ``user_id``."""

    engine = _ensure_engine()
    with engine.connect() as conn:
        value = conn.execute(
            text("SELECT tokens FROM balances WHERE user_id = :uid"),
            {"uid": int(user_id)},
        ).scalar()
    if value is None:
        return None
    return int(value)


def log_audit(
    user_id: int,
    action: str,
    amount: int,
    *,
    actor_id: Optional[int] = None,
    meta: Optional[Dict[str, object]] = None,
) -> None:
    """Write an entry to the audit log."""

    engine = _ensure_engine()
    payload = json.dumps(meta or {}, ensure_ascii=False)
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO audit_log (user_id, actor_id, action, amount, meta)
                VALUES (:user_id, :actor_id, :action, :amount, :meta::jsonb)
                """
            ),
            {
                "user_id": int(user_id),
                "actor_id": int(actor_id) if actor_id is not None else None,
                "action": action,
                "amount": int(amount),
                "meta": payload,
            },
        )


def restore_referrals(records: Iterable[Tuple[int, int]]) -> int:
    """Restore referral relations from ``(user_id, referrer_id)`` records."""

    restored = 0
    engine = _ensure_engine()
    with engine.begin() as conn:
        for referred, referrer in records:
            referred_id = int(referred)
            referrer_id = int(referrer)
            if referred_id == referrer_id or referred_id <= 0 or referrer_id <= 0:
                continue
            try:
                conn.execute(
                    text(
                        """
                        INSERT INTO users (id)
                        VALUES (:referred)
                        ON CONFLICT (id) DO NOTHING
                        """
                    ),
                    {"referred": referred_id},
                )
                conn.execute(
                    text(
                        """
                        INSERT INTO users (id)
                        VALUES (:referrer)
                        ON CONFLICT (id) DO NOTHING
                        """
                    ),
                    {"referrer": referrer_id},
                )
                updated = conn.execute(
                    text(
                        """
                        UPDATE users
                           SET referrer_id = COALESCE(users.referrer_id, :referrer)
                         WHERE id = :referred
                    RETURNING 1
                        """
                    ),
                    {"referred": referred_id, "referrer": referrer_id},
                ).rowcount
                conn.execute(
                    text(
                        """
                        INSERT INTO referrals (referrer_id, referred_id)
                        VALUES (:referrer, :referred)
                        ON CONFLICT (referrer_id, referred_id) DO NOTHING
                        """
                    ),
                    {"referrer": referrer_id, "referred": referred_id},
                )
                if updated:
                    restored += 1
            except SQLAlchemyError as exc:
                log.warning(
                    "postgres.restore_referral_failed | referrer=%s referred=%s err=%s",
                    referrer_id,
                    referred_id,
                    exc,
                )
    return restored


__all__ = [
    "configure",
    "ensure_tables",
    "get_engine",
    "check_health",
    "export_balances_snapshot",
    "ensure_user",
    "set_referrer",
    "increment_referral_earnings",
    "get_referral_stats",
    "list_referrals",
    "get_user_balance",
    "log_audit",
    "restore_referrals",
    "get_referral_total",
    "log_transaction",
]
