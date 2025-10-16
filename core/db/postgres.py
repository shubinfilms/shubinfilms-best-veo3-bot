"""PostgreSQL connection helpers with resilient pooling."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import psycopg
from psycopg.conninfo import make_conninfo
from psycopg_pool import ConnectionPool

from db.postgres import mask_dsn as _mask_dsn
from db.postgres import normalize_dsn as _normalize_dsn

log = logging.getLogger("core.db.postgres")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw.strip())
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


_DEFAULT_POOL_MAX = _env_int("PG_POOL_MAX", 10)
_DEFAULT_POOL_RECYCLE = _env_int("PG_POOL_RECYCLE_SEC", 300)
_DEFAULT_STATEMENT_TIMEOUT = _env_int("PG_CONN_STATEMENT_TIMEOUT_MS", 15_000)
_DEFAULT_KEEPALIVES_IDLE = _env_int("PG_KEEPALIVES_IDLE", 30)
_DEFAULT_KEEPALIVES_INTERVAL = _env_int("PG_KEEPALIVES_INTERVAL", 10)
_DEFAULT_KEEPALIVES_COUNT = _env_int("PG_KEEPALIVES_COUNT", 3)
_DEFAULT_TCP_USER_TIMEOUT = _env_int("PG_TCP_USER_TIMEOUT_MS", 15_000)


def normalize_dsn(raw: str) -> str:
    """Expose the shared DSN normaliser."""

    return _normalize_dsn(raw)


def mask_dsn(dsn: str) -> str:
    """Expose the shared DSN masker."""

    return _mask_dsn(dsn)


def ensure_conninfo(raw_dsn: str, *, application_name: Optional[str] = None) -> str:
    """Append keepalive and timeout parameters to a PostgreSQL DSN."""

    base = normalize_dsn(raw_dsn)
    conn_kwargs: Dict[str, Any] = {
        "sslmode": "require",
        "keepalives": 1,
        "keepalives_idle": _DEFAULT_KEEPALIVES_IDLE,
        "keepalives_interval": _DEFAULT_KEEPALIVES_INTERVAL,
        "keepalives_count": _DEFAULT_KEEPALIVES_COUNT,
        "tcp_user_timeout": _DEFAULT_TCP_USER_TIMEOUT,
        "options": f"-c statement_timeout={_DEFAULT_STATEMENT_TIMEOUT}",
    }
    if application_name:
        conn_kwargs["application_name"] = application_name
    return make_conninfo(base, **conn_kwargs)


class ResilientConnectionPool(ConnectionPool):
    """A psycopg connection pool that discards closed connections."""

    def getconn(self, timeout: Optional[float] = None) -> psycopg.Connection[Any]:  # type: ignore[override]
        while True:
            conn = super().getconn(timeout=timeout)
            if getattr(conn, "closed", False):
                super().putconn(conn, close=True)
                continue
            return conn

    def putconn(
        self, conn: psycopg.Connection[Any], close: bool = False
    ) -> None:  # type: ignore[override]
        close = close or getattr(conn, "closed", False)
        super().putconn(conn, close=close)


def create_connection_pool(
    raw_dsn: str,
    *,
    application_name: str = "bot",
    max_size: Optional[int] = None,
    timeout: float = 10.0,
    max_lifetime: Optional[int] = None,
    reuse_threshold: int = 1000,
) -> ResilientConnectionPool:
    """Create a :class:`ResilientConnectionPool` with sensible defaults."""

    conninfo = ensure_conninfo(raw_dsn, application_name=application_name)
    pool_max = max_size or _DEFAULT_POOL_MAX
    lifetime = max_lifetime or _DEFAULT_POOL_RECYCLE
    pool = ResilientConnectionPool(
        conninfo=conninfo,
        max_size=pool_max,
        timeout=timeout,
        max_lifetime=lifetime,
        reuse_threshold=reuse_threshold,
        kwargs={"autocommit": False},
    )
    pool.wait()
    log.info(
        "DB_READY",
        extra={
            "dsn": mask_dsn(conninfo),
            "pool_max": pool_max,
            "recycle": lifetime,
            "timeout": timeout,
            "reuse_threshold": reuse_threshold,
        },
    )
    return pool
