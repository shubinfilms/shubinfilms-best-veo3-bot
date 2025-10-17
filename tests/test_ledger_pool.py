import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

import psycopg


def _install_db_stub() -> None:
    db_module = ModuleType("db")
    postgres_module = ModuleType("db.postgres")

    def _identity(value: str) -> str:
        return value

    postgres_module.mask_dsn = _identity  # type: ignore[attr-defined]
    postgres_module.normalize_dsn = _identity  # type: ignore[attr-defined]
    db_module.postgres = postgres_module  # type: ignore[attr-defined]
    sys.modules.setdefault("db", db_module)
    sys.modules.setdefault("db.postgres", postgres_module)


_install_db_stub()

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ledger import _PostgresLedgerStorage  # noqa: E402


class _DummyPool:
    def __init__(self) -> None:
        self.calls = 0

    @contextmanager
    def connection(self):
        self.calls += 1
        yield object()


def test_with_connection_retries(monkeypatch):
    storage = _PostgresLedgerStorage("postgresql://user:pass@localhost/db")
    dummy_pool = _DummyPool()
    storage._pool = dummy_pool  # type: ignore[attr-defined]

    attempts = {"count": 0}

    def flaky(conn):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise psycopg.OperationalError("SSL connection has been closed unexpectedly")
        return "ok"

    monkeypatch.setattr("ledger.time.sleep", lambda _: None)

    result = storage._with_connection(flaky, op="test_retry")

    assert result == "ok"
    assert attempts["count"] == 2
    assert dummy_pool.calls == 2
