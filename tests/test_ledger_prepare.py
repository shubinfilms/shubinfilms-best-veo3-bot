import importlib
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class _DummyTransaction:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _DummyCursor:
    def __init__(self, statements: list[str]):
        self._statements = statements

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, statement: str) -> None:
        self._statements.append(statement)


class _DummyConnection:
    def __init__(self, statements: list[str]):
        self._statements = statements

    def transaction(self) -> _DummyTransaction:
        return _DummyTransaction()

    def cursor(self) -> _DummyCursor:
        return _DummyCursor(self._statements)


def test_prepare_is_idempotent(monkeypatch):
    postgres_stub = types.ModuleType("core.db.postgres")
    postgres_stub.normalize_dsn = lambda value: value  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "core", types.ModuleType("core"))
    monkeypatch.setitem(sys.modules, "core.db", types.ModuleType("core.db"))
    monkeypatch.setitem(sys.modules, "core.db.postgres", postgres_stub)

    ledger_module = importlib.import_module("ledger")
    storage = ledger_module._PostgresLedgerStorage("postgresql://user:pass@localhost/db")
    executed: list[str] = []
    dummy_conn = _DummyConnection(executed)

    def fake_with_connection(fn, *, op, retries=None, **ctx):
        fn(dummy_conn)

    monkeypatch.setattr(storage, "_with_connection", fake_with_connection)

    storage._prepare()
    storage._prepare()

    guard_statements = [stmt for stmt in executed if "transactions_user_type_key_key" in stmt]
    assert guard_statements, "constraint guard must execute"
    assert any("DO $$" in stmt for stmt in guard_statements)
