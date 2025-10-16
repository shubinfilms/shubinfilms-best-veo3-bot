from __future__ import annotations

import contextlib
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ledger import _PostgresLedgerStorage


class FakeDB:
    def __init__(self) -> None:
        self.users: Dict[int, Dict[str, Any]] = {}
        self.balances: Dict[int, Dict[str, Any]] = {}
        self.ledger: Dict[str, Dict[str, Any]] = {}
        self.transactions: Dict[Tuple[int, str, Optional[str]], Dict[str, Any]] = {}
        self.transaction_entries: List[Tuple[int, str, int, str]] = []


class FakeCursor:
    def __init__(self, db: FakeDB) -> None:
        self.db = db
        self._results: List[Any] = []
        self.rowcount: int = 0

    def execute(self, sql: str, params: Tuple[Any, ...] | Any = ()) -> None:
        text = " ".join(sql.split()).lower()
        self._results = []
        self.rowcount = 0

        if text.startswith("insert into users"):
            uid, username, referrer_id = params
            record = self.db.users.setdefault(uid, {"username": None, "referrer_id": None})
            if username is not None:
                record["username"] = username
            if record.get("referrer_id") is None and referrer_id is not None:
                record["referrer_id"] = referrer_id
            self.rowcount = 1
        elif text.startswith("insert into balances"):
            uid = params[0]
            self.db.balances.setdefault(
                uid, {"tokens": 0, "signup_bonus_granted": False}
            )
            self.rowcount = 1
        elif "select tokens, signup_bonus_granted from balances" in text:
            uid = params[0]
            bal = self.db.balances.get(uid, {"tokens": 0, "signup_bonus_granted": False})
            self._results = [(bal["tokens"], bal.get("signup_bonus_granted", False))]
        elif text.startswith("select 1 from ledger"):
            op_id = params[0]
            self._results = [(1,)] if op_id in self.db.ledger else []
        elif text.startswith("update balances set signup_bonus_granted = true"):
            uid = params[0]
            bal = self.db.balances.setdefault(
                uid, {"tokens": 0, "signup_bonus_granted": False}
            )
            bal["signup_bonus_granted"] = True
            self.rowcount = 1
        elif "update balances" in text and "tokens = tokens +" in text:
            amount, uid = params
            bal = self.db.balances.setdefault(
                uid, {"tokens": 0, "signup_bonus_granted": False}
            )
            bal["tokens"] += int(amount)
            bal["signup_bonus_granted"] = True
            self.rowcount = 1
            self._results = [(bal["tokens"],)]
        elif "update balances" in text and "tokens = tokens -" in text:
            amount, uid = params
            bal = self.db.balances.setdefault(
                uid, {"tokens": 0, "signup_bonus_granted": False}
            )
            bal["tokens"] -= int(amount)
            self.rowcount = 1
            self._results = [(bal["tokens"],)]
        elif text.startswith("insert into ledger"):
            uid, amount, reason, op_id, _meta = params
            if op_id in self.db.ledger:
                raise RuntimeError("duplicate ledger op")
            self.db.ledger[op_id] = {
                "user_id": uid,
                "amount": int(amount),
                "reason": reason,
            }
            self.rowcount = 1
        elif text.startswith("insert into transactions"):
            uid, tx_type, amount, reason, key = params
            key_tuple = (uid, tx_type, key)
            if key is not None and key_tuple in self.db.transactions:
                self.rowcount = 0
            else:
                if key is not None:
                    self.db.transactions[key_tuple] = {
                        "amount": int(amount),
                        "reason": reason,
                    }
                else:
                    self.db.transaction_entries.append((uid, tx_type, int(amount), reason))
                self.rowcount = 1
        elif "select tokens from balances where user_id=%s for update" in text or (
            text.startswith("select tokens from balances where user_id=%s")
            and "for update" not in text
        ):
            uid = params[0]
            bal = self.db.balances.get(uid, {"tokens": 0})
            self._results = [(bal["tokens"],)]
        else:  # pragma: no cover - ensure tests keep SQL coverage in sync
            raise AssertionError(f"Unhandled SQL: {sql}")

    def fetchone(self) -> Any:
        if self._results:
            return self._results.pop(0)
        return None


class FakeConnection:
    def __init__(self, db: FakeDB) -> None:
        self.db = db
        self.closed = False

    @contextlib.contextmanager
    def cursor(self) -> Any:
        yield FakeCursor(self.db)

    @contextlib.contextmanager
    def transaction(self) -> Any:
        yield

    def close(self) -> None:
        self.closed = True


class FakePool:
    def __init__(self, db: FakeDB) -> None:
        self.db = db

    @contextlib.contextmanager
    def connection(self) -> Any:
        conn = FakeConnection(self.db)
        try:
            yield conn
        finally:
            conn.close()


def make_storage() -> tuple[_PostgresLedgerStorage, FakeDB]:
    db = FakeDB()
    storage = object.__new__(_PostgresLedgerStorage)
    storage.pool = FakePool(db)  # type: ignore[attr-defined]
    storage.log = logging.getLogger("test.ledger")  # type: ignore[attr-defined]
    return storage, db


def test_ensure_user_idempotent_updates_username_only():
    storage, db = make_storage()

    storage.ensure_user(1, username="alice", referrer_id=42)
    storage.ensure_user(1, username="bob", referrer_id=100)

    assert db.users == {1: {"username": "bob", "referrer_id": 42}}
    assert 1 in db.balances


def test_grant_signup_bonus_idempotent():
    storage, db = make_storage()

    result_first = storage.grant_signup_bonus(5, 100)
    assert result_first.applied is True
    assert db.balances[5]["tokens"] == 100
    assert db.balances[5]["signup_bonus_granted"] is True
    assert (5, "credit", "signup") in db.transactions

    result_second = storage.grant_signup_bonus(5, 100)
    assert result_second.applied is False
    assert result_second.duplicate is True
    assert db.balances[5]["tokens"] == 100
    assert len(db.transactions) == 1
