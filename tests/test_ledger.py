import pytest

from ledger import InsufficientBalance, LedgerStorage


def test_memory_ledger_operations():
    ledger = LedgerStorage(None)
    assert ledger.backend_name == "memory"

    credit_result = ledger.credit(1, 100, "deposit", "mem-op-1")
    assert credit_result.applied is True
    assert credit_result.balance == 100

    duplicate_credit = ledger.credit(1, 100, "deposit", "mem-op-1")
    assert duplicate_credit.applied is False
    assert duplicate_credit.duplicate is True

    debit_result = ledger.debit(1, 25, "withdraw", "mem-op-2")
    assert debit_result.applied is True
    assert debit_result.balance == 75

    with pytest.raises(InsufficientBalance):
        ledger.debit(1, 1000, "overdraft", "mem-op-3")

    promo_result = ledger.apply_promo(2, "FREE", 30, None)
    assert promo_result.applied is True
    assert promo_result.balance == 30

    other_user = ledger.apply_promo(3, "FREE", 30, None)
    assert other_user.applied is False
    assert other_user.duplicate is True


def test_sqlite_ledger_persistence(tmp_path):
    db_path = tmp_path / "ledger.sqlite3"
    dsn = f"sqlite:///{db_path}"

    ledger = LedgerStorage(dsn)
    assert ledger.backend_name == "sqlite"

    ledger.credit(10, 50, "deposit", "sql-op-1")
    assert ledger.get_balance(10) == 50

    ledger_second = LedgerStorage(dsn)
    assert ledger_second.get_balance(10) == 50

    ledger_second.debit(10, 5, "spend", "sql-op-2")
    assert ledger_second.get_balance(10) == 45

    ledger_third = LedgerStorage(dsn)
    assert ledger_third.get_balance(10) == 45
