from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Optional

try:  # pragma: no cover - optional dependency in tests
    from ledger import LedgerStorage
except Exception:  # pragma: no cover - missing dependency fallback
    LedgerStorage = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency in tests
    from redis_utils import get_balance as redis_get_balance
except Exception:  # pragma: no cover - missing dependency fallback
    redis_get_balance = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency in tests
    from settings import DATABASE_URL, LEDGER_BACKEND
except Exception:  # pragma: no cover - default fallback
    DATABASE_URL = ""
    LEDGER_BACKEND = "memory"

__all__ = [
    "BALANCE_PLACEHOLDER",
    "BALANCE_WARNING",
    "BalanceSnapshot",
    "aget_balance_snapshot",
    "get_balance_snapshot",
    "set_ledger_storage",
]

log = logging.getLogger(__name__)

BALANCE_PLACEHOLDER = "—"
BALANCE_WARNING = "⚠️ Сервер недоступен. Попробуйте позже."


@dataclass(frozen=True)
class BalanceSnapshot:
    value: Optional[int]
    display: str
    warning: Optional[str] = None

    @property
    def is_available(self) -> bool:
        return self.value is not None


_LEDGER_STORAGE: Optional[Any] = None


def set_ledger_storage(storage: LedgerStorage) -> None:
    global _LEDGER_STORAGE
    _LEDGER_STORAGE = storage


def _get_ledger_storage() -> Optional[LedgerStorage]:
    global _LEDGER_STORAGE
    if (_LEDGER_STORAGE is None) and DATABASE_URL and LedgerStorage is not None:
        try:
            _LEDGER_STORAGE = LedgerStorage(DATABASE_URL, backend=LEDGER_BACKEND)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("balance.ledger_init_failed | err=%s", exc)
            _LEDGER_STORAGE = None
    return _LEDGER_STORAGE


def _build_snapshot(value: Optional[int], warning: Optional[str] = None) -> BalanceSnapshot:
    if value is None:
        return BalanceSnapshot(value=None, display=BALANCE_PLACEHOLDER, warning=warning or BALANCE_WARNING)
    return BalanceSnapshot(value=int(value), display=str(int(value)), warning=None)


def get_balance_snapshot(user_id: int, *, retries: int = 2) -> BalanceSnapshot:
    attempts = max(int(retries), 1)
    if redis_get_balance is not None:
        for attempt in range(attempts):
            try:
                value = redis_get_balance(user_id)
            except Exception as exc:
                log.warning(
                    "balance.fetch_retry | user=%s attempt=%s err=%s",
                    user_id,
                    attempt + 1,
                    exc,
                )
            else:
                return _build_snapshot(value)
    else:
        log.warning("balance.redis_unavailable | user=%s", user_id)

    ledger = _get_ledger_storage()
    if ledger is not None and hasattr(ledger, "get_balance"):
        try:
            value = ledger.get_balance(user_id)
        except Exception as exc:  # pragma: no cover - database unavailable
            log.warning("balance.fetch_db_failed | user=%s err=%s", user_id, exc)
        else:
            return _build_snapshot(value)

    return _build_snapshot(None)


async def aget_balance_snapshot(user_id: int, *, retries: int = 2) -> BalanceSnapshot:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(get_balance_snapshot, user_id, retries=retries))
