"""Async helpers for database operations."""

from __future__ import annotations

import asyncio
from functools import partial
from typing import Any, Callable, Optional, Sequence

try:  # pragma: no cover - optional dependency in tests
    from db import postgres as db_postgres
except Exception:  # pragma: no cover - missing dependency fallback
    db_postgres = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency in tests
    from ledger import LedgerStorage
except Exception:  # pragma: no cover - missing dependency fallback
    LedgerStorage = None  # type: ignore[assignment]

__all__ = [
    "get_user_balance_async",
    "list_transactions_async",
]


async def _run_blocking(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(func, *args, **kwargs))


async def get_user_balance_async(user_id: int) -> Optional[int]:
    """Return user balance via :mod:`db.postgres` in a worker thread."""

    if db_postgres is None:
        raise RuntimeError("db.postgres module is not available")
    return await _run_blocking(db_postgres.get_user_balance, int(user_id))


async def list_transactions_async(
    storage: LedgerStorage,
    user_id: int,
    *,
    limit: int = 5,
) -> Sequence[dict[str, Any]]:
    """Return recent transactions from :class:`ledger.LedgerStorage`."""

    if storage is None:
        return []
    return await _run_blocking(storage.get_history, int(user_id), int(limit))

