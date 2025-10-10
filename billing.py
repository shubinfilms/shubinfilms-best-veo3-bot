"""Minimal billing facade used across the bot."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, List

from redis_utils import credit_balance, debit_try

logger = logging.getLogger(__name__)


class BillingError(RuntimeError):
    """Base error for billing operations."""


class NotEnoughFunds(BillingError):
    """Raised when user balance is insufficient for a charge."""


async def _run_in_executor(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


async def charge(user_id: int, amount: int, reason: str = "") -> int:
    """Charge ``amount`` tokens from ``user_id``.

    Returns the new balance or raises :class:`NotEnoughFunds` if the user
    doesn't have enough tokens.
    """

    if amount < 0:
        raise ValueError("amount must be non-negative")

    def _debit() -> tuple[bool, int]:
        return debit_try(user_id, int(amount), reason or "debit")

    ok, new_balance = await _run_in_executor(_debit)
    if not ok:
        logger.debug(
            "billing.charge.insufficient",
            extra={"user_id": user_id, "amount": amount, "balance": new_balance},
        )
        raise NotEnoughFunds(f"insufficient funds for user {user_id}")
    logger.debug(
        "billing.charge.ok",
        extra={"user_id": user_id, "amount": amount, "balance": new_balance},
    )
    return new_balance


async def refund(user_id: int, amount: int, reason: str = "") -> int:
    """Return ``amount`` tokens back to ``user_id``."""

    if amount < 0:
        raise ValueError("amount must be non-negative")

    def _credit() -> int:
        return credit_balance(user_id, int(amount), reason or "refund")

    new_balance = await _run_in_executor(_credit)
    logger.debug(
        "billing.refund.ok",
        extra={"user_id": user_id, "amount": amount, "balance": new_balance},
    )
    return new_balance


async def get_history(user_id: int) -> List[dict[str, Any]]:
    """Return a list of billing operations for ``user_id``.

    The default implementation returns an empty list. Applications can
    monkeypatch or replace this module in tests to provide mock data.
    """

    return []


__all__ = [
    "BillingError",
    "NotEnoughFunds",
    "charge",
    "refund",
    "get_history",
]
