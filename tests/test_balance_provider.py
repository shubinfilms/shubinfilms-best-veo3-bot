from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.balance_provider import aget_balance_snapshot, get_balance_snapshot


@patch("core.balance_provider._get_ledger_storage")
@patch("core.balance_provider.redis_get_balance", return_value=None)
def test_balance_provider_fallback_on_redis_miss(
    redis_mock: MagicMock,
    ledger_mock: MagicMock,
    user_id: int,
) -> None:
    ledger_mock.return_value.get_balance.return_value = 42
    snapshot = get_balance_snapshot(user_id)
    assert snapshot.is_available
    assert snapshot.value == 42


@pytest.mark.anyio
@patch("core.balance_provider._get_ledger_storage")
@patch("core.balance_provider.redis_get_balance", return_value=None)
async def test_balance_provider_fallback_on_redis_miss_async(
    redis_mock: MagicMock,
    ledger_mock: MagicMock,
    user_id: int,
) -> None:
    ledger_mock.return_value.get_balance.return_value = 42
    snapshot = await aget_balance_snapshot(user_id)
    assert snapshot.is_available
    assert snapshot.value == 42
