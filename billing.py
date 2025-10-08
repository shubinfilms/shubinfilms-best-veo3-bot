"""Minimal billing service facade used by profile handlers.

The real project can replace this module with a proper implementation.
"""
from __future__ import annotations

from typing import Any, List


async def get_history(user_id: int) -> List[dict[str, Any]]:
    """Return a list of billing operations for ``user_id``.

    The default implementation returns an empty list. Applications can
    monkeypatch or replace this module in tests to provide mock data.
    """

    return []
