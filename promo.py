"""Simple promo-code validation helpers for the bot."""
from __future__ import annotations


def apply(user_id: int, code: str) -> bool:
    """Validate ``code`` for ``user_id``.

    By default, codes that start with ``"VE"`` (case-insensitive) are
    considered valid. Integrations can monkeypatch this function in tests
    or replace the module with a real implementation.
    """

    if not code:
        return False
    return code.strip().upper().startswith("VE")
