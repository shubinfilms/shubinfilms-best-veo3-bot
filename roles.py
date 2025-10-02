"""Role helpers for privileged accounts."""
from __future__ import annotations

import logging
from typing import Any

from settings import SUPPORT_USER_ID

_log = logging.getLogger("roles")


def is_support(user_id: Any) -> bool:
    """Return ``True`` if ``user_id`` matches the configured support account."""

    if user_id is None:
        return False
    try:
        numeric_id = int(user_id)
    except (TypeError, ValueError):
        return False
    if SUPPORT_USER_ID <= 0:
        return False
    return numeric_id == SUPPORT_USER_ID


__all__ = ["SUPPORT_USER_ID", "is_support"]
