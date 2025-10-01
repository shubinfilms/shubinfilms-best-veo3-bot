"""Role helpers for privileged accounts."""
from __future__ import annotations

import logging
import os
from typing import Any

_log = logging.getLogger("roles")


def _env_int(name: str, default: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    text = raw.strip()
    if not text:
        return default
    try:
        return int(text)
    except ValueError:
        _log.warning("Invalid integer for %s: %s", name, text)
        return default


SUPPORT_USER_ID: int = _env_int("SUPPORT_USER_ID", 0)


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
