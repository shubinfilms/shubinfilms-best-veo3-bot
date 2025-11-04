"""Shared helpers for async error handling."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("async-errors")


async def handle_async_error(exc: Exception, context: str) -> dict[str, Any]:
    """Log ``exc`` with ``context`` and return a structured payload."""

    logger.error("[%s] %s: %s", context, type(exc).__name__, exc)
    return {"ok": False, "error": str(exc)}


__all__ = ["handle_async_error"]
