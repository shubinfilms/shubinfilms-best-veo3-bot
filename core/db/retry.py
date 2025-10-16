"""Retry helpers for transient PostgreSQL errors."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Optional

import psycopg

RETRYABLE_MESSAGES = (
    "SSL connection has been closed unexpectedly",
    "Connection reset by peer",
    "server closed the connection unexpectedly",
    "terminating connection due to administrator command",
)


def is_retryable_db_error(exc: BaseException) -> bool:
    message = str(exc)
    return isinstance(exc, psycopg.OperationalError) and any(
        fragment in message for fragment in RETRYABLE_MESSAGES
    )


def with_db_retries(
    fn: Callable[[], Any],
    *,
    attempts: int = 3,
    backoff: float = 0.2,
    logger: Optional[logging.Logger] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Any:
    """Execute ``fn`` retrying on transient PostgreSQL failures."""

    context_meta: Dict[str, Any] = dict(context or {})
    last_exc: Optional[BaseException] = None
    for attempt in range(1, attempts + 1):
        try:
            result = fn()
            if logger and attempt > 1:
                logger.info("DB_RETRY_OK", extra={"attempt": attempt, **context_meta})
            return result
        except Exception as exc:  # noqa: BLE001 - deliberate broad catch for retry logic
            last_exc = exc
            if not is_retryable_db_error(exc):
                if logger:
                    logger.error(
                        "DB_RETRY_GIVEUP",
                        extra={"err": str(exc), "attempt": attempt, **context_meta},
                    )
                raise
            if attempt == attempts:
                if logger:
                    logger.error(
                        "DB_RETRY_GIVEUP",
                        extra={"err": str(exc), "attempt": attempt, **context_meta},
                    )
                raise
            if logger:
                logger.warning(
                    "DB_RETRY",
                    extra={"err": str(exc), "attempt": attempt, **context_meta},
                )
            time.sleep(backoff * attempt)
    if last_exc is not None:  # pragma: no cover - defensive, loop should return or raise
        raise last_exc
