"""Reusable helpers for resilient API interactions."""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any, Awaitable, Callable, Optional, TypeVar

T = TypeVar("T")


def _default_retry_filter(exc: BaseException) -> bool:
    """Retry on generic transient errors by default."""

    return isinstance(exc, Exception)


async def request_with_retries(
    operation: Callable[[], Awaitable[T] | T],
    *,
    attempts: int = 3,
    base_delay: float = 0.8,
    max_delay: float = 6.0,
    backoff_factor: float = 2.0,
    jitter: float | tuple[float, float] | None = None,
    max_total_delay: Optional[float] = None,
    logger: Optional[logging.Logger] = None,
    log_context: Optional[dict[str, Any]] = None,
    retry_filter: Optional[Callable[[BaseException], bool]] = None,
) -> T:
    """Execute ``operation`` with retries and exponential backoff.

    ``operation`` may be synchronous or asynchronous. If it raises an exception
    deemed retryable by ``retry_filter`` it will be re-run until ``attempts`` are
    exhausted. Non-retryable errors are re-raised immediately.
    """

    if attempts <= 0:
        raise ValueError("attempts must be positive")

    retry_checker = retry_filter or _default_retry_filter
    log_extra = dict(log_context or {})
    attempt = 0
    last_error: Optional[BaseException] = None
    total_delay = 0.0

    while attempt < attempts:
        attempt += 1
        try:
            result = operation()
            if asyncio.iscoroutine(result) or isinstance(result, Awaitable):
                return await result  # type: ignore[return-value]
            return result  # type: ignore[return-value]
        except BaseException as exc:  # noqa: PERF203 - intentional broad catch
            last_error = exc
            if not retry_checker(exc) or attempt >= attempts:
                raise
            delay = max(0.0, base_delay * (backoff_factor ** (attempt - 1)))
            delay = min(max_delay, delay)
            if jitter:
                low: float
                high: float
                if isinstance(jitter, tuple):
                    low, high = jitter
                else:
                    low, high = 0.0, float(jitter)
                if high < low:
                    low, high = high, low
                jitter_value = random.uniform(low, high)
                delay = min(max(delay + jitter_value, 0.0), max_delay)
            if max_total_delay is not None:
                remaining = max_total_delay - total_delay
                if remaining <= 0:
                    break
                delay = min(delay, max(remaining, 0.0))
            if logger:
                logger.warning(
                    "api.retry",  # noqa: TRY400 - structured logging key
                    extra={
                        **log_extra,
                        "attempt": attempt,
                        "max_attempts": attempts,
                        "delay": round(delay, 3),
                        "error": str(exc),
                    },
                )
            total_delay += delay
            if delay > 0:
                await asyncio.sleep(delay)

    # In practice the loop returns or raises, but mypy expects a fallback.
    if last_error is not None:  # pragma: no cover - defensive guard
        raise last_error
    raise RuntimeError("request_with_retries exhausted without executing")


__all__ = ["request_with_retries"]

