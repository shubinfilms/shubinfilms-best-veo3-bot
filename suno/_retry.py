"""Utility module providing tenacity-style retry helpers."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Iterable, Tuple, Type

try:  # pragma: no cover - prefer real tenacity
    from tenacity import (  # type: ignore
        RetryError,
        Retrying,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
    )
except Exception:  # pragma: no cover - fallback implementation
    class RetryError(Exception):
        """Raised when retry attempts are exhausted."""

    @dataclass
    class _StopConfig:
        max_attempts: int

    @dataclass
    class _RetryConfig:
        exceptions: Tuple[Type[BaseException], ...]

    @dataclass
    class _WaitConfig:
        multiplier: float
        minimum: float
        maximum: float

    def stop_after_attempt(attempts: int) -> _StopConfig:
        return _StopConfig(max_attempts=attempts)

    def retry_if_exception_type(exceptions: Iterable[Type[BaseException]] | Type[BaseException]) -> _RetryConfig:
        if isinstance(exceptions, type) and issubclass(exceptions, BaseException):
            exc_tuple = (exceptions,)
        else:
            exc_tuple = tuple(exceptions)
        return _RetryConfig(exceptions=exc_tuple)

    def wait_exponential(multiplier: float = 1.0, min: float = 0.0, max: float = 60.0) -> _WaitConfig:
        return _WaitConfig(multiplier=multiplier, minimum=min, maximum=max)

    class Retrying:
        def __init__(self, stop: _StopConfig, wait: _WaitConfig, retry: _RetryConfig, reraise: bool = True) -> None:
            self._max_attempts = max(1, stop.max_attempts)
            self._exceptions = retry.exceptions or (Exception,)
            self._wait = wait
            self._reraise = reraise

        def __call__(self, fn: Callable, *args, **kwargs):
            delay = self._wait.minimum or 0
            last_exc: BaseException | None = None
            for attempt in range(1, self._max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except self._exceptions as exc:  # type: ignore[misc]
                    last_exc = exc
                    if attempt >= self._max_attempts:
                        break
                    delay = delay * 2 if delay else max(self._wait.multiplier, 0.1)
                    delay = min(delay, self._wait.maximum)
                    time.sleep(delay)
            if self._reraise and last_exc:
                raise last_exc
            raise RetryError("Maximum retry attempts reached")

        def call(self, fn: Callable, *args, **kwargs):
            return self.__call__(fn, *args, **kwargs)

__all__ = [
    "RetryError",
    "Retrying",
    "retry_if_exception_type",
    "stop_after_attempt",
    "wait_exponential",
]
