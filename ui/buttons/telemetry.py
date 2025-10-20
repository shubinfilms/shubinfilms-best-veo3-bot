from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

from prometheus_client import Counter, Histogram


_click_total = Counter(
    "ui_button_click_total",
    "Total number of button clicks grouped by id and tier.",
    labelnames=("button_id", "user_tier"),
)
_success_total = Counter(
    "ui_button_handler_success_total",
    "Number of successful button handler executions.",
    labelnames=("button_id",),
)
_error_total = Counter(
    "ui_button_handler_error_total",
    "Number of button handler errors grouped by kind.",
    labelnames=("button_id", "error_kind"),
)
_latency_seconds = Histogram(
    "ui_button_latency_seconds",
    "Execution latency of button handlers in seconds.",
    labelnames=("button_id",),
)


def log_click(button_id: str, user_tier: str) -> None:
    _click_total.labels(button_id=button_id, user_tier=user_tier).inc()


def log_success(button_id: str) -> None:
    _success_total.labels(button_id=button_id).inc()


def log_error(button_id: str, error_kind: str) -> None:
    _error_total.labels(button_id=button_id, error_kind=error_kind).inc()


@contextmanager
def measure_latency(button_id: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = max(time.perf_counter() - start, 0.0)
        _latency_seconds.labels(button_id=button_id).observe(duration)
