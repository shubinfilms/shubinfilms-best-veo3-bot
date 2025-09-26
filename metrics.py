"""Prometheus metrics helpers shared across bot and web services."""
from __future__ import annotations

import time
from typing import Iterable

from prometheus_client import CollectorRegistry, Counter, Gauge, generate_latest

REGISTRY = CollectorRegistry()

suno_callback_total = Counter(
    "suno_callback_total",
    "Total Suno callbacks processed",
    labelnames=("type", "code"),
    registry=REGISTRY,
)

suno_callback_download_fail_total = Counter(
    "suno_callback_download_fail_total",
    "Number of failed callback asset downloads",
    labelnames=("reason",),
    registry=REGISTRY,
)

suno_task_store_total = Counter(
    "suno_task_store_total",
    "Suno task storage operations",
    labelnames=("result",),
    registry=REGISTRY,
)

bot_telegram_send_fail_total = Counter(
    "bot_telegram_send_fail_total",
    "Telegram send failures from the bot",
    labelnames=("method",),
    registry=REGISTRY,
)

process_uptime_seconds = Gauge(
    "process_uptime_seconds",
    "Process uptime in seconds",
    registry=REGISTRY,
)

_START_TIME = time.time()


def render_metrics() -> bytes:
    """Return the current metrics payload in Prometheus text format."""

    process_uptime_seconds.set(max(0.0, time.time() - _START_TIME))
    return generate_latest(REGISTRY)


__all__: Iterable[str] = [
    "REGISTRY",
    "suno_callback_total",
    "suno_callback_download_fail_total",
    "suno_task_store_total",
    "bot_telegram_send_fail_total",
    "process_uptime_seconds",
    "render_metrics",
]
