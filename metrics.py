"""Prometheus metrics helpers shared across bot and web services."""
from __future__ import annotations

import os
import time
from typing import Iterable

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

REGISTRY = CollectorRegistry()

_ENV = (os.getenv("APP_ENV") or "prod").strip() or "prod"


def _labels(service: str) -> dict[str, str]:
    return {"env": _ENV, "service": service}

suno_requests_total = Counter(
    "suno_requests_total",
    "Total Suno requests grouped by outcome",
    labelnames=("result", "reason", "env", "service"),
    registry=REGISTRY,
)

suno_callback_total = Counter(
    "suno_callback_total",
    "Total Suno callbacks processed",
    labelnames=("status", "env", "service"),
    registry=REGISTRY,
)

telegram_send_total = Counter(
    "telegram_send_total",
    "Telegram send attempts grouped by kind/result",
    labelnames=("kind", "result", "env", "service"),
    registry=REGISTRY,
)

suno_latency_seconds = Histogram(
    "suno_latency_seconds",
    "Latency from task start to callback",
    labelnames=("env", "service"),
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

suno_notify_ok = Counter(
    "suno_notify_ok",
    "Successful Suno launch notifications",
    labelnames=("env", "service"),
    registry=REGISTRY,
)

suno_notify_fail = Counter(
    "suno_notify_fail",
    "Failed Suno launch notifications grouped by error type",
    labelnames=("type", "env", "service"),
    registry=REGISTRY,
)

suno_notify_duration_seconds = Histogram(
    "suno_notify_duration_seconds",
    "Duration of Suno launch acknowledgement notifications",
    labelnames=("env", "service"),
    registry=REGISTRY,
)

suno_enqueue_duration_seconds = Histogram(
    "suno_enqueue_duration_seconds",
    "Duration of Suno enqueue operations",
    labelnames=("env", "service"),
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
    "suno_requests_total",
    "suno_callback_download_fail_total",
    "suno_task_store_total",
    "bot_telegram_send_fail_total",
    "suno_callback_total",
    "telegram_send_total",
    "suno_latency_seconds",
    "suno_notify_ok",
    "suno_notify_fail",
    "suno_notify_duration_seconds",
    "suno_enqueue_duration_seconds",
    "process_uptime_seconds",
    "render_metrics",
]
