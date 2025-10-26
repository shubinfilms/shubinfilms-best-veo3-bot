"""Prometheus metrics helpers shared across bot and web services."""
from __future__ import annotations

import logging
import os
import re
import time
from typing import Iterable

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

REGISTRY = CollectorRegistry()

_ENV = (os.getenv("APP_ENV") or "prod").strip() or "prod"
_log = logging.getLogger(__name__)


def _labels(service: str) -> dict[str, str]:
    return {"env": _ENV, "service": service}


def inc(
    metric_name: str,
    *,
    value: float = 1.0,
    tags: dict[str, str] | None = None,
    service: str = "bot",
) -> None:
    """Increment a Prometheus metric by name with provided tags."""

    metric = globals().get(metric_name)
    if metric is None:
        raise ValueError(f"unknown metric {metric_name!r}")

    labels = dict(tags or {})
    label_names = tuple(getattr(metric, "_labelnames", ()) or ())
    if "env" in label_names:
        labels.setdefault("env", _ENV)
    if "service" in label_names:
        labels.setdefault("service", service)
    metric.labels(**labels).inc(value)

suno_requests_total = Counter(
    "suno_requests_total",
    "Total Suno requests grouped by outcome",
    labelnames=("result", "reason", "api_version", "env", "service"),
    registry=REGISTRY,
)

suno_callback_total = Counter(
    "suno_callback_total",
    "Total Suno callbacks processed",
    labelnames=("status", "env", "service"),
    registry=REGISTRY,
)

suno_enqueue_total = Counter(
    "suno_enqueue_total",
    "Suno enqueue attempts grouped by outcome",
    labelnames=("outcome", "api", "env", "service"),
    registry=REGISTRY,
)

suno_notify_total = Counter(
    "suno_notify_total",
    "Launch acknowledgement notifications outcome",
    labelnames=("outcome", "env", "service"),
    registry=REGISTRY,
)

suno_notify_latency_ms = Histogram(
    "suno_notify_latency_ms",
    "Latency of Suno launch acknowledgements in milliseconds",
    labelnames=("env", "service"),
    registry=REGISTRY,
)

suno_refund_total = Counter(
    "suno_refund_total",
    "Total Suno refunds grouped by reason",
    labelnames=("reason", "env", "service"),
    registry=REGISTRY,
)

telegram_send_total = Counter(
    "telegram_send_total",
    "Telegram send attempts grouped by kind/result",
    labelnames=("kind", "result", "env", "service"),
    registry=REGISTRY,
)

ui_callback_ack_total = Counter(
    "ui_callback_ack_total",
    "Callback query acknowledgement attempts grouped by result.",
    labelnames=("result", "env", "service"),
    registry=REGISTRY,
)

ui_callback_ack_latency_ms = Histogram(
    "ui_callback_ack_latency_ms",
    "Latency of callback query acknowledgements in milliseconds.",
    labelnames=("env", "service"),
    registry=REGISTRY,
    buckets=(5, 10, 25, 50, 100, 150, 250, 500, 1000),
)

ui_ack_latency_ms = Histogram(
    "ui_ack_latency_ms",
    "Latency between receiving and acknowledging a callback query in milliseconds.",
    labelnames=("source", "env", "service"),
    registry=REGISTRY,
    buckets=(5, 10, 25, 50, 100, 150, 250, 500, 1000, 2000),
)

callback_ack_latency_ms = Histogram(
    "callback_ack_latency_ms",
    "End-to-end latency of callback acknowledgements in milliseconds.",
    labelnames=("source", "env", "service"),
    registry=REGISTRY,
    buckets=(5, 10, 25, 50, 75, 100, 150, 250, 400, 600, 1000),
)

ui_callback_total = Counter(
    "ui_callback_total",
    "Callback processing outcomes grouped by action and result.",
    labelnames=("action", "result", "env", "service"),
    registry=REGISTRY,
)

ui_callback_dedup_total = Counter(
    "ui_callback_dedup_total",
    "Callback query deduplication decisions grouped by action.",
    labelnames=("action", "env", "service"),
    registry=REGISTRY,
)

ui_callback_unmatched_total = Counter(
    "ui_callback_unmatched_total",
    "Callback queries that did not match any UI handler grouped by source.",
    labelnames=("source", "legacy", "env", "service"),
    registry=REGISTRY,
)

ui_callback_legacy_forwarded_total = Counter(
    "ui_callback_legacy_forwarded_total",
    "Legacy callback payloads forwarded to registered actions.",
    labelnames=("target", "env", "service"),
    registry=REGISTRY,
)

router_callback_legacy_total = Counter(
    "router_callback_legacy_total",
    "Legacy callback payloads forwarded to namespace router handlers.",
    labelnames=("namespace", "action"),
    registry=REGISTRY,
)

profile_open_total = Counter(
    "profile_open_total",
    "Profile open attempts grouped by source and result.",
    labelnames=("source", "result"),
    registry=REGISTRY,
)

profile_first_paint_ms = Histogram(
    "profile_first_paint_ms",
    "Latency between callback reception and first profile render in milliseconds.",
    labelnames=("source",),
    registry=REGISTRY,
    buckets=(25, 50, 100, 150, 200, 300, 400, 600, 1000, 1500),
)

router_callback_unmatched_total = Counter(
    "router_callback_unmatched_total",
    "Namespace router callbacks that were not matched to any handler.",
    labelnames=("data",),
    registry=REGISTRY,
)

ui_wait_clear_all_total = Counter(
    "ui_wait_clear_all_total",
    "Forced wait/input state clears grouped by reason.",
    labelnames=("reason", "env", "service"),
    registry=REGISTRY,
)

sum_open_total = Counter(
    "sum_open_total",
    "Summary card open attempts grouped by result.",
    labelnames=("result", "env", "service"),
    registry=REGISTRY,
)

sum_start_total = Counter(
    "sum_start_total",
    "Summary generation attempts grouped by outcome.",
    labelnames=("result", "env", "service"),
    registry=REGISTRY,
)

faq_root_views_total = Counter(
    "faq_root_views_total",
    "Total number of FAQ root menu views",
    labelnames=("env", "service"),
    registry=REGISTRY,
)

faq_views_total = Counter(
    "faq_views_total",
    "FAQ section views grouped by section",
    labelnames=("section", "env", "service"),
    registry=REGISTRY,
)

chat_messages_total = Counter(
    "chat_messages_total",
    "Total chat messages processed",
    labelnames=("outcome",),
    registry=REGISTRY,
)

chat_latency_ms = Histogram(
    "chat_latency_ms",
    "Chat roundtrip latency in milliseconds",
    buckets=(50, 100, 200, 400, 800, 1500, 3000, 6000, 10000),
    registry=REGISTRY,
)

chat_context_tokens = Gauge(
    "chat_context_tokens",
    "Estimated tokens in chat context (last observed)",
    registry=REGISTRY,
)

chat_autoswitch_total = Counter(
    "chat_autoswitch_total",
    "Automatic chat mode routing events grouped by outcome",
    labelnames=("outcome",),
    registry=REGISTRY,
)

stars_buy_total = Counter(
    "stars_buy_total",
    "Telegram Stars purchase attempts grouped by amount and result.",
    labelnames=("amount", "result"),
    registry=REGISTRY,
)

chat_first_hint_total = Counter(
    "chat_first_hint_total",
    "Total hint messages shown for automatic chat activation",
    registry=REGISTRY,
)

chat_voice_total = Counter(
    "chat_voice_total",
    "Voice messages processed in chat",
    labelnames=("outcome", "env", "service"),
    registry=REGISTRY,
)

chat_voice_latency_ms = Histogram(
    "chat_voice_latency_ms",
    "Latency of handling chat voice messages in milliseconds",
    labelnames=("env", "service"),
    registry=REGISTRY,
)

chat_transcribe_latency_ms = Histogram(
    "chat_transcribe_latency_ms",
    "Latency of audio transcription calls in milliseconds",
    labelnames=("env", "service"),
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

suno_poll_status_total = Counter(
    "suno_poll_status_total",
    "Suno record-info poll outcomes grouped by HTTP status and mapped state",
    labelnames=("http_status", "state", "env", "service"),
    registry=REGISTRY,
)

suno_ready_latency_seconds = Histogram(
    "suno_ready_latency_seconds",
    "Time from poll start to ready state",
    labelnames=("env", "service"),
    registry=REGISTRY,
)

suno_late_delivery_total = Counter(
    "suno_late_delivery_total",
    "Late Suno deliveries grouped by trigger",
    labelnames=("trigger", "env", "service"),
    registry=REGISTRY,
)

suno_delivery_total = Counter(
    "suno_delivery_total",
    "Suno delivery attempts grouped by result",
    labelnames=("result", "env", "service"),
    registry=REGISTRY,
)

suno_timeout_total = Counter(
    "suno_timeout_total",
    "Suno polling timeouts grouped by stage",
    labelnames=("stage", "env", "service"),
    registry=REGISTRY,
)

suno_refund_trigger_total = Counter(
    "suno_refund_trigger_total",
    "Suno refund triggers grouped by stage",
    labelnames=("stage", "env", "service"),
    registry=REGISTRY,
)

suno_refund_outcome_total = Counter(
    "suno_refund_outcome_total",
    "Suno refund outcomes grouped by result",
    labelnames=("result", "env", "service"),
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


def safe_label(value: str) -> str:
    """Return a sanitized label value suitable for Prometheus."""

    normalized = re.sub(r"[^a-zA-Z0-9:_\-\|\.]", "_", str(value))
    return normalized[:64]


def lbl_safe(metric, /, **labels):
    """Return a labelled child instance without propagating errors."""

    try:
        return metric.labels(**labels)
    except Exception:  # pragma: no cover - defensive guard
        _log.debug(
            "metrics.lbl_safe.failed",
            exc_info=True,
            extra={"metric": getattr(metric, "_name", None), "labels": labels},
        )
        return None


__all__: Iterable[str] = [
    "REGISTRY",
    "inc",
    "lbl_safe",
    "safe_label",
    "suno_requests_total",
    "suno_callback_download_fail_total",
    "suno_task_store_total",
    "bot_telegram_send_fail_total",
    "suno_callback_total",
    "suno_enqueue_total",
    "suno_notify_total",
    "suno_notify_latency_ms",
    "suno_refund_total",
    "telegram_send_total",
    "faq_root_views_total",
    "faq_views_total",
    "suno_latency_seconds",
    "suno_notify_ok",
    "suno_notify_fail",
    "suno_notify_duration_seconds",
    "suno_enqueue_duration_seconds",
    "chat_messages_total",
    "chat_latency_ms",
    "chat_context_tokens",
    "chat_autoswitch_total",
    "chat_first_hint_total",
    "chat_voice_total",
    "chat_voice_latency_ms",
    "chat_transcribe_latency_ms",
    "ui_ack_latency_ms",
    "ui_callback_total",
    "ui_callback_ack_total",
    "ui_callback_ack_latency_ms",
    "ui_callback_dedup_total",
    "ui_callback_unmatched_total",
    "profile_open_total",
    "profile_first_paint_ms",
    "router_callback_unmatched_total",
    "process_uptime_seconds",
    "render_metrics",
]
