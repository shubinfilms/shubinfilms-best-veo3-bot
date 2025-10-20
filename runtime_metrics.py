"""In-process counters and gauges for lightweight diagnostics."""
from __future__ import annotations

import threading
from typing import Dict, MutableMapping

_LOCK = threading.Lock()
_COUNTERS: Dict[str, int] = {}
_GAUGES: Dict[str, Dict[str, float]] = {}


def increment_counter(name: str, value: int = 1) -> int:
    """Increment a named counter and return the new value."""

    if not name:
        raise ValueError("counter name must be provided")
    if value == 0:
        return get_counter(name)
    with _LOCK:
        new_value = _COUNTERS.get(name, 0) + int(value)
        _COUNTERS[name] = new_value
        return new_value


def get_counter(name: str) -> int:
    """Return the current value of a named counter (defaults to ``0``)."""

    with _LOCK:
        return int(_COUNTERS.get(name, 0))


def set_counter(name: str, value: int) -> None:
    """Explicitly set a counter value."""

    if not name:
        raise ValueError("counter name must be provided")
    with _LOCK:
        _COUNTERS[name] = int(value)


def update_gauge(
    name: str,
    value: float,
    *,
    maximum: float | None = None,
    track_peak: bool = True,
) -> Dict[str, float]:
    """Update a gauge value and optionally record peak/maximum values."""

    if not name:
        raise ValueError("gauge name must be provided")
    numeric_value = float(value)
    with _LOCK:
        record = _GAUGES.setdefault(name, {"value": 0.0, "peak": 0.0, "max": 0.0})
        record["value"] = numeric_value
        if track_peak:
            record["peak"] = max(record.get("peak", 0.0), numeric_value)
        if maximum is not None:
            record["max"] = max(record.get("max", 0.0), float(maximum))
        return dict(record)


def get_gauge(name: str) -> Dict[str, float]:
    """Return a snapshot of a gauge value (value/peak/max)."""

    with _LOCK:
        record = _GAUGES.get(name, {"value": 0.0, "peak": 0.0, "max": 0.0})
        return dict(record)


def snapshot() -> Dict[str, MutableMapping[str, float]]:
    """Return a copy of current counters and gauges."""

    with _LOCK:
        return {
            "counters": dict(_COUNTERS),
            "gauges": {name: dict(payload) for name, payload in _GAUGES.items()},
        }


def increment_ui_callback_counter(kind: str, result: str) -> int:
    """Increment a structured callback counter (ui.callback.<kind>.<result>)."""

    if not kind:
        raise ValueError("kind must be provided")
    if not result:
        raise ValueError("result must be provided")
    name = f"ui.callback.{kind}.{result}"
    return increment_counter(name)
