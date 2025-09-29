import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.api_client import request_with_retries


def test_request_with_retries_respects_jitter_and_max(monkeypatch):
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    def fake_uniform(low: float, high: float) -> float:
        return high

    monkeypatch.setattr("utils.api_client.asyncio.sleep", fake_sleep)
    monkeypatch.setattr("utils.api_client.random.uniform", fake_uniform)

    attempts = {"count": 0}

    async def operation() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError(f"boom-{attempts['count']}")
        return "ok"

    result = asyncio.run(
        request_with_retries(
            operation,
            attempts=3,
            base_delay=1.0,
            max_delay=4.0,
            backoff_factor=2.0,
            jitter=0.5,
            max_total_delay=10.0,
            logger=None,
        )
    )

    assert result == "ok"
    assert attempts["count"] == 3
    assert len(delays) == 2
    assert delays[0] == pytest.approx(1.5)
    assert delays[1] == pytest.approx(3.0)
