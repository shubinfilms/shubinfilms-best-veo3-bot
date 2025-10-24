import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import utils.input_state as input_state
from helpers import debounce as debounce_helper
import redis_utils


def test_waits_are_cleared_on_profile_open(monkeypatch):
    wait_calls: list[tuple[str, int, str]] = []

    def fake_clear_wait_state(user_id: int, *, reason: str = "manual") -> None:
        wait_calls.append(("wait_state", int(user_id), reason))

    registry_calls: list[tuple[int, str]] = []

    class RegistryStub:
        def clear(self, chat_id: int, *, reason: str = "manual") -> None:
            registry_calls.append((int(chat_id), reason))

    release_calls: list[tuple[int, str]] = []

    def fake_release_user_lock(user_id: int, key: str) -> None:
        release_calls.append((int(user_id), key))

    mode_calls: list[int] = []

    async def fake_clear_mode_state(user_id: int) -> None:
        mode_calls.append(int(user_id))

    debounce_calls: list[int] = []

    def fake_debounce_reset(user_id: int) -> None:
        debounce_calls.append(int(user_id))

    metric_calls: list[tuple[str, dict | None]] = []

    def fake_metrics_inc(metric_name: str, *, value: float = 1.0, tags=None, service: str = "bot") -> None:
        metric_calls.append((metric_name, dict(tags or {})))

    monkeypatch.setattr(input_state, "clear_wait_state", fake_clear_wait_state)
    monkeypatch.setattr(input_state, "input_state", RegistryStub())
    monkeypatch.setattr(redis_utils, "release_user_lock", fake_release_user_lock)
    monkeypatch.setattr(redis_utils, "clear_mode_state", fake_clear_mode_state)
    monkeypatch.setattr(debounce_helper, "reset", fake_debounce_reset)
    monkeypatch.setattr(input_state, "metrics_inc", fake_metrics_inc)

    asyncio.run(input_state.force_clear_user_state(404, reason="profile_open"))

    assert wait_calls == [("wait_state", 404, "profile_open")]
    assert registry_calls == [(404, "profile_open")]
    assert mode_calls == [404]
    assert debounce_calls == [404]
    assert metric_calls == [("ui_wait_clear_all_total", {"reason": "profile_open"})]
    released_keys = {key for _, key in release_calls}
    assert released_keys == {"video_menu", "reply-nav", "dialog", "kb", "mode_reset"}
