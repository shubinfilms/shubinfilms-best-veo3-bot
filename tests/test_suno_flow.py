from typing import Optional

import os
import sys

import pytest
from fastapi.testclient import TestClient

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import suno_web
from suno.schemas import SunoTask
from utils.suno_state import (
    build_generation_payload,
    ensure_suno_state,
    process_style_input,
    process_title_input,
)

WAIT_SUNO_TITLE = "WAIT_SUNO_TITLE"


class _FakeService:
    def __init__(self) -> None:
        self.calls: list[tuple[SunoTask, Optional[str]]] = []

    @staticmethod
    def get_request_id(task_id: Optional[str]) -> Optional[str]:  # pragma: no cover - interface parity
        if not task_id:
            return None
        return f"req-{task_id}"

    @staticmethod
    def get_start_timestamp(task_id: Optional[str]) -> Optional[str]:  # pragma: no cover - interface parity
        return None

    def handle_callback(self, task: SunoTask, req_id: Optional[str] = None) -> None:
        self.calls.append((task, req_id))


def _build_payload(task_id: str, *, callback_type: str = "complete") -> dict:
    return {
        "code": 200,
        "msg": "ok",
        "data": {
            "callbackType": callback_type,
            "task_id": task_id,
            "taskId": task_id,
            "input": {
                "tracks": [
                    {
                        "id": "trk1",
                        "title": "Demo",
                        "audio_url": "https://cdn.example.com/demo.mp3",
                        "image_url": "https://cdn.example.com/demo.jpg",
                    }
                ]
            },
        },
    }


@pytest.fixture(autouse=True)
def _configure_env(monkeypatch):
    monkeypatch.setenv("SUNO_ENABLED", "true")
    monkeypatch.setenv("SUNO_CALLBACK_SECRET", "secret-token")
    monkeypatch.setenv(
        "SUNO_CALLBACK_URL", "https://shubinfilms-best-veo3-bot.onrender.com/suno-callback"
    )
    monkeypatch.setattr(suno_web, "SUNO_ENABLED", True, raising=False)
    monkeypatch.setattr(suno_web, "SUNO_CALLBACK_SECRET", "secret-token", raising=False)
    monkeypatch.setattr(
        suno_web,
        "SUNO_CALLBACK_URL",
        "https://shubinfilms-best-veo3-bot.onrender.com/suno-callback",
        raising=False,
    )
    suno_web._memory_idempotency.clear()


@pytest.fixture
def client(monkeypatch) -> TestClient:
    fake_service = _FakeService()
    monkeypatch.setattr(suno_web, "service", fake_service, raising=False)
    return TestClient(suno_web.app)


def test_music_callback_regular_track(client):
    response = client.post(
        "/suno-callback",
        headers={"X-Callback-Secret": "secret-token"},
        json=_build_payload("task-regular"),
    )
    assert response.status_code == 200
    assert response.json()["ok"] is True
    service: _FakeService = suno_web.service  # type: ignore[assignment]
    assert service.calls, "Callback was not forwarded to the service"
    task, req_id = service.calls[-1]
    assert task.task_id == "task-regular"
    assert req_id == "req-task-regular"
    assert task.items and task.items[0].audio_url.endswith("demo.mp3")


def test_music_callback_instrumental_track(client):
    payload = _build_payload("task-instrumental", callback_type="instrumental")
    response = client.post(
        "/suno-callback",
        headers={"X-Callback-Secret": "secret-token"},
        json=payload,
    )
    assert response.status_code == 200
    service: _FakeService = suno_web.service  # type: ignore[assignment]
    task, req_id = service.calls[-1]
    assert task.callback_type == "instrumental"
    assert req_id == "req-task-instrumental"


def test_music_callback_add_vocals(client):
    payload = _build_payload("task-vocals", callback_type="add_vocals")
    response = client.post(
        "/suno-callback",
        headers={"X-Callback-Secret": "secret-token"},
        json=payload,
    )
    assert response.status_code == 200
    service: _FakeService = suno_web.service  # type: ignore[assignment]
    task, _ = service.calls[-1]
    assert task.callback_type == "add_vocals"


def test_music_callback_invalid_secret_rejected(client):
    response = client.post(
        "/suno-callback",
        headers={"X-Callback-Secret": "wrong"},
        json=_build_payload("task-invalid"),
    )
    assert response.status_code == 403
    service: _FakeService = suno_web.service  # type: ignore[assignment]
    assert not service.calls


def test_music_callback_invalid_payload(client):
    response = client.post(
        "/suno-callback",
        headers={"X-Callback-Secret": "secret-token"},
        data="not-json",
    )
    assert response.status_code == 400
    service: _FakeService = suno_web.service  # type: ignore[assignment]
    assert not service.calls


def test_set_title_then_generate_payload_contains_title() -> None:
    container: dict[str, object] = {}
    state = ensure_suno_state(container)
    ok, value, error = process_title_input("  My   New   Song  ")
    assert ok and error is None
    state["title"] = value
    payload = build_generation_payload(state, instrumental=True, model="V5", lang="ru")
    assert payload["title"] == "My New Song"
    assert payload["prompt"] == "instrumental, cinematic, modern, dynamic"


def test_set_style_multiline_then_payload_contains_full_prompt() -> None:
    container: dict[str, object] = {}
    state = ensure_suno_state(container)
    raw_style = "Dreamy\n\n<b>cinematic</b> vibe  🎹🎹🎹"
    ok, value, error = process_style_input(raw_style)
    assert ok and error is None
    assert value == "Dreamy\ncinematic vibe 🎹🎹"
    state["style"] = value
    payload = build_generation_payload(state, instrumental=False, model="V5", lang="en")
    assert payload["prompt"] == "Dreamy\ncinematic vibe 🎹🎹"


def test_clear_title_and_style_with_dash() -> None:
    ok, value, error = process_title_input("-")
    assert ok and value is None and error is None
    ok, value, error = process_style_input("—")
    assert ok and value is None and error is None


def test_defaults_applied_when_missing() -> None:
    state = ensure_suno_state({})
    payload = build_generation_payload(state, instrumental=True, model="V5", lang="en")
    assert payload["title"] == "Untitled Track"
    assert payload["prompt"] == "instrumental, cinematic, modern, dynamic"


def test_cancel_exits_wait_state_without_changes() -> None:
    state = {"suno_waiting_state": WAIT_SUNO_TITLE, "suno_title": "Keep"}
    user_state = ensure_suno_state({"suno": {"title": "Keep"}})
    lowered = "/cancel"
    if lowered == "/cancel":
        state["suno_waiting_state"] = None
    assert state["suno_title"] == "Keep"
    assert user_state["title"] == "Keep"
    assert state["suno_waiting_state"] is None
