import json
import os
import sys
from pathlib import Path

import pytest
import requests
from fastapi.testclient import TestClient

os.environ.setdefault("SUNO_API_TOKEN", "test-token")
os.environ.setdefault("SUNO_CALLBACK_SECRET", "secret-token")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import suno_web
from suno_web import app
import suno.tempfiles as tempfiles


def _build_payload(task_id: str = "task-1", callback_type: str = "complete") -> dict:
    return {
        "code": 200,
        "msg": "ok",
        "data": {
            "callbackType": callback_type,
            "task_id": task_id,
            "data": [
                {
                    "id": "trk1",
                    "title": "Demo",
                    "audio_url": "https://cdn.example.com/demo.mp3",
                    "image_url": "https://cdn.example.com/demo.jpg",
                }
            ],
        },
    }


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch):
    monkeypatch.setenv("SUNO_CALLBACK_SECRET", "secret-token")
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    monkeypatch.delenv("TELEGRAM_TOKEN", raising=False)
    monkeypatch.delenv("ADMIN_IDS", raising=False)
    suno_web._memory_idempotency.clear()


def _client():
    return TestClient(app)


def test_callback_accepts_header_token(monkeypatch):
    received = {}

    def fake_handle(task):
        received["task"] = task

    monkeypatch.setattr(suno_web.service, "handle_callback", fake_handle)
    client = _client()
    response = client.post(
        "/suno-callback",
        headers={"X-Callback-Token": "secret-token"},
        json=_build_payload("header-token"),
    )
    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert received["task"].task_id == "header-token"


def test_callback_accepts_query_token(monkeypatch):
    received = {}

    def fake_handle(task):
        received["task"] = task

    monkeypatch.setattr(suno_web.service, "handle_callback", fake_handle)
    client = _client()
    response = client.post(
        "/suno-callback?token=secret-token",
        json=_build_payload("query-token"),
    )
    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert received["task"].task_id == "query-token"


def test_callback_rejects_wrong_token():
    client = _client()
    response = client.post(
        "/suno-callback",
        headers={"X-Callback-Token": "bad"},
        json=_build_payload(),
    )
    assert response.status_code == 403
    assert response.json()["error"] == "forbidden"


def test_duplicate_callbacks_processed_once(monkeypatch):
    count = {"calls": 0}

    def fake_handle(task):
        count["calls"] += 1

    monkeypatch.setattr(suno_web.service, "handle_callback", fake_handle)
    client = _client()
    payload = _build_payload("dup-task")
    for _ in range(2):
        response = client.post(
            "/suno-callback",
            headers={"X-Callback-Token": "secret-token"},
            json=payload,
        )
        assert response.status_code == 200
    assert count["calls"] == 1


def test_callback_rejects_payloads_over_limit(monkeypatch):
    invoked = {"called": False}

    def fake_handle(task):  # pragma: no cover - should not be called
        invoked["called"] = True

    monkeypatch.setattr(suno_web.service, "handle_callback", fake_handle)
    client = _client()
    oversized = "x" * (suno_web._MAX_JSON_BYTES + 10)
    body = json.dumps({"data": oversized})
    response = client.post(
        "/suno-callback",
        headers={"X-Callback-Token": "secret-token", "Content-Type": "application/json"},
        content=body,
    )
    assert response.status_code == 413
    assert response.json()["detail"] == "payload too large"
    assert invoked["called"] is False


def test_download_failure_falls_back_to_url(monkeypatch, tmp_path):
    captured = {}

    def fake_handle(task):
        captured["task"] = task

    monkeypatch.setattr(suno_web.service, "handle_callback", fake_handle)

    class FakeResp:
        def __init__(self, status_code: int, body: bytes = b"", ct: str = "application/octet-stream"):
            self.status_code = status_code
            self.headers = {"Content-Type": ct}
            self._body = body

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def iter_content(self, chunk_size=8192):
            yield self._body

    class FakeSession:
        def get(self, url, *args, **kwargs):
            if url.endswith("demo.mp3"):
                return FakeResp(403)
            return FakeResp(200, body=b"data", ct="image/jpeg")

    monkeypatch.setattr(suno_web, "_session", FakeSession(), raising=False)
    monkeypatch.setattr(tempfiles, "BASE_DIR", Path(tmp_path), raising=False)
    client = _client()
    response = client.post(
        "/suno-callback",
        headers={"X-Callback-Token": "secret-token"},
        json=_build_payload("asset-task"),
    )
    assert response.status_code == 200
    task = captured["task"]
    assert task.task_id == "asset-task"
    assert task.items[0].audio_url == "https://cdn.example.com/demo.mp3"
    image_path = task.items[0].image_url
    assert image_path
    assert Path(image_path).exists()
