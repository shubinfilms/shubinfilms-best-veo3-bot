import os
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import suno_web


@pytest.fixture(autouse=True)
def _patch_service(monkeypatch):
    monkeypatch.setattr(suno_web, "SUNO_CALLBACK_SECRET", "expected", raising=False)
    monkeypatch.setattr(suno_web, "rds", None, raising=False)
    monkeypatch.setattr(suno_web, "_prepare_assets", lambda task: None, raising=False)
    stub_service = SimpleNamespace(
        get_task_id_by_request=lambda *_args, **_kwargs: None,
        get_request_id=lambda *_args, **_kwargs: None,
        get_start_timestamp=lambda *_args, **_kwargs: None,
        handle_callback=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(suno_web, "service", stub_service, raising=False)
    suno_web._memory_idempotency.clear()
    yield


def test_callback_requires_secret():
    client = TestClient(suno_web.app)
    response = client.post("/suno-callback", json={})
    assert response.status_code == 403

    response = client.post(
        "/suno-callback",
        json={},
        headers={"X-Callback-Secret": "wrong"},
    )
    assert response.status_code == 403


def test_callback_ok_logs_summary(caplog):
    client = TestClient(suno_web.app)
    body = {
        "code": 200,
        "data": {
            "callbackType": "complete",
            "task_id": "t-123",
            "req_id": "req-abc",
            "items": [{"audio_url": "https://cdn.example/track.mp3"}],
        },
    }

    with caplog.at_level("INFO"):
        response = client.post(
            "/suno-callback",
            json=body,
            headers={"X-Callback-Secret": "expected"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True

    summary_records = [rec for rec in caplog.records if rec.getMessage() == "suno callback"]
    assert summary_records, "callback summary log not found"
    meta = summary_records[-1].__dict__.get("meta")
    assert meta["phase"] == "callback"
    assert meta["task_id"] == "t-123"
    assert meta["req_id"] == "req-abc"
    assert meta["items"] == 1
    assert meta["type"] == "complete"
