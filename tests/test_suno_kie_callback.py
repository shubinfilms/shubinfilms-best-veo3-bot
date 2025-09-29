import os
import sys
from types import SimpleNamespace

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import suno_web


def test_callback_ok(monkeypatch):
    monkeypatch.setattr(suno_web, "SUNO_CALLBACK_SECRET", "s3c", raising=False)
    monkeypatch.setattr(suno_web, "rds", None, raising=False)
    monkeypatch.setattr(suno_web, "_prepare_assets", lambda task: None, raising=False)
    stub_service = SimpleNamespace(
        get_task_id_by_request=lambda *_a, **_k: None,
        get_request_id=lambda *_a, **_k: None,
        get_start_timestamp=lambda *_a, **_k: None,
        handle_callback=lambda *_a, **_k: None,
    )
    monkeypatch.setattr(suno_web, "service", stub_service, raising=False)
    suno_web._memory_idempotency.clear()

    client = TestClient(suno_web.app)
    body = {
        "code": 200,
        "data": {
            "callbackType": "complete",
            "task_id": "t-123",
            "data": [{"audio_url": "https://x/y.mp3"}],
        },
    }
    response = client.post(
        "/suno-callback",
        json=body,
        headers={"X-Callback-Secret": "s3c"},
    )
    assert response.status_code == 200
    assert response.json()["ok"] is True
