import json
import os
import sys

import requests
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from suno_web import app


def test_suno_e2e_mocked(monkeypatch, caplog, tmp_path):
    # env
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("SUNO_CALLBACK_SECRET", "test-secret")
    monkeypatch.delenv("TELEGRAM_TOKEN", raising=False)
    monkeypatch.delenv("ADMIN_IDS", raising=False)

    caplog.set_level("DEBUG")

    client = TestClient(app)

    # mock HTTP GET for downloads
    class FakeResp:
        def __init__(self, status=200, body=b"ok", ct="audio/mpeg"):
            self.status_code = status
            self.ok = status == 200
            self.headers = {"Content-Type": ct}
            self._body = body

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def raise_for_status(self):
            if not self.ok:
                raise requests.HTTPError(self.status_code)

        def iter_content(self, chunk_size=8192):
            yield self._body

    def fake_get(url, *a, **k):
        if "forbidden.mp3" in url:
            return FakeResp(status=403)
        if url.endswith(".svg"):
            return FakeResp(body=b"<svg/>", ct="image/svg+xml")
        return FakeResp(body=b"FAKEAUDIO", ct="audio/mpeg")

    monkeypatch.setattr(requests, "get", fake_get)

    # emulate Suno callback
    payload = {
        "code": 200,
        "msg": "ok",
        "data": {
            "callbackType": "complete",
            "task_id": "demo123",
            "data": [
                {
                    "id": "trk1",
                    "title": "Demo",
                    "audio_url": "https://host/forbidden.mp3",  # will 403 -> link fallback
                    "image_url": "https://host/cover.svg",  # will 200 -> saved
                }
            ],
        },
    }

    r = client.post(
        "/suno-callback",
        headers={"X-Callback-Token": "test-secret", "Content-Type": "application/json"},
        data=json.dumps(payload),
    )
    assert r.status_code == 200
    assert r.json().get("status") == "received"

    joined = "\n".join(caplog.messages)
    assert "callback received" in joined
    assert "processed |" in joined
    # проверяем, что сервис не упал без сохранённого чата
    assert "No chat mapping" in joined
