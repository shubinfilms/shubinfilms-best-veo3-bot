import asyncio
import importlib
import json
import os
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import pytest
import requests
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from metrics import (
    suno_enqueue_duration_seconds,
    suno_enqueue_total,
    suno_notify_duration_seconds,
    suno_notify_fail,
    suno_notify_latency_ms,
    suno_notify_ok,
    suno_notify_total,
    suno_refund_total,
)
from telegram.error import Forbidden, TimedOut

os.environ.setdefault("SUNO_API_TOKEN", "test-token")
os.environ.setdefault("SUNO_CALLBACK_SECRET", "secret-token")
os.environ.setdefault("SUNO_ENABLED", "true")
os.environ.setdefault("SUNO_CALLBACK_URL", "https://callback.local/suno-callback")

import suno_web
from suno_web import app
import suno.tempfiles as tempfiles
from suno.client import SunoAPIError, SunoClient, SunoClientError
from suno.service import SunoService
from suno.schemas import CallbackEnvelope, SunoTask


def _build_payload(task_id: str = "task-1", callback_type: str = "complete") -> dict:
    return {
        "code": 200,
        "msg": "ok",
        "data": {
            "callbackType": callback_type,
            "callback_type": callback_type,
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


class StubService:
    def __init__(self, req_map: Optional[dict[str, str]] = None) -> None:
        self.req_map = req_map or {}
        self.captured: dict[str, Any] = {}

    def get_request_id(self, task_id: str) -> Optional[str]:
        return None

    def get_task_id_by_request(self, req_id: Optional[str]) -> Optional[str]:
        if not req_id:
            return None
        return self.req_map.get(req_id)

    def get_start_timestamp(self, task_id: str) -> Optional[str]:
        return None

    def handle_callback(self, task: SunoTask, req_id: Optional[str] = None) -> None:
        self.captured["task"] = task
        self.captured["req_id"] = req_id


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch):
    monkeypatch.setenv("SUNO_CALLBACK_SECRET", "secret-token")
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    monkeypatch.setenv("SUNO_ENABLED", "true")
    monkeypatch.setenv("SUNO_CALLBACK_URL", "https://callback.local/suno-callback")
    monkeypatch.delenv("TELEGRAM_TOKEN", raising=False)
    monkeypatch.delenv("ADMIN_IDS", raising=False)
    monkeypatch.setattr(suno_web, "SUNO_ENABLED", True, raising=False)
    suno_web._memory_idempotency.clear()


def _client():
    return TestClient(app)


def _metric_labels() -> dict[str, str]:
    env = (os.getenv("APP_ENV") or "prod").strip() or "prod"
    return {"env": env, "service": "bot"}


def _counter_value(counter, **labels) -> float:
    child = counter.labels(**labels)
    return child._value.get()


def _hist_sum(histogram, **labels) -> float:
    child = histogram.labels(**labels)
    return child._sum.get()


def _setup_client_env(monkeypatch) -> None:
    monkeypatch.setenv("SUNO_API_BASE", "https://api.example.com")
    monkeypatch.setenv("SUNO_API_TOKEN", "token")
    monkeypatch.setenv("SUNO_CALLBACK_URL", "https://callback.local/suno-callback")
    monkeypatch.setenv("SUNO_CALLBACK_SECRET", "secret-token")
    monkeypatch.setenv("SUNO_ENABLED", "true")
    monkeypatch.setattr(suno_web, "SUNO_ENABLED", True, raising=False)
    monkeypatch.setenv("SUNO_MODEL", "suno-v5")
    monkeypatch.setattr("suno.client.SUNO_CALLBACK_URL", "https://callback.local/suno-callback", raising=False)
    monkeypatch.setattr("suno.client.SUNO_CALLBACK_SECRET", "secret-token", raising=False)
    monkeypatch.setattr("suno.client.SUNO_MODEL", "suno-v5", raising=False)
    monkeypatch.setattr("suno.client.SUNO_GEN_PATH", "/api/v1/generate/add-vocals", raising=False)
    monkeypatch.setattr("suno.client.SUNO_TASK_STATUS_PATH", "/api/v1/generate/record-info", raising=False)
    monkeypatch.setattr("suno.client.SUNO_WAV_PATH", "/api/v1/wav/generate", raising=False)
    monkeypatch.setattr("suno.client.SUNO_WAV_INFO_PATH", "/api/v1/wav/record-info", raising=False)
    monkeypatch.setattr("suno.client.SUNO_MP4_PATH", "/api/v1/mp4/generate", raising=False)
    monkeypatch.setattr("suno.client.SUNO_MP4_INFO_PATH", "/api/v1/mp4/record-info", raising=False)
    monkeypatch.setattr("suno.client.SUNO_COVER_INFO_PATH", "/api/v1/suno/cover/record-info", raising=False)
    monkeypatch.setattr("suno.client.SUNO_LYRICS_PATH", "/api/v1/generate/get-timestamped-lyrics", raising=False)
    monkeypatch.setattr("suno.client.SUNO_STEM_PATH", "/api/v1/vocal-removal/generate", raising=False)
    monkeypatch.setattr("suno.client.SUNO_STEM_INFO_PATH", "/api/v1/vocal-removal/record-info", raising=False)
    monkeypatch.setattr("suno.client.SUNO_INSTR_PATH", "/api/v1/generate/add-instrumental", raising=False)
    monkeypatch.setattr("suno.client.SUNO_UPLOAD_EXTEND_PATH", "/api/v1/suno/upload-extend", raising=False)


def test_suno_v5_enqueue_success(monkeypatch, requests_mock):
    _setup_client_env(monkeypatch)
    client = SunoClient(base_url="https://api.example.com", token="token")
    requests_mock.post(
        "https://api.example.com/api/v1/generate/add-vocals",
        json={"task_id": "task-new"},
    )
    payload, version = client.create_music({"prompt": "hello", "style": "pop", "instrumental": False, "model": "V5"})
    assert version == "v5"
    assert payload["task_id"] == "task-new"
    assert requests_mock.call_count == 1
    sent = requests_mock.last_request.json()
    assert sent["model"] == "suno-v5"
    assert sent["input_text"] == "hello"
    assert sent["instrumental"] is False
    assert sent["style"] == "pop"
    assert sent["callbackUrl"] == "https://callback.local/suno-callback"


def test_payload_shape_v5(monkeypatch, requests_mock):
    _setup_client_env(monkeypatch)
    client = SunoClient(base_url="https://api.example.com", token="token")
    requests_mock.post(
        "https://api.example.com/api/v1/generate/add-instrumental",
        json={"task_id": "task-shape"},
    )
    client.create_music({"title": "Title", "prompt": "lyrics", "instrumental": True})
    body = requests_mock.last_request.json()
    assert body["model"] == "suno-v5"
    assert body["input_text"] == "lyrics"
    assert body["title"] == "Title"
    assert body["instrumental"] is True


def test_payload_uses_default_prompt(monkeypatch, requests_mock):
    _setup_client_env(monkeypatch)
    client = SunoClient(base_url="https://api.example.com", token="token")
    requests_mock.post(
        "https://api.example.com/api/v1/generate/add-vocals",
        json={"task_id": "task-default"},
    )
    client.create_music({"instrumental": False})
    body = requests_mock.last_request.json()
    assert body["input_text"] == "Untitled track"
    assert body["instrumental"] is False


def test_suno_v5_enqueue_404_raises(monkeypatch, requests_mock):
    _setup_client_env(monkeypatch)
    client = SunoClient(base_url="https://api.example.com", token="token")
    requests_mock.post(
        "https://api.example.com/api/v1/generate/add-vocals",
        status_code=404,
        json={"message": "not found"},
    )
    with pytest.raises(SunoAPIError) as exc:
        client.create_music({"prompt": "hello", "instrumental": False})
    assert exc.value.status == 404
    assert exc.value.api_version == "v5"
    assert isinstance(exc.value, SunoClientError)
    assert requests_mock.call_count == 1


def test_suno_v5_enqueue_code_404_raises(monkeypatch, requests_mock):
    _setup_client_env(monkeypatch)
    client = SunoClient(base_url="https://api.example.com", token="token")
    requests_mock.post(
        "https://api.example.com/api/v1/generate/add-vocals",
        json={"code": 404, "message": "not found"},
    )
    with pytest.raises(SunoAPIError) as exc:
        client.create_music({"prompt": "fallback"})
    assert exc.value.api_version == "v5"
    assert isinstance(exc.value, SunoClientError)
    assert requests_mock.call_count == 1


def test_suno_status_v5_404_raises(monkeypatch, requests_mock):
    _setup_client_env(monkeypatch)
    client = SunoClient(base_url="https://api.example.com", token="token")
    requests_mock.get(
        "https://api.example.com/api/v1/generate/record-info",
        status_code=404,
        json={"message": "not found"},
    )
    with pytest.raises(SunoAPIError) as exc:
        client.get_task_status("abc")
    assert exc.value.status == 404
    assert exc.value.api_version == "v5"
    assert isinstance(exc.value, SunoClientError)
    assert requests_mock.call_count == 1


def test_status_includes_request_id(monkeypatch, requests_mock):
    _setup_client_env(monkeypatch)
    client = SunoClient(base_url="https://api.example.com", token="token")
    requests_mock.get(
        "https://api.example.com/api/v1/generate/record-info",
        json={"status": "queued"},
    )
    client.get_task_status("abc", req_id="req-1")
    qs = {k.lower(): v for k, v in requests_mock.last_request.qs.items()}
    assert qs.get("taskid") == ["abc"]
    assert qs.get("requestid") == ["req-1"]


def test_suno_wav_helpers(monkeypatch, requests_mock):
    _setup_client_env(monkeypatch)
    client = SunoClient(base_url="https://api.example.com", token="token")
    requests_mock.post("https://api.example.com/api/v1/wav/generate", json={"ok": True})
    resp = client.build_wav("wav-123")
    assert resp["ok"] is True
    assert requests_mock.last_request.json()["taskId"] == "wav-123"
    requests_mock.get("https://api.example.com/api/v1/wav/record-info", json={"status": "ready"})
    info = client.get_wav_info("wav-123")
    assert info["status"] == "ready"
    assert "taskId=wav-123" in requests_mock.last_request.url


def test_suno_mp4_helpers(monkeypatch, requests_mock):
    _setup_client_env(monkeypatch)
    client = SunoClient(base_url="https://api.example.com", token="token")
    requests_mock.post("https://api.example.com/api/v1/mp4/generate", json={"queued": True})
    resp = client.build_mp4("mp4-1")
    assert resp["queued"] is True
    assert requests_mock.last_request.json()["taskId"] == "mp4-1"
    requests_mock.get("https://api.example.com/api/v1/mp4/record-info", json={"progress": 0.9})
    info = client.get_mp4_info("mp4-1")
    assert info["progress"] == 0.9
    assert "taskId=mp4-1" in requests_mock.last_request.url


def test_suno_extend(monkeypatch, requests_mock):
    _setup_client_env(monkeypatch)
    client = SunoClient(base_url="https://api.example.com", token="token")
    requests_mock.post("https://api.example.com/api/v1/suno/upload-extend", json={"task_id": "mp4-1"})
    payload = {"taskId": "mp4-1", "prompt": "more"}
    resp = client.upload_extend(payload, req_id="req-1")
    assert resp["task_id"] == "mp4-1"
    assert requests_mock.last_request.json()["prompt"] == "more"
    assert requests_mock.last_request.headers["X-Request-ID"] == "req-1"


class FakeRedis:
    def __init__(self):
        self.store: dict[str, str] = {}

    def setex(self, key, ttl, value):
        self.store[key] = value
        return True

    def get(self, key):
        return self.store.get(key)

    def delete(self, key):
        self.store.pop(key, None)

    def set(self, key, value, nx=False, ex=None):
        if nx and key in self.store:
            return False
        self.store[key] = value
        return True


@pytest.fixture
def bot_module(monkeypatch):
    monkeypatch.setenv("LEDGER_BACKEND", "memory")
    bot = importlib.import_module("bot")
    bot._SUNO_REFUND_MEMORY.clear()
    bot._SUNO_COOLDOWN_MEMORY.clear()
    bot._SUNO_PENDING_MEMORY.clear()
    bot._SUNO_REFUND_PENDING_MEMORY.clear()
    return bot


def test_callback_accepts_header_token(monkeypatch):
    received = {}

    def fake_handle(task, req_id=None):
        received["task"] = task
        received["req_id"] = req_id

    monkeypatch.setattr(suno_web.service, "handle_callback", fake_handle)
    client = _client()
    response = client.post(
        "/suno-callback",
        headers={"X-Callback-Secret": "secret-token", "X-Request-ID": "req-header"},
        json=_build_payload("header-token"),
    )
    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert received["task"].task_id == "header-token"
    assert received["req_id"] == "req-header"


def test_callback_accepts_query_token(monkeypatch):
    received = {}

    def fake_handle(task, req_id=None):
        received["task"] = task
        received["req_id"] = req_id

    monkeypatch.setattr(suno_web.service, "handle_callback", fake_handle)
    client = _client()
    response = client.post(
        "/suno-callback?secret=secret-token",
        json=_build_payload("query-token"),
    )
    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert received["task"].task_id == "query-token"


def test_callback_rejects_missing_token():
    client = _client()
    response = client.post(
        "/suno-callback",
        json=_build_payload(),
    )
    assert response.status_code == 403
    assert response.json()["error"] == "forbidden"


def test_callback_rejects_wrong_token():
    client = _client()
    response = client.post(
        "/suno-callback",
        headers={"X-Callback-Secret": "bad"},
        json=_build_payload(),
    )
    assert response.status_code == 403
    assert response.json()["error"] == "forbidden"


def test_duplicate_callbacks_processed_once(monkeypatch):
    count = {"calls": 0}

    def fake_handle(task, req_id=None):
        count["calls"] += 1

    monkeypatch.setattr(suno_web.service, "handle_callback", fake_handle)
    client = _client()
    payload = _build_payload("dup-task")
    for _ in range(2):
        response = client.post(
            "/suno-callback",
            headers={"X-Callback-Secret": "secret-token"},
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
        headers={"X-Callback-Secret": "secret-token", "Content-Type": "application/json"},
        content=body,
    )
    assert response.status_code == 413
    assert response.json()["detail"] == "payload too large"
    assert invoked["called"] is False


def test_download_failure_falls_back_to_url(monkeypatch, tmp_path):
    captured = {}

    def fake_handle(task, req_id=None):
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
        headers={"X-Callback-Secret": "secret-token"},
        json=_build_payload("asset-task"),
    )
    assert response.status_code == 200
    task = captured["task"]
    assert task.task_id == "asset-task"
    assert task.items[0].audio_url == "https://cdn.example.com/demo.mp3"
    image_path = task.items[0].image_url
    assert image_path
    assert Path(image_path).exists()


def test_suno_service_records_request_id(monkeypatch):
    class FakeClient:
        def __init__(self):
            self.captured_req_id = None

        def create_music(self, payload, *, req_id=None):
            self.captured_req_id = req_id
            return {"task_id": "task-req"}, "v5"

    service = SunoService(client=FakeClient(), redis=None, telegram_token="dummy")
    task = service.start_music(
        100,
        200,
        title="Demo",
        style="Pop",
        lyrics="Lyrics",
        instrumental=False,
        user_id=1,
        prompt="Demo",
        req_id="req-123",
    )
    assert task.task_id == "task-req"
    assert service.client.captured_req_id == "req-123"
    assert service.get_request_id(task.task_id) == "req-123"
    ts = service.get_start_timestamp(task.task_id)
    assert ts is not None


@pytest.mark.parametrize(
    "response, expected_req, expected_task",
    [
        ({"req_id": "req-1", "task_id": "task-1"}, "req-1", "task-1"),
        ({"data": {"req_id": "req-2", "task_id": "task-2"}}, "req-2", "task-2"),
        ({"requestId": "req-3", "taskId": "task-3"}, "req-3", "task-3"),
    ],
)
def test_suno_service_extracts_enqueue_ids(response, expected_req, expected_task):
    class FakeClient:
        def create_music(self, payload, *, req_id=None):
            return response, "v5"

    service = SunoService(client=FakeClient(), redis=None, telegram_token="dummy")
    task = service.start_music(
        10,
        20,
        title="Demo",
        style=None,
        lyrics=None,
        instrumental=False,
        user_id=1,
        prompt="Prompt",
        req_id=None,
    )
    assert task.task_id == expected_task
    assert service.get_request_id(expected_task) == expected_req
    assert service.get_task_id_by_request(expected_req) == expected_task


def test_suno_service_generates_and_handles_callback(monkeypatch, requests_mock):
    _setup_client_env(monkeypatch)
    requests_mock.post(
        "https://api.example.com/api/v1/generate/add-instrumental",
        json={"task_id": "task-flow"},
    )
    service = SunoService(client=SunoClient(base_url="https://api.example.com", token="token"), redis=None, telegram_token="dummy")

    sent_messages: list[tuple[int, str]] = []

    def fake_send_text(chat_id: int, text: str, *, reply_to: Optional[int] = None) -> None:
        sent_messages.append((chat_id, text))

    monkeypatch.setattr(service, "_send_text", fake_send_text)
    monkeypatch.setattr(service, "_send_audio", lambda *args, **kwargs: True)
    monkeypatch.setattr(service, "_send_image", lambda *args, **kwargs: True)

    task = service.start_music(
        42,
        99,
        title="Instrumental Test",
        style="Ambient",
        lyrics=None,
        model="V5",
        instrumental=True,
        user_id=101,
        prompt="Test instrumental track",
        req_id="req-flow",
    )

    assert task.task_id == "task-flow"
    sent_body = requests_mock.last_request.json()
    assert sent_body["input_text"] == "Test instrumental track"
    assert sent_body["instrumental"] is True

    envelope = CallbackEnvelope(
        code=200,
        msg="ok",
        data={
            "taskId": "task-flow",
            "callbackType": "complete",
            "response": {
                "tracks": [
                    {
                        "id": "trk-1",
                        "title": "Instrumental Test",
                        "audioUrl": "https://cdn.example.com/test.mp3",
                        "imageUrl": "https://cdn.example.com/test.jpg",
                    }
                ]
            },
        },
    )

    service.handle_callback(SunoTask.from_envelope(envelope), req_id="req-flow")
    assert any("Suno: этап complete" in text for _, text in sent_messages)
    assert any("https://cdn.example.com/test.mp3" in text for _, text in sent_messages)


def test_callback_restores_missing_req_id(monkeypatch):
    class FakeClient:
        def __init__(self, task_id: str):
            self.task_id = task_id

        def create_music(self, payload, *, req_id=None):
            return {"task_id": self.task_id}, "legacy"

    fake_client = FakeClient("restore-task")
    service = SunoService(client=fake_client, redis=None, telegram_token="dummy")
    service.start_music(1, 1, title="T", style="S", lyrics="L", instrumental=False, user_id=2, prompt="P", req_id="req-restore")

    captured = {}

    def fake_handle(task, req_id=None):
        captured["task"] = task
        captured["req_id"] = req_id

    monkeypatch.setattr(suno_web, "service", service, raising=False)
    monkeypatch.setattr(service, "handle_callback", fake_handle)
    monkeypatch.setattr(suno_web, "_prepare_assets", lambda task: None)

    client = _client()
    response = client.post(
        "/suno-callback?token=secret-token",
        json=_build_payload("restore-task"),
    )
    assert response.status_code == 200
    assert captured["task"].task_id == "restore-task"
    assert captured["req_id"] == "req-restore"


@pytest.mark.parametrize(
    "payload, expected_req, expected_task, req_map",
    [
        (
            {
                "code": 200,
                "msg": "ok",
                "payload": {
                    "requestId": "req-wrap",
                    "taskId": "task-wrap",
                    "status": "complete",
                    "results": [{"audioUrl": "https://cdn.example.com/a.mp3"}],
                },
            },
            "req-wrap",
            "task-wrap",
            {},
        ),
        (
            {
                "data": {
                    "payload": {
                        "data": {
                            "req_id": "req-nest",
                            "task_id": "task-nest",
                            "type": "complete",
                            "items": [
                                {
                                    "audio_url": "https://cdn.example.com/b.mp3",
                                }
                            ],
                        }
                    }
                }
            },
            "req-nest",
            "task-nest",
            {},
        ),
        (
            {
                "requestId": "req-map",
                "status": "complete",
                "results": [{"audioUrl": "https://cdn.example.com/c.mp3"}],
            },
            "req-map",
            "task-from-map",
            {"req-map": "task-from-map"},
        ),
    ],
)
def test_suno_callback_parses_wrapped_payload(monkeypatch, payload, expected_req, expected_task, req_map):
    stub_service = StubService(req_map)
    monkeypatch.setattr(suno_web, "service", stub_service, raising=False)
    monkeypatch.setattr(suno_web, "_prepare_assets", lambda task: None)
    client = _client()
    response = client.post(
        "/suno-callback",
        headers={"X-Callback-Secret": "secret-token"},
        json=payload,
    )
    assert response.status_code == 200
    assert stub_service.captured["req_id"] == expected_req
    assert stub_service.captured["task"].task_id == expected_task


def test_launch_suno_notify_ok_flow(monkeypatch, bot_module):
    bot = bot_module
    fake_redis = FakeRedis()
    monkeypatch.setattr(bot, "rds", fake_redis)
    monkeypatch.setattr(bot, "_suno_configured", lambda: True)
    monkeypatch.setattr(bot, "ensure_user", lambda uid: None)
    bot.SUNO_PER_USER_COOLDOWN_SEC = 0

    async def async_noop(*args, **kwargs):
        return None

    monkeypatch.setattr(bot, "refresh_suno_card", async_noop)
    monkeypatch.setattr(bot, "refresh_balance_card_if_open", async_noop)
    monkeypatch.setattr(bot, "show_balance_notification", async_noop)
    monkeypatch.setattr(bot, "_suno_update_last_debit_meta", lambda *args, **kwargs: None)
    monkeypatch.setattr(bot, "_suno_set_cooldown", lambda *args, **kwargs: None)

    labels = _metric_labels()
    ok_before = _counter_value(suno_notify_ok, **labels)
    notify_hist_before = _hist_sum(suno_notify_duration_seconds, **labels)
    notify_latency_before = _hist_sum(suno_notify_latency_ms, **labels)
    enqueue_hist_before = _hist_sum(suno_enqueue_duration_seconds, **labels)
    notify_total_before = _counter_value(suno_notify_total, outcome="success", **labels)
    enqueue_total_before = _counter_value(suno_enqueue_total, outcome="success", api="v5", **labels)

    debit_calls = {"count": 0}

    def fake_debit_try(user_id, price, reason, meta):
        debit_calls["count"] += 1
        return True, 90

    monkeypatch.setattr(bot, "debit_try", fake_debit_try)

    notify_calls = {"count": 0}
    edited_payloads: list[str] = []

    async def fake_notify(*args, **kwargs):
        notify_calls["count"] += 1
        return SimpleNamespace(message_id=111)

    monkeypatch.setattr(bot, "_suno_notify", fake_notify)

    async def fake_safe_edit_message(ctx_param, chat_id_param, message_id_param, new_text, **kwargs):
        edited_payloads.append(new_text)
        return True

    monkeypatch.setattr(bot, "safe_edit_message", fake_safe_edit_message)

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(bot.asyncio, "to_thread", fake_to_thread)

    start_calls = {"count": 0}

    class DummyTask:
        def __init__(self, task_id: str):
            self.task_id = task_id
            self.items = []
            self.callback_type = "start"
            self.msg = None
            self.code = None

    def fake_start_music(*args, **kwargs):
        start_calls["count"] += 1
        return DummyTask("task-xyz")

    monkeypatch.setattr(bot.SUNO_SERVICE, "start_music", fake_start_music)

    fixed_uuid = uuid.UUID("00000000-0000-0000-0000-000000000001")
    monkeypatch.setattr(uuid, "uuid4", lambda: fixed_uuid)

    ctx = SimpleNamespace(user_data={}, bot=SimpleNamespace())
    params = {"title": "Demo", "style": "Pop", "lyrics": "", "instrumental": True}
    user_id = 888
    user_id = 777
    user_id = 888
    user_id = 777
    user_id = 888
    user_id = 777
    user_id = 555

    async def _run():
        await bot._launch_suno_generation(
            chat_id=123,
            ctx=ctx,
            params=params,
            user_id=user_id,
            reply_to=None,
            trigger="test",
        )

    asyncio.run(_run())

    ok_after = _counter_value(suno_notify_ok, **labels)
    notify_hist_after = _hist_sum(suno_notify_duration_seconds, **labels)
    notify_latency_after = _hist_sum(suno_notify_latency_ms, **labels)
    enqueue_hist_after = _hist_sum(suno_enqueue_duration_seconds, **labels)
    notify_total_after = _counter_value(suno_notify_total, outcome="success", **labels)
    enqueue_total_after = _counter_value(suno_enqueue_total, outcome="success", api="v5", **labels)

    assert ok_after == ok_before + 1
    assert notify_hist_after > notify_hist_before
    assert notify_latency_after > notify_latency_before
    assert enqueue_hist_after > enqueue_hist_before
    assert notify_total_after == notify_total_before + 1
    assert enqueue_total_after == enqueue_total_before + 1
    assert notify_calls["count"] == 1
    assert edited_payloads and edited_payloads[-1].startswith("✅ Списано")
    assert start_calls["count"] == 1
    assert debit_calls["count"] == 1

    pending_key = f"{bot.REDIS_PREFIX}:suno:pending:suno:{user_id}:{str(fixed_uuid)}"
    assert pending_key in fake_redis.store
    record = json.loads(fake_redis.store[pending_key])
    assert record["notify_ok"] is True
    assert record["task_id"] == "task-xyz"
    assert record["status"] == "enqueued"
    assert record["req_short"] == "SUNO:5"
    state = bot.state(ctx)
    assert state["suno_generating"] is False
    assert state["suno_current_req_id"] is None


def test_launch_suno_notify_fail_continues(monkeypatch, bot_module):
    bot = bot_module
    fake_redis = FakeRedis()
    monkeypatch.setattr(bot, "rds", fake_redis)
    monkeypatch.setattr(bot, "_suno_configured", lambda: True)
    monkeypatch.setattr(bot, "ensure_user", lambda uid: None)
    bot.SUNO_PER_USER_COOLDOWN_SEC = 0

    async def async_noop(*args, **kwargs):
        return None

    monkeypatch.setattr(bot, "refresh_suno_card", async_noop)
    monkeypatch.setattr(bot, "refresh_balance_card_if_open", async_noop)
    monkeypatch.setattr(bot, "show_balance_notification", async_noop)
    monkeypatch.setattr(bot, "_suno_update_last_debit_meta", lambda *args, **kwargs: None)
    monkeypatch.setattr(bot, "_suno_set_cooldown", lambda *args, **kwargs: None)

    labels = _metric_labels()
    ok_before = _counter_value(suno_notify_ok, **labels)
    fail_labels = dict(labels, type="Forbidden")
    fail_before = _counter_value(suno_notify_fail, **fail_labels)
    notify_total_error_before = _counter_value(suno_notify_total, outcome="error", **labels)
    enqueue_total_before = _counter_value(suno_enqueue_total, outcome="success", api="v5", **labels)

    debit_calls = {"count": 0}

    def fake_debit_try(user_id, price, reason, meta):
        debit_calls["count"] += 1
        return True, 90

    monkeypatch.setattr(bot, "debit_try", fake_debit_try)

    async def failing_notify(*args, **kwargs):
        raise Forbidden("blocked")

    monkeypatch.setattr(bot, "_suno_notify", failing_notify)

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(bot.asyncio, "to_thread", fake_to_thread)

    class DummyTask:
        def __init__(self, task_id: str):
            self.task_id = task_id
            self.items = []
            self.callback_type = "start"
            self.msg = None
            self.code = None

    start_calls = {"count": 0}

    def fake_start_music(*args, **kwargs):
        start_calls["count"] += 1
        return DummyTask("task-fail-ok")

    monkeypatch.setattr(bot.SUNO_SERVICE, "start_music", fake_start_music)

    fixed_uuid = uuid.UUID("00000000-0000-0000-0000-000000000002")
    monkeypatch.setattr(uuid, "uuid4", lambda: fixed_uuid)

    ctx = SimpleNamespace(user_data={}, bot=SimpleNamespace())
    params = {"title": "Demo", "style": "Pop", "lyrics": "", "instrumental": True}
    user_id = 777
    async def _run():
        await bot._launch_suno_generation(
            chat_id=321,
            ctx=ctx,
            params=params,
            user_id=user_id,
            reply_to=None,
            trigger="test",
        )

    asyncio.run(_run())

    ok_after = _counter_value(suno_notify_ok, **labels)
    fail_after = _counter_value(suno_notify_fail, **fail_labels)
    notify_total_error_after = _counter_value(suno_notify_total, outcome="error", **labels)
    enqueue_total_after = _counter_value(suno_enqueue_total, outcome="success", api="v5", **labels)
    assert ok_after == ok_before
    assert fail_after == fail_before + 1
    assert notify_total_error_after == notify_total_error_before + 1
    assert enqueue_total_after == enqueue_total_before + 1
    assert start_calls["count"] == 1
    assert debit_calls["count"] == 1

    pending_key = f"{bot.REDIS_PREFIX}:suno:pending:suno:{user_id}:{str(fixed_uuid)}"
    record = json.loads(fake_redis.store[pending_key])
    assert record["notify_ok"] is False
    assert record["task_id"] == "task-fail-ok"
    assert record["status"] == "enqueued"
    state = bot.state(ctx)
    assert state["suno_generating"] is False


def test_launch_suno_failure_marks_refund(monkeypatch, bot_module):
    bot = bot_module
    fake_redis = FakeRedis()
    monkeypatch.setattr(bot, "rds", fake_redis)
    monkeypatch.setattr(bot, "_suno_configured", lambda: True)
    monkeypatch.setattr(bot, "ensure_user", lambda uid: None)
    bot.SUNO_PER_USER_COOLDOWN_SEC = 0

    async def async_noop(*args, **kwargs):
        return None

    monkeypatch.setattr(bot, "refresh_suno_card", async_noop)
    monkeypatch.setattr(bot, "refresh_balance_card_if_open", async_noop)
    monkeypatch.setattr(bot, "show_balance_notification", async_noop)
    monkeypatch.setattr(bot, "_suno_update_last_debit_meta", lambda *args, **kwargs: None)
    monkeypatch.setattr(bot, "_suno_set_cooldown", lambda *args, **kwargs: None)

    labels = _metric_labels()
    fail_labels = dict(labels, type="TimedOut")
    fail_before = _counter_value(suno_notify_fail, **fail_labels)

    debit_calls = {"count": 0}

    def fake_debit_try(user_id, price, reason, meta):
        debit_calls["count"] += 1
        return True, 90

    monkeypatch.setattr(bot, "debit_try", fake_debit_try)

    async def failing_notify(*args, **kwargs):
        raise TimedOut("late")

    monkeypatch.setattr(bot, "_suno_notify", failing_notify)

    async def fake_to_thread(func, *args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(bot.asyncio, "to_thread", fake_to_thread)

    refund_calls = {"count": 0}

    async def fake_issue_refund(*args, **kwargs):
        refund_calls["count"] += 1
        ctx_obj = args[0] if args else kwargs.get("ctx")
        if ctx_obj is not None:
            state = bot.state(ctx_obj)
            state["suno_generating"] = False
            state["suno_current_req_id"] = None

    monkeypatch.setattr(bot, "_suno_issue_refund", fake_issue_refund)

    fixed_uuid = uuid.UUID("00000000-0000-0000-0000-000000000003")
    monkeypatch.setattr(uuid, "uuid4", lambda: fixed_uuid)

    ctx = SimpleNamespace(user_data={}, bot=SimpleNamespace())
    params = {"title": "Demo", "style": "Pop", "lyrics": "", "instrumental": True}
    user_id = 888
    async def _run():
        await bot._launch_suno_generation(
            chat_id=654,
            ctx=ctx,
            params=params,
            user_id=user_id,
            reply_to=None,
            trigger="test",
        )

    asyncio.run(_run())

    fail_after = _counter_value(suno_notify_fail, **fail_labels)
    assert fail_after == fail_before + 1
    assert debit_calls["count"] == 1
    assert refund_calls["count"] == 1

    pending_key = f"{bot.REDIS_PREFIX}:suno:pending:suno:{user_id}:{str(fixed_uuid)}"
    record = json.loads(fake_redis.store[pending_key])
    assert record["status"] == "failed"
    assert record["notify_ok"] is False

    refund_key = f"{bot.REDIS_PREFIX}:suno:refund:pending:suno:{user_id}:{str(fixed_uuid)}"
    assert refund_key in fake_redis.store
    refund_record = json.loads(fake_redis.store[refund_key])
    assert refund_record["status"] == "failed"


def test_launch_suno_duplicate_req_id_no_double_charge(monkeypatch, bot_module):
    bot = bot_module
    fake_redis = FakeRedis()
    monkeypatch.setattr(bot, "rds", fake_redis)
    monkeypatch.setattr(bot, "_suno_configured", lambda: True)
    monkeypatch.setattr(bot, "ensure_user", lambda uid: None)
    bot.SUNO_PER_USER_COOLDOWN_SEC = 0

    async def async_noop(*args, **kwargs):
        return None

    monkeypatch.setattr(bot, "refresh_suno_card", async_noop)
    monkeypatch.setattr(bot, "refresh_balance_card_if_open", async_noop)
    monkeypatch.setattr(bot, "show_balance_notification", async_noop)
    monkeypatch.setattr(bot, "_suno_update_last_debit_meta", lambda *args, **kwargs: None)
    monkeypatch.setattr(bot, "_suno_set_cooldown", lambda *args, **kwargs: None)

    debit_calls = {"count": 0}

    def fake_debit_try(user_id, price, reason, meta):
        debit_calls["count"] += 1
        return True, 90

    monkeypatch.setattr(bot, "debit_try", fake_debit_try)

    async def fake_notify(*args, **kwargs):
        return SimpleNamespace(message_id=200)

    monkeypatch.setattr(bot, "_suno_notify", fake_notify)

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(bot.asyncio, "to_thread", fake_to_thread)

    class DummyTask:
        def __init__(self, task_id: str):
            self.task_id = task_id
            self.items = []
            self.callback_type = "start"
            self.msg = None
            self.code = None

    start_calls = {"count": 0}

    def fake_start_music(*args, **kwargs):
        start_calls["count"] += 1
        return DummyTask("task-dupe")

    monkeypatch.setattr(bot.SUNO_SERVICE, "start_music", fake_start_music)

    fixed_uuid = uuid.UUID("00000000-0000-0000-0000-000000000004")
    monkeypatch.setattr(uuid, "uuid4", lambda: fixed_uuid)

    ctx = SimpleNamespace(user_data={}, bot=SimpleNamespace())
    params = {"title": "Demo", "style": "Pop", "lyrics": "", "instrumental": True}

    async def _run():
        await bot._launch_suno_generation(
            chat_id=777,
            ctx=ctx,
            params=params,
            user_id=999,
            reply_to=None,
            trigger="test",
        )
        await bot._launch_suno_generation(
            chat_id=777,
            ctx=ctx,
            params=params,
            user_id=999,
            reply_to=None,
            trigger="test",
        )

    asyncio.run(_run())

    assert debit_calls["count"] == 1
    assert start_calls["count"] == 1
    pending_key = f"{bot.REDIS_PREFIX}:suno:pending:suno:999:{str(fixed_uuid)}"
    assert pending_key in fake_redis.store
