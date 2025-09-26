import asyncio
import importlib
import json
import os
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

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
from suno.client import SunoClient, SunoAPIError
from suno.service import SunoService


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
    monkeypatch.setattr("suno.client.SUNO_GEN_PATH", "/suno-api/generate", raising=False)
    monkeypatch.setattr("suno.client.SUNO_TASK_STATUS_PATH", "/suno-api/record-info", raising=False)


def test_suno_v5_enqueue_success(monkeypatch, requests_mock):
    _setup_client_env(monkeypatch)
    client = SunoClient(base_url="https://api.example.com", token="token")
    requests_mock.post(
        "https://api.example.com/suno-api/generate",
        json={"task_id": "task-new"},
    )
    payload, version = client.create_music({"prompt": "hello", "style": "pop", "instrumental": False, "model": "V5"})
    assert version == "v5"
    assert payload["task_id"] == "task-new"
    assert requests_mock.call_count == 1
    sent = requests_mock.last_request.json()
    assert sent["model"] == "suno-v5"
    assert sent["input"]["prompt"] == "hello"
    assert sent["input"]["instrumental"] is False
    assert sent["callbackUrl"] == "https://callback.local/suno-callback"


def test_payload_shape_v5(monkeypatch, requests_mock):
    _setup_client_env(monkeypatch)
    client = SunoClient(base_url="https://api.example.com", token="token")
    requests_mock.post("https://api.example.com/suno-api/generate", json={"task_id": "task-shape"})
    client.create_music({"title": "Title", "prompt": "lyrics", "instrumental": True})
    body = requests_mock.last_request.json()
    assert body["model"] == "suno-v5"
    assert body["input"]["prompt"] == "lyrics"
    assert body["input"]["title"] == "Title"
    assert body["input"]["instrumental"] is True


def test_suno_v5_enqueue_404_raises(monkeypatch, requests_mock):
    _setup_client_env(monkeypatch)
    client = SunoClient(base_url="https://api.example.com", token="token")
    requests_mock.post(
        "https://api.example.com/suno-api/generate",
        status_code=404,
        json={"message": "not found"},
    )
    with pytest.raises(SunoAPIError) as exc:
        client.create_music({"prompt": "hello", "instrumental": False})
    assert exc.value.status == 404
    assert exc.value.api_version == "v5"
    assert requests_mock.call_count == 1


def test_suno_v5_enqueue_code_404_raises(monkeypatch, requests_mock):
    _setup_client_env(monkeypatch)
    client = SunoClient(base_url="https://api.example.com", token="token")
    requests_mock.post(
        "https://api.example.com/suno-api/generate",
        json={"code": 404, "message": "not found"},
    )
    with pytest.raises(SunoAPIError) as exc:
        client.create_music({"prompt": "fallback"})
    assert exc.value.api_version == "v5"
    assert requests_mock.call_count == 1


def test_suno_status_v5_404_raises(monkeypatch, requests_mock):
    _setup_client_env(monkeypatch)
    client = SunoClient(base_url="https://api.example.com", token="token")
    requests_mock.get(
        "https://api.example.com/suno-api/record-info",
        status_code=404,
        json={"message": "not found"},
    )
    with pytest.raises(SunoAPIError) as exc:
        client.get_task_status("abc")
    assert exc.value.status == 404
    assert exc.value.api_version == "v5"
    assert requests_mock.call_count == 1


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
        headers={"X-Callback-Token": "secret-token", "X-Request-ID": "req-header"},
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
        "/suno-callback?token=secret-token",
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
        headers={"X-Callback-Token": "bad"},
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

    async def fake_notify(*args, **kwargs):
        notify_calls["count"] += 1
        return SimpleNamespace(message_id=111)

    monkeypatch.setattr(bot, "_suno_notify", fake_notify)

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

    async def _run():
        await bot._launch_suno_generation(
            chat_id=123,
            ctx=ctx,
            params=params,
            user_id=555,
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
    assert start_calls["count"] == 1
    assert debit_calls["count"] == 1

    pending_key = f"{bot.REDIS_PREFIX}:suno:pending:{str(fixed_uuid)}"
    assert pending_key in fake_redis.store
    record = json.loads(fake_redis.store[pending_key])
    assert record["notify_ok"] is True
    assert record["task_id"] == "task-xyz"
    assert record["status"] == "queued"
    assert record["req_short"] == "000000"
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

    async def _run():
        await bot._launch_suno_generation(
            chat_id=321,
            ctx=ctx,
            params=params,
            user_id=777,
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

    pending_key = f"{bot.REDIS_PREFIX}:suno:pending:{str(fixed_uuid)}"
    record = json.loads(fake_redis.store[pending_key])
    assert record["notify_ok"] is False
    assert record["task_id"] == "task-fail-ok"
    assert record["status"] == "queued"
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

    async def _run():
        await bot._launch_suno_generation(
            chat_id=654,
            ctx=ctx,
            params=params,
            user_id=888,
            reply_to=None,
            trigger="test",
        )

    asyncio.run(_run())

    fail_after = _counter_value(suno_notify_fail, **fail_labels)
    assert fail_after == fail_before + 1
    assert debit_calls["count"] == 1
    assert refund_calls["count"] == 1

    pending_key = f"{bot.REDIS_PREFIX}:suno:pending:{str(fixed_uuid)}"
    record = json.loads(fake_redis.store[pending_key])
    assert record["status"] == "failed"
    assert record["notify_ok"] is False

    refund_key = f"{bot.REDIS_PREFIX}:suno:refund:pending:{str(fixed_uuid)}"
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
    pending_key = f"{bot.REDIS_PREFIX}:suno:pending:{str(fixed_uuid)}"
    assert pending_key in fake_redis.store
