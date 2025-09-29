import json
import logging
import os
import time
from pathlib import Path

import json
import logging
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ.setdefault("SUNO_API_TOKEN", "test-token")
os.environ.setdefault("SUNO_CALLBACK_SECRET", "secret-token")
os.environ.setdefault("SUNO_CALLBACK_URL", "https://callback")

from logging_utils import JsonFormatter, refresh_secret_cache
from metrics import (
    render_metrics,
    suno_callback_download_fail_total,
    suno_callback_total,
    suno_task_store_total,
    suno_enqueue_duration_seconds,
    suno_notify_fail,
    suno_notify_ok,
    suno_notify_duration_seconds,
)
from settings import MAX_IN_LOG_BODY
from suno.client import SunoAPIError, SunoClient
from suno.tempfiles import BASE_DIR, cleanup_old_directories, task_directory


@pytest.fixture(autouse=True)
def _clean_tmp(tmp_path, monkeypatch):
    monkeypatch.setenv("SUNO_API_BASE", "https://example.com")
    monkeypatch.setenv("SUNO_API_TOKEN", "test-token")
    if BASE_DIR.exists():
        for item in BASE_DIR.glob("*"):
            if item.is_dir():
                for child in item.glob("**/*"):
                    child.unlink(missing_ok=True)
                item.rmdir()
    yield
    if BASE_DIR.exists():
        for item in BASE_DIR.glob("*"):
            if item.is_dir():
                for child in item.glob("**/*"):
                    child.unlink(missing_ok=True)
                item.rmdir()


def test_logging_redacts_secrets(monkeypatch):
    monkeypatch.setenv("EXTRA_TOKEN", "super-secret")
    refresh_secret_cache()
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=0,
        msg="token=%s",
        args=("super-secret",),
        exc_info=None,
    )
    payload = json.loads(formatter.format(record))
    assert "***" in payload["msg"]


def test_logging_truncates_long_messages():
    formatter = JsonFormatter()
    long = "x" * (MAX_IN_LOG_BODY + 100)
    record = logging.LogRecord("test", logging.INFO, __file__, 0, "%s", (long,), None)
    payload = json.loads(formatter.format(record))
    assert payload["msg"].endswith("…(truncated)")
    assert len(payload["msg"]) <= MAX_IN_LOG_BODY + len("…(truncated)")


@pytest.mark.parametrize(
    "status_codes,expected_calls",
    [
        ([403], 1),
        ([429, 429, 200], 3),
        ([500, 500, 200], 3),
    ],
)
def test_suno_client_retries(monkeypatch, requests_mock, status_codes, expected_calls):
    monkeypatch.setenv("SUNO_API_BASE", "https://example.com")
    monkeypatch.setenv("SUNO_API_TOKEN", "tkn")
    monkeypatch.setenv("SUNO_CALLBACK_SECRET", "cb")
    monkeypatch.setenv("SUNO_CALLBACK_URL", "https://callback")
    from suno import client as client_module

    monkeypatch.setattr(client_module.time, "sleep", lambda _: None)
    response_list = []
    for status in status_codes:
        if status == 200:
            response_list.append({"status_code": 200, "json": {"task_id": "ok"}})
        else:
            response_list.append({"status_code": status, "json": {"message": "err"}})
    requests_mock.post(
        "https://example.com/api/v1/generate/add-vocals",
        response_list=response_list,
    )
    suno_client = SunoClient(base_url="https://example.com", token="tkn", max_retries=3)
    if status_codes[0] == 403:
        with pytest.raises(SunoAPIError):
            suno_client.create_music({"instrumental": False, "userId": 7})
    else:
        payload, version = suno_client.create_music({"instrumental": False, "userId": 7})
        assert payload["task_id"] == "ok"
        assert version == "v5"
    assert requests_mock.call_count == expected_calls


def test_metrics_endpoint_outputs_counters(monkeypatch):
    env = (os.getenv("APP_ENV") or "prod").strip() or "prod"
    suno_callback_total.labels(status="ok", env=env, service="test").inc()
    suno_callback_download_fail_total.labels(reason="network").inc()
    suno_task_store_total.labels(result="memory").inc()
    env = (os.getenv("APP_ENV") or "prod").strip() or "prod"
    suno_notify_ok.labels(env=env, service="test").inc()
    suno_notify_fail.labels(type="Forbidden", env=env, service="test").inc()
    suno_notify_duration_seconds.labels(env=env, service="test").observe(0.1)
    suno_enqueue_duration_seconds.labels(env=env, service="test").observe(0.2)
    payload = render_metrics().decode("utf-8")
    assert "suno_callback_total" in payload
    assert 'status="ok"' in payload
    assert "process_uptime_seconds" in payload
    assert "suno_notify_ok" in payload
    assert "suno_notify_fail" in payload
    assert "suno_enqueue_duration_seconds" in payload


def test_cleanup_old_directories(tmp_path, monkeypatch):
    monkeypatch.setattr("suno.tempfiles.TMP_CLEANUP_HOURS", 1, raising=False)
    old_dir = task_directory("old-task")
    new_dir = task_directory("new-task")
    old_file = old_dir / "file.dat"
    new_file = new_dir / "file.dat"
    old_file.write_text("old")
    new_file.write_text("new")
    old_time = time.time() - 30 * 3600
    os.utime(old_dir, (old_time, old_time))
    cleanup_old_directories(now=time.time())
    assert not old_dir.exists()
    assert new_dir.exists()
