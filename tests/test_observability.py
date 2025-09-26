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
        ([408, 408, 200], 3),
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
    responses = []
    for status in status_codes:
        if status == 200:
            responses.append({"status_code": 200, "json": {"task_id": "ok"}})
        else:
            responses.append({"status_code": status, "json": {"message": "err"}})
    requests_mock.post("https://example.com/api/v1/generate/music", responses)
    suno_client = SunoClient(base_url="https://example.com", token="tkn", max_retries=3)
    if status_codes[0] == 403:
        with pytest.raises(SunoAPIError):
            suno_client.create_music({})
    else:
        suno_client.create_music({})
    assert requests_mock.call_count == expected_calls


def test_metrics_endpoint_outputs_counters(monkeypatch):
    suno_callback_total.labels(type="test", code="200").inc()
    suno_callback_download_fail_total.labels(reason="network").inc()
    suno_task_store_total.labels(result="memory").inc()
    payload = render_metrics().decode("utf-8")
    assert "suno_callback_total" in payload
    assert 'code="200"' in payload
    assert 'type="test"' in payload
    assert "process_uptime_seconds" in payload


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
