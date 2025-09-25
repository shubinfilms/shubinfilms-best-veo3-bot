from __future__ import annotations

import importlib
import json
import os
import sys
from typing import Callable, Dict
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def suno_modules(monkeypatch) -> Callable[..., tuple]:
    def _loader(**env: str) -> tuple:
        defaults: Dict[str, str] = {
            "SUNO_API_BASE": "https://api.test",
            "SUNO_API_TOKEN": "token-123",
            "SUNO_GEN_PATH": "/generate",
            "SUNO_INSTR_PATH": "/instrumental",
            "SUNO_UPLOAD_EXTEND_PATH": "/extend",
            "SUNO_TASK_STATUS_PATH": "/status",
        }
        defaults.update(env)
        for key, value in defaults.items():
            monkeypatch.setenv(key, value)
        # Ensure optional values are removed when explicitly set to None
        for key, value in env.items():
            if value is None:
                monkeypatch.delenv(key, raising=False)
        import settings
        import suno.client as suno_client
        import suno.service as suno_service

        importlib.reload(settings)
        importlib.reload(suno_client)
        importlib.reload(suno_service)
        return settings, suno_client, suno_service

    return _loader


def _mock_response(status: int, payload: dict) -> MagicMock:
    response = MagicMock()
    response.status_code = status
    body = json.dumps(payload).encode()
    response.content = body
    response.json.return_value = payload
    response.headers = {}
    return response


def test_create_music_builds_request(suno_modules, monkeypatch, tmp_path):
    settings, suno_client, suno_service = suno_modules(SUNO_CALLBACK_URL="https://callback.example/hook")
    session = MagicMock()
    payload = {
        "code": 200,
        "data": {
            "taskId": "task-42",
            "items": [
                {
                    "id": "song-1",
                    "title": "Neon",
                    "audio_url": "https://cdn.example/song.mp3",
                    "image_url": "https://cdn.example/song.jpg",
                    "duration_ms": 1234,
                }
            ],
        },
    }
    session.request.return_value = _mock_response(200, payload)

    http = suno_client.SunoHttp(session=session)
    store = MagicMock()
    service = suno_service.SunoService(store=store, download_dir=tmp_path, http=http)

    result = service.create_music(title="Song", style="Synth", lyrics="text", model="MODEL_X", mode="fast")

    assert result.task_id == "task-42"
    assert result.items and result.items[0].audio_url == "https://cdn.example/song.mp3"

    args, kwargs = session.request.call_args
    assert args[0] == "POST"
    assert args[1] == "https://api.test/generate"
    assert kwargs["json"]["callback_url"] == "https://callback.example/hook"
    assert kwargs["headers"]["Authorization"] == "Bearer token-123"
    assert kwargs["headers"]["Accept"] == "application/json"
    assert kwargs["headers"]["Content-Type"] == "application/json"
    assert kwargs["headers"]["User-Agent"].startswith("best-veo3-bot/1.0")


def test_retry_on_transient_errors(monkeypatch, suno_modules, tmp_path):
    monkeypatch.setattr("random.random", lambda: 0.0)
    sleep_calls = []
    monkeypatch.setattr("time.sleep", lambda value: sleep_calls.append(value))
    settings, suno_client, suno_service = suno_modules()

    session = MagicMock()
    session.request.side_effect = [
        _mock_response(429, {"code": 429, "message": "rate"}),
        _mock_response(502, {"code": 502, "message": "bad gateway"}),
        _mock_response(200, {"code": 200, "data": {"taskId": "ok"}}),
    ]

    http = suno_client.SunoHttp(session=session)
    service = suno_service.SunoService(store=MagicMock(), download_dir=tmp_path, http=http)
    result = service.create_music(title="Song")

    assert result.task_id == "ok"
    assert session.request.call_count == 3
    # two sleeps between three attempts
    assert len(sleep_calls) == 2
    assert pytest.approx(sleep_calls[0], rel=0.05) == 1.0
    assert pytest.approx(sleep_calls[1], rel=0.05) == 2.0


def test_not_found_raises(suno_modules, tmp_path):
    settings, suno_client, suno_service = suno_modules()
    session = MagicMock()
    session.request.return_value = _mock_response(404, {"message": "missing"})

    http = suno_client.SunoHttp(session=session)
    service = suno_service.SunoService(store=MagicMock(), download_dir=tmp_path, http=http)

    with pytest.raises(suno_client.SunoNotFound):
        service.get_status("task-id")


def test_callback_secret_header(suno_modules, tmp_path):
    settings, suno_client, suno_service = suno_modules(
        SUNO_CALLBACK_URL="https://callback.example/hook",
        SUNO_CALLBACK_SECRET="secret-key",
    )
    session = MagicMock()
    session.request.return_value = _mock_response(200, {"code": 200, "data": {}})

    http = suno_client.SunoHttp(session=session)
    service = suno_service.SunoService(store=MagicMock(), download_dir=tmp_path, http=http)

    service.create_music(title="Song")

    headers = session.request.call_args.kwargs["headers"]
    assert headers["X-Callback-Token"] == "secret-key"
