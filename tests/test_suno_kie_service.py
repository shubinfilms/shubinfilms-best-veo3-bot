import os
import sys

import pytest
import requests_mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import suno.client as suno_client_module
from suno.client import SunoClient, SunoClientError


@pytest.fixture(autouse=True)
def _suno_defaults(monkeypatch):
    monkeypatch.setattr(suno_client_module, "SUNO_API_TOKEN", "test-token", raising=False)
    monkeypatch.setattr(
        suno_client_module,
        "SUNO_CALLBACK_URL",
        "https://bot.example.com/suno-callback",
        raising=False,
    )
    monkeypatch.setattr(suno_client_module, "SUNO_CALLBACK_SECRET", "super-secret", raising=False)
    monkeypatch.setattr(suno_client_module, "SUNO_API_BASE", "https://api.kie.ai", raising=False)
    yield


def _generate_url() -> str:
    return "https://api.kie.ai/api/v1/generate"


def test_enqueue_music_builds_instrumental_payload():
    client = SunoClient()
    captured: dict[str, object] = {}

    with requests_mock.Mocker() as mocker:
        def _capture(request):
            captured["json"] = request.json()
            return True

        mocker.post(
            _generate_url(),
            json={"code": 200, "data": {"taskId": "task-1", "msg": "ok"}},
            additional_matcher=_capture,
        )

        result = client.enqueue_music(
            user_id=878622103,
            title="Ping",
            prompt="ambient",
            instrumental=True,
            has_lyrics=False,
            lyrics=None,
        )

    payload = captured.get("json")
    assert isinstance(payload, dict)
    assert payload["model"] == "V5"
    assert payload["userId"] == "878622103"
    assert payload["callBackUrl"] == "https://bot.example.com/suno-callback"
    assert "callBackSecret" not in payload
    assert payload["customMode"] is False
    assert payload["prompt_len"] == 16
    assert payload["instrumental"] is True
    assert payload["has_lyrics"] is False
    assert payload["negativeTags"] == []
    assert payload["tags"] == []
    assert "lyrics" not in payload
    assert result.task_id == "task-1"
    assert result.req_id == "task-1"
    assert result.custom_mode is False


def test_enqueue_music_builds_vocal_payload():
    client = SunoClient()
    captured: dict[str, object] = {}

    with requests_mock.Mocker() as mocker:
        def _capture(request):
            captured["json"] = request.json()
            return True

        mocker.post(
            _generate_url(),
            json={"code": 200, "data": {"taskId": "task-2", "msg": "ok"}},
            additional_matcher=_capture,
        )

        result = client.enqueue_music(
            user_id=111,
            title="Song",
            prompt="pop ballad",
            instrumental=False,
            has_lyrics=True,
            lyrics="hello world",
        )

    payload = captured.get("json")
    assert isinstance(payload, dict)
    assert payload["model"] == "V5"
    assert payload["customMode"] is False
    assert payload["has_lyrics"] is True
    assert payload["instrumental"] is False
    assert payload["lyrics"] == "hello world"
    assert result.task_id == "task-2"
    assert result.req_id == "task-2"
    assert result.custom_mode is False


def test_enqueue_music_returns_422_without_retry(monkeypatch):
    client = SunoClient()
    attempts = {"count": 0}

    def _matcher(request):
        attempts["count"] += 1
        return True

    with requests_mock.Mocker() as mocker:
        mocker.post(
            _generate_url(),
            status_code=422,
            json={
                "detail": [
                    {"loc": ["body", "lyrics"], "msg": "field required"},
                ]
            },
            additional_matcher=_matcher,
        )

        with pytest.raises(SunoClientError) as exc:
            client.enqueue_music(
                user_id=1,
                title="Song",
                prompt="pop",
                instrumental=False,
                has_lyrics=True,
                lyrics=None,
            )

    assert "Suno validation error" in str(exc.value)
    assert "lyrics" in str(exc.value)
    assert attempts["count"] == 1


def test_enqueue_music_retries_on_server_error():
    client = SunoClient()
    with requests_mock.Mocker() as mocker:
        mocker.post(
            _generate_url(),
            response_list=[
                {"status_code": 500, "json": {"message": "fail"}},
                {"status_code": 502, "json": {"message": "still failing"}},
                {"status_code": 200, "json": {"data": {"taskId": "ok"}}},
            ],
        )

        result = client.enqueue_music(
            user_id=2,
            title="Retry",
            prompt="ambient",
            instrumental=True,
            has_lyrics=False,
            lyrics=None,
        )

    assert result.task_id == "ok"
    assert result.req_id == "ok"
    assert result.custom_mode is False
    assert len(mocker.request_history) == 3
