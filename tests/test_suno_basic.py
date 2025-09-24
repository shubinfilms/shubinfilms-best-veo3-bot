from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Tuple
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from suno.callbacks import MusicCallback, parse_music_callback
from suno.client import ENDPOINTS, SunoClient
from suno.service import SunoService
from suno.store import InMemoryTaskStore


class ImmediateExecutor:
    def submit(self, fn, *args, **kwargs):
        fn(*args, **kwargs)
        return MagicMock()


def build_service(tmp_path: Path, collector: List[Tuple[str, Path]]):
    store = InMemoryTaskStore()

    def fake_downloader(url: str, dest_path, base_dir=None):
        base_dir = Path(base_dir) if base_dir else Path("downloads")
        if not isinstance(dest_path, Path):
            dest_path = Path(dest_path)
        collector.append((url, base_dir / dest_path))
        return base_dir / dest_path

    service = SunoService(
        store=store,
        download_dir=tmp_path,
        executor=ImmediateExecutor(),
        downloader=fake_downloader,
    )
    return service, store


def test_endpoint_url_building():
    session = MagicMock()
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"code": 200}
    session.request.return_value = response
    client = SunoClient(base_url="https://api.example.com", token="token", session=session)
    expected = "https://api.example.com" + ENDPOINTS["generate_music"]
    assert client._url_for("generate_music") == expected


def test_music_callback_extracts_assets(tmp_path):
    collected: List[Tuple[str, Path]] = []
    service, store = build_service(tmp_path, collected)
    payload = {
        "code": 200,
        "data": {
            "taskId": "task123",
            "callbackType": "first",
            "response": {
                "audios": [
                    {"audioId": "a1", "audioUrl": "https://cdn.example.com/a1.mp3"},
                    {"audioId": "a2", "audioUrl": "https://cdn.example.com/a2.wav"},
                ],
                "images": ["https://cdn.example.com/cover.png"],
            },
        },
    }
    service.handle_music_callback(payload)
    assert store.is_processed("task123", "first")
    assert len(collected) == 3
    urls = {item[0] for item in collected}
    assert "https://cdn.example.com/a1.mp3" in urls
    assert all(path.parent == tmp_path / "task123" for _, path in collected)


def test_duplicate_callback_is_ignored(tmp_path):
    collected: List[Tuple[str, Path]] = []
    service, store = build_service(tmp_path, collected)
    payload = {
        "code": 200,
        "data": {
            "taskId": "task123",
            "callbackType": "text",
            "response": {},
        },
    }
    service.handle_music_callback(payload)
    service.handle_music_callback(payload)
    assert len(collected) == 0
    assert store.is_processed("task123", "text")


def test_suno_client_requires_content():
    client = SunoClient(base_url="https://api.example.com", token="token", session=MagicMock())
    with pytest.raises(ValueError):
        client.style_generate("")


def test_generate_music_requires_prompt():
    client = SunoClient(base_url="https://api.example.com", token="token", session=MagicMock())
    with pytest.raises(ValueError):
        client.generate_music(
            prompt="",
            model="model",
            title="title",
            style="style",
            callBackUrl="https://callback",
        )


def test_parse_music_callback_normalizes_type():
    payload = {
        "code": 200,
        "data": {
            "taskId": "abc",
            "callbackType": "FIRST",
            "response": {
                "tracks": [
                    {
                        "audioId": "track1",
                        "audioUrl": "https://cdn.example.com/track1.mp3",
                        "imageUrl": "https://cdn.example.com/track1.jpg",
                    }
                ]
            },
        },
    }
    callback = parse_music_callback(payload)
    assert isinstance(callback, MusicCallback)
    assert callback.type == "first"
    assert callback.task_id == "abc"
    assert callback.tracks and callback.tracks[0].audio_id == "track1"


def test_client_post_includes_auth(monkeypatch):
    session = MagicMock()
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"code": 200, "data": {}}
    session.request.return_value = response
    client = SunoClient(base_url="https://api.example.com", token="secret", session=session)
    client.add_instrumental({"uploadUrl": "x", "callBackUrl": "cb"})
    args, kwargs = session.request.call_args
    assert kwargs["headers"]["Authorization"] == "Bearer secret"
    assert kwargs["headers"]["Content-Type"] == "application/json"


def test_generate_music_retries_on_429():
    session = MagicMock()
    first = MagicMock()
    first.status_code = 429
    first.text = "rate limit"
    first.json.return_value = {"code": 429, "msg": "rate"}
    second = MagicMock()
    second.status_code = 200
    second.json.return_value = {"code": 200, "data": {"taskId": "xyz"}}
    session.request.side_effect = [first, second]
    client = SunoClient(base_url="https://api.example.com", token="token", session=session)
    result = client.generate_music(
        prompt="melody",
        model="model",
        title="title",
        style="style",
        callBackUrl="https://callback",
    )
    assert result["data"]["taskId"] == "xyz"
    assert session.request.call_count == 2


def test_music_callback_deduplicates_audio_across_types(tmp_path):
    collected: List[Tuple[str, Path]] = []
    service, store = build_service(tmp_path, collected)
    base_payload = {
        "code": 200,
        "data": {
            "taskId": "task777",
            "callbackType": "first",
            "response": {
                "tracks": [
                    {
                        "audioId": "songA",
                        "audioUrl": "https://cdn.example.com/songA.mp3",
                        "imageUrl": "https://cdn.example.com/songA.jpg",
                    }
                ]
            },
        },
    }
    service.handle_music_callback(parse_music_callback(base_payload))
    follow_up = base_payload.copy()
    follow_up["data"] = dict(base_payload["data"])
    follow_up["data"]["callbackType"] = "complete"
    service.handle_music_callback(parse_music_callback(follow_up))
    assert store.is_processed("task777", "first")
    assert store.is_processed("task777", "complete")
    assert len(collected) == 2  # audio + image only once
    saved_paths = {path for _, path in collected}
    assert tmp_path / "task777" / "songA.mp3" in saved_paths
    assert tmp_path / "task777" / "songA.jpeg" in saved_paths
