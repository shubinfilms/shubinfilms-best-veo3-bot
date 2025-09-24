from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Tuple
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from suno.client import SunoClient
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
