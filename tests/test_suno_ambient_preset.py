import os
import sys
from typing import Any, Mapping, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from suno.client import (
    AMBIENT_NATURE_PRESET,
    AMBIENT_NATURE_PRESET_ID,
    SunoClient,
)
from suno.service import SunoService, TelegramMeta, TaskLink
from suno.schemas import SunoTask, SunoTrack
from utils.suno_state import SunoState, build_generation_payload


def _prepare_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUNO_API_BASE", "https://api.example.com")
    monkeypatch.setenv("SUNO_API_TOKEN", "token")
    monkeypatch.setenv("SUNO_CALLBACK_URL", "https://callback.local/suno-callback")
    monkeypatch.setenv("SUNO_CALLBACK_SECRET", "secret-token")
    monkeypatch.setenv("SUNO_ENABLED", "true")
    monkeypatch.setenv("SUNO_MODEL", "V5")
    monkeypatch.setattr("suno.client.SUNO_CALLBACK_URL", "https://callback.local/suno-callback", raising=False)
    monkeypatch.setattr("suno.client.SUNO_CALLBACK_SECRET", "secret-token", raising=False)
    monkeypatch.setattr("suno.client.SUNO_MODEL", "V5", raising=False)


def test_ambient_preset_auto_tags(monkeypatch: pytest.MonkeyPatch) -> None:
    _prepare_env(monkeypatch)
    client = SunoClient(base_url="https://api.example.com", token="token")

    state = SunoState()
    state.preset = AMBIENT_NATURE_PRESET_ID

    payload = build_generation_payload(state, model="V5", lang="en")
    result = client.build_payload(
        user_id="777",
        title=payload.get("title"),
        prompt=payload.get("prompt"),
        instrumental=payload.get("instrumental", True),
        has_lyrics=payload.get("has_lyrics", False),
        lyrics=payload.get("lyrics"),
        model=payload.get("model"),
        tags=payload.get("tags"),
        negative_tags=payload.get("negative_tags"),
        preset=payload.get("preset"),
    )

    expected_tags = [str(tag).strip().lower() for tag in AMBIENT_NATURE_PRESET["tags"]]
    expected_negative = [str(tag).strip().lower() for tag in AMBIENT_NATURE_PRESET["negative_tags"]]

    assert result["instrumental"] is True
    assert result["has_lyrics"] is False
    assert result["tags"] == expected_tags
    assert result["negativeTags"] == expected_negative
    assert "ocean waves" in result["prompt"].lower()
    assert result["title"].strip()


def test_ambient_preset_fallback_cover_and_caption(monkeypatch: pytest.MonkeyPatch) -> None:
    service = SunoService(redis=None, telegram_token="test-token")

    existing_record = {
        "preset": AMBIENT_NATURE_PRESET_ID,
        "title": "Oceanic Dreams",
        "user_title": "Oceanic Dreams",
    }

    monkeypatch.setattr(service, "_load_task_record", lambda task_id: existing_record)
    monkeypatch.setattr(
        service,
        "_load_mapping",
        lambda task_id: TelegramMeta(
            chat_id=123,
            msg_id=10,
            title="Oceanic Dreams",
            ts="now",
            req_id="req-ambient",
            user_title="Oceanic Dreams",
        ),
    )
    monkeypatch.setattr(
        service,
        "_load_user_link",
        lambda task_id: TaskLink(user_id=555, prompt="Preset Prompt", ts="now"),
    )
    monkeypatch.setattr(service, "_save_task_record", lambda *args, **kwargs: None)
    monkeypatch.setattr(service, "_send_text", lambda *args, **kwargs: None)
    monkeypatch.setattr(service, "_find_local_file", lambda *args, **kwargs: None)
    monkeypatch.setattr(service, "_log_delivery", lambda *args, **kwargs: None)

    events: list[tuple[str, Mapping[str, Any]]] = []

    def _fake_cover(**kwargs: Any) -> tuple[bool, Optional[str]]:
        events.append(("cover", kwargs))
        return True, None

    def _fake_audio(**kwargs: Any) -> tuple[bool, Optional[str]]:
        events.append(("audio", kwargs))
        return True, None

    monkeypatch.setattr(service, "_send_cover_url", _fake_cover)
    monkeypatch.setattr(service, "_send_audio_url_with_retry", _fake_audio)

    track = SunoTrack(
        id="take-1",
        title="",
        source_audio_url="https://cdn/audio.mp3",
        source_image_url=None,
        audio_url="https://cdn/audio.mp3",
        image_url=None,
        tags="ambient, cinematic, soundscape",
        duration=172.4,
    )

    task = SunoTask(
        task_id="ambient-task",
        callback_type="complete",
        items=[track],
        msg="ok",
        code=200,
    )

    service.handle_callback(task, req_id="req-ambient", delivery_via="webhook")

    cover_events = [event for event in events if event[0] == "cover"]
    audio_events = [event for event in events if event[0] == "audio"]
    assert cover_events and audio_events

    cover_payload = cover_events[0][1]
    assert "pollinations" in cover_payload["photo_url"]
    assert "surreal+ocean+horizon" in cover_payload["photo_url"]

    audio_payload = audio_events[0][1]
    caption = audio_payload["caption"]
    assert "Ambient Preset" in caption
    assert "ocean waves" in caption
    assert caption.startswith("🌊")
