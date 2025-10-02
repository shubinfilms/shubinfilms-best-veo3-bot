import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from suno.service import SunoService, TelegramMeta, TaskLink
from suno.schemas import SunoTask, SunoTrack


def test_suno_uses_user_title_in_caption(monkeypatch):
    service = SunoService(redis=None, telegram_token="test-token")

    meta = TelegramMeta(
        chat_id=321,
        msg_id=99,
        title="Generated",
        ts="now",
        req_id="req-77",
        user_title="My Custom Title",
    )
    link = TaskLink(user_id=42, prompt="Prompt", ts="now")

    monkeypatch.setattr(service, "_load_mapping", lambda task_id: meta)
    monkeypatch.setattr(service, "_load_user_link", lambda task_id: link)
    monkeypatch.setattr(service, "_save_task_record", lambda *args, **kwargs: None)
    monkeypatch.setattr(service, "_send_text", lambda *args, **kwargs: None)
    monkeypatch.setattr(service, "_find_local_file", lambda *args, **kwargs: None)
    monkeypatch.setattr(service, "_log_delivery", lambda *args, **kwargs: None)
    monkeypatch.setattr(service, "_send_cover_url", lambda **_: (True, None))

    captured = {}

    def _capture_audio(**kwargs):
        captured.update(kwargs)
        return True, None

    monkeypatch.setattr(service, "_send_audio_url_with_retry", _capture_audio)

    track = SunoTrack(
        id="take-5",
        title="Autogen",
        source_audio_url="https://cdn/audio.mp3",
        source_image_url="https://cdn/cover.jpg",
        duration=47.2,
        tags="synthwave, retro",
    )
    task = SunoTask(task_id="task-title", callback_type="complete", items=[track], msg="ok", code=200)

    service.handle_callback(task, req_id="req-77", delivery_via="webhook")

    assert captured
    caption = captured.get("caption")
    assert isinstance(caption, str)
    assert caption.startswith("My Custom Title (Take 1)")
    assert "synthwave" in caption
    assert captured.get("title") == "My Custom Title"
