import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("SUNO_API_BASE", "https://example.com")
os.environ.setdefault("SUNO_API_TOKEN", "token")
os.environ.setdefault("SUNO_CALLBACK_URL", "https://callback.example")
os.environ.setdefault("SUNO_CALLBACK_SECRET", "secret")

from suno.schemas import SunoTask, SunoTrack
from suno.service import SunoService, TelegramMeta


class FakeClient:
    def __init__(self, response: dict) -> None:
        self._response = response

    def get_lyrics(self, task_id: str, *, req_id=None, payload=None):  # type: ignore[override]
        return self._response

    def enqueue(self, payload, req_id=None):  # type: ignore[override]
        raise AssertionError("enqueue should not be called in final attempt")


class WarningService(SunoService):
    def __init__(self, response: dict) -> None:
        super().__init__(client=FakeClient(response), redis=None, telegram_token="dummy-token")
        self.retry_called = False
        self.messages: list[str] = []

    def _strict_retry_enqueue(self, **kwargs):  # type: ignore[override]
        self.retry_called = True
        return False

    def _send_text(self, chat_id: int, text: str, reply_to=None):  # type: ignore[override]
        self.messages.append(text)
        return True

    def _send_audio_url_with_retry(self, **kwargs):  # type: ignore[override]
        return True, None

    def _send_cover_url(self, **kwargs):  # type: ignore[override]
        return True, None

    def _send_file(self, *args, **kwargs):  # type: ignore[override]
        return True


def test_vocal_similarity_low_still_low_warn_user() -> None:
    service = WarningService({"data": {"lyrics": "not matching"}})
    meta_payload = {
        "chat_id": 1,
        "msg_id": 12,
        "title": "Demo",
        "ts": datetime.now(timezone.utc).isoformat(),
        "req_id": "req-2",
        "user_title": "Demo",
        "strict_enabled": True,
        "strict_threshold": 0.75,
        "original_lyrics": "hello world",
        "strict_payload": {"lyrics": "hello world"},
        "strict_lyrics": {
            "original": "hello world",
            "payload": {"lyrics": "hello world"},
            "attempts": 1,
            "threshold": 0.75,
        },
    }
    service._store_mapping("task-3", meta_payload)
    task = SunoTask(
        task_id="task-3",
        callback_type="complete",
        items=[
            SunoTrack(id="1", title="Take", audio_url="http://example.com/audio.mp3", image_url=None, tags=None, duration=12.0)
        ],
        code=200,
        msg="ok",
    )

    service.handle_callback(task, req_id="req-2")

    assert service.retry_called is False
    warning_messages = [msg for msg in service.messages if "⚠️" in msg]
    assert warning_messages, "warning message should be sent"
    assert "not matching" in warning_messages[-1]
