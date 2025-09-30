import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("SUNO_API_BASE", "https://example.com")
os.environ.setdefault("SUNO_API_TOKEN", "token")
os.environ.setdefault("SUNO_CALLBACK_URL", "https://callback.example")
os.environ.setdefault("SUNO_CALLBACK_SECRET", "secret")

from suno.schemas import SunoTask
from suno.service import SunoService, TelegramMeta


class FakeClient:
    def __init__(self, responses: list[dict]) -> None:
        self._responses = list(responses)

    def get_lyrics(self, task_id: str, *, req_id=None, payload=None):  # type: ignore[override]
        if self._responses:
            return self._responses.pop(0)
        return {"data": {"lyrics": ""}}


class RetryService(SunoService):
    def __init__(self, responses: list[dict]) -> None:
        super().__init__(client=FakeClient(responses), redis=None, telegram_token="dummy-token")
        self.retry_called = False

    def _strict_retry_enqueue(self, **kwargs):  # type: ignore[override]
        self.retry_called = True
        return True


def test_vocal_similarity_low_one_retry_then_ok() -> None:
    service = RetryService([{"data": {"lyrics": "different text"}}])
    meta = TelegramMeta(
        chat_id=1,
        msg_id=11,
        title="Demo",
        ts=datetime.now(timezone.utc).isoformat(),
        req_id="req-1",
        extras={
            "strict_enabled": True,
            "strict_threshold": 0.75,
            "original_lyrics": "hello world",
            "strict_payload": {"lyrics": "hello world"},
            "strict_lyrics": {"original": "hello world", "payload": {"lyrics": "hello world"}, "attempts": 0},
        },
    )
    context = service._strict_context_from_meta(meta)
    assert context is not None

    result = service._process_strict_delivery(
        task=SunoTask(task_id="task-2", callback_type="complete", items=[]),
        meta=meta,
        link=None,
        strict_context=context,
        req_id="req-1",
    )

    assert result == "retry"
    assert service.retry_called is True
