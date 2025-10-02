import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from suno.service import SunoService, TelegramMeta, TaskLink
from suno.schemas import SunoTask, SunoTrack


class _MemoryRedis:
    def __init__(self):
        self.store = {}

    def set(self, key, value, nx=False, ex=None):
        if nx:
            if key in self.store:
                return False
            self.store[key] = value
            return True
        self.store[key] = value
        return True

    # Compatibility no-ops
    def setex(self, *args, **kwargs):
        return True

    def delete(self, *args, **kwargs):
        return True

    def lpush(self, *args, **kwargs):
        return True

    def lrange(self, *args, **kwargs):
        return []

    def get(self, key):
        return self.store.get(key)


@pytest.mark.parametrize("delivery_order", [("poll", "webhook"), ("webhook", "poll")])
def test_suno_dedupe_shared_store(monkeypatch, delivery_order):
    redis = _MemoryRedis()
    service_a = SunoService(redis=redis, telegram_token="test-token")
    service_b = SunoService(redis=redis, telegram_token="test-token")

    meta = TelegramMeta(
        chat_id=123,
        msg_id=77,
        title="Demo",
        ts="now",
        req_id="req-1",
        user_title="Custom",
    )
    link = TaskLink(user_id=555, prompt="Prompt", ts="now")

    def _setup_service(service):
        monkeypatch.setattr(service, "_load_mapping", lambda task_id: meta)
        monkeypatch.setattr(service, "_load_user_link", lambda task_id: link)
        monkeypatch.setattr(service, "_save_task_record", lambda *args, **kwargs: None)
        monkeypatch.setattr(service, "_send_text", lambda *args, **kwargs: None)
        monkeypatch.setattr(service, "_find_local_file", lambda *args, **kwargs: None)
        monkeypatch.setattr(service, "_log_delivery", lambda *args, **kwargs: None)

    _setup_service(service_a)
    _setup_service(service_b)

    events_a = []
    events_b = []

    monkeypatch.setattr(
        service_a,
        "_send_audio_url_with_retry",
        lambda **kwargs: (events_a.append(kwargs), None) and (True, None),
    )
    monkeypatch.setattr(
        service_b,
        "_send_audio_url_with_retry",
        lambda **kwargs: (events_b.append(kwargs), None) and (True, None),
    )
    monkeypatch.setattr(service_a, "_send_cover_url", lambda **_: (True, None))
    monkeypatch.setattr(service_b, "_send_cover_url", lambda **_: (True, None))

    track = SunoTrack(
        id="take-1",
        title="",
        source_audio_url="https://cdn/audio.mp3",
        source_image_url="https://cdn/cover.jpg",
        duration=30.0,
        tags="rock",
    )
    task = SunoTask(task_id="task-1", callback_type="complete", items=[track], msg="ok", code=200)

    first_via, second_via = delivery_order
    first_service = service_a if first_via == "poll" else service_b
    second_service = service_b if first_service is service_a else service_a

    first_service.handle_callback(task, req_id="req-1", delivery_via=first_via)
    second_service.handle_callback(task, req_id="req-1", delivery_via=second_via)

    assert len(events_a) + len(events_b) == 1
    if events_a:
        assert events_b == []
    else:
        assert len(events_b) == 1
