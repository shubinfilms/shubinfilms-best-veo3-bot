import asyncio
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def bot_module(monkeypatch):
    monkeypatch.setenv("TELEGRAM_TOKEN", "test-token")
    monkeypatch.setenv("SUNO_API_BASE", "https://example.com")
    monkeypatch.setenv("SUNO_API_TOKEN", "dummy-token")
    monkeypatch.setenv("LEDGER_BACKEND", "memory")
    monkeypatch.setenv("DATABASE_URL", "postgres://test")
    module = importlib.import_module("bot")
    module = importlib.reload(module)
    module.mj_log.disabled = True
    return module


def _make_update(data: str):
    async def answer(text: Optional[str] = None, show_alert: bool = False):
        return None

    query = SimpleNamespace(
        data=data,
        message=SimpleNamespace(chat=SimpleNamespace(id=100), message_id=0),
        answer=answer,
        from_user=SimpleNamespace(id=200, language_code="ru"),
    )
    return SimpleNamespace(
        callback_query=query,
        effective_chat=SimpleNamespace(id=100),
        effective_user=SimpleNamespace(id=200, language_code="ru"),
    )


def test_mj_upscale_callback(monkeypatch, bot_module):
    grid = {"task_id": "grid", "result_urls": ["a", "b", "c", "d"]}
    monkeypatch.setattr(bot_module, "_load_mj_grid_snapshot", lambda *_: grid)
    monkeypatch.setattr(bot_module, "acquire_ttl_lock", lambda *args, **kwargs: True)
    monkeypatch.setattr(bot_module, "release_ttl_lock", lambda *args, **kwargs: None)

    gallery_calls = []

    def fake_gallery(chat_id, message_id):
        gallery_calls.append((chat_id, message_id))
        return [
            {"source_url": "a", "file_name": "midjourney_01.jpeg", "bytes_len": 2048, "mime": "image/jpeg", "sent_message_id": 11},
            {"source_url": "b", "file_name": "midjourney_02.jpeg", "bytes_len": 2048, "mime": "image/jpeg", "sent_message_id": 12},
            {"source_url": "c", "file_name": "midjourney_03.jpeg", "bytes_len": 2048, "mime": "image/jpeg", "sent_message_id": 13},
            {"source_url": "d", "file_name": "midjourney_04.jpeg", "bytes_len": 2048, "mime": "image/jpeg", "sent_message_id": 14},
        ]

    monkeypatch.setattr(bot_module, "get_mj_gallery", fake_gallery)

    launch_calls = []

    async def fake_launch(chat_id, ctx, *, user_id, grid, image_index, locale, source, **kwargs):
        launch_calls.append(
            {
                "chat_id": chat_id,
                "user_id": user_id,
                "grid": grid,
                "index": image_index,
                "locale": locale,
                "source": source,
            }
        )
        return True

    monkeypatch.setattr(bot_module, "_launch_mj_upscale", fake_launch)

    async def fake_send_message(*args, **kwargs):  # pragma: no cover - should not trigger
        raise AssertionError("send_message should not be used")

    ctx = SimpleNamespace(
        bot=SimpleNamespace(send_message=fake_send_message),
        user_data={},
    )

    asyncio.run(bot_module.handle_mj_upscale_choice(_make_update("mj.upscale:grid:2"), ctx))

    assert gallery_calls == [(100, 0)]
    assert launch_calls
    assert launch_calls[0]["index"] == 1
    assert launch_calls[0]["grid"] is grid
