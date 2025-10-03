import asyncio
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

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


def test_mj_document_not_empty(monkeypatch, bot_module):
    base_urls = [
        "https://cdn.example/mj/1.jpeg",
        "https://cdn.example/mj/2.jpeg",
        "https://cdn.example/mj/3.jpeg",
        "https://cdn.example/mj/4.jpeg",
    ]

    call_counts = {url: 0 for url in base_urls}

    def fake_download(url: str, index: int):
        call_counts[url] += 1
        return b"x" * (2048 + index), f"midjourney_{index:02d}.jpeg", "image/jpeg", url

    monkeypatch.setattr(bot_module, "_download_mj_image_bytes", fake_download)

    sent_docs = []
    sent_messages = []
    gallery_store = []

    async def fake_send_document(chat_id, document, **kwargs):
        sent_docs.append(document)
        return SimpleNamespace(message_id=len(sent_docs) + 10)

    async def fake_send_message(chat_id, text, reply_markup=None):
        sent_messages.append((text, reply_markup))
        return SimpleNamespace(message_id=999)

    monkeypatch.setattr(bot_module, "set_mj_gallery", lambda *args: gallery_store.append(args))

    bot = SimpleNamespace(send_document=fake_send_document, send_message=fake_send_message)
    ctx = SimpleNamespace(bot=bot, user_data={})

    state = bot_module.state(ctx)
    state["mj_locale"] = "ru"

    result = asyncio.run(
        bot_module._deliver_mj_grid_documents(
            ctx,
            chat_id=123,
            user_id=456,
            grid_id="grid123",
            urls=base_urls,
            prompt="test",
        )
    )

    assert sum(call_counts.values()) == 4
    assert len(sent_docs) == 4
    for idx, doc in enumerate(sent_docs, start=1):
        assert doc.filename == f"midjourney_{idx:02d}.jpeg"
        assert len(doc.input_file_content) > 1024
    assert sent_messages[0][0] == "Галерея сгенерирована."
    assert gallery_store, "gallery metadata should be persisted"
