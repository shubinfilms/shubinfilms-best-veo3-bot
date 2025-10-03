import asyncio
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def bot_module(monkeypatch):
    monkeypatch.setenv("TELEGRAM_TOKEN", "dummy-token")
    monkeypatch.setenv("SUNO_API_BASE", "https://example.com")
    monkeypatch.setenv("SUNO_API_TOKEN", "token")
    monkeypatch.setenv("LEDGER_BACKEND", "memory")
    monkeypatch.setenv("DATABASE_URL", "postgres://test")
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://bot.example")
    module = importlib.import_module("bot")
    return importlib.reload(module)


class DummyBot:
    def __init__(self):
        self.sent_stickers = []
        self.deleted_messages = []
        self.sent_documents = []
        self._next_id = 100

    async def send_sticker(self, chat_id, sticker_id):  # pragma: no cover - async stub
        self.sent_stickers.append((chat_id, sticker_id))
        self._next_id += 1
        return SimpleNamespace(message_id=self._next_id)

    async def delete_message(self, chat_id, message_id):  # pragma: no cover - async stub
        self.deleted_messages.append((chat_id, message_id))

    async def send_document(self, chat_id, document):  # pragma: no cover - async stub
        self.sent_documents.append((chat_id, getattr(document, "name", None)))


def test_wait_sticker_replace(bot_module, tmp_path):
    bot = DummyBot()
    ctx = SimpleNamespace(bot=bot)

    wait_id = asyncio.run(bot_module.show_wait_sticker(ctx, 123, "sticker"))
    assert bot.sent_stickers == [(123, "sticker")]

    file_path = tmp_path / "result.mp4"
    file_path.write_bytes(b"data")

    asyncio.run(bot_module.replace_wait_with_docs(ctx, 123, wait_id, [str(file_path)]))

    assert bot.deleted_messages == [(123, wait_id)]
    assert bot.sent_documents == [(123, str(file_path))]
