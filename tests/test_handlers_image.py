import asyncio
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
    monkeypatch.setenv("TELEGRAM_TOKEN", "test-token")
    monkeypatch.setenv("SUNO_API_BASE", "https://example.com")
    monkeypatch.setenv("SUNO_API_TOKEN", "dummy-token")
    monkeypatch.setenv("LEDGER_BACKEND", "memory")
    monkeypatch.setenv("DATABASE_URL", "postgres://test")
    module = importlib.import_module("bot")
    return importlib.reload(module)


def test_banana_deliver_photo_and_document(monkeypatch, tmp_path, bot_module):
    path = tmp_path / "banana.png"
    path.write_bytes(b"image-bytes")

    class _Bot:
        def __init__(self):
            self.photo_calls = []
            self.doc_calls = []

        async def send_photo(self, **kwargs):
            self.photo_calls.append(kwargs)
            return SimpleNamespace(message_id=101)

        async def send_document(self, **kwargs):
            self.doc_calls.append(kwargs)
            return SimpleNamespace(message_id=202)

    bot = _Bot()
    caption = "Preview"
    delivered = asyncio.run(
        bot_module._deliver_banana_media(  # type: ignore[attr-defined]
            bot,
            chat_id=1,
            user_id=2,
            file_path=path,
            caption=caption,
            reply_markup=None,
            send_document=True,
        )
    )

    assert delivered is True
    assert not path.exists()
    assert len(bot.photo_calls) == 1
    assert bot.photo_calls[0]["caption"] == caption
    assert len(bot.doc_calls) == 1
    doc_call = bot.doc_calls[0]
    assert doc_call["caption"] is None
    assert doc_call["disable_notification"] is True
    assert doc_call["document"].filename == "result.png"


def test_banana_deliver_skips_document(monkeypatch, tmp_path, bot_module):
    path = tmp_path / "banana.png"
    path.write_bytes(b"image-bytes")

    class _Bot:
        def __init__(self):
            self.photo_calls = []
            self.document_called = False

        async def send_photo(self, **kwargs):
            self.photo_calls.append(kwargs)
            return SimpleNamespace(message_id=303)

        async def send_document(self, **kwargs):
            self.document_called = True
            raise AssertionError("document should not be sent")

    bot = _Bot()
    delivered = asyncio.run(
        bot_module._deliver_banana_media(  # type: ignore[attr-defined]
            bot,
            chat_id=10,
            user_id=20,
            file_path=path,
            caption="Caption",
            reply_markup=None,
            send_document=False,
        )
    )

    assert delivered is True
    assert not path.exists()
    assert len(bot.photo_calls) == 1
    assert bot.document_called is False


def test_mj_documents_sent_without_compression(monkeypatch, bot_module):
    calls: list[dict[str, object]] = []

    async def _fake_send(bot, chat_id, data, filename, *, caption=None, reply_markup=None, req_id=None):
        calls.append({"filename": filename, "caption": caption})
        return SimpleNamespace(message_id=len(calls))

    monkeypatch.setattr(bot_module, "send_image_as_document", _fake_send)

    delivered = asyncio.run(
        bot_module._deliver_mj_media(  # type: ignore[attr-defined]
            SimpleNamespace(),
            chat_id=55,
            user_id=66,
            caption="MJ",
            items=[(b"a", "1.png"), (b"b", "2.png")],
            send_as_album=True,
            task_id="task-3",
        )
    )

    assert delivered is True
    assert [call["caption"] for call in calls] == ["MJ", None]
    assert calls[0]["filename"] == "midjourney_task-3_01.png"
    assert calls[1]["filename"] == "midjourney_task-3_02.png"
