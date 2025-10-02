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


def test_mj_deliver_as_album(bot_module):
    class _Bot:
        def __init__(self):
            self.media_calls = []
            self.photo_calls = []

        async def send_media_group(self, **kwargs):
            self.media_calls.append(kwargs)
            return [SimpleNamespace(message_id=i) for i in range(4)]

        async def send_photo(self, **kwargs):
            self.photo_calls.append(kwargs)
            return SimpleNamespace(message_id=999)

    bot = _Bot()
    items = [(b"a", "1.png"), (b"b", "2.png"), (b"c", "3.png"), (b"d", "4.png")]
    delivered = asyncio.run(
        bot_module._deliver_mj_media(  # type: ignore[attr-defined]
            bot,
            chat_id=77,
            user_id=88,
            caption="MJ",
            items=items,
            send_as_album=True,
            task_id="task-1",
        )
    )

    assert delivered is True
    assert len(bot.media_calls) == 1
    payload = bot.media_calls[0]
    assert len(payload["media"]) == 4
    assert payload["media"][0].caption == "MJ"
    assert all(item.caption is None for item in payload["media"][1:])
    assert not bot.photo_calls


def test_mj_album_fallback_on_error(bot_module, monkeypatch):
    class _Bot:
        def __init__(self):
            self.media_calls = []
            self.photo_calls = []

        async def send_media_group(self, **kwargs):
            self.media_calls.append(kwargs)
            raise RuntimeError("network fail")

        async def send_photo(self, **kwargs):
            self.photo_calls.append(kwargs)
            return SimpleNamespace(message_id=len(self.photo_calls))

    log_records = []

    class _MJLog:
        def info(self, msg, *, extra=None):  # type: ignore[override]
            log_records.append(("info", msg, extra))

        def warning(self, msg, *, extra=None):  # type: ignore[override]
            log_records.append(("warning", msg, extra))

    monkeypatch.setattr(bot_module, "mj_log", _MJLog())

    bot = _Bot()
    items = [(b"a", "1.png"), (b"b", "2.png"), (b"c", "3.png"), (b"d", "4.png")]
    delivered = asyncio.run(
        bot_module._deliver_mj_media(  # type: ignore[attr-defined]
            bot,
            chat_id=101,
            user_id=202,
            caption="MJ",
            items=items,
            send_as_album=True,
            task_id="task-2",
        )
    )

    assert delivered is True
    assert len(bot.media_calls) == 1
    assert len(bot.photo_calls) == 4
    assert any(name == "warning" and msg == "mj.album.fallback_single" for name, msg, _ in log_records)


def test_mj_fallback_to_single_photo(bot_module):
    class _Bot:
        def __init__(self):
            self.media_calls = []
            self.photo_calls = []

        async def send_media_group(self, **kwargs):
            self.media_calls.append(kwargs)
            return [SimpleNamespace(message_id=1)]

        async def send_photo(self, **kwargs):
            self.photo_calls.append(kwargs)
            return SimpleNamespace(message_id=len(self.photo_calls))

    bot = _Bot()
    items = [(b"a", "1.png")]
    delivered = asyncio.run(
        bot_module._deliver_mj_media(  # type: ignore[attr-defined]
            bot,
            chat_id=55,
            user_id=66,
            caption="MJ",
            items=items,
            send_as_album=True,
            task_id="task-3",
        )
    )

    assert delivered is True
    assert not bot.media_calls  # альбом не отправлялся из-за недостатка картинок
    assert len(bot.photo_calls) == 1
    assert bot.photo_calls[0]["caption"] == "MJ"
