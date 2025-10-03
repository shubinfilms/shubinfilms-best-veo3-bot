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
    ctx = SimpleNamespace(bot=SimpleNamespace(send_document=None, send_message=None))
    doc_calls: list[dict[str, object]] = []
    menu_calls: list[dict[str, object]] = []

    async def fake_send_document(chat_id, *, document, **kwargs):
        doc_calls.append({"chat_id": chat_id, "document": document, "kwargs": kwargs})

    async def fake_send_message(chat_id, text, *, reply_markup=None):
        menu_calls.append({"chat_id": chat_id, "text": text, "markup": reply_markup})

    monkeypatch.setattr(ctx.bot, "send_document", fake_send_document)
    monkeypatch.setattr(ctx.bot, "send_message", fake_send_message)
    monkeypatch.setattr(bot_module, "_save_mj_grid_snapshot", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        bot_module,
        "_download_mj_image_bytes",
        lambda url, index: (b"x" * 2048, f"midjourney_{index:02d}.png", "image/png", url),
    )

    delivered = asyncio.run(
        bot_module._deliver_mj_grid_documents(
            ctx,
            chat_id=55,
            user_id=66,
            grid_id="task-3",
            urls=["https://cdn/1.png", "https://cdn/2.png"],
            prompt="MJ",
        )
    )

    assert delivered is True
    assert [call["document"].filename for call in doc_calls] == [
        "midjourney_01.png",
        "midjourney_02.png",
    ]
    assert menu_calls and menu_calls[0]["markup"].inline_keyboard[0][0].callback_data.startswith("mj.upscale.menu:")
