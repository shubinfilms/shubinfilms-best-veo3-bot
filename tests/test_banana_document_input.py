import asyncio
import importlib
import sys
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _make_image_bytes(fmt: str = "PNG") -> bytes:
    image = Image.new("RGB", (4, 4), color=(200, 10, 10))
    buffer = BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


class _FakeFile:
    def __init__(self, file_path: str, data: bytes) -> None:
        self.file_path = file_path
        self._data = data

    async def download_as_bytearray(self) -> bytearray:
        return bytearray(self._data)


class _FakeBot:
    def __init__(self, files: dict[str, tuple[str, bytes]]) -> None:
        self._files = files
        self.get_file_calls: list[str] = []
        self.send_messages: list[dict[str, object]] = []
        self.photo_calls: list[dict[str, object]] = []
        self.document_calls: list[dict[str, object]] = []

    async def get_file(self, file_id: str) -> _FakeFile:  # type: ignore[override]
        self.get_file_calls.append(file_id)
        path, data = self._files[file_id]
        return _FakeFile(path, data)

    async def send_message(self, chat_id: int, text: str, **kwargs: object):  # type: ignore[override]
        payload = {"chat_id": chat_id, "text": text, **kwargs}
        self.send_messages.append(payload)
        return SimpleNamespace(message_id=len(self.send_messages))

    async def send_photo(self, **kwargs: object):  # type: ignore[override]
        self.photo_calls.append(kwargs)
        return SimpleNamespace(message_id=len(self.photo_calls))

    async def send_document(self, **kwargs: object):  # type: ignore[override]
        self.document_calls.append(kwargs)
        return SimpleNamespace(message_id=len(self.document_calls))


class _DummyMessage:
    def __init__(self, document=None, photo=None, caption: str | None = None) -> None:
        self.document = document
        self.photo = photo or []
        self.caption = caption
        self.replies: list[str] = []
        self.reply_kwargs: list[dict[str, object]] = []

    async def reply_text(self, text: str, **kwargs: object) -> None:  # type: ignore[override]
        self.replies.append(text)
        self.reply_kwargs.append(kwargs)


class _DummyDocument:
    def __init__(self, file_id: str, *, mime: str, name: str, size: int) -> None:
        self.file_id = file_id
        self.mime_type = mime
        self.file_name = name
        self.file_size = size


class _DummyPhoto:
    def __init__(self, file_id: str, *, width: int = 64, height: int = 64) -> None:
        self.file_id = file_id
        self.file_unique_id = f"u-{file_id}"
        self.width = width
        self.height = height


@pytest.fixture
def bot_module(monkeypatch):
    monkeypatch.setenv("TELEGRAM_TOKEN", "test-token")
    monkeypatch.setenv("SUNO_API_BASE", "https://example.com")
    monkeypatch.setenv("SUNO_API_TOKEN", "dummy-token")
    monkeypatch.setenv("LEDGER_BACKEND", "memory")
    monkeypatch.setenv("DATABASE_URL", "postgres://test")
    module = importlib.import_module("bot")
    return importlib.reload(module)


def _build_update(message, chat_id: int = 1, user_id: int = 10):
    return SimpleNamespace(
        effective_message=message,
        message=message,
        effective_chat=SimpleNamespace(id=chat_id),
        effective_user=SimpleNamespace(id=user_id),
    )


def test_document_png_added_to_banana(monkeypatch, bot_module):
    files = {"doc-1": ("file/documents/test.png", _make_image_bytes("PNG"))}
    bot = _FakeBot(files)
    ctx = SimpleNamespace(bot=bot, user_data={}, application=None)
    state = bot_module.state(ctx)
    state["mode"] = "banana"

    document = _DummyDocument("doc-1", mime="image/png", name="test.png", size=len(files["doc-1"][1]))
    message = _DummyMessage(document=document)

    monkeypatch.setattr(bot_module, "show_banana_card", lambda *args, **kwargs: asyncio.sleep(0))

    asyncio.run(bot_module.on_document(_build_update(message), ctx))

    images = bot_module.state(ctx)["banana_images"]
    assert len(images) == 1
    entry = images[0]
    assert entry["source"] == "document"
    assert entry["mime"] == "image/png"
    assert entry["url"].startswith("https://api.telegram.org/file/bottest-token/")
    assert message.replies == ["📸 Фото принято (1/4)."]


def test_document_non_image_rejected(bot_module):
    files = {"doc-2": ("file/docs/sample.pdf", b"%PDF-1.4")}
    bot = _FakeBot(files)
    ctx = SimpleNamespace(bot=bot, user_data={}, application=None)
    state = bot_module.state(ctx)
    state["mode"] = "banana"

    document = _DummyDocument("doc-2", mime="application/pdf", name="doc.pdf", size=2048)
    message = _DummyMessage(document=document)

    asyncio.run(bot_module.on_document(_build_update(message), ctx))

    assert message.replies == ["Нужно прислать изображение (PNG/JPG/WEBP) как файл-документ."]
    assert not bot_module.state(ctx)["banana_images"]
    assert not bot.get_file_calls


def test_document_too_large(bot_module):
    files = {"doc-3": ("file/documents/huge.png", _make_image_bytes("PNG"))}
    bot = _FakeBot(files)
    ctx = SimpleNamespace(bot=bot, user_data={}, application=None)
    state = bot_module.state(ctx)
    state["mode"] = "banana"

    document = _DummyDocument(
        "doc-3",
        mime="image/png",
        name="huge.png",
        size=(20 * 1024 * 1024) + 1,
    )
    message = _DummyMessage(document=document)

    asyncio.run(bot_module.on_document(_build_update(message), ctx))

    assert message.replies == ["Файл слишком большой (лимит 20 MB)."]
    assert not bot.get_file_calls


def test_mixed_photo_and_document_trigger_generation(monkeypatch, tmp_path, bot_module):
    png_bytes = _make_image_bytes("PNG")
    jpeg_bytes = _make_image_bytes("JPEG")
    files = {
        "photo-1": ("photos/test.jpg", jpeg_bytes),
        "doc-4": ("documents/photo.png", png_bytes),
    }
    bot = _FakeBot(files)
    ctx = SimpleNamespace(bot=bot, user_data={}, application=None)
    state = bot_module.state(ctx)
    state["mode"] = "banana"

    monkeypatch.setattr(bot_module, "show_banana_card", lambda *args, **kwargs: asyncio.sleep(0))

    photo_message = _DummyMessage(photo=[_DummyPhoto("photo-1")])
    asyncio.run(bot_module.on_photo(_build_update(photo_message), ctx))

    document = _DummyDocument("doc-4", mime="image/png", name="doc.png", size=len(png_bytes))
    doc_message = _DummyMessage(document=document)
    asyncio.run(bot_module.on_document(_build_update(doc_message), ctx))

    images = bot_module.state(ctx)["banana_images"]
    assert len(images) == 2
    assert images[0]["source"] == "photo"
    assert images[1]["source"] == "document"

    created_urls: list[list[str]] = []

    def _fake_create(prompt, image_urls, *args, **kwargs):
        created_urls.append(list(image_urls))
        return "task-xyz"

    monkeypatch.setattr(bot_module, "create_banana_task", _fake_create)
    monkeypatch.setattr(
        bot_module,
        "wait_for_banana_result",
        lambda *args, **kwargs: ["https://cdn.example.com/result.png"],
    )
    monkeypatch.setattr(
        bot_module,
        "_download_binary",
        lambda url: (png_bytes, "image/png"),
    )

    def _save_bytes(data, suffix=".png"):
        target = tmp_path / f"banana{suffix}"
        target.write_bytes(data)
        return target

    monkeypatch.setattr(bot_module, "save_bytes_to_temp", _save_bytes)

    asyncio.run(
        bot_module._banana_run_and_send(  # type: ignore[attr-defined]
            chat_id=1,
            ctx=ctx,
            src_images=list(images),
            prompt="clean background",
            price=5,
            user_id=101,
        )
    )

    assert len(created_urls) == 1
    assert len(created_urls[0]) == 2
    assert created_urls[0][0].startswith("https://api.telegram.org/file/bottest-token/")
    assert created_urls[0][1].startswith("https://api.telegram.org/file/bottest-token/")

