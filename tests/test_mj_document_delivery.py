import asyncio
import importlib
import sys
from types import SimpleNamespace
from pathlib import Path

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


def _build_items(count: int) -> list[tuple[bytes, str]]:
    return [(f"data-{i}".encode(), f"file{i}.png") for i in range(count)]


def test_documents_sent_sequentially(monkeypatch, bot_module):
    calls: list[dict[str, object]] = []

    async def _fake_send_image(bot, chat_id, data, filename, *, caption=None, reply_markup=None, req_id=None):
        calls.append(
            {
                "chat_id": chat_id,
                "data": data,
                "filename": filename,
                "caption": caption,
            }
        )
        return SimpleNamespace(message_id=len(calls))

    monkeypatch.setattr(bot_module, "send_image_as_document", _fake_send_image)

    bot = SimpleNamespace()
    delivered = asyncio.run(
        bot_module._deliver_mj_media(  # type: ignore[attr-defined]
            bot,
            chat_id=100,
            user_id=200,
            caption="Caption",
            items=_build_items(3),
            send_as_album=True,
            task_id="task-ABC",
        )
    )

    assert delivered is True
    assert [call["caption"] for call in calls] == ["Caption", None, None]
    assert [call["filename"] for call in calls] == [
        "midjourney_task-ABC_01.png",
        "midjourney_task-ABC_02.png",
        "midjourney_task-ABC_03.png",
    ]


def test_document_filenames_use_original_extension(monkeypatch, bot_module):
    calls: list[dict[str, object]] = []

    async def _fake_send(bot, chat_id, data, filename, *, caption=None, reply_markup=None, req_id=None):
        calls.append({"filename": filename})
        return SimpleNamespace(message_id=len(calls))

    monkeypatch.setattr(bot_module, "send_image_as_document", _fake_send)

    items = [(b"a", "grid.JPEG"), (b"b", "noext"), (b"c", "image.webp")]
    asyncio.run(
        bot_module._deliver_mj_media(  # type: ignore[attr-defined]
            SimpleNamespace(),
            chat_id=1,
            user_id=2,
            caption="",
            items=items,
            send_as_album=False,
            task_id="tid",
        )
    )

    filenames = [call["filename"] for call in calls]
    assert filenames == [
        "midjourney_tid_01.jpeg",
        "midjourney_tid_02.png",
        "midjourney_tid_03.webp",
    ]


def test_document_delivery_handles_partial_failures(monkeypatch, bot_module):
    calls: list[dict[str, object]] = []

    async def _fake_send(bot, chat_id, data, filename, *, caption=None, reply_markup=None, req_id=None):
        if len(calls) == 0:
            raise RuntimeError("boom")
        calls.append({"filename": filename})
        return SimpleNamespace(message_id=len(calls))

    async def _wrapped(bot, chat_id, data, filename, *, caption=None, reply_markup=None, req_id=None):
        try:
            return await _fake_send(bot, chat_id, data, filename, caption=caption, reply_markup=reply_markup, req_id=req_id)
        finally:
            if len(calls) < 1:
                calls.append({"filename": filename, "failed": True})

    monkeypatch.setattr(bot_module, "send_image_as_document", _wrapped)

    delivered = asyncio.run(
        bot_module._deliver_mj_media(  # type: ignore[attr-defined]
            SimpleNamespace(),
            chat_id=5,
            user_id=6,
            caption="hi",
            items=_build_items(2),
            send_as_album=False,
            task_id="tid",
        )
    )

    assert delivered is True
    assert any(call.get("failed") for call in calls)


def test_document_delivery_returns_false_if_all_fail(monkeypatch, bot_module):
    async def _fail(*args, **kwargs):
        raise RuntimeError("nope")

    monkeypatch.setattr(bot_module, "send_image_as_document", _fail)

    delivered = asyncio.run(
        bot_module._deliver_mj_media(  # type: ignore[attr-defined]
            SimpleNamespace(),
            chat_id=7,
            user_id=8,
            caption="",
            items=_build_items(2),
            send_as_album=False,
            task_id="tid",
        )
    )

    assert delivered is False


def test_document_caption_trimmed(monkeypatch, bot_module):
    captured: list[str | None] = []

    async def _fake_send(bot, chat_id, data, filename, *, caption=None, reply_markup=None, req_id=None):
        captured.append(caption)
        return SimpleNamespace(message_id=len(captured))

    monkeypatch.setattr(bot_module, "send_image_as_document", _fake_send)

    long_caption = "A" * 1100
    asyncio.run(
        bot_module._deliver_mj_media(  # type: ignore[attr-defined]
            SimpleNamespace(),
            chat_id=9,
            user_id=10,
            caption=long_caption,
            items=_build_items(2),
            send_as_album=True,
            task_id="tid",
        )
    )

    assert captured
    assert isinstance(captured[0], str)
    assert len(captured[0]) == 1024
    assert captured[0].endswith("…")
    assert captured[1] is None
