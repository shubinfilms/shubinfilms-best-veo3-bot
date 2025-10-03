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
    module = importlib.reload(module)
    module.mj_log.disabled = True
    return module


def _make_ctx():
    bot = SimpleNamespace(send_document=None, send_message=None)
    return SimpleNamespace(bot=bot)


def test_grid_delivery_sends_documents_and_menu(monkeypatch, bot_module):
    ctx = _make_ctx()
    monkeypatch.setattr(
        bot_module,
        "mj_log",
        SimpleNamespace(info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None),
    )
    doc_calls: list[dict[str, object]] = []
    menu_calls: list[dict[str, object]] = []
    saved_snapshots: list[tuple[str, list[str], str | None]] = []

    async def fake_send_document(chat_id, *, document, **kwargs):
        doc_calls.append({"chat_id": chat_id, "document": document, "kwargs": kwargs})

    async def fake_send_message(chat_id, text, *, reply_markup=None):
        menu_calls.append({"chat_id": chat_id, "text": text, "markup": reply_markup})

    monkeypatch.setattr(ctx.bot, "send_document", fake_send_document)
    monkeypatch.setattr(ctx.bot, "send_message", fake_send_message)
    monkeypatch.setattr(
        bot_module,
        "_download_mj_image_bytes",
        lambda url, index: (b"x" * (2048 + index), f"midjourney_{index:02d}.jpeg", "image/jpeg", url),
    )

    def fake_save(grid_id, urls, *, prompt=None):
        saved_snapshots.append((grid_id, list(urls), prompt))

    monkeypatch.setattr(bot_module, "_save_mj_grid_snapshot", fake_save)

    urls = [
        "https://cdn.example.com/a.jpeg",
        "https://cdn.example.com/b.png",
        "https://cdn.example.com/c",
    ]

    delivered = asyncio.run(
        bot_module._deliver_mj_grid_documents(
            ctx,
            chat_id=100,
            user_id=55,
            grid_id="grid123",
            urls=urls,
            prompt="prompt text",
        )
    )

    assert delivered is True
    assert len(doc_calls) == 3
    assert [call["document"].filename for call in doc_calls] == [
        "midjourney_01.jpeg",
        "midjourney_02.jpeg",
        "midjourney_03.jpeg",
    ]
    assert all(len(call["document"].input_file_content) > 1024 for call in doc_calls)
    assert menu_calls and menu_calls[0]["markup"].inline_keyboard[0][0].callback_data.startswith("mj.upscale.menu:")
    assert saved_snapshots == [("grid123", urls, "prompt text")]


def test_grid_delivery_returns_false_when_no_urls(monkeypatch, bot_module):
    ctx = _make_ctx()
    monkeypatch.setattr(
        bot_module,
        "mj_log",
        SimpleNamespace(info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None),
    )
    monkeypatch.setattr(ctx.bot, "send_document", lambda *args, **kwargs: None)
    monkeypatch.setattr(ctx.bot, "send_message", lambda *args, **kwargs: None)

    delivered = asyncio.run(
        bot_module._deliver_mj_grid_documents(
            ctx,
            chat_id=1,
            user_id=None,
            grid_id="grid",
            urls=[],
        )
    )

    assert delivered is False


def test_grid_delivery_handles_failed_documents(monkeypatch, bot_module):
    ctx = _make_ctx()
    monkeypatch.setattr(
        bot_module,
        "mj_log",
        SimpleNamespace(info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None),
    )
    attempts = {"count": 0}

    async def flaky_send(chat_id, *, document, **kwargs):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("boom")

    monkeypatch.setattr(ctx.bot, "send_document", flaky_send)

    async def noop_message(*args, **kwargs):
        return None

    monkeypatch.setattr(ctx.bot, "send_message", noop_message)
    monkeypatch.setattr(bot_module, "_save_mj_grid_snapshot", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        bot_module,
        "_download_mj_image_bytes",
        lambda url, index: (b"y" * 2048, f"midjourney_{index:02d}.png", "image/png", url),
    )

    delivered = asyncio.run(
        bot_module._deliver_mj_grid_documents(
            ctx,
            chat_id=10,
            user_id=3,
            grid_id="grid",
            urls=["https://cdn/x.png", "https://cdn/y.png"],
        )
    )

    assert delivered is True
    assert attempts["count"] == 2


def test_grid_delivery_returns_false_when_menu_fails(monkeypatch, bot_module):
    ctx = _make_ctx()

    async def fake_send_document(chat_id, *, document, **kwargs):
        return None

    async def failing_menu(*args, **kwargs):
        raise RuntimeError("fail")

    monkeypatch.setattr(ctx.bot, "send_document", fake_send_document)
    monkeypatch.setattr(ctx.bot, "send_message", failing_menu)
    monkeypatch.setattr(bot_module, "_save_mj_grid_snapshot", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        bot_module,
        "_download_mj_image_bytes",
        lambda url, index: (b"z" * 2048, f"midjourney_{index:02d}.png", "image/png", url),
    )

    delivered = asyncio.run(
        bot_module._deliver_mj_grid_documents(
            ctx,
            chat_id=1,
            user_id=2,
            grid_id="grid",
            urls=["https://cdn/a.png"],
        )
    )

    assert delivered is False
