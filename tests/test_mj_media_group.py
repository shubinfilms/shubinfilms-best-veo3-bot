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


def _build_items(count: int) -> list[tuple[bytes, str]]:
    return [(f"data-{i}".encode(), f"{i:02d}.png") for i in range(count)]


def test_album_sends_four_photos(bot_module):
    class _Bot:
        def __init__(self):
            self.media_calls = []
            self.photo_calls = []

        async def send_media_group(self, **kwargs):
            self.media_calls.append(kwargs)
            return [SimpleNamespace(message_id=i) for i in range(len(kwargs["media"]))]

        async def send_photo(self, **kwargs):
            self.photo_calls.append(kwargs)
            return SimpleNamespace(message_id=len(self.photo_calls))

    bot = _Bot()
    items = _build_items(4)
    delivered = asyncio.run(
        bot_module._deliver_mj_media(  # type: ignore[attr-defined]
            bot,
            chat_id=1,
            user_id=2,
            caption="Caption",
            items=items,
            send_as_album=True,
            task_id="album-1",
        )
    )

    assert delivered is True
    assert len(bot.media_calls) == 1
    payload = bot.media_calls[0]
    assert len(payload["media"]) == 4
    assert payload["media"][0].caption == "Caption"
    assert all(item.caption is None for item in payload["media"][1:])
    assert not bot.photo_calls


def test_single_photo_uses_send_photo(bot_module):
    class _Bot:
        def __init__(self):
            self.media_calls = []
            self.photo_calls = []

        async def send_media_group(self, **kwargs):
            self.media_calls.append(kwargs)
            return []

        async def send_photo(self, **kwargs):
            self.photo_calls.append(kwargs)
            return SimpleNamespace(message_id=1)

    bot = _Bot()
    delivered = asyncio.run(
        bot_module._deliver_mj_media(  # type: ignore[attr-defined]
            bot,
            chat_id=5,
            user_id=6,
            caption="Only",
            items=_build_items(1),
            send_as_album=True,
            task_id="single-1",
        )
    )

    assert delivered is True
    assert not bot.media_calls
    assert len(bot.photo_calls) == 1
    assert bot.photo_calls[0]["caption"] == "Only"


def test_more_than_ten_chunked(bot_module):
    class _Bot:
        def __init__(self):
            self.media_calls = []
            self.photo_calls = []

        async def send_media_group(self, **kwargs):
            self.media_calls.append(kwargs)
            return [SimpleNamespace(message_id=i) for i in range(len(kwargs["media"]))]

        async def send_photo(self, **kwargs):
            self.photo_calls.append(kwargs)
            return SimpleNamespace(message_id=len(self.photo_calls))

    bot = _Bot()
    delivered = asyncio.run(
        bot_module._deliver_mj_media(  # type: ignore[attr-defined]
            bot,
            chat_id=9,
            user_id=10,
            caption="Chunk",
            items=_build_items(13),
            send_as_album=True,
            task_id="chunk-1",
        )
    )

    assert delivered is True
    assert len(bot.media_calls) == 2
    chunk_lengths = [len(call["media"]) for call in bot.media_calls]
    assert chunk_lengths == [10, 3]
    assert not bot.photo_calls


def test_album_error_fallbacks_to_single(bot_module, monkeypatch):
    class _Bot:
        def __init__(self):
            self.media_calls = []
            self.photo_calls = []

        async def send_media_group(self, **kwargs):
            self.media_calls.append(kwargs)
            raise RuntimeError("fail")

        async def send_photo(self, **kwargs):
            self.photo_calls.append(kwargs)
            return SimpleNamespace(message_id=len(self.photo_calls))

    records = []

    class _MJLog:
        def info(self, msg, *, extra=None):  # type: ignore[override]
            records.append(("info", msg, extra))

        def warning(self, msg, *, extra=None):  # type: ignore[override]
            records.append(("warning", msg, extra))

    monkeypatch.setattr(bot_module, "mj_log", _MJLog())

    bot = _Bot()
    delivered = asyncio.run(
        bot_module._deliver_mj_media(  # type: ignore[attr-defined]
            bot,
            chat_id=11,
            user_id=12,
            caption="Fallback",
            items=_build_items(4),
            send_as_album=True,
            task_id="fail-1",
        )
    )

    assert delivered is True
    assert len(bot.media_calls) == 1
    assert len(bot.photo_calls) == 4
    assert any(name == "warning" and msg == "mj.album.fallback_single" for name, msg, _ in records)


def test_caption_is_trimmed(bot_module):
    class _Bot:
        def __init__(self):
            self.media_calls = []
            self.photo_calls = []

        async def send_media_group(self, **kwargs):
            self.media_calls.append(kwargs)
            return [SimpleNamespace(message_id=i) for i in range(len(kwargs["media"]))]

        async def send_photo(self, **kwargs):
            self.photo_calls.append(kwargs)
            return SimpleNamespace(message_id=len(self.photo_calls))

    bot = _Bot()
    long_caption = "A" * 1100
    asyncio.run(
        bot_module._deliver_mj_media(  # type: ignore[attr-defined]
            bot,
            chat_id=21,
            user_id=22,
            caption=long_caption,
            items=_build_items(2),
            send_as_album=True,
            task_id="trim-1",
        )
    )

    assert len(bot.media_calls) == 1
    caption = bot.media_calls[0]["media"][0].caption
    assert isinstance(caption, str)
    assert len(caption) == 1024
    assert caption.endswith("…")
    assert all(item.caption is None for item in bot.media_calls[0]["media"][1:])


def test_follow_up_message_after_album(monkeypatch, bot_module):
    class _Bot:
        def __init__(self):
            self.messages = []

        async def send_message(self, chat_id, text, reply_markup=None, **kwargs):
            self.messages.append({
                "chat_id": chat_id,
                "text": text,
                "reply_markup": reply_markup,
                "kwargs": kwargs,
            })
            return SimpleNamespace(message_id=len(self.messages))

    bot = _Bot()

    def _fake_status(task_id):
        return True, 1, {}

    monkeypatch.setattr(bot_module, "mj_status", _fake_status)
    monkeypatch.setattr(bot_module, "_extract_mj_image_urls", lambda payload: ["u1", "u2"])
    monkeypatch.setattr(
        bot_module,
        "_download_mj_image_bytes",
        lambda url, index: (b"data", f"img{index}.png"),
    )

    deliver_calls = []

    async def _fake_deliver(*args, **kwargs):
        deliver_calls.append(kwargs)
        return True

    monkeypatch.setattr(bot_module, "_deliver_mj_media", _fake_deliver)

    ctx = SimpleNamespace(bot=bot, user_data={"state": {"mj_locale": "ru"}})

    asyncio.run(
        bot_module.poll_mj_and_send_photos(  # type: ignore[attr-defined]
            chat_id=33,
            task_id="tid-1",
            ctx=ctx,
            prompt="prompt",
            aspect="16:9",
            user_id=44,
            price=5,
        )
    )

    assert deliver_calls
    assert bot.messages
    last_message = bot.messages[-1]
    assert last_message["text"] == "Галерея сгенерирована."
    markup = last_message["reply_markup"]
    assert markup is not None
    buttons = [[button.text for button in row] for row in markup.inline_keyboard]
    assert buttons == [["Повторить"], ["Назад в меню"]]
