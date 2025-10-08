from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("TELEGRAM_TOKEN", "test-token")
os.environ.setdefault("REDIS_URL", "memory://test")
os.environ.setdefault("KIE_API_KEY", "test-kie-key")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from handlers import video as video_module


class DummyBot:
    def __init__(self) -> None:
        self.sent_messages: list[SimpleNamespace] = []
        self.sent_videos: list[tuple[int, str]] = []
        self.sent_documents: list[tuple[int, str]] = []

    async def send_message(
        self, *, chat_id: int, text: str, reply_markup=None
    ) -> SimpleNamespace:  # type: ignore[override]
        self.sent_messages.append(
            SimpleNamespace(chat_id=chat_id, text=text, reply_markup=reply_markup)
        )
        return SimpleNamespace(message_id=len(self.sent_messages))

    async def send_video(self, *, chat_id: int, video: str) -> SimpleNamespace:  # type: ignore[override]
        self.sent_videos.append((chat_id, video))
        return SimpleNamespace(message_id=len(self.sent_videos))

    async def send_document(self, *, chat_id: int, document: str) -> SimpleNamespace:  # type: ignore[override]
        self.sent_documents.append((chat_id, document))
        return SimpleNamespace(message_id=len(self.sent_documents))


def _make_update(chat_id: int = 101, user_id: int = 202) -> SimpleNamespace:
    return SimpleNamespace(
        effective_chat=SimpleNamespace(id=chat_id),
        effective_user=SimpleNamespace(id=user_id),
        effective_message=None,
        callback_query=None,
    )


def _make_context(bot: DummyBot, state: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(bot=bot, user_data={"state": state or {}}, chat_data={})


def test_veo_animate_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    bot = DummyBot()
    ctx = _make_context(bot, {"last_image_url": "https://img.example/photo.jpg", "last_prompt": "smile"})
    update = _make_update()

    calls: list[tuple[str, str | None]] = []

    async def fake_start(image_url: str, prompt: str | None) -> str:
        calls.append((image_url, prompt))
        return "job-1"

    async def fake_wait(job_id: str) -> tuple[list[str], dict]:
        assert job_id == "job-1"
        return ["https://cdn.example/video.mp4"], {"status": "done"}

    async def fake_fetch(url: str) -> int:
        return 10 * 1024 * 1024

    monkeypatch.setattr(video_module, "_start_animation", fake_start)
    monkeypatch.setattr(video_module, "_wait_for_result", fake_wait)
    monkeypatch.setattr(video_module, "_fetch_content_length", fake_fetch)
    stored: list[tuple[int, str]] = []

    def fake_remember(user_id: int, job_id: str, *, ttl: int = 0) -> None:
        stored.append((user_id, job_id))

    monkeypatch.setattr(video_module, "remember_veo_anim_job", fake_remember)

    asyncio.run(video_module.veo_animate(update, ctx))

    assert calls == [("https://img.example/photo.jpg", "smile")]
    assert bot.sent_videos == [(101, "https://cdn.example/video.mp4")]
    assert bot.sent_documents == []
    assert bot.sent_messages == []
    assert stored == [(202, "job-1")]


def test_veo_animate_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    bot = DummyBot()
    ctx = _make_context(bot, {"last_image_url": "https://img.example/photo.jpg"})
    update = _make_update()

    async def fake_start(image_url: str, prompt: str | None) -> str:
        return "job-2"

    async def fake_wait(job_id: str) -> tuple[list[str], dict]:
        raise video_module.VeoAnimateTimeout("timeout")

    monkeypatch.setattr(video_module, "_start_animation", fake_start)
    monkeypatch.setattr(video_module, "_wait_for_result", fake_wait)
    async def fake_fetch(url: str) -> int:
        return 0

    monkeypatch.setattr(video_module, "_fetch_content_length", fake_fetch)
    monkeypatch.setattr(video_module, "remember_veo_anim_job", lambda *args, **kwargs: None)

    asyncio.run(video_module.veo_animate(update, ctx))

    assert bot.sent_videos == []
    assert bot.sent_documents == []
    assert len(bot.sent_messages) == 1
    message = bot.sent_messages[0]
    assert message.chat_id == 101
    assert (
        message.text
        == "Сервис сейчас отвечает дольше обычного. Нажмите «Повторить» или попробуйте позже."
    )
    assert message.reply_markup is not None
    button = message.reply_markup.inline_keyboard[0][0]
    assert button.text == "🔁 Повторить"


def test_veo_animate_bad_request(monkeypatch: pytest.MonkeyPatch) -> None:
    bot = DummyBot()
    ctx = _make_context(bot, {"last_image_url": "https://img.example/photo.jpg"})
    update = _make_update()

    async def fake_start(image_url: str, prompt: str | None) -> str:
        raise video_module.VeoAnimateBadRequest("bad")

    monkeypatch.setattr(video_module, "_start_animation", fake_start)
    async def fake_fetch(url: str) -> int:
        return 0

    monkeypatch.setattr(video_module, "_fetch_content_length", fake_fetch)
    monkeypatch.setattr(video_module, "remember_veo_anim_job", lambda *args, **kwargs: None)

    asyncio.run(video_module.veo_animate(update, ctx))

    assert len(bot.sent_messages) == 1
    message = bot.sent_messages[0]
    assert message.chat_id == 101
    assert (
        message.text
        == "Не получилось выполнить запрос. Похоже, в тексте есть запрещённый или "
        "негативный контент. Попробуйте переформулировать короче и нейтральнее."
    )
    assert message.reply_markup is None
    assert bot.sent_videos == []


def test_veo_animate_list_of_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    bot = DummyBot()
    ctx = _make_context(bot, {"last_image_url": "https://img.example/photo.jpg"})
    update = _make_update()

    async def fake_start(image_url: str, prompt: str | None) -> str:
        return "job-3"

    async def fake_wait(job_id: str) -> tuple[list[str], dict]:
        return ["https://cdn.example/a.mp4", "https://cdn.example/b.mp4"], {"status": "done"}

    monkeypatch.setattr(video_module, "_start_animation", fake_start)
    monkeypatch.setattr(video_module, "_wait_for_result", fake_wait)
    async def fake_fetch(url: str) -> None:
        return None

    monkeypatch.setattr(video_module, "_fetch_content_length", fake_fetch)
    monkeypatch.setattr(video_module, "remember_veo_anim_job", lambda *args, **kwargs: None)

    asyncio.run(video_module.veo_animate(update, ctx))

    assert bot.sent_videos == [(101, "https://cdn.example/a.mp4")]
    assert all("http" in msg for _, msg in bot.sent_videos)
    assert bot.sent_messages == []


def test_veo_animate_waits_for_photo(monkeypatch: pytest.MonkeyPatch) -> None:
    bot = DummyBot()
    ctx = _make_context(bot)
    update = _make_update()

    call_counter: list[str] = []

    async def fake_start(image_url: str, prompt: str | None) -> str:
        call_counter.append(image_url)
        return "job-4"

    async def fake_wait(job_id: str) -> tuple[list[str], dict]:
        return ["https://cdn.example/auto.mp4"], {"status": "done"}

    monkeypatch.setattr(video_module, "_start_animation", fake_start)
    monkeypatch.setattr(video_module, "_wait_for_result", fake_wait)
    async def fake_fetch(url: str) -> int:
        return 0

    monkeypatch.setattr(video_module, "_fetch_content_length", fake_fetch)
    monkeypatch.setattr(video_module, "remember_veo_anim_job", lambda *args, **kwargs: None)

    asyncio.run(video_module.veo_animate(update, ctx))
    assert len(bot.sent_messages) == 1
    message = bot.sent_messages[0]
    assert message.chat_id == 101
    assert (
        message.text
        == "Нужна фотография или корректный промт. Пришлите изображение и повторите."
    )
    assert message.reply_markup is None
    assert video_module._WAIT_FLAG in ctx.user_data["state"]

    asyncio.run(
        video_module.handle_veo_animate_photo(
            update, ctx, image_url="https://img.example/new.jpg"
        )
    )

    assert call_counter == ["https://img.example/new.jpg"]
    assert bot.sent_videos[-1] == (101, "https://cdn.example/auto.mp4")
