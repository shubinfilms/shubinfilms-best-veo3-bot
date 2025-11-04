from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from handlers.banana_async_handler import BananaAsyncHandler, _build_cache_key
from services.kie_api_async import KieAPITimeoutError
from utils.banana_state import BananaState, save

_ACK = "🟡 Processing your Banana edit... please wait."
_FAILURE = "⚠️ Something went wrong on Banana servers. Please try again later."


class FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.published: list[tuple[str, str]] = []

    async def get(self, key: str) -> str | None:
        return self.store.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.store[key] = value

    def publish(self, channel: str, payload: str) -> None:  # pragma: no cover - debug only
        self.published.append((channel, payload))


class DummyBot:
    def __init__(self) -> None:
        self.messages: list[dict[str, object]] = []
        self.media_edits: list[dict[str, object]] = []
        self.text_edits: list[dict[str, object]] = []

    async def send_message(self, chat_id, text, **kwargs):
        self.messages.append({"chat_id": chat_id, "text": text, "kwargs": kwargs})
        return SimpleNamespace(message_id=len(self.messages))

    async def edit_message_media(self, chat_id, message_id, media, reply_markup=None):
        entry = {
            "chat_id": chat_id,
            "message_id": message_id,
            "media": media,
            "reply_markup": reply_markup,
        }
        self.media_edits.append(entry)
        return SimpleNamespace(
            message_id=message_id,
            document=SimpleNamespace(file_id="banana-file"),
            photo=[],
        )

    async def edit_message_text(self, chat_id, message_id, text, **kwargs):
        entry = {
            "chat_id": chat_id,
            "message_id": message_id,
            "text": text,
            "kwargs": kwargs,
        }
        self.text_edits.append(entry)
        return SimpleNamespace(message_id=message_id, text=text)


class SuccessfulClient:
    def __init__(self) -> None:
        self.create_calls: list[dict[str, object]] = []

    async def request_json(self, method, path, json_payload=None, params=None, headers=None):
        if path == "/api/v1/jobs/createTask":
            self.create_calls.append({"method": method, "payload": json_payload})
            return {"taskId": "banana-task"}
        raise AssertionError(f"Unexpected path {path}")

    async def poll_job(self, path, *, task_id=None, params=None, interval=0.0, timeout=0.0):
        yield {
            "data": {
                "state": "success",
                "resultUrls": ["https://cdn.example/result.png"],
            }
        }


class TimeoutClient(SuccessfulClient):
    async def poll_job(self, path, *, task_id=None, params=None, interval=0.0, timeout=0.0):
        if False:  # pragma: no cover - satisfy async generator contract
            yield {}
        raise KieAPITimeoutError("poll timeout")


def test_successful_async_generation_flow(monkeypatch):
    async def scenario():
        redis = FakeRedis()
        bot = DummyBot()
        handler = BananaAsyncHandler(redis=redis, client=SuccessfulClient())

        async def fake_prepare(self, bot_obj, photos):
            return ["https://cdn.example/upload.png"]

        monkeypatch.setattr(handler, "_prepare_image_urls", fake_prepare.__get__(handler, BananaAsyncHandler))

        user_id = 42
        chat_id = 100
        state = BananaState(photos=["photo-1"], prompt="Clean background")
        await save(redis, user_id, state)

        update = SimpleNamespace(
            effective_chat=SimpleNamespace(id=chat_id),
            effective_user=SimpleNamespace(id=user_id),
        )
        context = SimpleNamespace(bot=bot, redis=redis)

        await handler.run(update, context)

        assert bot.messages and bot.messages[0]["text"] == _ACK
        assert bot.media_edits, "expected result edit"
        media = bot.media_edits[0]["media"]
        assert getattr(media, "media", None) == "https://cdn.example/result.png"

        cache_key = _build_cache_key(user_id, state.prompt or "", state.photos)
        cached_raw = redis.store.get(cache_key)
        assert cached_raw, "result should be cached"
        cached = json.loads(cached_raw)
        assert cached["file_id"] == "banana-file"

    asyncio.run(scenario())


def test_cache_hit_reuses_file(monkeypatch):
    async def scenario():
        redis = FakeRedis()
        bot = DummyBot()
        client = SuccessfulClient()
        handler = BananaAsyncHandler(redis=redis, client=client)

        user_id = 7
        chat_id = 55
        prompt = "Retouch"
        photos = ["img-1", "img-2"]
        state = BananaState(photos=list(photos), prompt=prompt)
        await save(redis, user_id, state)

        cache_key = _build_cache_key(user_id, prompt, photos)
        await redis.set(
            cache_key,
            json.dumps({"file_id": "cached-file", "caption": "✅ Banana edit ready!"}),
        )

        update = SimpleNamespace(
            effective_chat=SimpleNamespace(id=chat_id),
            effective_user=SimpleNamespace(id=user_id),
        )
        context = SimpleNamespace(bot=bot, redis=redis)

        await handler.run(update, context)

        assert client.create_calls == [], "backend should not be invoked on cache hit"
        assert bot.media_edits and bot.media_edits[0]["media"].media == "cached-file"
        assert bot.messages and bot.messages[0]["text"] == _ACK

    asyncio.run(scenario())


def test_backend_failure(monkeypatch):
    async def scenario():
        redis = FakeRedis()
        bot = DummyBot()
        handler = BananaAsyncHandler(redis=redis, client=TimeoutClient())

        async def fake_prepare(self, bot_obj, photos):
            return ["https://cdn.example/upload.png"]

        monkeypatch.setattr(handler, "_prepare_image_urls", fake_prepare.__get__(handler, BananaAsyncHandler))

        user_id = 99
        chat_id = 77
        state = BananaState(photos=["photo"], prompt="Touch up")
        await save(redis, user_id, state)

        update = SimpleNamespace(
            effective_chat=SimpleNamespace(id=chat_id),
            effective_user=SimpleNamespace(id=user_id),
        )
        context = SimpleNamespace(bot=bot, redis=redis)

        await handler.run(update, context)

        assert bot.messages and bot.messages[0]["text"] == _ACK
        assert bot.text_edits and bot.text_edits[0]["text"] == _FAILURE
        assert not bot.media_edits, "no media edits expected on failure"

    asyncio.run(scenario())
