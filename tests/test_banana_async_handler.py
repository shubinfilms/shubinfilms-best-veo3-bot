from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from handlers import banana
from utils.banana_state import BananaState, save


class FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    async def get(self, key: str):
        return self.store.get(key)

    async def set(self, key: str, value: str, ex: int | None = None):
        self.store[key] = value

    async def delete(self, key: str):
        self.store.pop(key, None)


class DummyBot:
    def __init__(self) -> None:
        self.messages: list[dict[str, object]] = []
        self.documents: list[dict[str, object]] = []

    async def send_message(self, chat_id, text, **kwargs):
        self.messages.append({"chat_id": chat_id, "text": text, "kwargs": kwargs})

    async def send_document(self, chat_id, document, **kwargs):
        entry = {
            "chat_id": chat_id,
            "document": document,
            "kwargs": kwargs,
        }
        message = SimpleNamespace(
            message_id=len(self.documents) + 100,
            document=SimpleNamespace(file_id=f"file-{len(self.documents)+1}"),
            photo=[],
        )
        self.documents.append(entry)
        return message


class FakeQuery:
    def __init__(self, user_id: int) -> None:
        self.from_user = SimpleNamespace(id=user_id)

    async def answer(self):
        return None


def test_banana_generation_success(monkeypatch):
    async def scenario():
        user_id = 42
        redis = FakeRedis()
        bot = DummyBot()
        state = BananaState(photos=["tg-file-1"], prompt="Clean the background")
        await save(redis, user_id, state)

        update = SimpleNamespace(
            callback_query=FakeQuery(user_id),
            effective_user=SimpleNamespace(id=user_id),
        )
        context = SimpleNamespace(bot=bot, redis=redis)

        async def fake_fetch(bot_obj, file_id: str):
            return b"image-bytes"

        async def fake_upload(data: bytes, *, filename: str):
            return f"https://cdn.example/{filename}"

        async def fake_balance(uid: int):
            return 100

        async def fake_request_json(method, path, **kwargs):
            if path == banana._BANANA_CREATE_PATH:
                return {"taskId": "banana-task-1"}
            return {"data": {"state": "success", "resultUrls": ["https://cdn.example/result.png"]}}

        monkeypatch.setattr(banana, "_fetch_file_bytes", fake_fetch)
        monkeypatch.setattr(banana, "_upload_image_bytes", fake_upload)
        monkeypatch.setattr(banana, "validate_image", lambda data: None)
        monkeypatch.setattr(banana, "get_user_balance_async", fake_balance)
        monkeypatch.setattr(banana._client, "request_json", fake_request_json)

        await banana.on_banana_start(update, context)
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert bot.messages, "Expected acknowledgement messages"
        assert bot.documents, "Result document should be sent"

        cache_key = banana._build_cache_key(user_id, state.prompt, state.photos)
        cached_raw = redis.store.get(cache_key)
        assert cached_raw, "Cached entry should be stored"
        cached = json.loads(cached_raw)
        assert cached.get("file_id")

    asyncio.run(scenario())


def test_banana_uses_cache(monkeypatch):
    async def scenario():
        user_id = 7
        redis = FakeRedis()
        bot = DummyBot()
        state = BananaState(photos=["file-a"], prompt="Retouch")
        await save(redis, user_id, state)

        cache_key = banana._build_cache_key(user_id, state.prompt, state.photos)
        await redis.set(cache_key, json.dumps({"file_id": "cached-file"}))

        update = SimpleNamespace(
            callback_query=FakeQuery(user_id),
            effective_user=SimpleNamespace(id=user_id),
        )
        context = SimpleNamespace(bot=bot, redis=redis)

        async def boom(*args, **kwargs):  # pragma: no cover - should not be called
            raise AssertionError("Backend should not be invoked when cache is hit")

        monkeypatch.setattr(banana, "_prepare_image_urls", boom)

        await banana.on_banana_start(update, context)

        assert bot.documents and bot.documents[0]["document"] == "cached-file"
        assert any("готов" in msg["text"].lower() for msg in bot.messages)

    asyncio.run(scenario())


def test_banana_backend_failure(monkeypatch):
    async def scenario():
        user_id = 99
        redis = FakeRedis()
        bot = DummyBot()
        state = BananaState(photos=["f1"], prompt="Touch up")
        await save(redis, user_id, state)

        update = SimpleNamespace(
            callback_query=FakeQuery(user_id),
            effective_user=SimpleNamespace(id=user_id),
        )
        context = SimpleNamespace(bot=bot, redis=redis)

        async def fake_prepare(bot_obj, files):
            return ["https://cdn.example/img.png"]

        async def fake_submit(prompt, urls):
            raise banana.BananaBackendError("service down")

        errors: list[dict[str, object]] = []

        async def capture_error(ctx, kind, *, details=None, retry_cb=None):
            errors.append({"kind": kind, "details": details, "retry": retry_cb})

        monkeypatch.setattr(banana, "_prepare_image_urls", fake_prepare)
        monkeypatch.setattr(banana, "_submit_job", fake_submit)
        monkeypatch.setattr(banana, "get_user_balance_async", lambda uid: 0)
        monkeypatch.setattr(banana, "send_user_error", capture_error)

        await banana.on_banana_start(update, context)
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert errors
        assert errors[0]["kind"] == "backend_fail"

    asyncio.run(scenario())

