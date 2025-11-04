import asyncio
from types import SimpleNamespace

from handlers import midjourney


class DummyBot:
    def __init__(self) -> None:
        self.messages: list[dict[str, object]] = []
        self.photos: list[dict[str, object]] = []

    async def send_message(self, chat_id, text, **kwargs):
        self.messages.append({"chat_id": chat_id, "text": text, "kwargs": kwargs})

    async def send_photo(self, chat_id, photo, **kwargs):
        self.photos.append({"chat_id": chat_id, "photo": photo, "kwargs": kwargs})


class DummyMessage:
    def __init__(self, chat_id: int, text: str) -> None:
        self.chat = SimpleNamespace(id=chat_id)
        self.chat_id = chat_id
        self.text = text
        self.reply_text_calls: list[str] = []

    async def reply_text(self, text: str):
        self.reply_text_calls.append(text)


def test_midjourney_prompt_success(monkeypatch):
    async def scenario():
        bot = DummyBot()
        message = DummyMessage(chat_id=42, text="A futuristic city skyline")
        update = SimpleNamespace(
            effective_message=message,
            effective_chat=message.chat,
            effective_user=SimpleNamespace(id=777),
        )
        context = SimpleNamespace(bot=bot, chat_data={})

        async def fake_balance(user_id):
            return 900

        async def fake_request_json(method, path, **kwargs):
            if "generate" in path:
                return {"taskId": "mj-123"}
            return {"flag": 1, "resultUrls": ["https://cdn.example/img.png"]}

        monkeypatch.setattr(midjourney, "get_user_balance_async", fake_balance)
        monkeypatch.setattr(midjourney._client, "request_json", fake_request_json)

        await midjourney.handle_midjourney_prompt(update, context, aspect="16:9")
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert message.reply_text_calls
        assert any("Начинаю генерацию" in text for text in message.reply_text_calls)
        assert bot.photos and bot.photos[0]["photo"] == "https://cdn.example/img.png"
        assert context.chat_data["midjourney_progress"]["success"] is True

    asyncio.run(scenario())


def test_midjourney_bad_request(monkeypatch):
    async def scenario():
        bot = DummyBot()
        message = DummyMessage(chat_id=11, text="")
        update = SimpleNamespace(
            effective_message=message,
            effective_chat=message.chat,
            effective_user=SimpleNamespace(id=7),
        )
        context = SimpleNamespace(bot=bot, chat_data={})

        recorded: list[dict] = []

        async def capture_error(ctx, kind, *, details=None, retry_cb=None):
            recorded.append({"kind": kind, "details": details, "retry": retry_cb})

        monkeypatch.setattr(midjourney, "send_user_error", capture_error)

        await midjourney.handle_midjourney_prompt(update, context)

        assert recorded
        assert recorded[0]["kind"] == "invalid_input"

    asyncio.run(scenario())


def test_midjourney_backend_error(monkeypatch):
    async def scenario():
        bot = DummyBot()
        message = DummyMessage(chat_id=55, text="Space whale swimming")
        update = SimpleNamespace(
            effective_message=message,
            effective_chat=message.chat,
            effective_user=SimpleNamespace(id=8),
        )
        context = SimpleNamespace(bot=bot, chat_data={})

        async def fake_balance(user_id):
            return 100

        async def fake_request_json(method, path, **kwargs):
            raise midjourney.KieAPIHTTPError(status=500, payload={"message": "server error"})

        errors: list[dict[str, object]] = []

        async def capture_error(ctx, kind, *, details=None, retry_cb=None):
            errors.append({"kind": kind, "details": details})

        monkeypatch.setattr(midjourney, "get_user_balance_async", fake_balance)
        monkeypatch.setattr(midjourney._client, "request_json", fake_request_json)
        monkeypatch.setattr(midjourney, "send_user_error", capture_error)

        await midjourney.handle_midjourney_prompt(update, context)
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert errors
        assert errors[0]["kind"] == "backend_fail"

    asyncio.run(scenario())
