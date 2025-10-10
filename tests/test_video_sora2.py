import asyncio
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.input_state import (  # noqa: E402
    WaitInputState,
    WaitKind,
    clear_wait_state,
    get_wait_state,
    set_wait_state,
)
from telegram.ext import ApplicationHandlerStop  # noqa: E402


@pytest.fixture
def bot_module(monkeypatch):
    monkeypatch.setenv("TELEGRAM_TOKEN", "dummy-token")
    monkeypatch.setenv("SUNO_API_BASE", "https://example.com")
    monkeypatch.setenv("SUNO_API_TOKEN", "token")
    monkeypatch.setenv("LEDGER_BACKEND", "memory")
    monkeypatch.setenv("DATABASE_URL", "postgres://test")
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://bot.example")
    monkeypatch.setenv("SORA2_ENABLED", "true")
    monkeypatch.setenv("KIE_API_KEY", "kie-key")
    monkeypatch.setenv("REDIS_URL", "redis://localhost/0")
    settings_module = importlib.import_module("settings")
    importlib.reload(settings_module)
    module = importlib.import_module("bot")
    return importlib.reload(module)


class DummyAsyncLock:
    async def __aenter__(self):
        return None

    async def __aexit__(self, exc_type, exc, tb):
        return False


class DummyBot:
    def __init__(self):
        self.sent_messages = []
        self.sent_videos = []

    async def send_message(self, chat_id, text, **kwargs):
        self.sent_messages.append({"chat_id": chat_id, "text": text, "kwargs": kwargs})

    async def send_video(self, chat_id, video, caption=None, **kwargs):
        self.sent_videos.append(
            {"chat_id": chat_id, "video": video, "caption": caption, "kwargs": kwargs}
        )


class DummyMessage:
    def __init__(self, chat_id: int, text: str = "Scene") -> None:
        self.chat = SimpleNamespace(id=chat_id)
        self.chat_id = chat_id
        self.message_id = 999
        self.text = text
        self.reply_calls: list[str] = []
        self.from_user = SimpleNamespace(id=777)

    async def reply_text(self, text: str):
        self.reply_calls.append(text)
        return SimpleNamespace(message_id=1000)


def _dummy_query(data: str, message: DummyMessage, user_id: int = 42):
    async def answer(*args, **kwargs):
        return None

    return SimpleNamespace(data=data, message=message, from_user=SimpleNamespace(id=user_id), answer=answer)


def _dummy_context(bot=None):
    return SimpleNamespace(bot=bot or DummyBot(), user_data={}, application=None)


def _install_lock_stub(monkeypatch, bot_module):
    monkeypatch.setattr(bot_module, "with_menu_lock", lambda *args, **kwargs: DummyAsyncLock())


def _install_safe_edit_stub(monkeypatch, bot_module, store):
    async def fake_safe_edit(ctx, **kwargs):
        store.append(kwargs)
        return kwargs.get("fallback_message_id", 123)

    monkeypatch.setattr(bot_module, "safe_edit_or_send_menu", fake_safe_edit)


def _prepare_wait_state(user_id: int, chat_id: int) -> None:
    wait_state = WaitInputState(
        kind=WaitKind.SORA2_PROMPT,
        card_msg_id=1,
        chat_id=chat_id,
        meta={"mode": "sora2_simple", "suppress_ack": True},
    )
    set_wait_state(user_id, wait_state)


def _make_update(query):
    return SimpleNamespace(callback_query=query, effective_user=query.from_user, effective_chat=query.message.chat)


def test_video_sora2_intro_card(monkeypatch, bot_module):
    ctx = _dummy_context()
    message = DummyMessage(chat_id=555)
    query = _dummy_query("video:type:sora2", message)
    update = _make_update(query)

    _install_lock_stub(monkeypatch, bot_module)
    captured = []
    _install_safe_edit_stub(monkeypatch, bot_module, captured)

    asyncio.run(bot_module.video_menu_callback(update, ctx))

    assert captured, "menu edit was not triggered"
    payload = captured[0]
    assert "🎬 *Sora2 — генерация видео по тексту*" in payload["text"]
    markup = payload["reply_markup"]
    buttons = markup.inline_keyboard
    assert buttons[0][0].callback_data == "sora2:start"
    assert buttons[1][0].callback_data == "video:menu"


def test_video_sora2_start_sets_wait_state(monkeypatch, bot_module):
    ctx = _dummy_context()
    message = DummyMessage(chat_id=777)
    query = _dummy_query("sora2:start", message, user_id=101)
    update = _make_update(query)

    _install_lock_stub(monkeypatch, bot_module)
    captured = []
    _install_safe_edit_stub(monkeypatch, bot_module, captured)

    asyncio.run(bot_module.video_menu_callback(update, ctx))

    wait_state = get_wait_state(101)
    assert wait_state is not None
    assert wait_state.kind == WaitKind.SORA2_PROMPT
    assert wait_state.meta.get("mode") == "sora2_simple"
    assert wait_state.meta.get("suppress_ack") is True
    assert "Введите текст" in captured[0]["text"]

    clear_wait_state(101)


def test_sora2_prompt_without_funds(monkeypatch, bot_module):
    user_id = 202
    chat_id = 808
    _prepare_wait_state(user_id, chat_id)

    async def failing_charge(*args, **kwargs):
        raise bot_module.billing.NotEnoughFunds()

    monkeypatch.setattr(bot_module.billing, "charge", failing_charge)
    monkeypatch.setattr(bot_module.billing, "refund", lambda *args, **kwargs: asyncio.sleep(0))

    bot = DummyBot()
    ctx = _dummy_context(bot)
    message = DummyMessage(chat_id=chat_id)
    message.from_user = SimpleNamespace(id=user_id)
    update = SimpleNamespace(effective_message=message, effective_user=message.from_user)

    with pytest.raises(ApplicationHandlerStop):
        asyncio.run(bot_module.handle_card_input(update, ctx))

    assert bot.sent_messages, "user should be notified about insufficient funds"
    assert "Недостаточно токенов" in bot.sent_messages[0]["text"]
    assert get_wait_state(user_id) is None


def test_sora2_prompt_success_flow(monkeypatch, bot_module):
    user_id = 303
    chat_id = 909
    _prepare_wait_state(user_id, chat_id)

    async def ok_charge(*args, **kwargs):
        return 1000

    async def no_refund(*args, **kwargs):
        return 1000

    async def fake_create(ctx, prompt, **kwargs):
        return "task-1"

    async def fake_poll(ctx, task_id, **kwargs):
        return {"resultJson": {"resultUrls": ["https://example.com/video.mp4"]}}

    monkeypatch.setattr(bot_module.billing, "charge", ok_charge)
    monkeypatch.setattr(bot_module.billing, "refund", no_refund)
    monkeypatch.setattr(bot_module, "kie_create_sora2_task", fake_create)
    monkeypatch.setattr(bot_module, "kie_poll_sora2", fake_poll)

    bot = DummyBot()
    ctx = _dummy_context(bot)
    message = DummyMessage(chat_id=chat_id)
    message.from_user = SimpleNamespace(id=user_id)
    update = SimpleNamespace(effective_message=message, effective_user=message.from_user)

    with pytest.raises(ApplicationHandlerStop):
        asyncio.run(bot_module.handle_card_input(update, ctx))

    assert bot.sent_videos, "video must be sent on success"
    assert bot.sent_videos[0]["video"].startswith("https://example.com")
    assert get_wait_state(user_id) is None


def test_sora2_prompt_failure_refunds(monkeypatch, bot_module):
    user_id = 404
    chat_id = 111
    _prepare_wait_state(user_id, chat_id)

    async def ok_charge(*args, **kwargs):
        return 900

    refunds = []

    async def record_refund(user, amount, **kwargs):
        refunds.append((user, amount))
        return 1000

    async def fake_create(ctx, prompt, **kwargs):
        return "task-err"

    async def fake_poll(ctx, task_id, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(bot_module.billing, "charge", ok_charge)
    monkeypatch.setattr(bot_module.billing, "refund", record_refund)
    monkeypatch.setattr(bot_module, "kie_create_sora2_task", fake_create)
    monkeypatch.setattr(bot_module, "kie_poll_sora2", fake_poll)

    bot = DummyBot()
    ctx = _dummy_context(bot)
    message = DummyMessage(chat_id=chat_id)
    message.from_user = SimpleNamespace(id=user_id)
    update = SimpleNamespace(effective_message=message, effective_user=message.from_user)

    with pytest.raises(ApplicationHandlerStop):
        asyncio.run(bot_module.handle_card_input(update, ctx))

    assert refunds and refunds[0][0] == user_id
    assert bot.sent_messages
    assert "Токены возвращены" in bot.sent_messages[0]["text"]
    assert get_wait_state(user_id) is None


def test_sora2_prompt_requires_text(monkeypatch, bot_module):
    user_id = 505
    chat_id = 222
    _prepare_wait_state(user_id, chat_id)

    bot = DummyBot()
    ctx = _dummy_context(bot)
    message = DummyMessage(chat_id=chat_id, text="   ")
    message.from_user = SimpleNamespace(id=user_id)
    update = SimpleNamespace(effective_message=message, effective_user=message.from_user)

    with pytest.raises(ApplicationHandlerStop):
        asyncio.run(bot_module.handle_card_input(update, ctx))

    assert message.reply_calls and "Введите текст" in message.reply_calls[0]
    clear_wait_state(user_id)
