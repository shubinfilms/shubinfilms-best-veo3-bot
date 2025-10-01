import asyncio
import os
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("LEDGER_BACKEND", "memory")
os.environ.setdefault("WELCOME_BONUS", "10")
os.environ.setdefault("WELCOME_BONUS_ENABLED", "true")

from tests.suno_test_utils import FakeBot, bot_module  # noqa: E402


class DummyRedis:
    def __init__(self) -> None:
        self.storage: dict[str, str] = {}

    def set(self, key: str, value: str, nx: bool = False, ex: int | None = None):
        if nx and key in self.storage:
            return False
        self.storage[key] = value
        return True

    def delete(self, key: str) -> None:
        self.storage.pop(key, None)

    class _Pipeline:
        def sadd(self, *_args, **_kwargs):
            return self

        def setnx(self, *_args, **_kwargs):
            return self

        def execute(self) -> None:
            return None

    def pipeline(self):
        return self._Pipeline()


def _make_context(bot: FakeBot | None = None) -> SimpleNamespace:
    return SimpleNamespace(bot=bot or FakeBot(), user_data={}, args=[])


def _make_update(user_id: int, chat_id: int, *, text: str = "/start") -> SimpleNamespace:
    message = SimpleNamespace(
        text=text,
        caption=None,
        chat_id=chat_id,
    )
    return SimpleNamespace(
        effective_user=SimpleNamespace(id=user_id),
        effective_chat=SimpleNamespace(id=chat_id),
        message=message,
        effective_message=message,
        callback_query=None,
    )


def test_bonus_once(monkeypatch):
    user_id = 4242001
    ctx = _make_context()
    bot = ctx.bot

    dummy_redis = DummyRedis()
    monkeypatch.setattr(bot_module, "redis_client", dummy_redis)
    monkeypatch.setattr(bot_module, "WELCOME_BONUS_ENABLED", True)
    monkeypatch.setattr(bot_module, "WELCOME_BONUS", 10)
    monkeypatch.setattr(bot_module, "WELCOME_BONUS_AMOUNT", 10)
    monkeypatch.setattr(bot_module, "_welcome_bonus_memory", {})

    granted_first = asyncio.run(bot_module.ensure_signup_bonus_once(ctx, user_id))
    granted_second = asyncio.run(bot_module.ensure_signup_bonus_once(ctx, user_id))

    assert granted_first is True
    assert granted_second is False
    assert len(bot.sent) == 1
    bonus_payload = bot.sent[0]
    assert bonus_payload["chat_id"] == user_id
    assert bonus_payload["text"] == "🎁 <b>Добро пожаловать!</b> Начислил <b>+10💎</b> на баланс."
    parse_mode = bonus_payload.get("parse_mode")
    parse_mode_value = getattr(parse_mode, "value", parse_mode)
    assert parse_mode_value == "HTML"
    assert bot_module.ledger_storage.get_balance(user_id) == 10


def test_menu_command(monkeypatch):
    ctx = _make_context()
    bot = ctx.bot

    async def fake_hub(*_args, **_kwargs):
        return None

    monkeypatch.setattr(bot_module, "show_emoji_hub_for_chat", fake_hub)

    update = _make_update(user_id=501, chat_id=777, text="/menu")
    asyncio.run(bot_module.on_menu(update, ctx))

    menu_messages = [
        payload
        for payload in bot.sent
        if isinstance(payload, dict)
        and str(payload.get("text", "")).startswith("👋 Добро пожаловать!")
    ]
    assert menu_messages, "main menu message should be sent"
    markup = menu_messages[-1]["reply_markup"]
    rows = markup.inline_keyboard
    assert [[btn.callback_data for btn in row] for row in rows] == [
        ["menu:video"],
        ["menu:image"],
        ["menu:music"],
        ["menu:buy"],
        ["menu:lang"],
        ["menu:help"],
        ["menu:faq"],
    ]


def test_start_flow(monkeypatch):
    ctx = _make_context()
    bot = ctx.bot

    dummy_redis = DummyRedis()
    monkeypatch.setattr(bot_module, "redis_client", dummy_redis)
    monkeypatch.setattr(bot_module, "WELCOME_BONUS", 10)
    monkeypatch.setattr(bot_module, "WELCOME_BONUS_AMOUNT", 10)
    monkeypatch.setattr(bot_module, "_welcome_bonus_memory", {})

    async def fake_hub(*_args, **_kwargs):
        return None

    monkeypatch.setattr(bot_module, "show_emoji_hub_for_chat", fake_hub)

    update = _make_update(user_id=9001, chat_id=42)
    asyncio.run(bot_module.on_start(update, ctx))

    bonus_messages = [
        payload
        for payload in bot.sent
        if isinstance(payload, dict)
        and payload.get("text", "").startswith("🎁 <b>Добро пожаловать!</b>")
    ]
    assert bonus_messages
    bonus_payload = bonus_messages[0]
    assert bonus_payload["chat_id"] == 9001
    assert bonus_payload["text"] == "🎁 <b>Добро пожаловать!</b> Начислил <b>+10💎</b> на баланс."
    parse_mode = bonus_payload.get("parse_mode")
    parse_mode_value = getattr(parse_mode, "value", parse_mode)
    assert parse_mode_value == "HTML"

    menu_messages = [
        payload
        for payload in bot.sent
        if isinstance(payload, dict)
        and str(payload.get("text", "")).startswith("👋 Добро пожаловать!")
    ]
    assert len(menu_messages) == 1

    asyncio.run(bot_module.on_start(update, ctx))

    bonus_messages_after = [
        payload
        for payload in bot.sent
        if isinstance(payload, dict)
        and payload.get("text", "").startswith("🎁 <b>Добро пожаловать!</b>")
    ]
    assert len(bonus_messages_after) == 1, "bonus should not be sent twice"
    menu_messages_after = [
        payload
        for payload in bot.sent
        if isinstance(payload, dict)
        and str(payload.get("text", "")).startswith("👋 Добро пожаловать!")
    ]
    assert len(menu_messages_after) == 2


def test_callbacks(monkeypatch):
    ctx = _make_context()

    calls: list[str] = []

    async def record(action_name: str):
        calls.append(action_name)

    for action in ("video", "image", "music", "prompt", "chat", "balance"):
        async def _handler(update, _ctx, name=action):
            await record(name)

        monkeypatch.setitem(bot_module.MAIN_ACTIONS, action, _handler)

    async def fake_menu(update, _ctx):
        calls.append("menu")

    monkeypatch.setattr(bot_module, "handle_menu", fake_menu)

    async def fake_render(update, _ctx, edit=False):
        calls.append(f"lang:{edit}")
        return None

    monkeypatch.setattr(bot_module, "render_main_menu", fake_render)

    def _make_callback(data: str, message_id: int = 500) -> SimpleNamespace:
        async def answer(*_args, **_kwargs):
            return None

        message = SimpleNamespace(chat_id=77, message_id=message_id)
        return SimpleNamespace(data=data, message=message, answer=answer)

    update_base = SimpleNamespace(
        effective_user=SimpleNamespace(id=321),
        effective_chat=SimpleNamespace(id=77),
    )

    for action in ("video", "image", "music", "prompt", "chat", "balance"):
        update = SimpleNamespace(**update_base.__dict__)
        update.callback_query = _make_callback(f"act:{action}")
        asyncio.run(bot_module.on_action(update, ctx))

    update_menu = SimpleNamespace(**update_base.__dict__)
    update_menu.callback_query = _make_callback("act:menu")
    asyncio.run(bot_module.on_action(update_menu, ctx))

    state_dict = bot_module.state(ctx)
    state_dict["last_ui_msg_id_menu"] = 900

    update_lang = SimpleNamespace(**update_base.__dict__)
    update_lang.callback_query = _make_callback("lang:en", message_id=900)
    asyncio.run(bot_module.on_action(update_lang, ctx))

    assert calls[:6] == ["video", "image", "music", "prompt", "chat", "balance"]
    assert "menu" in calls
    assert any(entry.startswith("lang:") for entry in calls)
    assert ctx.user_data.get("preferred_language") == "en"


def test_html_escape(monkeypatch):
    ctx = _make_context()
    bot = ctx.bot
    dummy_redis = DummyRedis()
    monkeypatch.setattr(bot_module, "redis_client", dummy_redis)
    monkeypatch.setattr(bot_module, "WELCOME_BONUS", 10)
    monkeypatch.setattr(bot_module, "WELCOME_BONUS_AMOUNT", 10)
    monkeypatch.setattr(bot_module, "_welcome_bonus_memory", {})

    asyncio.run(bot_module.ensure_signup_bonus_once(ctx, 701))

    assert bot.sent
    payload = bot.sent[0]
    assert payload["chat_id"] == 701
    assert payload["text"] == "🎁 <b>Добро пожаловать!</b> Начислил <b>+10💎</b> на баланс."
    parse_mode = payload.get("parse_mode")
    assert getattr(parse_mode, "value", parse_mode) == "HTML"
