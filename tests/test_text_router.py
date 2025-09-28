import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from telegram.error import BadRequest
from telegram.ext import ApplicationHandlerStop

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("SUNO_API_BASE", "https://example.com")
os.environ.setdefault("SUNO_API_TOKEN", "token")
os.environ.setdefault("SUNO_CALLBACK_URL", "https://callback.example")
os.environ.setdefault("SUNO_CALLBACK_SECRET", "secret")
os.environ.setdefault("TELEGRAM_TOKEN", "dummy-token")
os.environ.setdefault("KIE_API_KEY", "test-key")
os.environ.setdefault("KIE_BASE_URL", "https://example.com")
os.environ.setdefault("DATABASE_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("LEDGER_BACKEND", "memory")
os.environ.setdefault("LOG_JSON", "false")
os.environ.setdefault("LOG_LEVEL", "WARNING")

import bot as bot_module
from handlers import prompt_master_handle_text
from utils.input_state import (
    WaitInputState,
    WaitKind,
    clear_wait_state,
    get_wait_state,
    set_wait_state,
)
from utils.suno_state import load as load_suno_state
from utils.telegram_safe import safe_edit_message


class DummyMessage:
    def __init__(self, chat_id: int, text: str) -> None:
        self.chat_id = chat_id
        self.text = text
        self.replies: list[str] = []

    async def reply_text(self, text: str, **_: object) -> None:  # type: ignore[override]
        self.replies.append(text)


class NeutralBot:
    def __init__(self) -> None:
        self.edit_calls: list[dict[str, object]] = []
        self.reply_markup_calls: list[dict[str, object]] = []
        self.sent: list[dict[str, object]] = []

    async def edit_message_text(self, **kwargs):  # type: ignore[override]
        self.edit_calls.append(kwargs)
        return SimpleNamespace(message_id=kwargs.get("message_id"))

    async def edit_message_reply_markup(self, **kwargs):  # type: ignore[override]
        self.reply_markup_calls.append(kwargs)
        return SimpleNamespace(message_id=kwargs.get("message_id"))

    async def send_message(self, **kwargs):  # type: ignore[override]
        self.sent.append(kwargs)
        return SimpleNamespace(message_id=kwargs.get("message_id", 1000))


class NotModifiedBot(NeutralBot):
    async def edit_message_text(self, **kwargs):  # type: ignore[override]
        self.edit_calls.append(kwargs)
        raise BadRequest("Message is not modified")

    async def edit_message_reply_markup(self, **kwargs):  # type: ignore[override]
        self.reply_markup_calls.append(kwargs)
        raise BadRequest("Message is not modified")


def _run(coro):
    try:
        asyncio.run(coro)
    except ApplicationHandlerStop:
        pass


def test_router_updates_veo_prompt() -> None:
    ctx = SimpleNamespace(bot=None, user_data={})
    state_dict = bot_module.state(ctx)
    state_dict["last_ui_msg_id_veo"] = 123
    user_id = 101
    wait_state = WaitInputState(kind=WaitKind.VEO_PROMPT, card_msg_id=123, chat_id=777, meta={})
    set_wait_state(user_id, wait_state)

    calls: list[int] = []
    original_show = bot_module.show_veo_card

    async def fake_show(chat_id: int, ctx_param):
        calls.append(chat_id)
        state_dict["last_ui_msg_id_veo"] = 456

    bot_module.show_veo_card = fake_show  # type: ignore[assignment]

    message = DummyMessage(chat_id=777, text=" Test prompt ")
    update = SimpleNamespace(effective_message=message, effective_user=SimpleNamespace(id=user_id))

    try:
        _run(bot_module.handle_card_input(update, ctx))
    finally:
        bot_module.show_veo_card = original_show  # type: ignore[assignment]
        clear_wait_state(user_id)

    assert state_dict["last_prompt"] == "Test prompt"
    assert calls == [777]
    assert not get_wait_state(user_id)
    assert message.replies == ["✅ Принято"]


def test_router_updates_banana_prompt() -> None:
    ctx = SimpleNamespace(bot=None, user_data={})
    state_dict = bot_module.state(ctx)
    state_dict["last_ui_msg_id_banana"] = 42
    user_id = 909
    wait_state = WaitInputState(kind=WaitKind.BANANA_PROMPT, card_msg_id=42, chat_id=321, meta={})
    set_wait_state(user_id, wait_state)

    calls: list[int] = []
    original_show = bot_module.show_banana_card

    async def fake_show(chat_id: int, ctx_param):
        calls.append(chat_id)
        state_dict["last_ui_msg_id_banana"] = 84

    bot_module.show_banana_card = fake_show  # type: ignore[assignment]

    message = DummyMessage(chat_id=321, text="  Touch up portrait  ")
    update = SimpleNamespace(effective_message=message, effective_user=SimpleNamespace(id=user_id))

    try:
        _run(bot_module.handle_card_input(update, ctx))
    finally:
        bot_module.show_banana_card = original_show  # type: ignore[assignment]
        clear_wait_state(user_id)

    assert state_dict["last_prompt"] == "Touch up portrait"
    assert calls == [321]
    assert not get_wait_state(user_id)
    assert message.replies == ["✅ Принято"]


def test_router_updates_mj_prompt() -> None:
    ctx = SimpleNamespace(bot=None, user_data={})
    state_dict = bot_module.state(ctx)
    state_dict["last_ui_msg_id_mj"] = 999
    user_id = 202
    wait_state = WaitInputState(kind=WaitKind.MJ_PROMPT, card_msg_id=999, chat_id=888, meta={})
    set_wait_state(user_id, wait_state)

    calls: list[int] = []
    original_show = bot_module.show_mj_prompt_card

    async def fake_show(chat_id: int, ctx_param):
        calls.append(chat_id)
        state_dict["last_ui_msg_id_mj"] = 1001

    bot_module.show_mj_prompt_card = fake_show  # type: ignore[assignment]

    message = DummyMessage(chat_id=888, text="Midjourney idea")
    update = SimpleNamespace(effective_message=message, effective_user=SimpleNamespace(id=user_id))

    try:
        _run(bot_module.handle_card_input(update, ctx))
    finally:
        bot_module.show_mj_prompt_card = original_show  # type: ignore[assignment]
        clear_wait_state(user_id)

    assert state_dict["last_prompt"] == "Midjourney idea"
    assert calls == [888]
    assert not get_wait_state(user_id)
    assert message.replies == ["✅ Принято"]


def test_router_updates_suno_fields() -> None:
    ctx = SimpleNamespace(bot=None, user_data={})
    state_dict = bot_module.state(ctx)

    refreshed: list[int] = []
    original_refresh = bot_module.refresh_suno_card

    async def fake_refresh(ctx_param, chat_id: int, state_payload: dict[str, object], *, price: int):
        refreshed.append(chat_id)
        state_payload["last_ui_msg_id_suno"] = 321
        return 321

    bot_module.refresh_suno_card = fake_refresh  # type: ignore[assignment]

    try:
        user_id = 303
        # Title
        set_wait_state(
            user_id,
            WaitInputState(kind=WaitKind.SUNO_TITLE, card_msg_id=0, chat_id=444, meta={}),
        )
        message = DummyMessage(chat_id=444, text="  My Song  ")
        update = SimpleNamespace(effective_message=message, effective_user=SimpleNamespace(id=user_id))
        _run(bot_module.handle_card_input(update, ctx))
        suno_state = load_suno_state(ctx)
        assert suno_state.title == "My Song"
        assert message.replies == ["✅ Принято"]

        # Style
        set_wait_state(
            user_id,
            WaitInputState(kind=WaitKind.SUNO_STYLE, card_msg_id=0, chat_id=444, meta={}),
        )
        style_message = DummyMessage(chat_id=444, text="Epic cinematic score")
        update_style = SimpleNamespace(
            effective_message=style_message,
            effective_user=SimpleNamespace(id=user_id),
        )
        _run(bot_module.handle_card_input(update_style, ctx))
        suno_state = load_suno_state(ctx)
        assert suno_state.style == "Epic cinematic score"
        assert style_message.replies == ["✅ Принято"]

        # Lyrics
        set_wait_state(
            user_id,
            WaitInputState(kind=WaitKind.SUNO_LYRICS, card_msg_id=0, chat_id=444, meta={}),
        )
        lyrics_message = DummyMessage(chat_id=444, text="Line one\nLine two")
        update_lyrics = SimpleNamespace(
            effective_message=lyrics_message,
            effective_user=SimpleNamespace(id=user_id),
        )
        _run(bot_module.handle_card_input(update_lyrics, ctx))
        suno_state = load_suno_state(ctx)
        assert suno_state.lyrics == "Line one\nLine two"
        assert lyrics_message.replies == ["✅ Принято"]
    finally:
        bot_module.refresh_suno_card = original_refresh  # type: ignore[assignment]
        clear_wait_state(303)

    assert refreshed == [444, 444, 444]


def test_router_handles_repeated_text_without_error() -> None:
    bot = NotModifiedBot()
    ctx = SimpleNamespace(bot=bot, user_data={})
    state_dict = bot_module.state(ctx)
    state_dict["last_ui_msg_id_veo"] = 555
    state_dict["_last_text_veo"] = None
    state_dict["last_prompt"] = "Same"

    user_id = 404
    set_wait_state(
        user_id,
        WaitInputState(kind=WaitKind.VEO_PROMPT, card_msg_id=555, chat_id=999, meta={}),
    )

    message = DummyMessage(chat_id=999, text="Same")
    update = SimpleNamespace(effective_message=message, effective_user=SimpleNamespace(id=user_id))

    try:
        _run(bot_module.handle_card_input(update, ctx))
    finally:
        clear_wait_state(user_id)

    assert bot.edit_calls  # attempted edit
    assert not bot.reply_markup_calls
    assert message.replies == ["✅ Принято"]


def test_wait_state_blocks_prompt_master() -> None:
    ctx = SimpleNamespace(bot=None, user_data={})
    message = DummyMessage(chat_id=111, text="Prompt text")
    update = SimpleNamespace(effective_message=message, message=message, effective_user=SimpleNamespace(id=505))

    set_wait_state(505, WaitInputState(kind=WaitKind.VEO_PROMPT, card_msg_id=0, chat_id=111, meta={}))

    try:
        asyncio.run(prompt_master_handle_text(update, ctx))
    finally:
        clear_wait_state(505)

    assert "pm_state" not in ctx.user_data
    assert message.replies == []


def test_safe_edit_message_handles_not_modified() -> None:
    bot = NotModifiedBot()
    ctx = SimpleNamespace(bot=bot)

    markup = SimpleNamespace()
    result = asyncio.run(
        safe_edit_message(
            ctx,
            chat_id=1,
            message_id=10,
            new_text="Hello",
            reply_markup=markup,
        )
    )

    assert result is False
    assert bot.edit_calls
    assert not bot.reply_markup_calls
