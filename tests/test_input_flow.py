from __future__ import annotations

import asyncio
from types import SimpleNamespace

from telegram.error import BadRequest

import os
from pathlib import Path
import sys

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

from telegram_utils import safe_edit
from utils.input_state import WaitInputState, WaitKind, clear_wait_state, set_wait_state, get_wait_state

import bot as bot_module


class DummyMessage:
    def __init__(self, chat_id: int, text: str) -> None:
        self.chat_id = chat_id
        self.text = text
        self.replies: list[str] = []

    async def reply_text(self, text: str, **kwargs) -> None:  # type: ignore[override]
        self.replies.append(text)


class FakeBot:
    def __init__(self) -> None:
        self.sent: list[dict[str, object]] = []
        self.edits: list[dict[str, object]] = []
        self.deleted: list[tuple[int, int]] = []
        self._errors: list[Exception] = []
        self._next_message_id = 1000

    def queue_edit_error(self, exc: Exception) -> None:
        self._errors.append(exc)

    async def edit_message_text(self, **kwargs):  # type: ignore[override]
        if self._errors:
            raise self._errors.pop(0)
        self.edits.append(kwargs)
        return SimpleNamespace(message_id=kwargs.get("message_id"))

    async def send_message(self, **kwargs):  # type: ignore[override]
        self.sent.append(kwargs)
        self._next_message_id += 1
        return SimpleNamespace(message_id=self._next_message_id)

    async def delete_message(self, chat_id: int, message_id: int):  # type: ignore[override]
        self.deleted.append((chat_id, message_id))


async def _run_apply(
    wait_state: WaitInputState,
    message: DummyMessage,
    ctx: SimpleNamespace,
    user_id: int,
) -> tuple[bool, object]:
    return await bot_module._apply_wait_state_input(ctx, message, wait_state, user_id=user_id)


def test_wait_state_updates_veo_prompt() -> None:
    ctx = SimpleNamespace(bot=None, user_data={})
    state_dict = bot_module.state(ctx)
    state_dict["last_ui_msg_id_veo"] = 321
    wait_state = WaitInputState(kind=WaitKind.VEO_PROMPT, card_msg_id=321, chat_id=777, meta={})
    user_id = 555
    set_wait_state(user_id, wait_state)

    calls: list[tuple[int, object]] = []

    original_show = bot_module.show_veo_card

    async def fake_show(chat_id: int, ctx_param, *, force_new: bool = False):
        calls.append((chat_id, ctx_param))
        state_dict["last_ui_msg_id_veo"] = 654

    bot_module.show_veo_card = fake_show  # type: ignore[assignment]

    message = DummyMessage(chat_id=777, text=" Test prompt ")

    try:
        handled, _ = asyncio.run(_run_apply(wait_state, message, ctx, user_id))
        state_after = get_wait_state(user_id)
    finally:
        bot_module.show_veo_card = original_show  # type: ignore[assignment]
        clear_wait_state(user_id)

    assert handled is True
    assert state_dict["last_prompt"] == "Test prompt"
    assert calls and calls[-1][0] == 777
    assert state_after is not None and state_after.kind == WaitKind.VEO_PROMPT
    assert message.replies == []


def test_wait_state_updates_banana_prompt() -> None:
    ctx = SimpleNamespace(bot=None, user_data={})
    state_dict = bot_module.state(ctx)
    state_dict["last_ui_msg_id_banana"] = 111
    wait_state = WaitInputState(kind=WaitKind.BANANA_PROMPT, card_msg_id=111, chat_id=888, meta={})
    user_id = 556
    set_wait_state(user_id, wait_state)

    calls: list[tuple[int, object]] = []

    original_show = bot_module.show_banana_card

    async def fake_show(chat_id: int, ctx_param, *, force_new: bool = False):
        calls.append((chat_id, ctx_param))
        state_dict["last_ui_msg_id_banana"] = 222

    bot_module.show_banana_card = fake_show  # type: ignore[assignment]

    message = DummyMessage(chat_id=888, text="  Fix face blemishes  ")

    try:
        handled, _ = asyncio.run(_run_apply(wait_state, message, ctx, user_id))
        state_after = get_wait_state(user_id)
    finally:
        bot_module.show_banana_card = original_show  # type: ignore[assignment]
        clear_wait_state(user_id)

    assert handled is True
    assert state_dict["last_prompt"] == "Fix face blemishes"
    assert calls and calls[-1][0] == 888
    assert state_after is not None and state_after.kind == WaitKind.BANANA_PROMPT
    assert message.replies == []


def test_wait_state_suno_title_updates_card() -> None:
    ctx = SimpleNamespace(bot=None, user_data={})
    state_dict = bot_module.state(ctx)
    wait_state = WaitInputState(kind=WaitKind.SUNO_TITLE, card_msg_id=205, chat_id=888, meta={})
    user_id = 991
    set_wait_state(user_id, wait_state)

    original_refresh = bot_module.refresh_suno_card
    refreshed: list[tuple[int, dict[str, object]]] = []

    async def fake_refresh(
        ctx_param,
        chat_id: int,
        state_dict_param: dict[str, object],
        *,
        price: int,
        state_key: str = "last_ui_msg_id_suno",
        force_new: bool = False,
    ):
        refreshed.append((chat_id, state_dict_param))
        state_dict_param[state_key] = 777
        return 777

    bot_module.refresh_suno_card = fake_refresh  # type: ignore[assignment]

    message = DummyMessage(chat_id=888, text="  <b>My Song</b>  ")

    try:
        handled, _ = asyncio.run(_run_apply(wait_state, message, ctx, user_id))
        state_after = get_wait_state(user_id)
    finally:
        bot_module.refresh_suno_card = original_refresh  # type: ignore[assignment]
        clear_wait_state(user_id)

    assert handled is True
    suno_state = bot_module.load_suno_state(ctx)
    assert suno_state.title == "My Song"
    assert refreshed and refreshed[-1][0] == 888
    assert state_after is not None and state_after.kind == WaitKind.SUNO_TITLE
    assert message.replies == []


def test_safe_edit_resends_after_not_modified() -> None:
    bot = FakeBot()
    bot.queue_edit_error(BadRequest("Message is not modified"))
    bot.queue_edit_error(BadRequest("Message is not modified"))
    state: dict[str, object] = {}

    async def scenario() -> None:
        result = await safe_edit(
            bot,
            chat_id=42,
            message_id=101,
            text="Hello",
            reply_markup=None,
            parse_mode="HTML",
            disable_web_page_preview=True,
            state=state,
            resend_on_not_modified=True,
        )
        assert result.status == "resent"
        assert result.reason == "not_modified"

    asyncio.run(scenario())

    assert bot.deleted == [(42, 101)]
    assert bot.sent and bot.sent[-1]["text"] == "Hello"
    assert state.get("msg_id") != 101
    assert not bot._errors

