import asyncio
from types import SimpleNamespace
import os
import sys
from pathlib import Path

import pytest
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
from utils.input_state import WaitInputState, WaitKind, clear_wait_state, set_wait_state


def _run(coro) -> None:
    asyncio.run(coro)


class DummyMessage:
    def __init__(self, chat_id: int, text: str) -> None:
        self.chat_id = chat_id
        self.text = text
        self.replies: list[str] = []

    async def reply_text(self, text: str, **_: object) -> None:  # type: ignore[override]
        self.replies.append(text)


def test_card_prompt_empty_on_open() -> None:
    ctx = SimpleNamespace(bot=None, user_data={})
    state = bot_module.state(ctx)
    state["last_prompt"] = None
    state["aspect"] = "16:9"

    veo_text = bot_module.veo_card_text(state)
    assert "<code>—</code>" not in veo_text
    assert "<code> </code>" in veo_text

    mj_text = bot_module._mj_prompt_card_text("16:9", state.get("last_prompt"))
    assert "Промпт: <i>—</i>" not in mj_text
    assert "Промпт: <i> </i>" in mj_text


def test_menu_labels_not_saved_as_prompt() -> None:
    ctx = SimpleNamespace(bot=None, user_data={})
    state = bot_module.state(ctx)
    state["last_ui_msg_id_mj"] = 321
    user_id = 909
    wait_state = WaitInputState(kind=WaitKind.MJ_PROMPT, card_msg_id=321, chat_id=777, meta={})
    set_wait_state(user_id, wait_state)

    message = DummyMessage(chat_id=777, text=" 🎨 ГЕНЕРАЦИЯ ИЗОБРАЖЕНИЙ ")
    update = SimpleNamespace(effective_message=message, effective_user=SimpleNamespace(id=user_id))

    with pytest.raises(ApplicationHandlerStop):
        _run(bot_module.handle_card_input(update, ctx))
    clear_wait_state(user_id)

    assert state["last_prompt"] is None
    assert message.replies == []


def test_user_text_saved_and_acknowledged() -> None:
    ctx = SimpleNamespace(bot=None, user_data={})
    state = bot_module.state(ctx)
    state["last_ui_msg_id_mj"] = 654
    user_id = 404
    wait_state = WaitInputState(kind=WaitKind.MJ_PROMPT, card_msg_id=654, chat_id=555, meta={})
    set_wait_state(user_id, wait_state)

    message = DummyMessage(chat_id=555, text="  Magic idea  ")
    update = SimpleNamespace(effective_message=message, effective_user=SimpleNamespace(id=user_id))

    original_show = bot_module.show_mj_prompt_card

    async def fake_show(chat_id_param: int, ctx_param, *, force_new: bool = False) -> None:
        state["last_ui_msg_id_mj"] = 654

    bot_module.show_mj_prompt_card = fake_show  # type: ignore[assignment]

    try:
        with pytest.raises(ApplicationHandlerStop):
            _run(bot_module.handle_card_input(update, ctx))
    finally:
        bot_module.show_mj_prompt_card = original_show  # type: ignore[assignment]
        clear_wait_state(user_id)

    assert state["last_prompt"] == "Magic idea"
    assert message.replies == ["✅ Принято"]
