import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace

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
from utils.input_state import (  # noqa: E402
    WaitInputState,
    WaitKind,
    clear_wait_state,
    set_wait_state,
)


class DummyMessage:
    def __init__(self, chat_id: int, text: str) -> None:
        self.chat_id = chat_id
        self.text = text
        self.replies: list[str] = []

    async def reply_text(self, text: str, **_: object) -> None:  # type: ignore[override]
        self.replies.append(text)


def _run(coro):
    try:
        asyncio.run(coro)
    except ApplicationHandlerStop:
        pass


def test_veo_card_opens_with_empty_prompt() -> None:
    ctx = SimpleNamespace(bot=None, user_data={})
    captured: list[str] = []

    async def fake_upsert(
        ctx_param,
        chat_id: int,
        state_dict: dict,
        state_key: str,
        text: str,
        reply_markup,
        *,
        force_new: bool = False,
        parse_mode=None,
        disable_web_page_preview: bool = True,
    ) -> int:
        captured.append(text)
        state_dict[state_key] = 123
        return 123

    async def fake_ensure(update_param):
        return None

    def fake_set_mode(user_id: int, on: bool) -> None:  # type: ignore[override]
        return None

    original_upsert = bot_module.upsert_card
    original_ensure = bot_module.ensure_user_record
    original_set_mode = bot_module.set_mode

    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=555),
        effective_user=SimpleNamespace(id=555),
        message=SimpleNamespace(),
        effective_message=SimpleNamespace(),
    )

    try:
        bot_module.upsert_card = fake_upsert  # type: ignore[assignment]
        bot_module.ensure_user_record = fake_ensure  # type: ignore[assignment]
        bot_module.set_mode = fake_set_mode  # type: ignore[assignment]
        _run(bot_module.handle_video_entry(update, ctx))
    finally:
        bot_module.upsert_card = original_upsert  # type: ignore[assignment]
        bot_module.ensure_user_record = original_ensure  # type: ignore[assignment]
        bot_module.set_mode = original_set_mode  # type: ignore[assignment]

    assert captured, "card render not captured"
    assert "Введите промпт…" in captured[-1]


def test_mj_card_opens_with_empty_prompt() -> None:
    ctx = SimpleNamespace(bot=None, user_data={})
    captured: list[str] = []

    async def fake_upsert(
        ctx_param,
        chat_id: int,
        state_dict: dict,
        state_key: str,
        text: str,
        reply_markup,
        *,
        force_new: bool = False,
        parse_mode=None,
        disable_web_page_preview: bool = True,
    ) -> int:
        captured.append(text)
        state_dict[state_key] = 321
        return 321

    async def fake_ensure(update_param):
        return None

    def fake_set_mode(user_id: int, on: bool) -> None:  # type: ignore[override]
        return None

    original_upsert = bot_module.upsert_card
    original_ensure = bot_module.ensure_user_record
    original_set_mode = bot_module.set_mode

    state_dict = bot_module.state(ctx)
    state_dict["image_engine"] = "mj"

    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=777),
        effective_user=SimpleNamespace(id=777),
        message=SimpleNamespace(),
        effective_message=SimpleNamespace(),
    )

    try:
        bot_module.upsert_card = fake_upsert  # type: ignore[assignment]
        bot_module.ensure_user_record = fake_ensure  # type: ignore[assignment]
        bot_module.set_mode = fake_set_mode  # type: ignore[assignment]
        _run(bot_module.handle_image_entry(update, ctx))
        _run(bot_module.show_mj_prompt_card(777, ctx))
    finally:
        bot_module.upsert_card = original_upsert  # type: ignore[assignment]
        bot_module.ensure_user_record = original_ensure  # type: ignore[assignment]
        bot_module.set_mode = original_set_mode  # type: ignore[assignment]

    assert captured, "card render not captured"
    assert "Введите промпт…" in captured[-1]


def test_menu_label_not_saved_as_prompt() -> None:
    ctx = SimpleNamespace(bot=None, user_data={})
    state_dict = bot_module.state(ctx)
    state_dict["last_prompt"] = "existing"

    user_id = 909
    wait_state = WaitInputState(kind=WaitKind.VEO_PROMPT, card_msg_id=0, chat_id=101, meta={})
    set_wait_state(user_id, wait_state)

    message = DummyMessage(chat_id=101, text="Баланс")
    update = SimpleNamespace(effective_message=message, effective_user=SimpleNamespace(id=user_id))

    try:
        _run(bot_module.handle_card_input(update, ctx))
    finally:
        clear_wait_state(user_id)

    assert state_dict["last_prompt"] == "existing"
    assert message.replies == []
