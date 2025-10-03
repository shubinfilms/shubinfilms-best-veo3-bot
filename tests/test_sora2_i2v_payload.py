import asyncio
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def bot_module(monkeypatch):
    monkeypatch.setenv("TELEGRAM_TOKEN", "dummy-token")
    monkeypatch.setenv("SUNO_API_BASE", "https://example.com")
    monkeypatch.setenv("SUNO_API_TOKEN", "token")
    monkeypatch.setenv("LEDGER_BACKEND", "memory")
    monkeypatch.setenv("DATABASE_URL", "postgres://test")
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://bot.example")
    module = importlib.import_module("bot")
    return importlib.reload(module)


class DummyMessage:
    def __init__(self, chat_id: int, user_id: int, text: str):
        self.chat = SimpleNamespace(id=chat_id)
        self.chat_id = chat_id
        self.message_id = 500
        self.from_user = SimpleNamespace(id=user_id)
        self.text = text
        self.reply_calls = []

    async def reply_text(self, text: str):
        self.reply_calls.append(text)
        return SimpleNamespace(message_id=901)


def test_sora2_i2v_payload(monkeypatch, bot_module):
    ctx = SimpleNamespace(bot=SimpleNamespace(), user_data={}, application=None)
    state = bot_module.state(ctx)
    state["mode"] = "sora2_itv"
    state["sora2_prompt"] = None
    state["sora2_image_urls"] = []

    async def fake_show_card(*args, **kwargs):
        return None

    monkeypatch.setattr(bot_module, "show_sora2_card", fake_show_card)
    monkeypatch.setattr(bot_module, "refresh_card_pointer", lambda *args, **kwargs: None)
    monkeypatch.setattr(bot_module, "classify_wait_input", lambda text: (True, None))

    user_id = 101
    chat_id = 606
    bot_module.set_wait(user_id, "sora2_prompt", card_msg_id=321, chat_id=chat_id)

    text = (
        "Story with refs https://img.one/a.png and https://img.two/b.png "
        "plus https://img.three/c.png https://img.four/d.png and https://img.five/e.png"
    )
    expected_urls = bot_module._extract_http_urls(text)
    assert expected_urls

    message = DummyMessage(chat_id, user_id, text)
    wait_state = bot_module.get_wait(user_id)
    asyncio.run(bot_module._apply_wait_state_input(ctx, message, wait_state, user_id=user_id))
    asyncio.run(bot_module._wait_acknowledge(message))

    state = bot_module.state(ctx)
    assert state["sora2_image_urls"] == expected_urls[: bot_module.SORA2_MAX_IMAGES]
    assert state["sora2_prompt"]
    assert any(call == "✅ Принято" for call in message.reply_calls)

    clear_message = DummyMessage(chat_id, user_id, "clear")
    wait_state_clear = bot_module.get_wait(user_id)
    asyncio.run(bot_module._apply_wait_state_input(ctx, clear_message, wait_state_clear, user_id=user_id))
    asyncio.run(bot_module._wait_acknowledge(clear_message))

    state = bot_module.state(ctx)
    assert state["sora2_image_urls"] == []
    assert state["sora2_prompt"] is None
    assert any(call == "✅ Принято" for call in clear_message.reply_calls)
