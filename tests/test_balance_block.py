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
def balance_module(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    module = importlib.import_module("balance")
    return importlib.reload(module)


class DummyBot:
    def __init__(self):
        self.messages = []

    async def send_message(self, **kwargs):  # pragma: no cover - async stub
        self.messages.append(kwargs)
        return SimpleNamespace()


def test_balance_block_shows_topup(monkeypatch, balance_module):
    monkeypatch.setattr(balance_module, "get_balance", lambda user_id: 0)

    bot = DummyBot()
    ctx = SimpleNamespace(bot=bot)

    result = asyncio.run(balance_module.ensure_tokens(ctx, chat_id=11, user_id=22, need=100))
    assert result is False
    assert len(bot.messages) == 1

    keyboard = bot.messages[0]["reply_markup"]
    button_texts = [button.text for row in keyboard.inline_keyboard for button in row]
    assert "💳 Пополнить баланс" in button_texts
