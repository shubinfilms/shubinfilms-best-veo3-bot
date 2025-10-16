import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.telegram_sender import TelegramSender


def test_sender_submit_rejects_chat_id_kwarg():
    sender = TelegramSender()

    async def dummy(**kwargs):  # pragma: no cover - should not run
        return kwargs

    async def runner() -> None:
        with pytest.raises(TypeError, match="Do not pass 'chat_id' in kwargs"):
            await sender.submit(123, dummy, method_name="x", kind="y", chat_id=123)

    asyncio.run(runner())


def test_sender_submit_injects_chat_id():
    sender = TelegramSender()
    received: dict[str, object] = {}

    async def dummy(**kwargs):
        received.update(kwargs)
        return "ok"

    async def runner() -> None:
        result = await sender.submit(987, dummy, method_name="foo", kind="bar", text="hi")
        assert result == "ok"
        assert received["chat_id"] == 987
        assert received["text"] == "hi"
        await sender.stop()

    asyncio.run(runner())
