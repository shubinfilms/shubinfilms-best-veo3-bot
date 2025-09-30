import asyncio
import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace


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

bot_module = importlib.import_module("bot")


class FakeBot:
    def __init__(self) -> None:
        self.sent: list[dict[str, object]] = []
        self.edited: list[dict[str, object]] = []
        self.deleted: list[dict[str, object]] = []
        self._next_message_id = 100

    async def send_message(self, **kwargs):  # type: ignore[override]
        self.sent.append(kwargs)
        message_id = self._next_message_id
        self._next_message_id += 1
        return SimpleNamespace(message_id=message_id)

    async def edit_message_text(self, **kwargs):  # type: ignore[override]
        self.edited.append(kwargs)
        return SimpleNamespace(message_id=kwargs.get("message_id"))

    async def delete_message(self, *args, **kwargs):  # type: ignore[override]
        if args:
            chat_id = args[0] if len(args) > 0 else kwargs.get("chat_id")
            message_id = args[1] if len(args) > 1 else kwargs.get("message_id")
            payload = {"chat_id": chat_id, "message_id": message_id}
        else:
            payload = kwargs
        self.deleted.append(payload)
        return True

    async def send_chat_action(self, **_kwargs):  # type: ignore[override]
        return None

    async def get_file(self, file_id: str):  # type: ignore[override]
        async def _download_as_bytearray() -> bytes:
            return b"telegram-audio"

        return SimpleNamespace(
            file_path=f"audio/{file_id}.mp3",
            download_as_bytearray=_download_as_bytearray,
        )


class DummyMessage:
    def __init__(self, text: str, chat_id: int) -> None:
        self.text = text
        self.chat_id = chat_id
        self.replies: list[str] = []
        self.voice = None
        self.audio = None

    async def reply_text(self, text: str, **_kwargs):  # type: ignore[override]
        self.replies.append(text)
        return SimpleNamespace(message_id=900 + len(self.replies))


def setup_cover_context(chat_id: int = 777):
    bot = FakeBot()
    ctx = SimpleNamespace(bot=bot, user_data={})
    state_dict = bot_module.state(ctx)
    asyncio.run(bot_module.refresh_suno_card(ctx, chat_id=chat_id, state_dict=state_dict, price=bot_module.PRICE_SUNO))
    return ctx, state_dict, bot


__all__ = [
    "DummyMessage",
    "FakeBot",
    "bot_module",
    "setup_cover_context",
]
