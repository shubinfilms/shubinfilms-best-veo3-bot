import asyncio
import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

bot_module = importlib.import_module("bot")


class DummyBot:
    def __init__(self) -> None:
        self.sent: list[dict[str, object]] = []
        self.files: dict[str, object] = {}

    async def send_message(self, **kwargs):  # type: ignore[override]
        self.sent.append(kwargs)
        return SimpleNamespace(message_id=kwargs.get("message_id", 100))

    async def edit_message_text(self, **kwargs):  # type: ignore[override]
        return SimpleNamespace(message_id=kwargs.get("message_id", 101))

    async def send_chat_action(self, **kwargs):  # type: ignore[override]
        return None

    async def get_file(self, file_id: str):  # type: ignore[override]
        return SimpleNamespace(file_path=f"audio/{file_id}.mp3")


class DummyMessage:
    def __init__(self, text: str, chat_id: int = 123) -> None:
        self.text = text
        self.chat_id = chat_id
        self.replies: list[str] = []

    async def reply_text(self, text: str, **_kwargs) -> None:  # type: ignore[override]
        self.replies.append(text)


def _setup_context() -> tuple[SimpleNamespace, dict[str, object]]:
    bot = DummyBot()
    ctx = SimpleNamespace(bot=bot, user_data={})
    state_dict = bot_module.state(ctx)
    return ctx, state_dict


def test_instrumental_auto_style_default() -> None:
    ctx, state_dict = _setup_context()
    chat_id = 321

    asyncio.run(
        bot_module._music_begin_flow(chat_id, ctx, state_dict, flow="instrumental", user_id=42)
    )

    title_message = DummyMessage("Dreamscape")
    asyncio.run(
        bot_module._handle_suno_waiting_input(
            ctx,
            chat_id,
            title_message,
            state_dict,
            bot_module.WAIT_SUNO_TITLE,
            user_id=42,
        )
    )

    message = DummyMessage("")
    asyncio.run(
        bot_module._handle_suno_waiting_input(
            ctx,
            chat_id,
            message,
            state_dict,
            bot_module.WAIT_SUNO_STYLE,
            user_id=42,
        )
    )
    suno_state = bot_module.load_suno_state(ctx)
    assert suno_state.style == bot_module._INSTRUMENTAL_DEFAULT_STYLE
    assert state_dict.get("suno_step") == "ready"
    assert any("стиль по умолчанию" in reply for reply in message.replies)


def test_lyrics_manual_and_auto_generation() -> None:
    ctx, state_dict = _setup_context()
    chat_id = 555

    asyncio.run(
        bot_module._music_begin_flow(chat_id, ctx, state_dict, flow="lyrics", user_id=99)
    )

    title_msg = DummyMessage("City Lights")
    asyncio.run(
        bot_module._handle_suno_waiting_input(
            ctx,
            chat_id,
            title_msg,
            state_dict,
            bot_module.WAIT_SUNO_TITLE,
            user_id=99,
        )
    )

    # manual lyrics
    manual_msg = DummyMessage("First line\nSecond line")
    asyncio.run(
        bot_module._handle_suno_waiting_input(
            ctx,
            chat_id,
            manual_msg,
            state_dict,
            bot_module.WAIT_SUNO_LYRICS,
            user_id=99,
        )
    )
    manual_state = bot_module.load_suno_state(ctx)
    assert manual_state.lyrics == "First line\nSecond line"
    assert not state_dict.get("suno_auto_lyrics_pending")

    # restart flow to test auto lyrics
    asyncio.run(
        bot_module._music_begin_flow(chat_id, ctx, state_dict, flow="lyrics", user_id=99)
    )
    title_msg_auto = DummyMessage("City Lights")
    asyncio.run(
        bot_module._handle_suno_waiting_input(
            ctx,
            chat_id,
            title_msg_auto,
            state_dict,
            bot_module.WAIT_SUNO_TITLE,
            user_id=99,
        )
    )
    auto_msg = DummyMessage("")
    asyncio.run(
        bot_module._handle_suno_waiting_input(
            ctx,
            chat_id,
            auto_msg,
            state_dict,
            bot_module.WAIT_SUNO_LYRICS,
            user_id=99,
        )
    )
    assert state_dict.get("suno_auto_lyrics_pending") is True

    # provide style to trigger auto lyrics
    style_msg = DummyMessage("dream pop, neon")
    asyncio.run(
        bot_module._handle_suno_waiting_input(
            ctx,
            chat_id,
            style_msg,
            state_dict,
            bot_module.WAIT_SUNO_STYLE,
            user_id=99,
        )
    )
    auto_state = bot_module.load_suno_state(ctx)
    assert auto_state.lyrics
    assert state_dict.get("suno_auto_lyrics_generated") is True


def test_cover_upload_flow_accepts_audio() -> None:
    ctx, state_dict = _setup_context()
    chat_id = 777

    asyncio.run(
        bot_module._music_begin_flow(chat_id, ctx, state_dict, flow="cover", user_id=7)
    )
    state_dict["mode"] = "suno"

    title_cover_msg = DummyMessage("Cover Song", chat_id)
    asyncio.run(
        bot_module._handle_suno_waiting_input(
            ctx,
            chat_id,
            title_cover_msg,
            state_dict,
            bot_module.WAIT_SUNO_TITLE,
            user_id=7,
        )
    )

    class Audio:
        def __init__(self) -> None:
            self.file_id = "file123"
            self.file_name = "demo.mp3"
            self.file_size = 1000
            self.duration = 5

    message = DummyMessage("", chat_id)
    message.audio = Audio()
    message.voice = None
    update = SimpleNamespace(message=message, effective_user=SimpleNamespace(id=7))
    asyncio.run(bot_module.handle_voice(update, ctx))

    suno_state = bot_module.load_suno_state(ctx)
    assert suno_state.cover_source_url
    assert suno_state.cover_source_label == "demo.mp3"
    assert state_dict.get("suno_step") == "style"


def test_artist_name_error_message() -> None:
    text = bot_module._suno_error_message(400, "Artist name detected")
    assert "Please remove artist names" in text
