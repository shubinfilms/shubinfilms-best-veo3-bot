import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from suno.cover_source import CoverSourceClientError, CoverSourceUnavailableError, CoverSourceValidationError
from utils.suno_state import load as load_suno_state

from tests.suno_test_utils import DummyMessage, bot_module, setup_cover_context


class InvalidAudio:
    def __init__(self) -> None:
        self.file_id = "bad123"
        self.file_name = "track.txt"
        self.mime_type = "text/plain"
        self.file_size = 1024
        self.duration = 2


def _prepare_cover_state(ctx, state_dict, chat_id):
    state_dict["mode"] = "suno"
    asyncio.run(bot_module._music_begin_flow(chat_id, ctx, state_dict, flow="cover", user_id=77))


def test_cover_upload_invalid_url(monkeypatch):
    ctx, state_dict, _bot = setup_cover_context()
    chat_id = 888
    _prepare_cover_state(ctx, state_dict, chat_id)

    async def fake_ensure(_url: str) -> str:
        raise CoverSourceValidationError("bad-url")

    monkeypatch.setattr(bot_module, "ensure_cover_audio_url", fake_ensure)

    message = DummyMessage("not a link", chat_id)
    result = asyncio.run(
        bot_module._cover_process_url_input(
            ctx,
            chat_id,
            message,
            state_dict,
            message.text,
            user_id=77,
        )
    )

    assert result is True
    assert message.replies[-1] == bot_module._COVER_INVALID_INPUT_MESSAGE
    assert load_suno_state(ctx).kie_file_id is None


def test_cover_upload_invalid_audio(monkeypatch):
    ctx, state_dict, _bot = setup_cover_context()
    chat_id = 889
    _prepare_cover_state(ctx, state_dict, chat_id)

    message = DummyMessage("", chat_id)
    message.audio = InvalidAudio()

    result = asyncio.run(
        bot_module._cover_process_audio_input(
            ctx,
            chat_id,
            message,
            state_dict,
            message.audio,
            user_id=77,
        )
    )

    assert result is True
    assert message.replies[-1] == bot_module._COVER_INVALID_INPUT_MESSAGE
    assert load_suno_state(ctx).kie_file_id is None


def test_cover_upload_service_errors(monkeypatch):
    ctx, state_dict, _bot = setup_cover_context()
    chat_id = 890
    _prepare_cover_state(ctx, state_dict, chat_id)

    async def fake_ensure(url: str) -> str:
        return url

    async def fake_upload(_url: str, **_kwargs) -> str:
        raise CoverSourceClientError("status:400")

    monkeypatch.setattr(bot_module, "ensure_cover_audio_url", fake_ensure)
    monkeypatch.setattr(bot_module, "upload_cover_url", fake_upload)

    message = DummyMessage("https://example.com/track.mp3", chat_id)
    result = asyncio.run(
        bot_module._cover_process_url_input(
            ctx,
            chat_id,
            message,
            state_dict,
            message.text,
            user_id=77,
        )
    )

    assert result is True
    assert message.replies[-1] == bot_module._COVER_UPLOAD_CLIENT_ERROR_MESSAGE

    async def fake_upload_unavailable(_url: str, **_kwargs) -> str:
        raise CoverSourceUnavailableError("status:503")

    monkeypatch.setattr(bot_module, "upload_cover_url", fake_upload_unavailable)
    message_error = DummyMessage("https://example.com/track.mp3", chat_id)
    result_error = asyncio.run(
        bot_module._cover_process_url_input(
            ctx,
            chat_id,
            message_error,
            state_dict,
            message_error.text,
            user_id=77,
        )
    )

    assert result_error is True
    assert message_error.replies[-1] == bot_module._COVER_UPLOAD_SERVICE_ERROR_MESSAGE
