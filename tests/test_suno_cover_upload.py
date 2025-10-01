import asyncio
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from suno.cover_source import CoverSourceUnavailableError
from dataclasses import replace

from tests.suno_test_utils import DummyMessage, bot_module, setup_cover_context


class DummyAudio:
    def __init__(self) -> None:
        self.file_id = "cover-audio"
        self.file_name = "cover.mp3"
        self.mime_type = "audio/mpeg"
        self.file_size = 4096
        self.duration = 12


def _prepare_state(chat_id: int):
    ctx, state_dict, bot = setup_cover_context(chat_id=chat_id)
    base_override = os.getenv("SUNO_API_BASE")
    if base_override:
        try:
            bot_module.SUNO_CONFIG = replace(bot_module.SUNO_CONFIG, base=base_override.rstrip("/"))
        except Exception:
            pass
    asyncio.run(
        bot_module._music_begin_flow(
            chat_id,
            ctx,
            state_dict,
            flow="cover",
            user_id=333,
        )
    )
    return ctx, state_dict, bot


def test_cover_upload_base64_fallback(monkeypatch):
    ctx, state_dict, bot = _prepare_state(1001)

    captured: dict[str, object] = {}

    async def fake_download(url: str, **_kwargs) -> bytes:
        captured["download"] = url
        return b"AUDIO"

    async def fail_stream(*_args, **_kwargs):
        raise CoverSourceUnavailableError("stream")

    async def fail_url(*_args, **_kwargs):
        raise CoverSourceUnavailableError("url")

    async def ok_base64(data: bytes, *_args, **_kwargs) -> str:
        captured["base64"] = data
        return "kie-cover-base64"

    monkeypatch.setattr(bot_module, "_download_telegram_file", fake_download)
    monkeypatch.setattr(bot_module, "upload_cover_stream", fail_stream)
    monkeypatch.setattr(bot_module, "upload_cover_url", fail_url)
    monkeypatch.setattr(bot_module, "upload_cover_base64", ok_base64)

    audio = DummyAudio()
    message = DummyMessage("", 1001)
    message.audio = audio

    result = asyncio.run(
        bot_module._cover_process_audio_input(
            ctx,
            1001,
            message,
            state_dict,
            audio,
            user_id=333,
        )
    )

    assert result is True
    assert message.replies[-1] == "✅ Принято"
    assert "base64" in captured
    assert captured["base64"] == b"AUDIO"

    payloads = []
    payloads.extend(item for item in bot.sent if isinstance(item, dict))
    payloads.extend(item for item in bot.edited if isinstance(item, dict))
    assert any("Источник: <i>🎧 Файл</i>" in item.get("text", "") for item in payloads)


def test_cover_upload_service_unavailable(monkeypatch):
    monkeypatch.setenv("SUNO_API_BASE", "https://service.local")
    monkeypatch.setenv("TELEGRAM_TOKEN", "token-real")
    monkeypatch.setattr(bot_module, "TELEGRAM_TOKEN", "token-real")
    ctx, state_dict, _bot = _prepare_state(1002)

    async def fake_download(_url: str, **_kwargs) -> bytes:
        return b"data"

    async def fail_stream(*_args, **_kwargs):
        raise CoverSourceUnavailableError("stream")

    async def fail_url(*_args, **_kwargs):
        raise CoverSourceUnavailableError("url")

    async def fail_base64(*_args, **_kwargs):
        raise CoverSourceUnavailableError("base64")

    monkeypatch.setattr(bot_module, "_download_telegram_file", fake_download)
    monkeypatch.setattr(bot_module, "upload_cover_stream", fail_stream)
    monkeypatch.setattr(bot_module, "upload_cover_url", fail_url)
    monkeypatch.setattr(bot_module, "upload_cover_base64", fail_base64)

    audio = DummyAudio()
    message = DummyMessage("", 1002)
    message.audio = audio

    result = asyncio.run(
        bot_module._cover_process_audio_input(
            ctx,
            1002,
            message,
            state_dict,
            audio,
            user_id=333,
        )
    )

    assert result is True
    assert message.replies[-1] == bot_module._COVER_UPLOAD_SERVICE_ERROR_MESSAGE
