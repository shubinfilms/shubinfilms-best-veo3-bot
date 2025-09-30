import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui_helpers import render_suno_card
from utils.suno_state import load as load_suno_state, save as save_suno_state, set_style as set_suno_style, set_title as set_suno_title

from tests.suno_test_utils import DummyMessage, bot_module, setup_cover_context


class DummyAudio:
    def __init__(self) -> None:
        self.file_id = "file123"
        self.file_name = "demo.mp3"
        self.mime_type = "audio/mpeg"
        self.file_size = 1024
        self.duration = 10


def test_cover_upload_stream_ok(monkeypatch):
    ctx, state_dict, _bot = setup_cover_context()
    chat_id = 555
    state_dict["mode"] = "suno"

    asyncio.run(bot_module._music_begin_flow(chat_id, ctx, state_dict, flow="cover", user_id=51))

    suno_state = load_suno_state(ctx)
    set_suno_title(suno_state, "Ocean waves")
    set_suno_style(suno_state, "calm ambient")
    save_suno_state(ctx, suno_state)
    state_dict["suno_state"] = suno_state.to_dict()

    captured: dict[str, object] = {}

    async def fake_download(url: str, **_kwargs) -> bytes:
        captured["download_url"] = url
        return b"1234"

    async def fake_upload(data: bytes, filename: str, mime_type: str, **_kwargs) -> str:
        captured["uploaded_bytes"] = data
        captured["filename"] = filename
        captured["mime"] = mime_type
        return "kie-audio-999"

    monkeypatch.setattr(bot_module, "_download_telegram_file", fake_download)
    monkeypatch.setattr(bot_module, "upload_cover_stream", fake_upload)

    message = DummyMessage("", chat_id)
    message.audio = DummyAudio()

    result = asyncio.run(
        bot_module._cover_process_audio_input(
            ctx,
            chat_id,
            message,
            state_dict,
            message.audio,
            user_id=51,
        )
    )

    assert result is True
    assert message.replies[-1] == "🎧 Источник загружен. Можно продолжать."
    assert captured["filename"] == "demo.mp3"
    assert captured["mime"] == "audio/mpeg"
    assert captured["uploaded_bytes"] == b"1234"

    updated_state = load_suno_state(ctx)
    assert updated_state.kie_file_id == "kie-audio-999"
    assert updated_state.cover_source_label == "demo.mp3"

    text, _markup, ready = render_suno_card(
        updated_state,
        price=bot_module.PRICE_SUNO,
        balance=None,
        generating=False,
        waiting_enqueue=False,
    )
    assert "загружено ✅ (id: kie-audio-999)" in text
    assert ready is True
