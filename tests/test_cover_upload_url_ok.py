import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui_helpers import render_suno_card
from utils.suno_state import load as load_suno_state, save as save_suno_state, set_style as set_suno_style, set_title as set_suno_title

from tests.suno_test_utils import DummyMessage, bot_module, setup_cover_context


def test_cover_upload_url_ok(monkeypatch):
    ctx, state_dict, _bot = setup_cover_context()
    chat_id = 777
    state_dict["mode"] = "suno"

    asyncio.run(bot_module._music_begin_flow(chat_id, ctx, state_dict, flow="cover", user_id=42))

    suno_state = load_suno_state(ctx)
    set_suno_title(suno_state, "My Cover")
    set_suno_style(suno_state, "ambient dream pop")
    save_suno_state(ctx, suno_state)
    state_dict["suno_state"] = suno_state.to_dict()

    captured: dict[str, str] = {}

    async def fake_upload(url: str, **_kwargs) -> str:
        captured["uploaded"] = url
        return "kie-test-123"

    async def fake_ensure(url: str) -> str:
        captured["ensured"] = url
        return url

    monkeypatch.setattr(bot_module, "upload_cover_url", fake_upload)
    monkeypatch.setattr(bot_module, "ensure_cover_audio_url", fake_ensure)

    message = DummyMessage("https://example.com/song.mp3", chat_id)
    result = asyncio.run(
        bot_module._cover_process_url_input(
            ctx,
            chat_id,
            message,
            state_dict,
            message.text,
            user_id=42,
        )
    )

    assert result is True
    assert message.replies[-1] == "✅ Принято"
    assert captured["uploaded"] == "https://example.com/song.mp3"
    assert captured["ensured"] == "https://example.com/song.mp3"

    updated_state = load_suno_state(ctx)
    assert updated_state.kie_file_id == "kie-test-123"
    assert updated_state.source_url == "https://example.com/song.mp3"

    text, _markup, ready = render_suno_card(
        updated_state,
        price=bot_module.PRICE_SUNO,
        balance=None,
        generating=False,
        waiting_enqueue=False,
    )
    assert "загружено ✅ (id: kie-test-123)" in text
    assert ready is True
