import asyncio
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from tests.suno_test_utils import DummyMessage, bot_module, setup_cover_context


def _find_card_text(bot) -> str:
    sources = []
    sources.extend(payload for payload in bot.sent if isinstance(payload, dict))
    sources.extend(payload for payload in bot.edited if isinstance(payload, dict))
    for payload in reversed(sources):
        text = payload.get("text") if isinstance(payload, dict) else None
        if isinstance(text, str) and "Источник текста" in text:
            return text
    return ""


def test_user_lyrics_persist_and_display():
    chat_id = 950
    user_id = 1010
    ctx, state_dict, bot = setup_cover_context(chat_id=chat_id)

    asyncio.run(
        bot_module._music_begin_flow(
            chat_id,
            ctx,
            state_dict,
            flow="lyrics",
            user_id=user_id,
        )
    )

    suno_state = bot_module.load_suno_state(ctx)
    bot_module.set_suno_lyrics_source(suno_state, bot_module.LyricsSource.USER)
    bot_module.save_suno_state(ctx, suno_state)
    state_dict["suno_state"] = suno_state.to_dict()

    waiting_field = state_dict.get("suno_waiting_state")
    assert waiting_field == bot_module.WAIT_SUNO_TITLE

    message = DummyMessage("City Lights", chat_id)
    asyncio.run(
        bot_module._handle_suno_waiting_input(
            ctx,
            chat_id,
            message,
            state_dict,
            waiting_field,
            user_id=user_id,
        )
    )

    waiting_field = state_dict.get("suno_waiting_state")
    assert waiting_field == bot_module.WAIT_SUNO_LYRICS

    lyrics_value = "Line one\nLine two"
    lyrics_msg = DummyMessage(lyrics_value, chat_id)
    asyncio.run(
        bot_module._handle_suno_waiting_input(
            ctx,
            chat_id,
            lyrics_msg,
            state_dict,
            waiting_field,
            user_id=user_id,
        )
    )

    waiting_field = state_dict.get("suno_waiting_state")
    assert waiting_field == bot_module.WAIT_SUNO_STYLE

    style_msg = DummyMessage("Dream pop", chat_id)
    asyncio.run(
        bot_module._handle_suno_waiting_input(
            ctx,
            chat_id,
            style_msg,
            state_dict,
            waiting_field,
            user_id=user_id,
        )
    )

    prompt_texts = [
        item.get("text")
        for item in bot.sent
        if isinstance(item, dict) and isinstance(item.get("text"), str)
    ]
    assert any("Пришлите текст песни" in text for text in prompt_texts)
    assert any("Шаг 2/3 (стиль)" in text for text in prompt_texts)

    waiting_field = state_dict.get("suno_waiting_state") or bot_module.WAIT_SUNO_STYLE
    state_dict["suno_waiting_state"] = waiting_field

    updated_state = bot_module.load_suno_state(ctx)
    assert updated_state.lyrics == lyrics_value
    assert updated_state.lyrics_source == bot_module.LyricsSource.USER
    assert updated_state.lyrics_hash is not None
    assert not state_dict.get("suno_auto_lyrics_pending")
    assert state_dict.get("suno_current_req_id") is None

    card_text = _find_card_text(bot)
    assert "Источник текста: <i>🧾 Мой текст</i>" in card_text
