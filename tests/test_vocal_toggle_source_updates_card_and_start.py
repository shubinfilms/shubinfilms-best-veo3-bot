import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("SUNO_API_BASE", "https://example.com")
os.environ.setdefault("SUNO_API_TOKEN", "token")
os.environ.setdefault("SUNO_CALLBACK_URL", "https://callback.example")
os.environ.setdefault("SUNO_CALLBACK_SECRET", "secret")

from texts import t
from ui_helpers import render_suno_card
from utils.suno_state import (
    LyricsSource,
    SunoState,
    set_lyrics,
    set_lyrics_source,
    set_style,
    set_title,
)


def test_vocal_toggle_source_updates_card_and_start() -> None:
    state = SunoState(mode="lyrics")
    set_title(state, "Orbit")
    set_style(state, "Synthwave")

    text_ai, _, ready_ai = render_suno_card(
        state,
        price=30,
        balance=None,
        generating=False,
        waiting_enqueue=False,
    )
    assert t("suno.lyrics_source.ai") in text_ai
    assert ready_ai is True

    set_lyrics_source(state, LyricsSource.USER)
    text_user, _, ready_user = render_suno_card(
        state,
        price=30,
        balance=None,
        generating=False,
        waiting_enqueue=False,
    )
    assert t("suno.lyrics_source.user") in text_user
    assert ready_user is False

    set_lyrics(state, "First line")
    _, _, ready_final = render_suno_card(
        state,
        price=30,
        balance=None,
        generating=False,
        waiting_enqueue=False,
    )
    assert ready_final is True
