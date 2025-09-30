import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("SUNO_API_BASE", "https://example.com")
os.environ.setdefault("SUNO_API_TOKEN", "token")
os.environ.setdefault("SUNO_CALLBACK_URL", "https://callback.example")
os.environ.setdefault("SUNO_CALLBACK_SECRET", "secret")

from utils.suno_state import (
    LyricsSource,
    SunoState,
    set_lyrics,
    set_lyrics_source,
    set_style,
    set_title,
    suno_is_ready_to_start,
)


def test_vocal_user_source_requires_lyrics() -> None:
    state = SunoState(mode="lyrics")
    set_title(state, "Ballad")
    set_style(state, "Indie pop")
    set_lyrics_source(state, LyricsSource.USER)

    assert not suno_is_ready_to_start(state)

    set_lyrics(state, "Line one\nLine two")

    assert suno_is_ready_to_start(state)
