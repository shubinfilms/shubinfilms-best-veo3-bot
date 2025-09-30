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
    set_lyrics_source,
    set_style,
    set_title,
    suno_is_ready_to_start,
)


def test_vocal_ai_source_ready_without_lyrics() -> None:
    state = SunoState(mode="lyrics")
    set_title(state, "My Track")
    set_style(state, "Dream pop")
    set_lyrics_source(state, LyricsSource.AI)

    assert suno_is_ready_to_start(state)
