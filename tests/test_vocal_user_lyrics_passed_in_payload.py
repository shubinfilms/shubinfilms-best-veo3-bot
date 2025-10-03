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
    build_generation_payload,
    set_lyrics,
    set_lyrics_source,
    set_style,
    set_title,
)


def test_vocal_user_lyrics_passed_in_payload() -> None:
    state = SunoState(mode="lyrics")
    set_title(state, "Skyline")
    set_style(state, "Alt pop")
    set_lyrics_source(state, LyricsSource.USER)
    set_lyrics(state, "Line one\nLine two")

    payload = build_generation_payload(state, model="V5", lang="ru")

    assert payload.get("lyrics") == "Line one\nLine two"
    assert payload.get("lyrics_source") == LyricsSource.USER.value
    assert payload.get("prompt") == "Line one\nLine two"

    set_lyrics_source(state, LyricsSource.AI)
    payload_ai = build_generation_payload(state, model="V5", lang="ru")

    assert "lyrics" not in payload_ai or payload_ai.get("lyrics") in (None, "")
    assert payload_ai.get("lyrics_source") == LyricsSource.AI.value
    assert payload_ai.get("prompt") == "Alt pop"
