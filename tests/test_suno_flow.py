from types import SimpleNamespace

import pytest

from ui_helpers import render_suno_card
from utils.suno_state import (
    SunoState,
    build_generation_payload,
    clear_style,
    clear_title,
    load,
    save,
    set_lyrics,
    set_style,
    set_title,
)


def _render(state: SunoState, *, price: int = 30, balance: int | None = None):
    text, markup = render_suno_card(
        state,
        price=price,
        balance=balance,
        generating=False,
    )
    return text, markup


def test_set_title_updates_card_text() -> None:
    state = SunoState()
    set_title(state, "  Test Track  ")
    text, _ = _render(state)
    assert "Название: Test Track" in text


def test_clear_title_displays_dash() -> None:
    state = SunoState()
    set_title(state, "Some Title")
    clear_title(state)
    text, _ = _render(state)
    assert "Название: —" in text


def test_set_style_reflects_in_card() -> None:
    state = SunoState()
    set_style(state, "Lo-fi chill, piano")
    text, _ = _render(state)
    assert "Стиль: Lo-fi chill, piano" in text


def test_lyrics_mode_shows_button_and_preview() -> None:
    state = SunoState(mode="lyrics")
    long_text = "\n".join(["Line" * 10 for _ in range(40)])
    set_lyrics(state, long_text)
    text, markup = _render(state)
    assert "Текст:" in text
    assert any(
        button.text == "📝 Текст песни"
        for row in markup.inline_keyboard
        for button in row
    )


def test_instrumental_mode_hides_lyrics_and_payload() -> None:
    state = SunoState(mode="lyrics")
    set_lyrics(state, "Verse one\nVerse two")
    state.mode = "instrumental"
    text, markup = _render(state)
    assert "Текст:" not in text
    assert all(
        button.text != "📝 Текст песни"
        for row in markup.inline_keyboard
        for button in row
    )
    payload = build_generation_payload(state, model="V5", lang="ru")
    assert payload["instrumental"] is True
    assert "lyrics" not in payload


def test_state_persistence_allows_cancel_flow() -> None:
    ctx = SimpleNamespace(user_data={})
    state = load(ctx)
    set_title(state, "Keep")
    save(ctx, state)
    reloaded = load(ctx)
    assert reloaded.title == "Keep"
    clear_title(reloaded)
    clear_style(reloaded)
    save(ctx, reloaded)
    assert load(ctx).title is None


@pytest.mark.parametrize("mode", ["instrumental", "lyrics"])
def test_generation_payload_contains_expected_fields(mode: str) -> None:
    state = SunoState(mode=mode)
    set_title(state, "My Song")
    set_style(state, "Dreamy pop")
    if mode == "lyrics":
        set_lyrics(state, "Sing about the stars")
    payload = build_generation_payload(state, model="V5", lang="en")
    assert payload["model"] == "V5"
    assert payload["title"] == "My Song"
    assert payload["style"] == "Dreamy pop"
    if mode == "lyrics":
        assert payload["lyrics"] == "Sing about the stars"
        assert payload["instrumental"] is False
        assert payload["has_lyrics"] is True
    else:
        assert payload["instrumental"] is True
        assert payload["has_lyrics"] is False
        assert "lyrics" not in payload
