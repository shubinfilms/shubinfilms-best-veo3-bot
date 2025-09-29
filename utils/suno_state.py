from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, MutableMapping, Optional

import html
import re


TITLE_MAX_LENGTH = 300
STYLE_MAX_LENGTH = 500
LYRICS_MAX_LENGTH = 2000
LYRICS_PREVIEW_LIMIT = 160
_STORAGE_KEY = "suno_state"


_SPACE_RE = re.compile(r"\s+")
_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _collapse_spaces(value: str) -> str:
    return _SPACE_RE.sub(" ", value).strip()


def _strip_html(value: str, *, keep_newlines: bool = True) -> str:
    if not value:
        return ""
    text = str(value)
    if keep_newlines:
        text = _BR_RE.sub("\n", text)
    else:
        text = _BR_RE.sub(" ", text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = html.unescape(text)
    return text


def _normalize_multiline(value: str) -> str:
    normalized_lines: list[str] = []
    stripped = _strip_html(value, keep_newlines=True)
    for raw_line in stripped.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        collapsed = _collapse_spaces(raw_line)
        if collapsed:
            normalized_lines.append(collapsed)
    return "\n".join(normalized_lines).strip()


def _normalize_lyrics(value: str) -> str:
    cleaned = _strip_html(value, keep_newlines=True)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in cleaned.split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines).strip()


def _apply_limit(value: str, max_length: int) -> str:
    if len(value) <= max_length:
        return value
    return value[: max_length].rstrip()


def _clean_title(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = _strip_html(str(value), keep_newlines=False)
    text = _collapse_spaces(text.replace("\n", " "))
    if not text:
        return None
    return _apply_limit(text, TITLE_MAX_LENGTH)


def _clean_style(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = _normalize_multiline(str(value))
    if not text:
        return None
    return _apply_limit(text, STYLE_MAX_LENGTH)


def _clean_lyrics(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = _normalize_lyrics(str(value))
    if not text:
        return None
    return _apply_limit(text, LYRICS_MAX_LENGTH)


@dataclass
class SunoState:
    mode: Literal["instrumental", "lyrics"] = "instrumental"
    title: Optional[str] = None
    style: Optional[str] = None
    lyrics: Optional[str] = None
    card_message_id: Optional[int] = None
    card_text_hash: Optional[str] = None
    card_markup_hash: Optional[str] = None
    card_chat_id: Optional[int] = None
    last_card_hash: Optional[str] = None

    @property
    def has_lyrics(self) -> bool:
        return self.mode == "lyrics"

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "title": self.title,
            "style": self.style,
            "lyrics": self.lyrics,
            "has_lyrics": self.has_lyrics,
            "card_message_id": self.card_message_id,
            "card_text_hash": self.card_text_hash,
            "card_markup_hash": self.card_markup_hash,
            "card_chat_id": self.card_chat_id,
            "last_card_hash": self.last_card_hash,
        }


def _from_mapping(payload: Mapping[str, Any]) -> SunoState:
    raw_mode = payload.get("mode")
    mode: Literal["instrumental", "lyrics"]
    if isinstance(raw_mode, str) and raw_mode in {"instrumental", "lyrics"}:
        mode = raw_mode  # type: ignore[assignment]
    else:
        mode = "lyrics" if bool(payload.get("has_lyrics")) else "instrumental"
    state = SunoState(mode=mode)
    set_title(state, payload.get("title"))
    set_style(state, payload.get("style"))
    set_lyrics(state, payload.get("lyrics"))
    raw_msg_id = payload.get("card_message_id")
    if isinstance(raw_msg_id, int):
        state.card_message_id = raw_msg_id
    raw_text_hash = payload.get("card_text_hash")
    if isinstance(raw_text_hash, str):
        state.card_text_hash = raw_text_hash
    raw_markup_hash = payload.get("card_markup_hash")
    if isinstance(raw_markup_hash, str):
        state.card_markup_hash = raw_markup_hash
    raw_chat_id = payload.get("card_chat_id")
    if isinstance(raw_chat_id, int):
        state.card_chat_id = raw_chat_id
    raw_last_hash = payload.get("last_card_hash")
    if isinstance(raw_last_hash, str):
        state.last_card_hash = raw_last_hash
    return state


def load(ctx: Any) -> SunoState:
    user_data = getattr(ctx, "user_data", None)
    if isinstance(user_data, MutableMapping):
        raw = user_data.get(_STORAGE_KEY)
        if isinstance(raw, Mapping):
            return _from_mapping(raw)
        legacy = user_data.get("suno")
        if isinstance(legacy, Mapping):
            return _from_mapping(legacy)
    return SunoState()


def save(ctx: Any, state: SunoState) -> None:
    user_data = getattr(ctx, "user_data", None)
    if not isinstance(user_data, MutableMapping):
        return
    user_data[_STORAGE_KEY] = state.to_dict()


def set_title(state: SunoState, value: Optional[str]) -> SunoState:
    state.title = _clean_title(value)
    return state


def clear_title(state: SunoState) -> SunoState:
    state.title = None
    return state


def set_style(state: SunoState, value: Optional[str]) -> SunoState:
    state.style = _clean_style(value)
    return state


def clear_style(state: SunoState) -> SunoState:
    state.style = None
    return state


def set_lyrics(state: SunoState, value: Optional[str]) -> SunoState:
    state.lyrics = _clean_lyrics(value)
    return state


def clear_lyrics(state: SunoState) -> SunoState:
    state.lyrics = None
    return state


def style_preview(value: Optional[str], limit: int = 120) -> str:
    if not value:
        return ""
    text = value.strip()
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 1)].rstrip() + "…"


def lyrics_preview(value: Optional[str], limit: int = LYRICS_PREVIEW_LIMIT) -> str:
    if not value:
        return ""
    first_line: Optional[str] = None
    for raw_line in value.split("\n"):
        collapsed = _collapse_spaces(raw_line)
        if collapsed:
            first_line = collapsed
            break
    if not first_line:
        return ""
    if len(first_line) <= limit:
        return first_line
    return first_line[: max(1, limit - 1)].rstrip() + "…"


def build_generation_payload(
    state: SunoState,
    *,
    model: str,
    lang: Optional[str] = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "title": state.title or "",
        "style": state.style or "",
        "instrumental": not state.has_lyrics,
        "has_lyrics": state.has_lyrics,
    }
    tags: list[str] = []
    if state.style:
        for raw in re.split(r"[\s,]+", state.style):
            tag = raw.strip().strip("#")
            if not tag:
                continue
            lowered = tag.lower()
            if lowered not in tags:
                tags.append(lowered)
    payload["tags"] = tags
    if state.has_lyrics:
        payload["lyrics"] = state.lyrics or ""
    if lang:
        payload["lang"] = lang
    if state.style:
        payload["prompt"] = state.style
    elif state.lyrics and state.has_lyrics:
        payload["prompt"] = state.lyrics
    elif state.title:
        payload["prompt"] = state.title
    return payload


def sanitize_payload_for_log(payload: Mapping[str, Any]) -> dict[str, Any]:
    preview: dict[str, Any] = {}
    for key in ("title", "style", "lyrics", "instrumental", "has_lyrics", "lang"):
        if key not in payload:
            continue
        value = payload[key]
        if isinstance(value, str):
            preview[key] = style_preview(value, limit=120)
        else:
            preview[key] = value
    return preview


__all__ = [
    "LYRICS_MAX_LENGTH",
    "LYRICS_PREVIEW_LIMIT",
    "STYLE_MAX_LENGTH",
    "TITLE_MAX_LENGTH",
    "SunoState",
    "build_generation_payload",
    "clear_lyrics",
    "clear_style",
    "clear_title",
    "lyrics_preview",
    "load",
    "save",
    "sanitize_payload_for_log",
    "set_lyrics",
    "set_style",
    "set_title",
    "style_preview",
]
