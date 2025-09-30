from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, MutableMapping, Optional

import html
import re

from suno.client import get_preset_config
from utils.suno_modes import get_mode_config as get_suno_mode_config


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
    mode: Literal["instrumental", "lyrics", "cover"] = "instrumental"
    title: Optional[str] = None
    style: Optional[str] = None
    lyrics: Optional[str] = None
    card_message_id: Optional[int] = None
    card_text_hash: Optional[str] = None
    card_markup_hash: Optional[str] = None
    card_chat_id: Optional[int] = None
    last_card_hash: Optional[str] = None
    preset: Optional[str] = None
    cover_source_url: Optional[str] = None
    cover_source_label: Optional[str] = None
    source_file_id: Optional[str] = None
    source_url: Optional[str] = None
    kie_file_id: Optional[str] = None
    start_msg_id: Optional[int] = None

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
            "preset": self.preset,
            "cover_source_url": self.cover_source_url,
            "cover_source_label": self.cover_source_label,
            "source_file_id": self.source_file_id,
            "source_url": self.source_url,
            "kie_file_id": self.kie_file_id,
            "start_msg_id": self.start_msg_id,
        }


def _from_mapping(payload: Mapping[str, Any]) -> SunoState:
    raw_mode = payload.get("mode")
    mode: Literal["instrumental", "lyrics", "cover"]
    if isinstance(raw_mode, str) and raw_mode in {"instrumental", "lyrics", "cover"}:
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
    raw_preset = payload.get("preset")
    if isinstance(raw_preset, str):
        text = raw_preset.strip().lower()
        if text:
            state.preset = text
    raw_cover_url = payload.get("cover_source_url")
    if isinstance(raw_cover_url, str):
        state.cover_source_url = raw_cover_url.strip() or None
    raw_cover_label = payload.get("cover_source_label")
    if isinstance(raw_cover_label, str):
        state.cover_source_label = raw_cover_label.strip() or None
    raw_source_file_id = payload.get("source_file_id")
    if isinstance(raw_source_file_id, str):
        state.source_file_id = raw_source_file_id.strip() or None
    raw_source_url = payload.get("source_url")
    if isinstance(raw_source_url, str):
        state.source_url = raw_source_url.strip() or None
    raw_kie_file_id = payload.get("kie_file_id")
    if isinstance(raw_kie_file_id, str):
        state.kie_file_id = raw_kie_file_id.strip() or None
    if state.source_url is None and state.cover_source_url:
        state.source_url = state.cover_source_url
    raw_start_msg_id = payload.get("start_msg_id")
    if isinstance(raw_start_msg_id, int):
        state.start_msg_id = raw_start_msg_id
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
    if state.style:
        state.preset = None
    return state


def clear_style(state: SunoState) -> SunoState:
    state.style = None
    return state


def set_cover_source(
    state: SunoState,
    url: Optional[str],
    label: Optional[str] = None,
    *,
    file_id: Optional[str] = None,
    source_url: Optional[str] = None,
    kie_file_id: Optional[str] = None,
) -> SunoState:
    state.cover_source_url = (url or "").strip() or None
    state.cover_source_label = (label or "").strip() or None
    if file_id is not None:
        text = str(file_id).strip()
        state.source_file_id = text or None
    if source_url is not None:
        text = str(source_url).strip()
        state.source_url = text or None
    elif url is not None:
        state.source_url = state.cover_source_url
    if kie_file_id is not None:
        text = str(kie_file_id).strip()
        state.kie_file_id = text or None
    return state


def clear_cover_source(state: SunoState) -> SunoState:
    state.cover_source_url = None
    state.cover_source_label = None
    state.source_file_id = None
    state.source_url = None
    state.kie_file_id = None
    return state


def reset_suno_card_state(
    state: SunoState,
    mode: Literal["instrumental", "lyrics", "cover"],
    *,
    card_message_id: Optional[int] = None,
    card_chat_id: Optional[int] = None,
) -> SunoState:
    state.mode = mode
    state.title = None
    state.style = None
    state.lyrics = None
    state.preset = None
    state.cover_source_url = None
    state.cover_source_label = None
    state.source_file_id = None
    state.source_url = None
    state.kie_file_id = None
    state.card_message_id = card_message_id
    state.card_text_hash = None
    state.card_markup_hash = None
    state.card_chat_id = card_chat_id
    state.last_card_hash = None
    state.start_msg_id = None
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
    if state.preset:
        payload["preset"] = state.preset
    if state.mode == "cover":
        payload["operationType"] = "upload-and-cover-audio"
        if state.cover_source_url:
            payload["inputAudioUrl"] = state.cover_source_url
        payload["instrumental"] = False
        payload["has_lyrics"] = False
    elif state.has_lyrics:
        payload["lyrics"] = state.lyrics or ""
    if lang:
        payload["lang"] = lang
    if state.style:
        payload["prompt"] = state.style
    elif state.lyrics and state.has_lyrics:
        payload["prompt"] = state.lyrics
    elif state.title:
        payload["prompt"] = state.title
    if state.preset:
        cfg = get_preset_config(state.preset)
        if cfg:
            preset_tags: list[str] = []
            for tag in cfg.get("tags", []):
                text = str(tag).strip().lower()
                if text and text not in preset_tags:
                    preset_tags.append(text)
            if not payload.get("tags"):
                payload["tags"] = preset_tags
            preset_negative: list[str] = []
            for tag in cfg.get("negative_tags", []):
                text = str(tag).strip().lower()
                if text and text not in preset_negative:
                    preset_negative.append(text)
            if preset_negative:
                payload["negative_tags"] = preset_negative
            if cfg.get("instrumental"):
                payload["instrumental"] = True
                payload["has_lyrics"] = False
            if not payload.get("prompt"):
                prompt_source = cfg.get("prompt") or ", ".join(cfg.get("tags", []))
                payload["prompt"] = str(prompt_source or "").strip()
            if not payload.get("title") and cfg.get("title_suggestions"):
                payload["title"] = cfg.get("title_suggestions")[0]
    if not payload.get("prompt"):
        payload["prompt"] = payload.get("title", "")
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


def suno_is_ready_to_start(state: SunoState) -> bool:
    config = get_suno_mode_config(state.mode)
    for field in config.required_fields:
        if field == "title" and not state.title:
            return False
        if field == "style" and not state.style:
            return False
        if field == "lyrics" and not state.lyrics:
            return False
        if field == "reference" and not state.kie_file_id:
            return False
    return True


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
    "suno_is_ready_to_start",
    "set_lyrics",
    "set_style",
    "set_cover_source",
    "set_title",
    "style_preview",
    "clear_cover_source",
    "reset_suno_card_state",
]
