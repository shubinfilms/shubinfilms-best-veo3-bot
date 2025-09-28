from __future__ import annotations

import re
import unicodedata
from typing import Any, Mapping, MutableMapping, Optional

TITLE_MAX_LENGTH = 60
STYLE_MAX_LENGTH = 500
STYLE_PREVIEW_LIMIT = 120
_DEFAULT_INSTRUMENTAL_PROMPT = "instrumental, cinematic, modern, dynamic"
_DEFAULT_VOCAL_PROMPT = "pop, modern, upbeat"

# Unicode zero-width and formatting characters that should be stripped
_INVISIBLE_RE = re.compile(r"[\u200B\u200C\u200D\uFEFF]")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


_DEFAULT_STATE: dict[str, Any] = {
    "title": None,
    "style": None,
    "lyrics": None,
    "mode": "instrumental",
    "has_lyrics": False,
}


def ensure_suno_state(container: MutableMapping[str, Any]) -> dict[str, Any]:
    """Ensure that the mutable mapping contains a SUNO state dictionary."""

    state = container.get("suno")
    if not isinstance(state, dict):
        state = {}
        container["suno"] = state
    for key, value in _DEFAULT_STATE.items():
        state.setdefault(key, value)
    return state


def _strip_html(text: str) -> str:
    return _HTML_TAG_RE.sub("", text)


def _strip_invisible(text: str) -> str:
    cleaned = _INVISIBLE_RE.sub("", text)
    return "".join(
        ch
        for ch in cleaned
        if (ch == "\n")
        or (ch == "\r")
        or (ch == "\t")
        or not unicodedata.category(ch).startswith("C")
    )


def _collapse_spaces(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def _normalize_whitespace_lines(text: str) -> str:
    normalized: list[str] = []
    for line in text.splitlines():
        collapsed = _WHITESPACE_RE.sub(" ", line).strip()
        if collapsed:
            normalized.append(collapsed)
    return "\n".join(normalized).strip()


def _collapse_emoji_runs(text: str, *, max_run: int = 2) -> str:
    result: list[str] = []
    last = ""
    run = 0
    for ch in text:
        code = ord(ch)
        is_emoji = (
            0x1F000 <= code <= 0x1FAFF
            or 0x2600 <= code <= 0x27BF
            or 0x1F900 <= code <= 0x1F9FF
        )
        if is_emoji and ch == last:
            run += 1
            if run >= max_run:
                continue
        else:
            run = 0
        result.append(ch)
        last = ch
    return "".join(result)


def sanitize_title(raw: str) -> str:
    text = _strip_html(raw)
    text = _strip_invisible(text)
    text = _collapse_spaces(text)
    if len(text) > TITLE_MAX_LENGTH:
        return ""
    return text


def sanitize_style(raw: str) -> str:
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = _strip_html(text)
    text = _strip_invisible(text)
    text = _collapse_emoji_runs(text)
    text = _normalize_whitespace_lines(text)
    if len(text) > STYLE_MAX_LENGTH:
        return ""
    return text


def process_title_input(raw: str) -> tuple[bool, Optional[str], Optional[str]]:
    stripped = raw.strip()
    if stripped in {"-", "—"}:
        return True, None, None
    sanitized = sanitize_title(stripped)
    if not sanitized:
        cleaned = _collapse_spaces(_strip_invisible(_strip_html(stripped)))
        if cleaned and len(cleaned) > TITLE_MAX_LENGTH:
            return (
                False,
                None,
                f"⚠️ Название слишком длинное — {len(cleaned)} символов. Сократите до {TITLE_MAX_LENGTH}.",
            )
        return True, None, None
    return True, sanitized, None


def process_style_input(raw: str) -> tuple[bool, Optional[str], Optional[str]]:
    stripped = raw.strip()
    if stripped in {"-", "—"}:
        return True, None, None
    sanitized = sanitize_style(raw)
    if not sanitized:
        cleaned = _normalize_whitespace_lines(_strip_invisible(_strip_html(raw)))
        if cleaned and len(cleaned) > STYLE_MAX_LENGTH:
            return (
                False,
                None,
                f"⚠️ Стиль слишком длинный — {len(cleaned)} символов. Сократите до {STYLE_MAX_LENGTH}.",
            )
        return True, None, None
    return True, sanitized, None


def style_preview(value: Optional[str], limit: int = STYLE_PREVIEW_LIMIT) -> str:
    if not value:
        return ""
    text = value.strip()
    if len(text) <= limit:
        return text
    clipped = text[: limit - 1].rstrip()
    return clipped + "…"


def build_generation_payload(
    state: Mapping[str, Any],
    *,
    instrumental: bool,
    model: str,
    lang: str,
) -> dict[str, Any]:
    title = (state.get("title") or "").strip()
    style = (state.get("style") or "").strip()
    lyrics = (state.get("lyrics") or "").strip()
    has_lyrics = bool(state.get("has_lyrics")) if not instrumental else False

    final_title = title or "Untitled Track"
    final_prompt = style or (
        _DEFAULT_INSTRUMENTAL_PROMPT if instrumental else _DEFAULT_VOCAL_PROMPT
    )

    payload: dict[str, Any] = {
        "model": model,
        "instrumental": bool(instrumental),
        "title": final_title,
        "prompt": final_prompt,
        "has_lyrics": bool(has_lyrics),
        "lang": lang or "en",
    }

    if style:
        payload["style"] = style
    if not instrumental and lyrics:
        payload["lyrics"] = lyrics
    return payload


def sanitize_payload_for_log(payload: Mapping[str, Any]) -> dict[str, Any]:
    preview: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, str):
            preview[key] = style_preview(value, limit=120)
        else:
            preview[key] = value
    return preview


__all__ = [
    "TITLE_MAX_LENGTH",
    "STYLE_MAX_LENGTH",
    "STYLE_PREVIEW_LIMIT",
    "ensure_suno_state",
    "process_title_input",
    "process_style_input",
    "sanitize_title",
    "sanitize_style",
    "style_preview",
    "build_generation_payload",
    "sanitize_payload_for_log",
]
