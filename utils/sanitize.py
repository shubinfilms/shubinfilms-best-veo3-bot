from __future__ import annotations

import html
import re
from typing import Iterable

__all__ = ["escape_html", "escape_md", "normalize_input"]

_INVALID_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")
_MD_SPECIALS = set("_*[]()~`>#+-=|{}.!")
_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"[\t\f]+")
_MULTI_SPACE_RE = re.compile(r" {2,}")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def _strip_invalid(text: str) -> str:
    return _INVALID_SURROGATE_RE.sub("", text)


def escape_html(value: str) -> str:
    text = _strip_invalid(value or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return html.escape(text)


def escape_md(value: str) -> str:
    text = _strip_invalid(value or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    escaped: list[str] = []
    for ch in text:
        if ch in _MD_SPECIALS:
            escaped.append("\\" + ch)
        else:
            escaped.append(ch)
    return "".join(escaped)


def normalize_input(value: str, *, allow_newlines: bool) -> str:
    """Normalize free-form input removing HTML artefacts."""

    text = str(value or "")
    text = _BR_RE.sub("\n" if allow_newlines else " ", text)
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not allow_newlines:
        text = text.replace("\n", " ")
    text = _WHITESPACE_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def truncate_text(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text
    return text[: max(0, max_length - 1)].rstrip()


def collapse_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def multiline_preview(lines: Iterable[str], *, limit: int = 50) -> str:
    joined = collapse_spaces(" ".join(lines))
    if len(joined) <= limit:
        return joined
    return joined[: max(1, limit - 1)].rstrip() + "…"

