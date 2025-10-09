from __future__ import annotations

from html import escape
from html.parser import HTMLParser
from typing import Iterable

from telegram.constants import ParseMode

ALLOWED: frozenset[str] = frozenset({
    "b",
    "i",
    "u",
    "s",
    "code",
    "pre",
    "a",
    "tg-spoiler",
    "blockquote",
})

_ALLOWED_ATTRS: dict[str, frozenset[str]] = {
    "a": frozenset({"href"}),
}

_MAX_TEXT_LENGTH = 3800


def tg_html_safe(text: str) -> str:
    """Return text adapted for Telegram HTML."""

    if not text:
        return ""

    safe = text.replace("<br/>", "\n").replace("<BR/>", "\n")
    if len(safe) > _MAX_TEXT_LENGTH:
        safe = safe[: _MAX_TEXT_LENGTH - 1].rstrip() + "…"
    return safe


class _ProfileHTMLSanitizer(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: Iterable[tuple[str, str | None]]) -> None:
        lower = tag.lower()
        if lower == "br":
            self._parts.append("\n")
            return
        if lower in ALLOWED:
            attr_str = self._sanitize_attrs(lower, attrs)
            self._parts.append(f"<{lower}{attr_str}>")
            return
        self._parts.append(escape(self.get_starttag_text() or "", quote=False))

    def handle_startendtag(self, tag: str, attrs: Iterable[tuple[str, str | None]]) -> None:
        lower = tag.lower()
        if lower == "br":
            self._parts.append("\n")
            return
        if lower in ALLOWED:
            attr_str = self._sanitize_attrs(lower, attrs)
            self._parts.append(f"<{lower}{attr_str}>")
            if lower not in {"code", "pre"}:
                self._parts.append(f"</{lower}>")
            return
        self._parts.append(escape(self.get_starttag_text() or "", quote=False))

    def handle_endtag(self, tag: str) -> None:
        lower = tag.lower()
        if lower == "br":
            return
        if lower in ALLOWED:
            self._parts.append(f"</{lower}>")
            return
        self._parts.append(escape(f"</{tag}>", quote=False))

    def handle_data(self, data: str) -> None:
        if data:
            self._parts.append(escape(data, quote=False))

    def handle_comment(self, data: str) -> None:
        if data:
            self._parts.append(escape(f"<!--{data}-->", quote=False))

    def get_value(self) -> str:
        return "".join(self._parts)

    @staticmethod
    def _sanitize_attrs(tag: str, attrs: Iterable[tuple[str, str | None]]) -> str:
        allowed = _ALLOWED_ATTRS.get(tag)
        if not allowed:
            return ""
        sanitized: list[str] = []
        for name, value in attrs:
            if name in allowed and value is not None:
                sanitized.append(f'{name}="{escape(value, quote=True)}"')
        if not sanitized:
            return ""
        return " " + " ".join(sanitized)


def sanitize_profile_html(text: str) -> tuple[str, ParseMode]:
    parser = _ProfileHTMLSanitizer()
    parser.feed(text or "")
    parser.close()
    safe_text = parser.get_value()
    safe_text = _ensure_length(safe_text)
    return safe_text, ParseMode.HTML


def strip_telegram_html(text: str) -> str:
    parser = _PlainTextExtractor()
    parser.feed(text or "")
    parser.close()
    return parser.get_value()


def _ensure_length(text: str) -> str:
    if len(text) <= _MAX_TEXT_LENGTH:
        return text
    plain = strip_telegram_html(text)
    if len(plain) > _MAX_TEXT_LENGTH - 1:
        plain = plain[: _MAX_TEXT_LENGTH - 1].rstrip()
        if plain:
            plain = f"{plain}…"
    return escape(plain, quote=False)


class _PlainTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: Iterable[tuple[str, str | None]]) -> None:
        if tag.lower() == "br":
            self._parts.append("\n")

    def handle_startendtag(self, tag: str, attrs: Iterable[tuple[str, str | None]]) -> None:
        if tag.lower() == "br":
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if data:
            self._parts.append(data)

    def get_value(self) -> str:
        return "".join(self._parts)
