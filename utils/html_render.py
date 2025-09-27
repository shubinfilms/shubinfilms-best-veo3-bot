"""Prompt-Master HTML rendering helpers."""

from __future__ import annotations

import html
import logging
import re
from html.parser import HTMLParser
from typing import Iterable, List

logger = logging.getLogger(__name__)

_ALLOWED_TAGS = {
    "b",
    "i",
    "u",
    "a",
    "code",
    "pre",
    "s",
    "blockquote",
    "br",
}
_ALLOWED_ATTRS = {"a": {"href"}}
_SELF_CLOSING_TAGS = {"br"}
_BLOCK_TAGS = {"pre", "blockquote"}

_CODE_FENCE_RE = re.compile(r"```(?:[a-z0-9_-]+)?\n(.*?)```", re.IGNORECASE | re.DOTALL)
_INLINE_CODE_RE = re.compile(r"(?<!\\)`([^`]+?)`")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_STRONG_RE = re.compile(r"__(.+?)__")
_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")
_EM_RE = re.compile(r"_(.+?)_")
_STRIKE_RE = re.compile(r"~~(.+?)~~")
_LINK_RE = re.compile(r"\[(.+?)\]\(([^\s)]+)\)")


class _HTMLSanitizer(HTMLParser):
    """HTML sanitizer that keeps only the allowed subset of tags."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.result: List[str] = []
        self._skip_depth = 0
        self._stack: List[str] = []

    # pylint: disable=too-many-branches
    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[override]
        tag = tag.lower()
        if self._skip_depth:
            self._skip_depth += 1
            return
        if tag in {"p"}:
            self._ensure_break()
            self._stack.append(tag)
            return
        if tag in {"ul", "ol"}:
            self._ensure_break()
            self._stack.append(tag)
            return
        if tag == "li":
            self._ensure_break()
            self.result.append("• ")
            self._stack.append(tag)
            return
        if tag not in _ALLOWED_TAGS:
            self._skip_depth = 1
            return
        attrs = self._filter_attrs(tag, attrs)
        attr_str = "".join(f' {name}="{html.escape(value, quote=True)}"' for name, value in attrs)
        if tag in _SELF_CLOSING_TAGS:
            self.result.append(f"<{tag}>")
            return
        self.result.append(f"<{tag}{attr_str}>")
        self._stack.append(tag)

    def handle_endtag(self, tag: str) -> None:  # type: ignore[override]
        tag = tag.lower()
        if self._skip_depth:
            self._skip_depth -= 1
            return
        if not self._stack:
            return
        top = self._stack.pop()
        if tag in {"p", "ul", "ol"}:
            self._ensure_break()
            return
        if tag == "li":
            self._ensure_break()
            return
        if top != tag:
            return
        if tag in _SELF_CLOSING_TAGS:
            return
        self.result.append(f"</{tag}>")
        if tag in _BLOCK_TAGS:
            self._ensure_break()

    def handle_data(self, data: str) -> None:  # type: ignore[override]
        if self._skip_depth:
            return
        if not data:
            return
        self.result.append(html.escape(data))

    def handle_entityref(self, name: str) -> None:  # type: ignore[override]
        if self._skip_depth:
            return
        self.result.append(f"&{name};")

    def handle_charref(self, name: str) -> None:  # type: ignore[override]
        if self._skip_depth:
            return
        self.result.append(f"&#{name};")

    def get_html(self) -> str:
        html_text = "".join(self.result)
        html_text = html_text.replace("<br />", "<br>").replace("<br/>", "<br>")
        html_text = re.sub(r"(?:<br>\s*){3,}", "<br><br>", html_text)
        return html_text.strip()

    def _ensure_break(self) -> None:
        if not self.result:
            return
        if self.result[-1].endswith("<br>"):
            return
        self.result.append("<br>")

    @staticmethod
    def _filter_attrs(tag: str, attrs) -> List[tuple[str, str]]:
        allowed = _ALLOWED_ATTRS.get(tag, set())
        filtered: List[tuple[str, str]] = []
        for name, value in attrs:
            if value is None:
                continue
            if name not in allowed:
                continue
            if tag == "a" and value.lower().startswith("javascript:"):
                continue
            filtered.append((name, value))
        return filtered


def _strip_markdown(text: str) -> str:
    return text.replace("\r\n", "\n")


def _escape_inline(text: str) -> str:
    codes: list[str] = []

    def repl_code(match: re.Match[str]) -> str:
        codes.append(match.group(1))
        return f"@@CODE{len(codes) - 1}@@"

    text = _INLINE_CODE_RE.sub(repl_code, text)
    text = html.escape(text)

    for idx, code in enumerate(codes):
        safe_code = html.escape(code)
        text = text.replace(f"@@CODE{idx}@@", f"<code>{safe_code}</code>")

    text = _BOLD_RE.sub(lambda m: f"<b>{m.group(1)}</b>", text)
    text = _STRONG_RE.sub(lambda m: f"<u>{m.group(1)}</u>", text)
    text = _ITALIC_RE.sub(lambda m: f"<i>{m.group(1)}</i>", text)
    text = _EM_RE.sub(lambda m: f"<i>{m.group(1)}</i>", text)
    text = _STRIKE_RE.sub(lambda m: f"<s>{m.group(1)}</s>", text)

    def link_repl(match: re.Match[str]) -> str:
        label, href = match.group(1), match.group(2)
        safe_label = label
        safe_href = href
        try:
            safe_label = html.escape(label)
        except Exception:  # pragma: no cover - defensive
            safe_label = label
        try:
            safe_href = html.escape(href, quote=True)
        except Exception:  # pragma: no cover - defensive
            safe_href = href
        if href.lower().startswith("javascript:"):
            return safe_label
        return f'<a href="{safe_href}">{safe_label}</a>'

    text = _LINK_RE.sub(link_repl, text)
    return text


def _markdown_to_html(markdown_text: str) -> str:
    markdown_text = _strip_markdown(markdown_text)
    parts: List[str] = []
    last_index = 0
    for match in _CODE_FENCE_RE.finditer(markdown_text):
        start, end = match.span()
        before = markdown_text[last_index:start]
        if before:
            parts.extend(_render_lines(before))
        code_content = match.group(1)
        safe_code = html.escape(code_content.strip("\n"))
        parts.append(f"<pre><code>{safe_code}</code></pre>")
        last_index = end
    remaining = markdown_text[last_index:]
    if remaining:
        parts.extend(_render_lines(remaining))
    return "".join(parts)


def _render_lines(block: str) -> Iterable[str]:
    lines = block.split("\n")
    rendered: List[str] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if rendered and not rendered[-1].endswith("<br>"):
                rendered.append("<br>")
            continue
        if line.startswith(('- ', '* ')):
            content = _escape_inline(line[2:].strip())
            rendered.append(f"• {content}<br>")
            continue
        if line.startswith('>'):
            content = _escape_inline(line[1:].strip())
            rendered.append(f"<blockquote>{content}</blockquote>")
            continue
        rendered.append(f"{_escape_inline(line)}<br>")
    return rendered


def render_pm_html(md_or_text: str) -> str:
    """Convert Markdown/HTML string to sanitized Telegram-ready HTML."""

    if not md_or_text:
        return ""
    try:
        if "<" in md_or_text and ">" in md_or_text:
            raw_html = md_or_text
        else:
            raw_html = _markdown_to_html(md_or_text)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("pm.render.error", exc_info=exc)
        raw_html = html.escape(md_or_text)

    raw_html = raw_html.replace("<br />", "<br>").replace("<br/>", "<br>")
    sanitizer = _HTMLSanitizer()
    sanitizer.feed(raw_html)
    sanitizer.close()
    result = sanitizer.get_html()
    if not result:
        return ""
    return result


def safe_lines(items: Iterable[str | None]) -> str:
    """Join text fragments ensuring safe new line placement."""

    parts: List[str] = []
    for item in items:
        if not item:
            continue
        text = str(item).strip()
        if not text:
            continue
        parts.append(text)
    return "\n".join(parts)


def html_to_plain(text: str) -> str:
    """Convert sanitized HTML into a readable plain-text string."""

    if not text:
        return ""
    normalized = text.replace("<br />", "<br>").replace("<br/>", "<br>")
    normalized = normalized.replace("<pre><code>", "\n").replace("</code></pre>", "\n")
    normalized = re.sub(r"<br>\s*", "\n", normalized)
    normalized = re.sub(r"<[^>]+>", "", normalized)
    return html.unescape(normalized).strip()


__all__ = ["render_pm_html", "safe_lines", "html_to_plain"]

