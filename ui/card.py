from __future__ import annotations

from html import escape
from typing import Iterable, Sequence

from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode

__all__ = ["build_card"]


def build_card(
    title: str,
    subtitle: str,
    rows: Sequence[Sequence[InlineKeyboardButton]],
    *,
    body_lines: Sequence[str] | None = None,
) -> dict:
    """Build a structured card payload for menu-like screens."""

    safe_rows: list[list[InlineKeyboardButton]] = [list(row) for row in rows]
    markup = InlineKeyboardMarkup(safe_rows)

    lines: list[str] = []
    if title:
        lines.append(f"<b>{escape(title)}</b>")
    if subtitle:
        lines.append(f"<i>{escape(subtitle)}</i>")

    extras: Iterable[str] = body_lines or ()
    clean_extras = [escape(line) for line in extras if isinstance(line, str) and line.strip()]
    if clean_extras:
        if subtitle:
            lines.append("")
        lines.extend(clean_extras)

    text = "\n".join(lines)

    return {
        "text": text,
        "reply_markup": markup,
        "parse_mode": ParseMode.HTML,
        "disable_web_page_preview": True,
    }
