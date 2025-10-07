from __future__ import annotations

import re
from typing import Optional

__all__ = ["normalize_btn_text"]


_LEADING_SYMBOLS_RE = re.compile(r"^[\W_]+", re.UNICODE)
_MULTI_SPACE_RE = re.compile(r"\s+")


def normalize_btn_text(value: Optional[str]) -> str:
    """Normalize button text for routing purposes."""

    if not value:
        return ""

    text = str(value).strip()
    if not text:
        return ""

    text = _LEADING_SYMBOLS_RE.sub("", text)
    if not text:
        return ""

    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text.lower()
