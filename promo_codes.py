"""Utilities for working with promo codes."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict

log = logging.getLogger(__name__)


# Default promo codes that ship with the bot. They can be overridden via
# environment variables or an external file – see :func:`load_promo_codes`.
DEFAULT_PROMO_CODES: Dict[str, int] = {
    "WELCOME50": 50,
    "FREE10": 10,
    "LABACCENT100": 100,
    "BONUS50": 50,
    "FRIENDS150": 150,
}


_WHITESPACE_RE = re.compile(r"\s+", re.UNICODE)


def normalize_promo_code(code: str | None) -> str:
    """Return a canonical representation of a promo code.

    The function removes surrounding whitespace, collapses inner whitespace and
    forces uppercase letters. Invisible whitespace characters (like zero-width
    spaces) are also stripped. The returned string can be safely used as a key
    in dictionaries, Redis or the database.
    """

    if not code:
        return ""
    text = str(code).strip()
    if not text:
        return ""
    # Remove any whitespace characters (including zero-width spaces) to avoid
    # issues when users copy promo codes from formatted messages.
    text = _WHITESPACE_RE.sub("", text)
    text = text.replace("\u200b", "")
    return text.upper()


def _parse_mapping_blob(blob: str, source: str) -> Dict[str, Any]:
    """Parse promo codes from a blob of text.

    The function tries JSON first and falls back to ``CODE=AMOUNT`` pairs
    separated by newlines, commas or semicolons. Whitespace is ignored.
    """

    text = (blob or "").strip()
    if not text:
        return {}

    # JSON format is the most explicit one. Expected structure: {"CODE": 10}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict):
        return parsed

    result: Dict[str, Any] = {}
    # Accept "CODE=10", "CODE:10" or "CODE 10" separated by newline/comma/semicolon.
    for raw in re.split(r"[\n,;]+", text):
        entry = raw.strip()
        if not entry:
            continue
        for delimiter in ("=", ":"):
            if delimiter in entry:
                code, value = entry.split(delimiter, 1)
                break
        else:
            parts = entry.split()
            if len(parts) == 2:
                code, value = parts
            else:
                log.warning("Ignoring malformed promo definition '%s' from %s", entry, source)
                continue
        result[code.strip()] = value.strip()
    return result


def _load_from_file(path: str) -> Dict[str, Any]:
    file_path = Path(path).expanduser()
    try:
        content = file_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        log.warning("Promo codes file '%s' not found", file_path)
        return {}
    except OSError as exc:
        log.warning("Cannot read promo codes file '%s': %s", file_path, exc)
        return {}
    return _parse_mapping_blob(content, f"file:{file_path}")


def load_promo_codes(defaults: Dict[str, int] | None = None) -> Dict[str, int]:
    """Load promo codes from defaults, environment variables and optional files."""

    merged: Dict[str, int] = {}

    def _apply(mapping: Dict[str, Any] | None, source: str) -> None:
        if not mapping:
            return
        for raw_code, raw_value in mapping.items():
            normalized = normalize_promo_code(raw_code)
            if not normalized:
                log.warning("Ignoring empty promo code from %s", source)
                continue
            try:
                amount = int(str(raw_value).strip())
            except (TypeError, ValueError):
                log.warning(
                    "Ignoring promo code '%s' from %s: invalid amount %r",
                    raw_code,
                    source,
                    raw_value,
                )
                continue
            if amount <= 0:
                log.warning(
                    "Ignoring promo code '%s' from %s: non-positive amount %s",
                    raw_code,
                    source,
                    amount,
                )
                continue
            merged[normalized] = amount

    _apply(defaults, "defaults")

    env_json = os.getenv("PROMO_CODES_JSON")
    if env_json:
        _apply(_parse_mapping_blob(env_json, "env:PROMO_CODES_JSON"), "env:PROMO_CODES_JSON")

    env_plain = os.getenv("PROMO_CODES")
    if env_plain:
        _apply(_parse_mapping_blob(env_plain, "env:PROMO_CODES"), "env:PROMO_CODES")

    file_path = os.getenv("PROMO_CODES_FILE")
    if file_path:
        _apply(_load_from_file(file_path), f"file:{file_path}")

    return merged


__all__ = ["DEFAULT_PROMO_CODES", "load_promo_codes", "normalize_promo_code"]

