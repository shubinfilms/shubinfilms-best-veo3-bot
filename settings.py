"""Shared configuration constants for Redis and Suno integrations."""

from __future__ import annotations

import logging
import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

_VALID_LEVELS = {name for name in logging._nameToLevel if isinstance(name, str)}

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
if LOG_LEVEL not in _VALID_LEVELS:
    LOG_LEVEL = "INFO"


def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def _parse_timeout(raw: Optional[str]) -> int:
    if not raw:
        return 45
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        return 45
    return max(1, min(value, 300))


_default_prefix = "veo3:prod"
_env_prefix = _get_env("REDIS_PREFIX", "") or ""
REDIS_PREFIX = _env_prefix or _default_prefix
SUNO_LOG_KEY = f"{REDIS_PREFIX}:suno:logs"

# Suno HTTP configuration --------------------------------------------------
SUNO_API_BASE = _get_env("SUNO_API_BASE")
SUNO_API_TOKEN = _get_env("SUNO_API_TOKEN")

SUNO_GEN_PATH = _get_env("SUNO_GEN_PATH")
SUNO_INSTR_PATH = _get_env("SUNO_INSTR_PATH")
SUNO_UPLOAD_EXTEND_PATH = _get_env("SUNO_UPLOAD_EXTEND_PATH")
SUNO_TASK_STATUS_PATH = _get_env("SUNO_TASK_STATUS_PATH")
SUNO_LYRICS_PATH = _get_env("SUNO_LYRICS_PATH")
SUNO_WAV_PATH = _get_env("SUNO_WAV_PATH")
SUNO_WAV_INFO_PATH = _get_env("SUNO_WAV_INFO_PATH")
SUNO_MP4_PATH = _get_env("SUNO_MP4_PATH")
SUNO_MP4_INFO_PATH = _get_env("SUNO_MP4_INFO_PATH")
SUNO_STEM_PATH = _get_env("SUNO_STEM_PATH")
SUNO_STEM_INFO_PATH = _get_env("SUNO_STEM_INFO_PATH")

SUNO_MODEL = _get_env("SUNO_MODEL")
SUNO_TIMEOUT_SEC = _parse_timeout(os.getenv("SUNO_TIMEOUT_SEC"))
SUNO_CALLBACK_URL = _get_env("SUNO_CALLBACK_URL")
SUNO_CALLBACK_SECRET = _get_env("SUNO_CALLBACK_SECRET")
