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


REDIS_PREFIX = (os.getenv("REDIS_PREFIX") or "suno:prod").strip() or "suno:prod"
SUNO_LOG_KEY = f"{REDIS_PREFIX}:suno:logs"

# Suno HTTP configuration --------------------------------------------------
SUNO_API_BASE = os.getenv("SUNO_API_BASE", "https://api.kie.ai")
SUNO_API_TOKEN = os.getenv("SUNO_API_TOKEN")
SUNO_CALLBACK_SECRET = os.getenv("SUNO_CALLBACK_SECRET")
try:
    SUNO_TIMEOUT_SEC = int(os.getenv("SUNO_TIMEOUT_SEC", "75"))
except ValueError:
    SUNO_TIMEOUT_SEC = 75
try:
    SUNO_MAX_RETRIES = int(os.getenv("SUNO_MAX_RETRIES", "5"))
except ValueError:
    SUNO_MAX_RETRIES = 5

SUNO_GEN_PATH = _get_env("SUNO_GEN_PATH", "/api/v1/generate/music")
SUNO_TASK_STATUS_PATH = _get_env("SUNO_TASK_STATUS_PATH", "/api/v1/generate/record-info")
SUNO_WAV_PATH = _get_env("SUNO_WAV_PATH", "/api/v1/wav/generate")
SUNO_WAV_INFO_PATH = _get_env("SUNO_WAV_INFO_PATH", "/api/v1/wav/record-info")
SUNO_MP4_PATH = _get_env("SUNO_MP4_PATH", "/api/v1/mp4/generate")
SUNO_MP4_INFO_PATH = _get_env("SUNO_MP4_INFO_PATH", "/api/v1/mp4/record-info")
SUNO_STEM_PATH = _get_env("SUNO_STEM_PATH", "/api/v1/vocal-removal/generate")
SUNO_STEM_INFO_PATH = _get_env("SUNO_STEM_INFO_PATH", "/api/v1/vocal-removal/record-info")
SUNO_LYRICS_PATH = _get_env("SUNO_LYRICS_PATH", "/api/v1/generate/get-timestamped-lyrics")
SUNO_UPLOAD_EXTEND_PATH = _get_env("SUNO_UPLOAD_EXTEND_PATH", "/api/v1/generate/upload-extend")
SUNO_COVER_INFO_PATH = _get_env("SUNO_COVER_INFO_PATH", "/api/v1/suno/cover/record-info")

SUNO_MODEL = _get_env("SUNO_MODEL")
SUNO_CALLBACK_URL = _get_env("SUNO_CALLBACK_URL")
