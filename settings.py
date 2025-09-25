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
SUNO_API_BASE = _get_env("SUNO_API_BASE", "https://api.kie.ai")
SUNO_API_TOKEN = _get_env("SUNO_API_TOKEN", "test-token")

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
SUNO_TIMEOUT_SEC = _parse_timeout(os.getenv("SUNO_TIMEOUT_SEC") or str(60))
try:
    SUNO_HTTP_RETRIES = int(os.getenv("SUNO_HTTP_RETRIES", "4"))
except ValueError:
    SUNO_HTTP_RETRIES = 4
try:
    SUNO_RETRY_BACKOFF_BASE = float(os.getenv("SUNO_RETRY_BACKOFF_BASE", "0.6"))
except ValueError:
    SUNO_RETRY_BACKOFF_BASE = 0.6
SUNO_CALLBACK_URL = _get_env("SUNO_CALLBACK_URL")
SUNO_CALLBACK_SECRET = _get_env("SUNO_CALLBACK_SECRET")
