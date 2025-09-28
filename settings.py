"""Shared configuration constants for Redis and Suno integrations."""
from __future__ import annotations

import logging
import os
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

load_dotenv()

_VALID_LEVELS = {name for name in logging._nameToLevel if isinstance(name, str)}


class _AppSettings(BaseModel):
    """Validated configuration loaded from environment variables."""

    model_config = ConfigDict(extra="ignore")

    LOG_LEVEL: str = Field(default="INFO")
    LOG_JSON: bool = Field(default=True)
    MAX_IN_LOG_BODY: int = Field(default=2048, ge=256, le=65536)

    HTTP_TIMEOUT_CONNECT: float = Field(default=10.0, ge=0.1, le=300.0)
    HTTP_TIMEOUT_READ: float = Field(default=60.0, ge=1.0, le=600.0)
    HTTP_TIMEOUT_TOTAL: float = Field(default=75.0, ge=1.0, le=900.0)
    HTTP_RETRY_ATTEMPTS: int = Field(default=3, ge=1, le=10)
    HTTP_POOL_CONNECTIONS: int = Field(default=50, ge=1, le=200)
    HTTP_POOL_PER_HOST: int = Field(default=10, ge=1, le=100)

    TMP_CLEANUP_HOURS: int = Field(default=24, ge=1, le=240)

    SUNO_ENABLED: bool = Field(default=False)
    SUNO_API_BASE: Optional[str] = Field(default="https://api.kie.ai")
    SUNO_API_TOKEN: Optional[str] = Field(default=None)
    SUNO_CALLBACK_SECRET: Optional[str] = Field(default=None)
    SUNO_CALLBACK_URL: Optional[str] = Field(default=None)
    SUNO_TIMEOUT_SEC: Optional[float] = Field(default=None)
    SUNO_MAX_RETRIES: Optional[int] = Field(default=None)

    @field_validator("LOG_LEVEL", mode="before")
    def _normalize_level(cls, value: object) -> str:
        if value is None:
            return "INFO"
        text = str(value).strip().upper()
        if text not in _VALID_LEVELS:
            return "INFO"
        return text

    @field_validator(
        "SUNO_API_BASE",
        "SUNO_API_TOKEN",
        "SUNO_CALLBACK_SECRET",
        "SUNO_CALLBACK_URL",
        mode="before",
    )
    def _strip_optional(cls, value: object) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @model_validator(mode="after")
    def _validate_timeouts(self) -> "_AppSettings":
        max_required = max(self.HTTP_TIMEOUT_CONNECT, self.HTTP_TIMEOUT_READ)
        if self.HTTP_TIMEOUT_TOTAL < max_required:
            raise ValueError("HTTP_TIMEOUT_TOTAL must be >= connect/read timeouts")
        return self

    @model_validator(mode="after")
    def _validate_suno(self) -> "_AppSettings":
        if self.SUNO_ENABLED:
            missing = [
                name
                for name in ("SUNO_API_BASE", "SUNO_API_TOKEN", "SUNO_CALLBACK_SECRET", "SUNO_CALLBACK_URL")
                if not getattr(self, name)
            ]
            if missing:
                joined = ", ".join(missing)
                raise ValueError(f"SUNO_ENABLED=true requires {joined}")
        return self


def _load_settings() -> _AppSettings:
    values: dict[str, str] = {}
    for field in _AppSettings.model_fields:
        if field in os.environ:
            values[field] = os.environ[field]
    try:
        return _AppSettings(**values)
    except ValidationError as exc:  # pragma: no cover - fail fast if config invalid
        raise RuntimeError(f"Invalid configuration: {exc}") from exc


_APP_SETTINGS = _load_settings()

LOG_LEVEL = _APP_SETTINGS.LOG_LEVEL
LOG_JSON = bool(_APP_SETTINGS.LOG_JSON)
MAX_IN_LOG_BODY = int(_APP_SETTINGS.MAX_IN_LOG_BODY)

HTTP_TIMEOUT_CONNECT = float(_APP_SETTINGS.HTTP_TIMEOUT_CONNECT)
HTTP_TIMEOUT_READ = float(_APP_SETTINGS.HTTP_TIMEOUT_READ)
_total_timeout = (
    float(_APP_SETTINGS.SUNO_TIMEOUT_SEC)
    if _APP_SETTINGS.SUNO_TIMEOUT_SEC is not None
    else float(_APP_SETTINGS.HTTP_TIMEOUT_TOTAL)
)
if _total_timeout < max(HTTP_TIMEOUT_CONNECT, HTTP_TIMEOUT_READ):
    _total_timeout = max(HTTP_TIMEOUT_CONNECT, HTTP_TIMEOUT_READ)
HTTP_TIMEOUT_TOTAL = float(_total_timeout)
HTTP_RETRY_ATTEMPTS = int(
    _APP_SETTINGS.SUNO_MAX_RETRIES
    if _APP_SETTINGS.SUNO_MAX_RETRIES is not None
    else _APP_SETTINGS.HTTP_RETRY_ATTEMPTS
)
HTTP_POOL_CONNECTIONS = int(_APP_SETTINGS.HTTP_POOL_CONNECTIONS)
HTTP_POOL_PER_HOST = int(_APP_SETTINGS.HTTP_POOL_PER_HOST)
TMP_CLEANUP_HOURS = int(_APP_SETTINGS.TMP_CLEANUP_HOURS)

REDIS_PREFIX = (os.getenv("REDIS_PREFIX") or "suno:prod").strip() or "suno:prod"
SUNO_LOG_KEY = f"{REDIS_PREFIX}:suno:logs"

SUNO_API_BASE = (_APP_SETTINGS.SUNO_API_BASE or "https://api.kie.ai").rstrip("/")
SUNO_API_TOKEN = _APP_SETTINGS.SUNO_API_TOKEN
SUNO_CALLBACK_SECRET = _APP_SETTINGS.SUNO_CALLBACK_SECRET
SUNO_CALLBACK_URL = _APP_SETTINGS.SUNO_CALLBACK_URL
SUNO_TIMEOUT_SEC = int(round(HTTP_TIMEOUT_TOTAL))
SUNO_MAX_RETRIES = max(1, HTTP_RETRY_ATTEMPTS)
SUNO_ENABLED = bool(_APP_SETTINGS.SUNO_ENABLED)


def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


SUNO_GEN_PATH = _get_env("SUNO_GEN_PATH", "/api/v1/suno/generate/music")
SUNO_TASK_STATUS_PATH = _get_env("SUNO_TASK_STATUS_PATH", "/api/v1/suno/record-info")
SUNO_WAV_PATH = _get_env("SUNO_WAV_PATH", "/api/v1/wav/generate")
SUNO_WAV_INFO_PATH = _get_env("SUNO_WAV_INFO_PATH", "/api/v1/wav/record-info")
SUNO_MP4_PATH = _get_env("SUNO_MP4_PATH", "/api/v1/mp4/generate")
SUNO_MP4_INFO_PATH = _get_env("SUNO_MP4_INFO_PATH", "/api/v1/mp4/record-info")
SUNO_STEM_PATH = _get_env("SUNO_STEM_PATH", "/api/v1/vocal-removal/generate")
SUNO_STEM_INFO_PATH = _get_env("SUNO_STEM_INFO_PATH", "/api/v1/vocal-removal/record-info")
SUNO_LYRICS_PATH = _get_env("SUNO_LYRICS_PATH", "/api/v1/generate/get-timestamped-lyrics")
SUNO_UPLOAD_EXTEND_PATH = _get_env("SUNO_UPLOAD_EXTEND_PATH", "/api/v1/suno/upload-extend")
SUNO_COVER_INFO_PATH = _get_env("SUNO_COVER_INFO_PATH", "/api/v1/suno/cover/record-info")
SUNO_INSTR_PATH = _get_env("SUNO_INSTR_PATH", "/api/v1/generate/add-instrumental")

SUNO_MODEL = _get_env("SUNO_MODEL")

__all__ = [
    "LOG_LEVEL",
    "LOG_JSON",
    "MAX_IN_LOG_BODY",
    "HTTP_TIMEOUT_CONNECT",
    "HTTP_TIMEOUT_READ",
    "HTTP_TIMEOUT_TOTAL",
    "HTTP_RETRY_ATTEMPTS",
    "HTTP_POOL_CONNECTIONS",
    "HTTP_POOL_PER_HOST",
    "TMP_CLEANUP_HOURS",
    "REDIS_PREFIX",
    "SUNO_LOG_KEY",
    "SUNO_API_BASE",
    "SUNO_API_TOKEN",
    "SUNO_CALLBACK_SECRET",
    "SUNO_CALLBACK_URL",
    "SUNO_TIMEOUT_SEC",
    "SUNO_MAX_RETRIES",
    "SUNO_GEN_PATH",
    "SUNO_TASK_STATUS_PATH",
    "SUNO_WAV_PATH",
    "SUNO_WAV_INFO_PATH",
    "SUNO_MP4_PATH",
    "SUNO_MP4_INFO_PATH",
    "SUNO_STEM_PATH",
    "SUNO_STEM_INFO_PATH",
    "SUNO_LYRICS_PATH",
    "SUNO_UPLOAD_EXTEND_PATH",
    "SUNO_COVER_INFO_PATH",
    "SUNO_INSTR_PATH",
    "SUNO_MODEL",
    "SUNO_ENABLED",
]
