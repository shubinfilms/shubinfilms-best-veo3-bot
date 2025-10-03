"""Shared configuration constants for Redis and Suno integrations."""
from __future__ import annotations

import logging
import os
import threading
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

load_dotenv()

log = logging.getLogger("config")
_CONFIG_WARNINGS: list[str] = []


def _push_warning(message: str) -> None:
    if message not in _CONFIG_WARNINGS:
        _CONFIG_WARNINGS.append(message)

_VALID_LEVELS = {name for name in logging._nameToLevel if isinstance(name, str)}


class _AppSettings(BaseModel):
    """Validated configuration loaded from environment variables."""

    model_config = ConfigDict(extra="ignore")

    LOG_LEVEL: str = Field(default="INFO")
    LOG_JSON: bool = Field(default=True)
    MAX_IN_LOG_BODY: int = Field(default=2048, ge=256, le=65536)
    SUPPORT_USERNAME: str = Field(default="BestAi_Support")
    SUPPORT_USER_ID: int = Field(default=7223448532)

    HTTP_TIMEOUT_CONNECT: float = Field(default=10.0, ge=0.1, le=300.0)
    HTTP_TIMEOUT_READ: float = Field(default=60.0, ge=1.0, le=600.0)
    HTTP_TIMEOUT_TOTAL: float = Field(default=75.0, ge=1.0, le=900.0)
    HTTP_RETRY_ATTEMPTS: int = Field(default=3, ge=1, le=10)
    HTTP_POOL_CONNECTIONS: int = Field(default=50, ge=1, le=200)
    HTTP_POOL_PER_HOST: int = Field(default=10, ge=1, le=100)

    BANANA_SEND_AS_DOCUMENT: bool = Field(default=True)
    MJ_SEND_AS_ALBUM: bool = Field(default=True)

    TMP_CLEANUP_HOURS: int = Field(default=24, ge=1, le=240)

    KIE_BASE_URL: str = Field(default="https://api.kie.ai")
    KIE_API_KEY: Optional[str] = Field(default=None)
    PUBLIC_BASE_URL: Optional[str] = Field(default=None)

    SUNO_ENABLED: bool = Field(default=False)
    SUNO_API_BASE: Optional[str] = Field(default="https://api.kie.ai")
    SUNO_API_TOKEN: Optional[str] = Field(default=None)
    SUNO_CALLBACK_SECRET: Optional[str] = Field(default=None)
    SUNO_CALLBACK_URL: Optional[str] = Field(default=None)
    SUNO_TIMEOUT_SEC: Optional[float] = Field(default=None)
    SUNO_MAX_RETRIES: Optional[int] = Field(default=None)
    SUNO_GEN_PATH: str = Field(default="/api/v1/generate")
    SUNO_TASK_STATUS_PATH: str = Field(default="/api/v1/generate/record-info")
    SUNO_WAV_PATH: str = Field(default="/api/v1/wav/generate")
    SUNO_WAV_INFO_PATH: str = Field(default="/api/v1/wav/record-info")
    SUNO_MP4_PATH: str = Field(default="/api/v1/mp4/generate")
    SUNO_MP4_INFO_PATH: str = Field(default="/api/v1/mp4/record-info")
    SUNO_STEM_PATH: str = Field(default="/api/v1/vocal-removal/generate")
    SUNO_STEM_INFO_PATH: str = Field(default="/api/v1/vocal-removal/record-info")
    SUNO_LYRICS_PATH: str = Field(default="/api/v1/generate/get-timestamped-lyrics")
    SUNO_UPLOAD_EXTEND_PATH: str = Field(default="/api/v1/suno/upload-extend")
    SUNO_COVER_INFO_PATH: str = Field(default="/api/v1/suno/cover/record-info")
    SUNO_INSTR_PATH: str = Field(default="/api/v1/generate/add-instrumental")
    SUNO_VOCAL_PATH: str = Field(default="/api/v1/generate/add-vocals")
    SUNO_MODEL: str = Field(default="V5")
    UPLOAD_BASE_URL: Optional[str] = Field(default=None)
    UPLOAD_STREAM_PATH: str = Field(default="/api/v1/upload/stream")
    UPLOAD_URL_PATH: str = Field(default="/api/v1/upload/url")
    UPLOAD_BASE64_PATH: str = Field(default="/api/v1/upload/base64")
    UPLOAD_FALLBACK_ENABLED: bool = Field(default=False)

    SORA2_ENABLED: bool = Field(default=False)
    SORA2_API_KEY: Optional[str] = Field(default=None)
    SORA2_GEN_PATH: str = Field(default="https://api.kie.ai/api/v1/jobs/createTask")
    SORA2_STATUS_PATH: str = Field(default="https://api.kie.ai/api/v1/jobs/queryTask")
    SORA2_WAIT_STICKER_ID: str = Field(default="5375464961822695044")
    SORA2_TIMEOUT_CONNECT: int = Field(default=20, ge=1, le=120)
    SORA2_TIMEOUT_READ: int = Field(default=30, ge=1, le=300)
    SORA2_TIMEOUT_WRITE: int = Field(default=30, ge=1, le=300)
    SORA2_TIMEOUT_POOL: int = Field(default=10, ge=1, le=120)

    YOOKASSA_SHOP_ID: Optional[str] = Field(default=None)
    YOOKASSA_SECRET_KEY: Optional[str] = Field(default=None)
    YOOKASSA_RETURN_URL: Optional[str] = Field(default=None)
    YOOKASSA_CURRENCY: str = Field(default="RUB")

    @field_validator("LOG_LEVEL", mode="before")
    def _normalize_level(cls, value: object) -> str:
        if value is None:
            return "INFO"
        text = str(value).strip().upper()
        if text not in _VALID_LEVELS:
            return "INFO"
        return text

    @field_validator(
        "KIE_BASE_URL",
        "KIE_API_KEY",
        "PUBLIC_BASE_URL",
        "SUNO_API_BASE",
        "SUNO_API_TOKEN",
        "SUNO_CALLBACK_SECRET",
        "SUNO_CALLBACK_URL",
        "YOOKASSA_SHOP_ID",
        "YOOKASSA_SECRET_KEY",
        "YOOKASSA_RETURN_URL",
        mode="before",
    )
    def _strip_optional(cls, value: object) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator(
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
        "SUNO_VOCAL_PATH",
        mode="before",
    )
    def _normalize_path(cls, value: object) -> str:
        text = str(value or "").strip()
        if not text:
            return "/"
        if text.startswith("http://") or text.startswith("https://"):
            return text
        if not text.startswith("/"):
            text = f"/{text}"
        while "//" in text:
            text = text.replace("//", "/")
        return text

    @field_validator("SUNO_MODEL", mode="before")
    def _normalize_model(cls, value: object) -> str:
        text = str(value or "V5").strip()
        if not text:
            return "V5"
        lowered = text.lower()
        if lowered in {"v5", "suno-v5"}:
            return "V5"
        return text

    @model_validator(mode="after")
    def _validate_timeouts(self) -> "_AppSettings":
        max_required = max(self.HTTP_TIMEOUT_CONNECT, self.HTTP_TIMEOUT_READ)
        if self.HTTP_TIMEOUT_TOTAL < max_required:
            raise ValueError("HTTP_TIMEOUT_TOTAL must be >= connect/read timeouts")
        return self

    @model_validator(mode="after")
    def _validate_suno(self) -> "_AppSettings":
        if not self.SUNO_API_BASE:
            self.SUNO_API_BASE = self.KIE_BASE_URL
        if not self.SUNO_API_TOKEN and self.KIE_API_KEY:
            self.SUNO_API_TOKEN = self.KIE_API_KEY
        if self.SUNO_ENABLED:
            missing = [
                name
                for name in ("SUNO_API_TOKEN", "SUNO_CALLBACK_SECRET", "SUNO_CALLBACK_URL")
                if not getattr(self, name)
            ]
            if missing:
                joined = ", ".join(missing)
                _push_warning(
                    f"SUNO_ENABLED=true but missing configuration: {joined}"
                )
        return self

    @field_validator("SUPPORT_USERNAME", mode="before")
    def _normalize_support_username(cls, value: object) -> str:
        text = str(value or "BestAi_Support").strip()
        text = text.lstrip("@")
        return text or "BestAi_Support"


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
UPLOAD_FALLBACK_ENABLED = bool(_APP_SETTINGS.UPLOAD_FALLBACK_ENABLED)
SUPPORT_USERNAME = _APP_SETTINGS.SUPPORT_USERNAME
SUPPORT_USER_ID = int(_APP_SETTINGS.SUPPORT_USER_ID)


def _strip_optional(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = value.strip()
    return text or None


KIE_BASE_URL = (_strip_optional(_APP_SETTINGS.KIE_BASE_URL) or "https://api.kie.ai").rstrip("/")
KIE_API_KEY = _strip_optional(_APP_SETTINGS.KIE_API_KEY)
PUBLIC_BASE_URL = _strip_optional(_APP_SETTINGS.PUBLIC_BASE_URL)
if PUBLIC_BASE_URL:
    PUBLIC_BASE_URL = PUBLIC_BASE_URL.rstrip("/")

_suno_base_candidate = _APP_SETTINGS.SUNO_API_BASE or KIE_BASE_URL
SUNO_API_BASE = (_strip_optional(_suno_base_candidate) or KIE_BASE_URL).rstrip("/")
SUNO_API_TOKEN = _strip_optional(_APP_SETTINGS.SUNO_API_TOKEN or KIE_API_KEY)
SUNO_CALLBACK_SECRET = _strip_optional(_APP_SETTINGS.SUNO_CALLBACK_SECRET)
SUNO_CALLBACK_URL = _strip_optional(_APP_SETTINGS.SUNO_CALLBACK_URL)
SUNO_TIMEOUT_SEC = int(round(HTTP_TIMEOUT_TOTAL))
SUNO_MAX_RETRIES = max(1, HTTP_RETRY_ATTEMPTS)
SUNO_ENABLED = bool(_APP_SETTINGS.SUNO_ENABLED)

YOOKASSA_SHOP_ID = _strip_optional(_APP_SETTINGS.YOOKASSA_SHOP_ID)
YOOKASSA_SECRET_KEY = _strip_optional(_APP_SETTINGS.YOOKASSA_SECRET_KEY)
YOOKASSA_RETURN_URL = _strip_optional(_APP_SETTINGS.YOOKASSA_RETURN_URL)
YOOKASSA_CURRENCY = (
    (_APP_SETTINGS.YOOKASSA_CURRENCY or "RUB").strip() or "RUB"
)

SUNO_GEN_PATH = _APP_SETTINGS.SUNO_GEN_PATH
SUNO_TASK_STATUS_PATH = _APP_SETTINGS.SUNO_TASK_STATUS_PATH
SUNO_WAV_PATH = _APP_SETTINGS.SUNO_WAV_PATH
SUNO_WAV_INFO_PATH = _APP_SETTINGS.SUNO_WAV_INFO_PATH
SUNO_MP4_PATH = _APP_SETTINGS.SUNO_MP4_PATH
SUNO_MP4_INFO_PATH = _APP_SETTINGS.SUNO_MP4_INFO_PATH
SUNO_STEM_PATH = _APP_SETTINGS.SUNO_STEM_PATH
SUNO_STEM_INFO_PATH = _APP_SETTINGS.SUNO_STEM_INFO_PATH
SUNO_LYRICS_PATH = _APP_SETTINGS.SUNO_LYRICS_PATH
SUNO_UPLOAD_EXTEND_PATH = _APP_SETTINGS.SUNO_UPLOAD_EXTEND_PATH
SUNO_COVER_INFO_PATH = _APP_SETTINGS.SUNO_COVER_INFO_PATH
SUNO_INSTR_PATH = _APP_SETTINGS.SUNO_INSTR_PATH
SUNO_VOCAL_PATH = _APP_SETTINGS.SUNO_VOCAL_PATH
SUNO_MODEL = _APP_SETTINGS.SUNO_MODEL or "V5"

UPLOAD_BASE_URL = (
    _strip_optional(_APP_SETTINGS.UPLOAD_BASE_URL) or SUNO_API_BASE
).rstrip("/")
UPLOAD_STREAM_PATH = _APP_SETTINGS.UPLOAD_STREAM_PATH
UPLOAD_URL_PATH = _APP_SETTINGS.UPLOAD_URL_PATH
UPLOAD_BASE64_PATH = _APP_SETTINGS.UPLOAD_BASE64_PATH
SUNO_READY = bool(
    SUNO_ENABLED and SUNO_API_TOKEN and SUNO_CALLBACK_SECRET and SUNO_CALLBACK_URL
)

SORA2_ENABLED = bool(_APP_SETTINGS.SORA2_ENABLED)
SORA2_GEN_PATH = _APP_SETTINGS.SORA2_GEN_PATH
SORA2_STATUS_PATH = _APP_SETTINGS.SORA2_STATUS_PATH
SORA2_API_KEY = _strip_optional(_APP_SETTINGS.SORA2_API_KEY) or KIE_API_KEY
SORA2_WAIT_STICKER_ID = (
    str(_APP_SETTINGS.SORA2_WAIT_STICKER_ID or "5375464961822695044").strip()
    or "5375464961822695044"
)
SORA2 = {
    "GEN_PATH": SORA2_GEN_PATH,
    "STATUS_PATH": SORA2_STATUS_PATH,
    "CALLBACK_URL": f"{PUBLIC_BASE_URL}/sora2-callback" if PUBLIC_BASE_URL else None,
    "API_KEY": SORA2_API_KEY,
}
SORA2_TIMEOUT_CONNECT = int(_APP_SETTINGS.SORA2_TIMEOUT_CONNECT)
SORA2_TIMEOUT_READ = int(_APP_SETTINGS.SORA2_TIMEOUT_READ)
SORA2_TIMEOUT_WRITE = int(_APP_SETTINGS.SORA2_TIMEOUT_WRITE)
SORA2_TIMEOUT_POOL = int(_APP_SETTINGS.SORA2_TIMEOUT_POOL)

BANANA_SEND_AS_DOCUMENT = bool(_APP_SETTINGS.BANANA_SEND_AS_DOCUMENT)
MJ_SEND_AS_ALBUM = bool(_APP_SETTINGS.MJ_SEND_AS_ALBUM)


def _emit_warnings() -> None:
    if not KIE_API_KEY:
        _push_warning("KIE_API_KEY is not configured")
    for name, value in {
        "SUNO_API_TOKEN": SUNO_API_TOKEN,
        "SUNO_CALLBACK_SECRET": SUNO_CALLBACK_SECRET,
        "SUNO_CALLBACK_URL": SUNO_CALLBACK_URL,
    }.items():
        if not value:
            _push_warning(f"{name} is not configured")
    for message in _CONFIG_WARNINGS:
        log.warning("config warning: %s", message)


_emit_warnings()


_OUTBOUND_IP: Optional[str] = None
_OUTBOUND_LOCK = threading.Lock()


def resolve_outbound_ip(*, force: bool = False, timeout: float = 5.0) -> Optional[str]:
    """Resolve and cache the container outbound IP address."""

    global _OUTBOUND_IP
    if not force and _OUTBOUND_IP:
        return _OUTBOUND_IP
    with _OUTBOUND_LOCK:
        if not force and _OUTBOUND_IP:
            return _OUTBOUND_IP
        url = os.getenv("OUTBOUND_IP_ECHO_URL") or "https://api.ipify.org"
        try:
            import httpx

            with httpx.Client(timeout=timeout) as client:
                response = client.get(url, params={"format": "text"})
                response.raise_for_status()
        except Exception as exc:  # pragma: no cover - network best effort
            log.warning("outbound ip detection failed: %s", exc)
            return _OUTBOUND_IP
        ip = (response.text or "").strip()
        if not ip:
            log.warning("outbound ip detection returned empty response")
            return _OUTBOUND_IP
        _OUTBOUND_IP = ip
        return _OUTBOUND_IP


def token_tail(token: Optional[str]) -> str:
    """Return masked token tail for logging/diagnostics."""

    if not token:
        return ""
    text = token.strip()
    if len(text) <= 4:
        return text
    return text[-4:]

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
    "KIE_BASE_URL",
    "KIE_API_KEY",
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
    "SUNO_VOCAL_PATH",
    "SUNO_MODEL",
    "SUNO_READY",
    "SUNO_ENABLED",
    "UPLOAD_BASE_URL",
    "UPLOAD_STREAM_PATH",
    "UPLOAD_URL_PATH",
    "UPLOAD_BASE64_PATH",
    "YOOKASSA_SHOP_ID",
    "YOOKASSA_SECRET_KEY",
    "YOOKASSA_RETURN_URL",
    "YOOKASSA_CURRENCY",
    "BANANA_SEND_AS_DOCUMENT",
    "MJ_SEND_AS_ALBUM",
    "SORA2_ENABLED",
    "SORA2_API_KEY",
    "SORA2_GEN_PATH",
    "SORA2_STATUS_PATH",
    "SORA2_WAIT_STICKER_ID",
    "SORA2",
    "SORA2_TIMEOUT_CONNECT",
    "SORA2_TIMEOUT_READ",
    "SORA2_TIMEOUT_WRITE",
    "SORA2_TIMEOUT_POOL",
    "resolve_outbound_ip",
    "token_tail",
]
