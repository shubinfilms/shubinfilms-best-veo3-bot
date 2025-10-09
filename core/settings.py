"""Centralised application configuration and environment validation."""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Mapping, MutableMapping, Optional

from pydantic import Field, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger("settings")


_SECRET_FIELDS = {
    "TELEGRAM_TOKEN",
    "REDIS_URL",
    "KIE_API_KEY",
    "SUNO_API_TOKEN",
    "SUNO_CALLBACK_SECRET",
    "SORA2_API_KEY",
    "YOOKASSA_SECRET_KEY",
}

_CRITICAL_ENDPOINT_FIELDS = {
    "KIE_BASE_URL",
    "KIE_GEN_PATH",
    "KIE_STATUS_PATH",
    "KIE_HD_PATH",
    "KIE_MJ_GEN_PATH",
    "KIE_MJ_STATUS_PATH",
    "SUNO_API_BASE",
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
    "SORA2_GEN_PATH",
    "SORA2_STATUS_PATH",
}


def _mask(value: Optional[str]) -> str:
    if not value:
        return ""
    text = str(value).strip()
    if len(text) <= 4:
        return text
    return f"***{text[-4:]}"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    TELEGRAM_TOKEN: str = Field(..., min_length=1)
    REDIS_URL: str = Field(..., min_length=1)
    KIE_API_KEY: str = Field(..., min_length=1)

    LOG_LEVEL: str = Field(default="INFO")
    LOG_JSON: bool = Field(default=True)
    MAX_IN_LOG_BODY: int = Field(default=2048, ge=256, le=65536)
    REDIS_PREFIX: str = Field(default="suno:prod")
    BOT_USERNAME: Optional[str] = None
    BOT_NAME: str = Field(default="")
    SUPPORT_USERNAME: str = Field(default="BestAi_Support")
    SUPPORT_USER_ID: int = Field(default=7223448532)
    REF_BONUS_HINT_ENABLED: bool = Field(default=False)

    HTTP_TIMEOUT_CONNECT: float = Field(default=10.0, ge=0.1, le=300.0)
    HTTP_TIMEOUT_READ: float = Field(default=60.0, ge=1.0, le=600.0)
    HTTP_TIMEOUT_TOTAL: float = Field(default=75.0, ge=1.0, le=900.0)
    HTTP_RETRY_ATTEMPTS: int = Field(default=3, ge=1, le=10)
    HTTP_POOL_CONNECTIONS: int = Field(default=50, ge=1, le=200)
    HTTP_POOL_PER_HOST: int = Field(default=10, ge=1, le=100)
    TMP_CLEANUP_HOURS: int = Field(default=24, ge=1, le=240)

    BANANA_SEND_AS_DOCUMENT: bool = Field(default=True)
    MJ_SEND_AS_ALBUM: bool = Field(default=True)

    KIE_BASE_URL: str = Field(default="https://api.kie.ai")
    KIE_GEN_PATH: str = Field(default="/api/v1/veo/generate")
    KIE_STATUS_PATH: str = Field(default="/api/v1/veo/record-info")
    KIE_HD_PATH: str = Field(default="/api/v1/veo/get-1080p-video")
    KIE_MJ_GEN_PATH: str = Field(default="/api/v1/mj/generate")
    KIE_MJ_STATUS_PATH: str = Field(default="/api/v1/mj/recordInfo")

    SUNO_ENABLED: bool = Field(default=False)
    SUNO_API_BASE: Optional[str] = Field(default=None)
    SUNO_API_TOKEN: Optional[str] = Field(default=None)
    SUNO_CALLBACK_SECRET: Optional[str] = Field(default=None)
    SUNO_CALLBACK_URL: Optional[str] = Field(default=None)
    SUNO_TIMEOUT_SEC: Optional[float] = Field(default=None, ge=1.0, le=600.0)
    SUNO_MAX_RETRIES: Optional[int] = Field(default=None, ge=1, le=10)
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

    PUBLIC_BASE_URL: Optional[str] = Field(default=None)

    TOPUP_URL: str = Field(default="")

    YOOKASSA_SHOP_ID: Optional[str] = Field(default=None)
    YOOKASSA_SECRET_KEY: Optional[str] = Field(default=None)
    YOOKASSA_RETURN_URL: Optional[str] = Field(default=None)
    YOOKASSA_CURRENCY: str = Field(default="RUB")
    CRYPTO_PAYMENT_URL: Optional[str] = Field(default=None)

    SORA2_ENABLED: bool = Field(default=False)
    SORA2_API_KEY: Optional[str] = Field(default=None)
    SORA2_GEN_PATH: str = Field(
        default="https://api.kie.ai/api/v1/jobs/createTask"
    )
    SORA2_STATUS_PATH: str = Field(
        default="https://api.kie.ai/api/v1/jobs/recordInfo"
    )
    SORA2_TIMEOUT_CONNECT: int = Field(default=20, ge=1, le=180)
    SORA2_TIMEOUT_READ: int = Field(default=30, ge=1, le=600)
    SORA2_TIMEOUT_WRITE: int = Field(default=30, ge=1, le=600)
    SORA2_TIMEOUT_POOL: int = Field(default=10, ge=1, le=180)

    VEO_WAIT_STICKER_ID: int = Field(default=5375464961822695044)
    SORA2_WAIT_STICKER_ID: int = Field(default=5375464961822695044)
    SUNO_WAIT_STICKER_ID: int = Field(default=5188621441926438751)
    MJ_WAIT_STICKER_ID: int = Field(default=5375074927252621134)
    PROMPTMASTER_WAIT_STICKER_ID: int = Field(default=5334882760735598374)
    PROMO_OK_STICKER_ID: int = Field(default=5199749070830197566)
    PURCHASE_OK_STICKER_ID: int = Field(default=5471952986970267163)

    WELCOME_BONUS_ENABLED: bool = Field(default=False)
    BOT_SINGLETON_DISABLED: bool = Field(default=False)
    ENABLE_VERTICAL_NORMALIZE: bool = Field(default=True)

    DIALOG_ENABLED: Optional[bool] = Field(default=None)

    # Runtime/computed attributes populated in ``model_post_init``
    SUNO_LOG_KEY: str = Field(default="", exclude=True)
    SUNO_READY: bool = Field(default=False, exclude=True)
    HTTP_TIMEOUT_TOTAL_EFFECTIVE: float = Field(default=0.0, exclude=True)
    HTTP_RETRY_ATTEMPTS_EFFECTIVE: int = Field(default=0, exclude=True)
    UPLOAD_BASE_URL_EFFECTIVE: str = Field(default="", exclude=True)
    SORA2_API_KEY_EFFECTIVE: Optional[str] = Field(default=None, exclude=True)

    @field_validator(
        "TELEGRAM_TOKEN",
        "REDIS_URL",
        "KIE_API_KEY",
        "KIE_BASE_URL",
        "KIE_GEN_PATH",
        "KIE_STATUS_PATH",
        "KIE_HD_PATH",
        "KIE_MJ_GEN_PATH",
        "KIE_MJ_STATUS_PATH",
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
        "SORA2_GEN_PATH",
        "SORA2_STATUS_PATH",
        mode="before",
    )
    def _strip_required(cls, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        return text

    @field_validator(
        "SUNO_API_BASE",
        "SUNO_API_TOKEN",
        "SUNO_CALLBACK_SECRET",
        "SUNO_CALLBACK_URL",
        "UPLOAD_BASE_URL",
        "PUBLIC_BASE_URL",
        "YOOKASSA_SHOP_ID",
        "YOOKASSA_SECRET_KEY",
        "YOOKASSA_RETURN_URL",
        "CRYPTO_PAYMENT_URL",
        "BOT_USERNAME",
        mode="before",
    )
    def _strip_optional(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator(
        "LOG_LEVEL",
        mode="before",
    )
    def _normalize_level(cls, value: Any) -> str:
        if value is None:
            return "INFO"
        text = str(value).strip().upper()
        if text not in logging._nameToLevel:  # type: ignore[attr-defined]
            return "INFO"
        return text

    @field_validator(
        "SUNO_MODEL",
        mode="before",
    )
    def _normalize_model(cls, value: Any) -> str:
        text = str(value or "V5").strip()
        if not text:
            return "V5"
        if text.lower() in {"v5", "suno-v5"}:
            return "V5"
        return text

    @field_validator(
        "SUPPORT_USERNAME",
        mode="before",
    )
    def _normalize_username(cls, value: Any) -> str:
        text = str(value or "BestAi_Support").strip()
        text = text.lstrip("@")
        return text or "BestAi_Support"

    @field_validator(
        "BOT_USERNAME",
        mode="after",
    )
    def _normalize_bot_username(cls, value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        return value.lstrip("@") or None

    @field_validator(
        "VEO_WAIT_STICKER_ID",
        "SORA2_WAIT_STICKER_ID",
        "SUNO_WAIT_STICKER_ID",
        "MJ_WAIT_STICKER_ID",
        "PROMPTMASTER_WAIT_STICKER_ID",
        "PROMO_OK_STICKER_ID",
        "PURCHASE_OK_STICKER_ID",
        mode="before",
    )
    def _coerce_sticker(cls, value: Any) -> int:
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            raise ValueError("Sticker identifiers must be integers")

    @field_validator("DIALOG_ENABLED", mode="before")
    def _parse_optional_bool(cls, value: Any) -> Optional[bool]:
        if value in (None, ""):
            return None
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
        return None

    @model_validator(mode="after")
    def _post_init(self) -> "Settings":
        for field in ("TELEGRAM_TOKEN", "REDIS_URL", "KIE_API_KEY"):
            value = getattr(self, field)
            if not value:
                msg = f"Missing required environment variable: {field}"
                logger.error(msg)
                raise RuntimeError(msg)

        self.KIE_BASE_URL = self.KIE_BASE_URL.rstrip("/")

        if not self.SUNO_API_BASE:
            self.SUNO_API_BASE = self.KIE_BASE_URL
        self.SUNO_API_BASE = self.SUNO_API_BASE.rstrip("/")

        self.SUNO_API_TOKEN = self.SUNO_API_TOKEN or self.KIE_API_KEY
        self.SUNO_CALLBACK_URL = (self.SUNO_CALLBACK_URL or "").rstrip("/") or None

        if self.PUBLIC_BASE_URL:
            self.PUBLIC_BASE_URL = self.PUBLIC_BASE_URL.rstrip("/")

        if self.UPLOAD_BASE_URL:
            self.UPLOAD_BASE_URL_EFFECTIVE = self.UPLOAD_BASE_URL.rstrip("/")
        else:
            self.UPLOAD_BASE_URL_EFFECTIVE = self.SUNO_API_BASE

        self.SUNO_LOG_KEY = f"{self.REDIS_PREFIX}:suno:logs"

        total_timeout = self.SUNO_TIMEOUT_SEC or self.HTTP_TIMEOUT_TOTAL
        if total_timeout < max(self.HTTP_TIMEOUT_CONNECT, self.HTTP_TIMEOUT_READ):
            total_timeout = max(self.HTTP_TIMEOUT_CONNECT, self.HTTP_TIMEOUT_READ)
        self.HTTP_TIMEOUT_TOTAL_EFFECTIVE = float(total_timeout)

        retry_attempts = self.SUNO_MAX_RETRIES or self.HTTP_RETRY_ATTEMPTS
        self.HTTP_RETRY_ATTEMPTS_EFFECTIVE = int(max(1, retry_attempts))

        self.SUNO_READY = bool(
            self.SUNO_ENABLED
            and self.SUNO_API_TOKEN
            and self.SUNO_CALLBACK_SECRET
            and self.SUNO_CALLBACK_URL
        )

        if not self.SORA2_API_KEY:
            self.SORA2_API_KEY_EFFECTIVE = self.KIE_API_KEY
        else:
            self.SORA2_API_KEY_EFFECTIVE = self.SORA2_API_KEY.strip()

        if not self.SORA2_STATUS_PATH.endswith("/recordInfo"):
            msg = (
                "SORA2_STATUS_PATH must point to /api/v1/jobs/recordInfo; "
                f"got '{self.SORA2_STATUS_PATH}'"
            )
            logger.error(msg)
            raise RuntimeError(msg)

        for field in _CRITICAL_ENDPOINT_FIELDS:
            value = getattr(self, field)
            if not value:
                msg = f"Critical endpoint '{field}' is not configured"
                logger.error(msg)
                raise RuntimeError(msg)

        return self

    def configuration_summary(self) -> Mapping[str, Any]:
        keys = {
            "REDIS_PREFIX": self.REDIS_PREFIX,
            "KIE_BASE_URL": self.KIE_BASE_URL,
            "SUNO_API_BASE": self.SUNO_API_BASE,
            "PUBLIC_BASE_URL": self.PUBLIC_BASE_URL,
            "SORA2_GEN_PATH": self.SORA2_GEN_PATH,
            "SORA2_STATUS_PATH": self.SORA2_STATUS_PATH,
        }
        for secret in sorted(_SECRET_FIELDS):
            value = getattr(self, secret, None)
            keys[secret] = _mask(value)
        return keys

    def critical_variables(self) -> Mapping[str, str]:
        data: MutableMapping[str, str] = {}
        for field in (
            "TELEGRAM_TOKEN",
            "REDIS_URL",
            "KIE_API_KEY",
            "SORA2_GEN_PATH",
            "SORA2_STATUS_PATH",
        ):
            value = getattr(self, field, "")
            data[field] = _mask(value) if field in _SECRET_FIELDS else str(value)
        for field in sorted(_CRITICAL_ENDPOINT_FIELDS):
            value = getattr(self, field, "")
            data[field] = _mask(value) if field in _SECRET_FIELDS else str(value)
        return data

    def sora2_payload_defaults(self) -> Mapping[str, Any]:
        return {
            "GEN_PATH": self.SORA2_GEN_PATH,
            "STATUS_PATH": self.SORA2_STATUS_PATH,
            "CALLBACK_URL": f"{self.PUBLIC_BASE_URL}/sora2-callback"
            if self.PUBLIC_BASE_URL
            else None,
            "API_KEY": self.SORA2_API_KEY_EFFECTIVE,
        }

    def token_tail(self, token: Optional[str]) -> str:
        if not token:
            return ""
        text = token.strip()
        if len(text) <= 4:
            return text
        return text[-4:]


def _load_settings() -> Settings:
    try:
        return Settings()
    except ValidationError as exc:  # pragma: no cover - fail fast
        errors = []
        for entry in exc.errors():
            loc = "::".join(str(part) for part in entry.get("loc", ()))
            msg = entry.get("msg", "invalid value")
            errors.append(f"{loc}: {msg}")
        message = "Invalid configuration: " + ", ".join(errors)
        logger.error(message)
        raise RuntimeError(message) from exc


settings = _load_settings()


def reload_settings() -> Settings:
    """Reload settings from the environment and update module globals."""

    global settings
    settings = _load_settings()
    return settings


_OUTBOUND_IP: Optional[str] = None
_OUTBOUND_LOCK = threading.Lock()


def resolve_outbound_ip(*, force: bool = False, timeout: float = 5.0) -> Optional[str]:
    """Resolve the container outbound IP address with caching."""

    global _OUTBOUND_IP
    if not force and _OUTBOUND_IP:
        return _OUTBOUND_IP
    with _OUTBOUND_LOCK:
        if not force and _OUTBOUND_IP:
            return _OUTBOUND_IP
        url = os.getenv("OUTBOUND_IP_ECHO_URL") or "https://api.ipify.org"
        try:
            import httpx

            response = httpx.get(url, timeout=max(1.0, timeout))
            response.raise_for_status()
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("outbound ip detection failed", extra={"error": str(exc)})
            return _OUTBOUND_IP
        ip = (response.text or "").strip()
        if not ip:
            logger.warning("outbound ip detection returned empty response")
            return _OUTBOUND_IP
        _OUTBOUND_IP = ip
        return _OUTBOUND_IP


def token_tail(token: Optional[str]) -> str:
    return settings.token_tail(token)


def configuration_summary_json() -> str:
    return json.dumps(settings.configuration_summary(), ensure_ascii=False)


__all__ = [
    "settings",
    "resolve_outbound_ip",
    "token_tail",
    "configuration_summary_json",
    "reload_settings",
]

