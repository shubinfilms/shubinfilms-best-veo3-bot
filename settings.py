"""Compatibility layer exporting configuration attributes."""

from __future__ import annotations

from core.settings import (
    configuration_summary_json,
    reload_settings as _reload_core_settings,
    resolve_outbound_ip,
    token_tail,
)

settings = _reload_core_settings()


def _populate_from_settings() -> None:
    g = globals()

    g["LOG_LEVEL"] = settings.LOG_LEVEL
    g["LOG_JSON"] = bool(settings.LOG_JSON)
    g["MAX_IN_LOG_BODY"] = int(settings.MAX_IN_LOG_BODY)

    g["HTTP_TIMEOUT_CONNECT"] = float(settings.HTTP_TIMEOUT_CONNECT)
    g["HTTP_TIMEOUT_READ"] = float(settings.HTTP_TIMEOUT_READ)
    g["HTTP_TIMEOUT_TOTAL"] = float(settings.HTTP_TIMEOUT_TOTAL_EFFECTIVE)
    g["HTTP_RETRY_ATTEMPTS"] = int(settings.HTTP_RETRY_ATTEMPTS_EFFECTIVE)
    g["HTTP_POOL_CONNECTIONS"] = int(settings.HTTP_POOL_CONNECTIONS)
    g["HTTP_POOL_PER_HOST"] = int(settings.HTTP_POOL_PER_HOST)
    g["TMP_CLEANUP_HOURS"] = int(settings.TMP_CLEANUP_HOURS)

    g["REDIS_PREFIX"] = settings.REDIS_PREFIX
    g["SUNO_LOG_KEY"] = settings.SUNO_LOG_KEY

    g["SUPPORT_USERNAME"] = settings.SUPPORT_USERNAME
    g["SUPPORT_USER_ID"] = int(settings.SUPPORT_USER_ID)
    g["BOT_USERNAME"] = settings.BOT_USERNAME
    g["BOT_NAME"] = settings.BOT_NAME or settings.BOT_USERNAME or ""
    g["REF_BONUS_HINT_ENABLED"] = bool(settings.REF_BONUS_HINT_ENABLED)

    g["KIE_BASE_URL"] = settings.KIE_BASE_URL
    g["KIE_API_KEY"] = settings.KIE_API_KEY
    g["KIE_GEN_PATH"] = settings.KIE_GEN_PATH
    g["KIE_STATUS_PATH"] = settings.KIE_STATUS_PATH
    g["KIE_HD_PATH"] = settings.KIE_HD_PATH

    g["TELEGRAM_TOKEN"] = settings.TELEGRAM_TOKEN
    g["REDIS_URL"] = settings.REDIS_URL

    g["SUNO_API_BASE"] = settings.SUNO_API_BASE
    g["SUNO_API_TOKEN"] = settings.SUNO_API_TOKEN
    g["SUNO_CALLBACK_SECRET"] = settings.SUNO_CALLBACK_SECRET
    g["SUNO_CALLBACK_URL"] = settings.SUNO_CALLBACK_URL
    g["SUNO_TIMEOUT_SEC"] = int(round(settings.HTTP_TIMEOUT_TOTAL_EFFECTIVE))
    g["SUNO_MAX_RETRIES"] = int(settings.HTTP_RETRY_ATTEMPTS_EFFECTIVE)
    g["SUNO_ENABLED"] = bool(settings.SUNO_ENABLED)
    g["SUNO_GEN_PATH"] = settings.SUNO_GEN_PATH
    g["SUNO_TASK_STATUS_PATH"] = settings.SUNO_TASK_STATUS_PATH
    g["SUNO_WAV_PATH"] = settings.SUNO_WAV_PATH
    g["SUNO_WAV_INFO_PATH"] = settings.SUNO_WAV_INFO_PATH
    g["SUNO_MP4_PATH"] = settings.SUNO_MP4_PATH
    g["SUNO_MP4_INFO_PATH"] = settings.SUNO_MP4_INFO_PATH
    g["SUNO_STEM_PATH"] = settings.SUNO_STEM_PATH
    g["SUNO_STEM_INFO_PATH"] = settings.SUNO_STEM_INFO_PATH
    g["SUNO_LYRICS_PATH"] = settings.SUNO_LYRICS_PATH
    g["SUNO_UPLOAD_EXTEND_PATH"] = settings.SUNO_UPLOAD_EXTEND_PATH
    g["SUNO_COVER_INFO_PATH"] = settings.SUNO_COVER_INFO_PATH
    g["SUNO_INSTR_PATH"] = settings.SUNO_INSTR_PATH
    g["SUNO_VOCAL_PATH"] = settings.SUNO_VOCAL_PATH
    g["SUNO_MODEL"] = settings.SUNO_MODEL
    g["SUNO_READY"] = bool(settings.SUNO_READY)

    g["UPLOAD_BASE_URL"] = settings.UPLOAD_BASE_URL_EFFECTIVE
    g["UPLOAD_STREAM_PATH"] = settings.UPLOAD_STREAM_PATH
    g["UPLOAD_URL_PATH"] = settings.UPLOAD_URL_PATH
    g["UPLOAD_BASE64_PATH"] = settings.UPLOAD_BASE64_PATH
    g["UPLOAD_FALLBACK_ENABLED"] = bool(settings.UPLOAD_FALLBACK_ENABLED)

    g["TOPUP_URL"] = settings.TOPUP_URL or ""
    g["STARS_PAYMENT_URL"] = settings.STARS_PAYMENT_URL or ""
    g["CARD_PAYMENT_URL"] = settings.CARD_PAYMENT_URL or ""

    g["YOOKASSA_SHOP_ID"] = settings.YOOKASSA_SHOP_ID
    g["YOOKASSA_SECRET_KEY"] = settings.YOOKASSA_SECRET_KEY
    g["YOOKASSA_RETURN_URL"] = settings.YOOKASSA_RETURN_URL
    g["YOOKASSA_CURRENCY"] = settings.YOOKASSA_CURRENCY
    g["CRYPTO_PAYMENT_URL"] = settings.CRYPTO_PAYMENT_URL or ""

    g["FEATURE_SORA2_ENABLED"] = bool(settings.FEATURE_SORA2_ENABLED)
    g["SORA2_ENABLED"] = bool(settings.SORA2_ENABLED)
    g["SORA2_API_KEY"] = settings.SORA2_API_KEY_EFFECTIVE
    g["SORA2_GEN_PATH"] = settings.SORA2_GEN_PATH
    g["SORA2_STATUS_PATH"] = settings.SORA2_STATUS_PATH
    g["SORA2_WAIT_STICKER_ID"] = int(settings.SORA2_WAIT_STICKER_ID)
    g["SORA2_TIMEOUT_CONNECT"] = int(settings.SORA2_TIMEOUT_CONNECT)
    g["SORA2_TIMEOUT_READ"] = int(settings.SORA2_TIMEOUT_READ)
    g["SORA2_TIMEOUT_WRITE"] = int(settings.SORA2_TIMEOUT_WRITE)
    g["SORA2_TIMEOUT_POOL"] = int(settings.SORA2_TIMEOUT_POOL)
    g["SORA2_PRICE"] = int(settings.SORA2_PRICE)
    g["SORA2_ALLOWED_AR"] = set(settings.SORA2_ALLOWED_AR)
    g["SORA2_MAX_DURATION"] = int(settings.SORA2_MAX_DURATION)
    g["SORA2_QUEUE_KEY"] = settings.SORA2_QUEUE_KEY
    g["KIA_SORA2_TIMEOUT"] = int(settings.KIA_SORA2_TIMEOUT)
    g["KIA_SORA2_RETRY"] = int(settings.KIA_SORA2_RETRY)
    g["SORA2"] = settings.sora2_payload_defaults()

    g["VEO_WAIT_STICKER_ID"] = int(settings.VEO_WAIT_STICKER_ID)
    g["SUNO_WAIT_STICKER_ID"] = int(settings.SUNO_WAIT_STICKER_ID)
    g["MJ_WAIT_STICKER_ID"] = int(settings.MJ_WAIT_STICKER_ID)
    g["PROMPTMASTER_WAIT_STICKER_ID"] = int(settings.PROMPTMASTER_WAIT_STICKER_ID)
    g["PROMO_OK_STICKER_ID"] = int(settings.PROMO_OK_STICKER_ID)
    g["PURCHASE_OK_STICKER_ID"] = int(settings.PURCHASE_OK_STICKER_ID)

    g["BANANA_SEND_AS_DOCUMENT"] = bool(settings.BANANA_SEND_AS_DOCUMENT)
    g["MJ_SEND_AS_ALBUM"] = bool(settings.MJ_SEND_AS_ALBUM)

    g["DIALOG_ENABLED"] = settings.DIALOG_ENABLED

    g["WELCOME_BONUS_ENABLED"] = bool(settings.WELCOME_BONUS_ENABLED)
    g["BOT_SINGLETON_DISABLED"] = bool(settings.BOT_SINGLETON_DISABLED)
    g["ENABLE_VERTICAL_NORMALIZE"] = bool(settings.ENABLE_VERTICAL_NORMALIZE)
    g["FEATURE_PROFILE_SIMPLE"] = bool(settings.FEATURE_PROFILE_SIMPLE)

    g["PUBLIC_BASE_URL"] = settings.PUBLIC_BASE_URL

    g["_EXPORTED_NAMES"] = [
        name
        for name in g
        if name.isupper()
    ]


def reload_settings() -> None:
    global settings
    settings = _reload_core_settings()
    _populate_from_settings()
    return settings


_populate_from_settings()

__all__ = sorted(_EXPORTED_NAMES) + [
    "configuration_summary_json",
    "resolve_outbound_ip",
    "token_tail",
    "settings",
    "reload_settings",
]

