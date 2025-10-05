# -*- coding: utf-8 -*-
# Best VEO3 Bot — PTB 21.x
# Версия: 2025-09-14r4
# Единственное изменение против прежней версии: надежная доставка VEO-видео в Telegram
# (освежение ссылки + повторная попытка + download&reupload с увеличенным таймаутом).
# Остальное (карточки, кнопки, тексты, цены, FAQ, промокоды, бонусы и т.д.) — без изменений.

# odex/fix-balance-reset-after-deploy
import logging
import os

from logging_utils import configure_logging, log_environment

os.environ.setdefault("PYTHONUNBUFFERED", "1")

configure_logging("bot")
log_environment(logging.getLogger("bot"))

import json, time, uuid, asyncio, tempfile, subprocess, re, signal, socket, hashlib, html, sys, math, random, copy, io, unicodedata
import threading
import atexit
from pathlib import Path
# main
from collections.abc import Mapping
from typing import (
    Dict,
    Any,
    Optional,
    List,
    Tuple,
    Callable,
    Awaitable,
    Union,
    MutableMapping,
    Sequence,
    Iterable,
)
from datetime import datetime, timezone
from contextlib import suppress
from urllib.parse import urlparse, urlunparse, urlencode, parse_qsl
from dataclasses import dataclass

import aiohttp
from aiohttp import ClientError, ClientResponseError, ClientTimeout
import requests
from telegram import (
    Update, InlineKeyboardButton, InlineKeyboardMarkup,
    InputFile, InputMediaVideo, LabeledPrice, ReplyKeyboardMarkup,
    KeyboardButton, BotCommand, User, Message
)
from telegram.constants import ParseMode, ChatAction
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    AIORateLimiter,
    PreCheckoutQueryHandler,
    ApplicationHandlerStop,
)
from telegram.error import BadRequest, Forbidden, RetryAfter, TimedOut, NetworkError, TelegramError

from handlers import (
    configure_faq,
    faq_callback,
    faq_command,
    help_command,
    get_pm_prompt,
    prompt_master_callback,
    prompt_master_handle_text,
    prompt_master_open,
    prompt_master_process,
    prompt_master_reset,
)

from prompt_master import (
    legacy_build_animate_prompt as build_animate_prompt,
    legacy_build_banana_json as build_banana_json,
    legacy_build_mj_json as build_mj_json,
    legacy_build_suno_prompt as build_suno_prompt,
    legacy_build_video_prompt as build_video_prompt,
)

# === KIE Banana wrapper ===
from kie_banana import (
    create_banana_task,
    wait_for_banana_result,
    KieBananaError,
    poll_veo_status,
    KIE_BAD_STATES,
    KIE_OK_STATES,
)

import redis

from ui_helpers import (
    upsert_card,
    refresh_balance_card_if_open,
    refresh_suno_card as _refresh_suno_card_raw,
    show_referral_card,
    pm_main_kb,
    pm_result_kb,
    sync_suno_start_message,
)
from stickers import delete_wait_sticker, send_ok_sticker, send_wait_sticker

from utils.suno_state import (
    LYRICS_MAX_LENGTH,
    LyricsSource,
    SunoState,
    build_generation_payload as build_suno_generation_payload,
    clear_lyrics as clear_suno_lyrics,
    clear_style as clear_suno_style,
    clear_title as clear_suno_title,
    load as load_suno_state,
    sanitize_payload_for_log,
    save as save_suno_state,
    set_lyrics as set_suno_lyrics,
    set_lyrics_source as set_suno_lyrics_source,
    set_style as set_suno_style,
    set_title as set_suno_title,
    set_cover_source as set_suno_cover_source,
    clear_cover_source as clear_suno_cover_source,
    reset_suno_card_state,
    suno_is_ready_to_start,
)

try:  # pragma: no cover - optional helper
    from utils.suno_state import style_preview as suno_style_preview
except Exception:  # pragma: no cover - defensive fallback
    suno_style_preview = None  # type: ignore[assignment]
try:  # pragma: no cover - optional helper
    from utils.suno_state import lyrics_preview as suno_lyrics_preview
except Exception:  # pragma: no cover - defensive fallback
    suno_lyrics_preview = None  # type: ignore[assignment]
from utils.suno_modes import (
    FIELD_LABELS as SUNO_FIELD_LABELS,
    default_style_text as suno_default_style_text,
    get_mode_config as get_suno_mode_config,
)
from utils.input_state import (
    WaitInputState,
    WaitKind,
    classify_wait_input,
    clear_wait_state,
    clear_wait,
    get_wait,
    input_state,
    refresh_card_pointer,
    set_wait,
    touch_wait,
)
from utils.telegram_utils import build_photo_album_media, label_to_command, should_capture_to_prompt
from utils.sanitize import collapse_spaces, normalize_input, truncate_text

from keyboards import (
    CB,
    CB_FAQ_PREFIX,
    CB_MAIN_AI_DIALOG,
    CB_MAIN_BACK,
    CB_MAIN_KNOWLEDGE,
    CB_MAIN_MUSIC,
    CB_MAIN_PHOTO,
    CB_MAIN_PROFILE,
    CB_MAIN_VIDEO,
    CB_PM_INSERT_PREFIX,
    CB_PM_PREFIX,
    CB_PROFILE_BACK,
    CB_PROFILE_TOPUP,
    CB_VIDEO_ENGINE_SORA2,
    CB_VIDEO_ENGINE_SORA2_DISABLED,
    CB_VIDEO_ENGINE_VEO,
    CB_VIDEO_BACK,
    CB_VIDEO_MENU,
    CB_VIDEO_MODE_FAST,
    CB_VIDEO_MODE_PHOTO,
    CB_VIDEO_MODE_QUALITY,
    CB_VIDEO_MODE_SORA_IMAGE,
    CB_VIDEO_MODE_SORA_TEXT,
    CB_AI_MODES,
    CB_CHAT_NORMAL,
    CB_CHAT_PROMPTMASTER,
    CB_PAY_CARD,
    CB_PAY_CRYPTO,
    CB_PAY_STARS,
    mj_upscale_root_keyboard,
    mj_upscale_select_keyboard,
    faq_keyboard,
    suno_modes_keyboard,
    suno_start_disabled_keyboard,
    kb_ai_dialog_modes,
    kb_profile_topup_entry,
    menu_pay_unified,
)
from texts import (
    SUNO_MODE_PROMPT,
    SUNO_START_READY_MESSAGE,
    SUNO_STARTING_MESSAGE,
    TXT_AI_DIALOG_NORMAL,
    TXT_AI_DIALOG_PM,
    TXT_AI_DIALOG_CHOOSE,
    TXT_CRYPTO_COMING_SOON,
    TXT_KB_AI_DIALOG,
    TXT_KB_PROFILE,
    TXT_KNOWLEDGE_INTRO,
    TXT_PROFILE_TITLE,
    TXT_TOPUP_CHOOSE,
    TXT_PAY_CRYPTO_OPEN_LINK,
    common_text,
    t,
)

from balance import ensure_tokens, insufficient_balance_keyboard
from payments.yookassa import (
    YOOKASSA_PACKS_ORDER,
    YookassaError,
    create_payment as yookassa_create_payment,
    pack_button_label as yookassa_pack_button_label,
)

from redis_utils import (
    credit,
    credit_balance,
    debit_balance,
    debit_try,
    ensure_user,
    add_user,
    clear_task_meta,
    get_balance,
    get_ledger_count,
    get_ledger_entries,
    get_all_user_ids,
    get_users_count,
    load_task_meta,
    mark_user_dead,
    mark_promo_used,
    remove_user,
    save_task_meta,
    is_promo_used,
    unmark_promo_used,
    user_exists,
    rds,
    USERS_KEY,
    get_inviter,
    set_inviter,
    add_ref_user,
    incr_ref_earned,
    get_ref_stats,
    set_last_mj_grid,
    get_last_mj_grid,
    clear_last_mj_grid,
    acquire_mj_upscale_lock,
    release_mj_upscale_lock,
    cache_get,
    cache_set,
    acquire_ttl_lock,
    release_ttl_lock,
    acquire_menu_lock,
    release_menu_lock,
    MenuLocked,
    with_menu_lock,
    acquire_sora2_lock,
    release_sora2_lock,
    mark_sora2_unavailable,
    clear_sora2_unavailable,
    is_sora2_unavailable,
    save_menu_message,
    get_menu_message,
    clear_menu_message,
    set_mj_gallery,
    get_mj_gallery,
    user_lock,
    release_user_lock,
    update_task_meta,
)

from ledger import (
    LedgerStorage,
    LedgerOpResult,
    BalanceRecalcResult,
    InsufficientBalance,
)
from settings import (
    BANANA_SEND_AS_DOCUMENT,
    CRYPTO_PAYMENT_URL,
    REDIS_PREFIX,
    SUNO_CALLBACK_URL as SETTINGS_SUNO_CALLBACK_URL,
    SUNO_ENABLED as SETTINGS_SUNO_ENABLED,
    SUNO_API_TOKEN as SETTINGS_SUNO_API_TOKEN,
    SUNO_LOG_KEY,
    SUNO_READY,
    SORA2,
    SORA2_ENABLED,
    SORA2_WAIT_STICKER_ID as SETTINGS_SORA2_WAIT_STICKER_ID,
)
from suno.cover_source import (
    MAX_AUDIO_MB as COVER_MAX_AUDIO_MB,
    CoverSourceClientError,
    CoverSourceUnavailableError,
    CoverSourceValidationError,
    ensure_audio_url as ensure_cover_audio_url,
    upload_base64 as upload_cover_base64,
    upload_stream as upload_cover_stream,
    upload_url as upload_cover_url,
    validate_audio_file as validate_cover_audio_file,
)
from suno.service import SunoService, SunoAPIError, RecordInfoPollResult
from sora2_client import (
    CreateTaskResponse,
    QueryTaskResponse,
    Sora2BadRequestError,
    Sora2AuthError,
    Sora2Error,
    Sora2UnavailableError,
    create_task as sora2_create_task,
    query_task as sora2_query_task,
    upload_image_urls as sora2_upload_image_urls,
)
from suno.schemas import CallbackEnvelope, SunoTask
from suno.client import (
    AMBIENT_NATURE_PRESET_ID,
    AMBIENT_NATURE_PRESET,
    SunoServerError,
    get_preset_config,
)
from chat_service import (
    append_ctx,
    build_messages,
    call_llm,
    clear_ctx,
    estimate_tokens,
    load_ctx,
    rate_limit_hit,
    reply as chat_reply,
    set_mode,
    CTX_MAX_TOKENS,
    INPUT_MAX_CHARS,
)
from chat_mode import is_on as chat_mode_is_on, turn_on as chat_mode_turn_on
from metrics import (
    suno_enqueue_duration_seconds,
    suno_enqueue_total,
    suno_notify_duration_seconds,
    suno_notify_fail,
    suno_notify_latency_ms,
    suno_notify_ok,
    suno_notify_total,
    suno_refund_total,
    chat_messages_total,
    chat_latency_ms,
    chat_context_tokens,
    chat_autoswitch_total,
    chat_first_hint_total,
    chat_voice_total,
    chat_voice_latency_ms,
    chat_transcribe_latency_ms,
    faq_root_views_total,
    faq_views_total,
)
from telegram_utils import (
    build_hub_keyboard,
    build_hub_text,
    safe_send as tg_safe_send,
    safe_edit,
    safe_edit_text,
    safe_send_document,
    safe_send_photo,
    safe_send_text,
    safe_send_placeholder,
    safe_edit_markdown_v2,
    safe_send_sticker,
    run_ffmpeg,
    md2_escape,
    mask_tokens,
    send_image_as_document,
    download_image_from_update,
    TelegramImageError,
)
from utils.api_client import request_with_retries
from utils.safe_send import safe_delete_message
from utils.telegram_safe import safe_edit_message
from utils.tempfiles import cleanup_temp, save_bytes_to_temp
from voice_service import VoiceTranscribeError, transcribe as voice_transcribe
try:
    import redis.asyncio as redis_asyncio  # type: ignore
except Exception:  # pragma: no cover - fallback if asyncio interface unavailable
    redis_asyncio = None

# ==========================
#   ENV / INIT
# ==========================
APP_VERSION = "2025-09-14r4"


ACTIVE_TASKS: Dict[int, str] = {}
_SORA2_POLLERS: Dict[str, asyncio.Task[None]] = {}
SHUTDOWN_EVENT = threading.Event()

SUNO_SERVICE = SunoService()

_METRIC_ENV = (os.getenv("APP_ENV") or "prod").strip() or "prod"
_METRIC_LABELS = {"env": _METRIC_ENV, "service": "bot"}
_VOICE_METRIC_LABELS = {"env": _METRIC_ENV, "service": "bot"}


def _faq_track_root() -> None:
    faq_root_views_total.labels(**_METRIC_LABELS).inc()


def _faq_track_section(section: str) -> None:
    faq_views_total.labels(section=section, **_METRIC_LABELS).inc()


_SUNO_LOCK_TTL = 15 * 60
_SUNO_LOCK_GUARD = threading.Lock()
_SUNO_LOCK_MEMORY: set[int] = set()
_SUNO_START_LOCK_TTL = 300

_SUNO_PENDING_TTL = 20 * 60
_SUNO_PENDING_LOCK = threading.Lock()
_SUNO_PENDING_MEMORY: Dict[str, tuple[float, str]] = {}

_SUNO_REFUND_PENDING_LOCK = threading.Lock()
_SUNO_REFUND_PENDING_MEMORY: Dict[str, tuple[float, str]] = {}
_SUNO_ENQUEUE_MAX_ATTEMPTS = 4
_SUNO_ENQUEUE_MAX_DELAY = 15.0


def _suno_lock_key(user_id: int) -> str:
    return f"{REDIS_PREFIX}:lock:{int(user_id)}"


def _acquire_suno_lock(user_id: int) -> bool:
    key = _suno_lock_key(user_id)
    if rds:
        try:
            stored = rds.set(key, "1", nx=True, ex=_SUNO_LOCK_TTL)
            if stored:
                return True
        except Exception as exc:
            log.warning("Suno lock redis error | user=%s err=%s", user_id, exc)
    with _SUNO_LOCK_GUARD:
        if user_id in _SUNO_LOCK_MEMORY:
            return False
        _SUNO_LOCK_MEMORY.add(user_id)
        return True


def _suno_start_lock_key(user_id: int) -> str:
    return f"{REDIS_PREFIX}:suno:start:{int(user_id)}"


def _acquire_suno_start_lock(user_id: int) -> bool:
    if not REDIS_LOCK_ENABLED or not redis_client:
        return True
    key = _suno_start_lock_key(user_id)
    try:
        stored = redis_client.set(key, "1", nx=True, ex=_SUNO_START_LOCK_TTL)
        return bool(stored)
    except Exception as exc:
        log.warning("Suno start redis lock error | user=%s err=%s", user_id, exc)
        return True


def _release_suno_lock(user_id: int) -> None:
    key = _suno_lock_key(user_id)
    if rds:
        try:
            rds.delete(key)
        except Exception as exc:
            log.warning("Suno lock release redis error | user=%s err=%s", user_id, exc)
    with _SUNO_LOCK_GUARD:
        _SUNO_LOCK_MEMORY.discard(user_id)


def _generate_suno_request_id(user_id: int) -> str:
    """Return a deterministic prefix request id for Suno enqueue calls."""

    return f"suno:{int(user_id)}:{uuid.uuid4()}"


def _generate_cover_upload_request_id(user_id: Optional[int]) -> str:
    prefix = f"suno-cover:{int(user_id)}:" if isinstance(user_id, int) else "suno-cover:"
    return prefix + uuid.uuid4().hex


def _suno_cooldown_key(user_id: int) -> str:
    return f"{REDIS_PREFIX}:suno:last:{int(user_id)}"


def _suno_cooldown_remaining(user_id: int) -> int:
    if SUNO_PER_USER_COOLDOWN_SEC <= 0:
        return 0
    key = _suno_cooldown_key(user_id)
    if rds:
        try:
            ttl = rds.ttl(key)
        except Exception as exc:
            log.warning("Suno cooldown redis ttl error | user=%s err=%s", user_id, exc)
        else:
            if isinstance(ttl, int) and ttl > 0:
                return ttl
    expires_at = _SUNO_COOLDOWN_MEMORY.get(user_id)
    if not expires_at:
        return 0
    now = time.time()
    if expires_at > now:
        return int(math.ceil(expires_at - now))
    _SUNO_COOLDOWN_MEMORY.pop(user_id, None)
    return 0


def _suno_set_cooldown(user_id: int) -> None:
    if SUNO_PER_USER_COOLDOWN_SEC <= 0:
        return
    key = _suno_cooldown_key(user_id)
    if rds:
        try:
            rds.setex(key, SUNO_PER_USER_COOLDOWN_SEC, "1")
        except Exception as exc:
            log.warning("Suno cooldown redis set error | user=%s err=%s", user_id, exc)
    _SUNO_COOLDOWN_MEMORY[user_id] = time.time() + SUNO_PER_USER_COOLDOWN_SEC


def _suno_refund_key(task_id: str) -> str:
    return f"{REDIS_PREFIX}:refund:{task_id}"


def _suno_pending_key(req_id: str) -> str:
    return f"{REDIS_PREFIX}:suno:pending:{req_id}"


def _suno_refund_pending_key(req_id: str) -> str:
    return f"{REDIS_PREFIX}:suno:refund:pending:{req_id}"


def _suno_pending_store(req_id: str, payload: Dict[str, Any]) -> None:
    if not req_id:
        return
    key = _suno_pending_key(req_id)
    raw = json.dumps(payload, ensure_ascii=False)
    if rds:
        try:
            rds.setex(key, _SUNO_PENDING_TTL, raw)
            return
        except Exception as exc:
            log.warning("Suno pending redis error | req_id=%s err=%s", req_id, exc)
    expires_at = time.time() + _SUNO_PENDING_TTL
    with _SUNO_PENDING_LOCK:
        _SUNO_PENDING_MEMORY[key] = (expires_at, raw)


def _suno_pending_load(req_id: Optional[str]) -> Optional[Dict[str, Any]]:
    if not req_id:
        return None
    key = _suno_pending_key(req_id)
    raw: Optional[str] = None
    if rds:
        try:
            data = rds.get(key)
        except Exception as exc:
            log.warning("Suno pending redis get error | req_id=%s err=%s", req_id, exc)
        else:
            if isinstance(data, bytes):
                raw = data.decode("utf-8", errors="replace")
            elif data is not None:
                raw = str(data)
    if raw is None:
        now = time.time()
        with _SUNO_PENDING_LOCK:
            entry = _SUNO_PENDING_MEMORY.get(key)
            if entry:
                expires_at, value = entry
                if expires_at > now:
                    raw = value
                else:
                    _SUNO_PENDING_MEMORY.pop(key, None)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except Exception:
        log.warning("Suno pending decode error | req_id=%s", req_id)
        return None


def _suno_refund_pending_mark(req_id: str, payload: Dict[str, Any]) -> None:
    if not req_id:
        return
    key = _suno_refund_pending_key(req_id)
    raw = json.dumps(payload, ensure_ascii=False)
    if rds:
        try:
            rds.setex(key, _SUNO_PENDING_TTL, raw)
            return
        except Exception as exc:
            log.warning("Suno refund-pending redis error | req_id=%s err=%s", req_id, exc)
    expires_at = time.time() + _SUNO_PENDING_TTL
    with _SUNO_REFUND_PENDING_LOCK:
        _SUNO_REFUND_PENDING_MEMORY[key] = (expires_at, raw)


def _suno_refund_pending_load(req_id: Optional[str]) -> Optional[Dict[str, Any]]:
    if not req_id:
        return None
    key = _suno_refund_pending_key(req_id)
    raw: Optional[str] = None
    if rds:
        try:
            data = rds.get(key)
        except Exception as exc:
            log.warning("Suno refund-pending redis get error | req_id=%s err=%s", req_id, exc)
        else:
            if isinstance(data, bytes):
                raw = data.decode("utf-8", errors="replace")
            elif data is not None:
                raw = str(data)
    if raw is None:
        now = time.time()
        with _SUNO_REFUND_PENDING_LOCK:
            entry = _SUNO_REFUND_PENDING_MEMORY.get(key)
            if entry:
                expires_at, value = entry
                if expires_at > now:
                    raw = value
                else:
                    _SUNO_REFUND_PENDING_MEMORY.pop(key, None)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except Exception:
        log.warning("Suno refund-pending decode error | req_id=%s", req_id)
        return None


def _suno_refund_pending_clear(req_id: Optional[str]) -> None:
    if not req_id:
        return
    key = _suno_refund_pending_key(req_id)
    if rds:
        try:
            rds.delete(key)
        except Exception as exc:
            log.warning("Suno refund-pending redis delete error | req_id=%s err=%s", req_id, exc)
    with _SUNO_REFUND_PENDING_LOCK:
        _SUNO_REFUND_PENDING_MEMORY.pop(key, None)


def _suno_refund_req_key(req_id: str) -> str:
    return f"{REDIS_PREFIX}:suno:refund:req:{req_id}"


_SUNO_REFUND_REQ_MEMORY: Dict[str, float] = {}


def _suno_acquire_refund(task_id: Optional[str], *, req_id: Optional[str] = None) -> bool:
    key: Optional[str]
    if task_id:
        key = _suno_refund_key(task_id)
    elif req_id:
        key = _suno_refund_req_key(req_id)
    else:
        return True
    assert key is not None
    if rds:
        try:
            stored = rds.set(key, "1", nx=True, ex=_SUNO_REFUND_TTL)
        except Exception as exc:
            log.warning("Suno refund redis error | key=%s err=%s", key, exc)
        else:
            if stored:
                return True
            return False
    now = time.time()
    expires_at = now + _SUNO_REFUND_TTL
    if task_id:
        current = _SUNO_REFUND_MEMORY.get(key)
    else:
        current = _SUNO_REFUND_REQ_MEMORY.get(key)
    if current and current > now:
        return False
    if task_id:
        _SUNO_REFUND_MEMORY[key] = expires_at
    else:
        _SUNO_REFUND_REQ_MEMORY[key] = expires_at
    return True


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso8601(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _env(k: str, d: str = "") -> str:
    v = os.getenv(k)
    return (v if v is not None else d).strip()


def _env_float(k: str, default: float) -> float:
    raw = _env(k, str(default))
    if not raw:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: Optional[bool] = None) -> Optional[bool]:
    raw = os.getenv(name)
    if raw is None:
        return default
    text = raw.strip().lower()
    if not text:
        return default
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _env_int(k: str, default: int) -> int:
    raw = _env(k, str(default))
    if not raw:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


START_EMOJI_STICKER_ID = _env("START_EMOJI_STICKER_ID", "5188621441926438751")
START_EMOJI_FALLBACK = _env("START_EMOJI_FALLBACK", "🎬") or "🎬"


SUNO_PER_USER_COOLDOWN_SEC = max(0, _env_int("SUNO_PER_USER_COOLDOWN_SEC", 0))
_SUNO_COOLDOWN_MEMORY: Dict[int, float] = {}
_SUNO_REFUND_TTL = 24 * 60 * 60
_SUNO_REFUND_MEMORY: Dict[str, float] = {}

_SUNO_STRICT_ENABLED = bool(_env_bool("SUNO_STRICT_LYRICS_ENABLED", True))
_SUNO_LYRICS_RETRY_THRESHOLD = max(0.0, min(1.0, _env_float("SUNO_LYRICS_RETRY_THRESHOLD", 0.75)))
_SUNO_LYRICS_MODEL_LIMIT = LYRICS_MAX_LENGTH
_SUNO_LYRICS_MAXLEN = max(
    1,
    min(
        _SUNO_LYRICS_MODEL_LIMIT,
        _env_int("SUNO_LYRICS_MAXLEN", _SUNO_LYRICS_MODEL_LIMIT),
    ),
)
_SUNO_LYRICS_STRICT_TEMPERATURE = max(0.0, _env_float("SUNO_LYRICS_STRICT_TEMPERATURE", 0.3))
_SUNO_LYRICS_STRICT_FLAG = _env("SUNO_LYRICS_STRICT_FLAG", "").strip()
_SUNO_LYRICS_SEED_RAW = _env("SUNO_LYRICS_SEED", "").strip()
try:
    _SUNO_LYRICS_SEED: Optional[int] = int(_SUNO_LYRICS_SEED_RAW) if _SUNO_LYRICS_SEED_RAW else None
except ValueError:
    _SUNO_LYRICS_SEED = None

def _normalize_endpoint_values(*values: Any) -> List[str]:
    """Collect endpoint path candidates from strings / iterables."""

    seen: set[str] = set()
    result: List[str] = []

    def _add(value: Any):
        if value is None:
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                _add(item)
            return
        text = str(value).strip()
        if not text:
            return
        if "," in text or any(ch.isspace() for ch in text):
            parts = re.split(r"[,\s]+", text)
        else:
            parts = [text]
        for part in parts:
            p = part.strip()
            if not p or p in seen:
                continue
            seen.add(p)
            result.append(p)

    for v in values:
        _add(v)
    return result

@dataclass(frozen=True)
class SunoConfig:
    base: str
    prefix: str
    gen_path: str
    status_path: str
    extend_path: str
    lyrics_path: str
    model: str
    price: int
    timeout_sec: int
    enabled: bool
    has_key: bool

    @property
    def configured(self) -> bool:
        return self.enabled and self.has_key and bool(self.base)


def _normalize_suno_path(raw: str, default: str) -> str:
    text = (raw or default).strip() or default
    if "://" in text:
        return text
    if not text.startswith("/"):
        text = f"/{text}"
    return text


def _compose_suno_url(*parts: str) -> str:
    cleaned = [p.strip() for p in parts if p]
    if not cleaned:
        return ""
    # Explicit URLs take precedence
    for part in reversed(cleaned):
        if "://" in part:
            return part

    normalized = []
    for idx, part in enumerate(cleaned):
        fragment = part.replace("\\", "/")
        if idx == 0:
            normalized.append(fragment.rstrip("/"))
        else:
            normalized.append(fragment.strip("/"))

    joined = "/".join(normalized)
    if "://" not in joined:
        joined = f"https://{joined}"

    parsed = urlparse(joined)
    scheme = parsed.scheme or "https"
    netloc = parsed.netloc or normalized[0].split("://", 1)[-1].split("/", 1)[0]
    path = parsed.path.replace("//", "/")
    return urlunparse((scheme, netloc, path, "", "", ""))


def _normalize_prefix(value: str) -> str:
    prefix = (value or "").strip()
    if not prefix:
        return ""
    if "://" in prefix:
        return prefix
    return "/" + prefix.strip("/")


def _load_suno_config() -> SunoConfig:
    base = (_env("SUNO_API_BASE", "https://api.kie.ai") or "https://api.kie.ai").strip().rstrip("/")
    prefix = _normalize_prefix(_env("SUNO_API_PREFIX", ""))
    gen_path = _normalize_suno_path(_env("SUNO_GEN_PATH", "/suno-api/generate"), "/suno-api/generate")
    status_path = _normalize_suno_path(
        _env("SUNO_STATUS_PATH", "/suno-api/record-info"),
        "/suno-api/record-info",
    )
    extend_path = _normalize_suno_path(
        _env("SUNO_EXTEND_PATH", "/suno-api/generate/extend"),
        "/suno-api/generate/extend",
    )
    lyrics_path = _normalize_suno_path(
        _env("SUNO_LYRICS_PATH", "/suno-api/generate/get-timestamped-lyrics"),
        "/suno-api/generate/get-timestamped-lyrics",
    )
    model = os.getenv("SUNO_MODEL", "V5").upper()  # всегда "V5"
    price = _env_int("SUNO_PRICE", 30)
    timeout_sec = _env_int("SUNO_TIMEOUT_SEC", 180)
    enabled_value = _env_bool("SUNO_ENABLED", bool(SETTINGS_SUNO_ENABLED))
    enabled = bool(enabled_value)
    has_key = bool(_env("KIE_API_KEY"))
    config = SunoConfig(
        base=base,
        prefix=prefix,
        gen_path=gen_path,
        status_path=status_path,
        extend_path=extend_path,
        lyrics_path=lyrics_path,
        model=model,
        price=price,
        timeout_sec=timeout_sec,
        enabled=enabled,
        has_key=has_key,
    )
    if enabled:
        logging.getLogger("veo3-bot").info(
            "suno configuration",
            extra={
                "meta": {
                    "base": base,
                    "gen_path": gen_path,
                    "status_path": status_path,
                    "callback_url": SETTINGS_SUNO_CALLBACK_URL,
                    "enabled": enabled,
                }
            },
        )
    return config


SUNO_CONFIG = _load_suno_config()
SUNO_MODEL = (SUNO_CONFIG.model or "V5").upper()
SUNO_MODEL_LABEL = SUNO_MODEL
SUNO_BASE_URL = _compose_suno_url(SUNO_CONFIG.base or "https://api.kie.ai", SUNO_CONFIG.prefix)
SUNO_GEN_PATH = SUNO_CONFIG.gen_path
SUNO_STATUS_PATH = SUNO_CONFIG.status_path
SUNO_EXTEND_PATH = SUNO_CONFIG.extend_path
SUNO_LYRICS_PATH = SUNO_CONFIG.lyrics_path
SUNO_GEN_URL = _compose_suno_url(SUNO_BASE_URL, SUNO_GEN_PATH)
SUNO_STATUS_URL = _compose_suno_url(SUNO_BASE_URL, SUNO_STATUS_PATH)
SUNO_EXTEND_URL = _compose_suno_url(SUNO_BASE_URL, SUNO_EXTEND_PATH)
SUNO_LYRICS_URL = _compose_suno_url(SUNO_BASE_URL, SUNO_LYRICS_PATH)
SUNO_PRICE = SUNO_CONFIG.price
SUNO_MODE_AVAILABLE = bool(SUNO_READY)
SUNO_POLL_INTERVAL = 3.0


def _parse_backoff_series(raw: str) -> List[float]:
    values: List[float] = []
    for part in (raw or "").split(","):
        try:
            number = float(part.strip())
        except ValueError:
            continue
        if number > 0:
            values.append(number)
    return values or [8.0, 13.0, 21.0, 34.0]


SUNO_POLL_FIRST_DELAY = max(0.5, _env_float("SUNO_POLL_FIRST_DELAY_SEC", 5.0))
_SUNO_POLL_BACKOFF_SERIES_RAW = _env("SUNO_POLL_BACKOFF_SERIES", "5,8,13,21,34")
_parsed_backoff = _parse_backoff_series(_SUNO_POLL_BACKOFF_SERIES_RAW)
if _parsed_backoff and abs(_parsed_backoff[0] - SUNO_POLL_FIRST_DELAY) < 1e-3:
    _parsed_backoff = _parsed_backoff[1:]
SUNO_POLL_BACKOFF_SERIES = _parsed_backoff or [8.0, 13.0, 21.0, 34.0]
SUNO_POLL_TIMEOUT = max(
    SUNO_POLL_FIRST_DELAY,
    420.0,
    _env_float("SUNO_POLL_TIMEOUT_SEC", float(SUNO_CONFIG.timeout_sec or 420.0)),
)
SUNO_POLL_NOTIFY_AFTER = max(30.0, min(SUNO_POLL_TIMEOUT, _env_float("SUNO_POLL_NOTIFY_AFTER", 75.0)))
SUNO_POLL_BACKGROUND_LIMIT = max(
    SUNO_POLL_NOTIFY_AFTER,
    _env_float("SUNO_POLL_BACKGROUND_LIMIT", 600.0),
)
ENV_NAME            = _env("ENV_NAME", "prod") or "prod"
BOT_SINGLETON_DISABLED = _env("BOT_SINGLETON_DISABLED", "false").lower() == "true"
BOT_LEADER_TTL_MS   = 30_000
BOT_LEADER_STALE_MS = 45_000
BOT_LEADER_HEARTBEAT_INTERVAL_SEC = max(0.01, _env_float("BOT_LEADER_HEARTBEAT_INTERVAL_SEC", 10.0))
TELEGRAM_TOKEN      = _env("TELEGRAM_TOKEN")
TELEGRAM_TOKEN_HASH = hashlib.sha256(TELEGRAM_TOKEN.encode("utf-8")).hexdigest() if TELEGRAM_TOKEN else "no-token"
PROMPTS_CHANNEL_URL = _env("PROMPTS_CHANNEL_URL", "https://t.me/bestveo3promts")
STARS_BUY_URL       = _env("STARS_BUY_URL", "https://t.me/PremiumBot")
PROMO_ENABLED       = _env("PROMO_ENABLED", "true").lower() == "true"
DEV_MODE            = _env("DEV_MODE", "false").lower() == "true"


OPENAI_API_KEY = _env("OPENAI_API_KEY")
try:
    import openai  # type: ignore
    if OPENAI_API_KEY:
        openai.api_key = OPENAI_API_KEY
except Exception:
    openai = None

# ---- KIE base ----
KIE_API_KEY  = _env("KIE_API_KEY")
KIE_BASE_URL = _env("KIE_BASE_URL", "https://api.kie.ai")

# VEO
KIE_VEO_GEN_PATH = _env("KIE_VEO_GEN_PATH", "/api/v1/veo/generate")

_KIE_VEO_STATUS_DEFAULT = "/api/v1/veo/record-info"
_KIE_VEO_STATUS_RAW = _env("KIE_VEO_STATUS_PATH", _KIE_VEO_STATUS_DEFAULT)
KIE_VEO_STATUS_PATHS = _normalize_endpoint_values(
    _KIE_VEO_STATUS_RAW,
    _KIE_VEO_STATUS_DEFAULT,
    "/api/v1/veo/status",
    "/api/v1/veo/recordInfo",
)
if KIE_VEO_STATUS_PATHS:
    KIE_VEO_STATUS_PATH = KIE_VEO_STATUS_PATHS[0]
else:
    KIE_VEO_STATUS_PATH = _KIE_VEO_STATUS_DEFAULT
    KIE_VEO_STATUS_PATHS = [KIE_VEO_STATUS_PATH]

_KIE_VEO_1080_DEFAULT = "/api/v1/veo/get-1080p-video"
_KIE_VEO_1080_RAW = _env("KIE_VEO_1080_PATH", _KIE_VEO_1080_DEFAULT)
KIE_VEO_1080_PATHS = _normalize_endpoint_values(
    _KIE_VEO_1080_RAW,
    _KIE_VEO_1080_DEFAULT,
    "/api/v1/veo/video-1080p",
    "/api/v1/veo/video/1080p",
    "/api/v1/veo/get1080pVideo",
)
if KIE_VEO_1080_PATHS:
    KIE_VEO_1080_PATH = KIE_VEO_1080_PATHS[0]
else:
    KIE_VEO_1080_PATH = _KIE_VEO_1080_DEFAULT
    KIE_VEO_1080_PATHS = [KIE_VEO_1080_PATH]

KIE_1080_SESSION = requests.Session()
_kie_token = (KIE_API_KEY or "").strip()
if _kie_token and not _kie_token.lower().startswith("bearer "):
    _kie_token = f"Bearer {_kie_token}"
if _kie_token:
    KIE_1080_SESSION.headers.update({"Authorization": _kie_token})

# MJ
_KIE_MJ_GENERATE_DEFAULT = "/api/v1/mj/generate"
_KIE_MJ_GENERATE_RAW = _env("KIE_MJ_GENERATE", _KIE_MJ_GENERATE_DEFAULT)
KIE_MJ_GENERATE_PATHS = _normalize_endpoint_values(
    _KIE_MJ_GENERATE_RAW,
    _KIE_MJ_GENERATE_DEFAULT,
    "/api/v1/mj/createTask",
    "/api/v1/mj/create-task",
)
if KIE_MJ_GENERATE_PATHS:
    KIE_MJ_GENERATE = KIE_MJ_GENERATE_PATHS[0]
else:
    KIE_MJ_GENERATE = _KIE_MJ_GENERATE_DEFAULT
    KIE_MJ_GENERATE_PATHS = [KIE_MJ_GENERATE]

_KIE_MJ_STATUS_DEFAULT = "/api/v1/mj/recordInfo"
_KIE_MJ_STATUS_RAW = _env("KIE_MJ_STATUS", _KIE_MJ_STATUS_DEFAULT)
KIE_MJ_STATUS_PATHS = _normalize_endpoint_values(
    _KIE_MJ_STATUS_RAW,
    _KIE_MJ_STATUS_DEFAULT,
    "/api/v1/mj/record-info",
    "/api/v1/mj/status",
    "/api/v1/mj/recordinfo",
)
if KIE_MJ_STATUS_PATHS:
    KIE_MJ_STATUS = KIE_MJ_STATUS_PATHS[0]
else:
    KIE_MJ_STATUS = _KIE_MJ_STATUS_DEFAULT
    KIE_MJ_STATUS_PATHS = [KIE_MJ_STATUS]

_KIE_MJ_UPSCALE_DEFAULT = "/api/v1/mj/generateUpscale"
_KIE_MJ_UPSCALE_RAW = _env("KIE_MJ_UPSCALE", _KIE_MJ_UPSCALE_DEFAULT)
KIE_MJ_UPSCALE_PATHS = _normalize_endpoint_values(
    _KIE_MJ_UPSCALE_RAW,
    _KIE_MJ_UPSCALE_DEFAULT,
    "/api/v1/mj/upscale",
    "/api/v1/mj/uv",
)
if KIE_MJ_UPSCALE_PATHS:
    KIE_MJ_UPSCALE = KIE_MJ_UPSCALE_PATHS[0]
else:
    KIE_MJ_UPSCALE = _KIE_MJ_UPSCALE_DEFAULT
    KIE_MJ_UPSCALE_PATHS = [KIE_MJ_UPSCALE]

# Видео
FFMPEG_BIN                = _env("FFMPEG_BIN", "ffmpeg")
ENABLE_VERTICAL_NORMALIZE = _env("ENABLE_VERTICAL_NORMALIZE", "true").lower() == "true"
ALWAYS_FORCE_FHD          = _env("ALWAYS_FORCE_FHD", "true").lower() == "true"
MAX_TG_VIDEO_MB           = int(_env("MAX_TG_VIDEO_MB", "48"))
POLL_INTERVAL_SECS = int(_env("POLL_INTERVAL_SECS", "6"))
POLL_TIMEOUT_SECS  = int(_env("POLL_TIMEOUT_SECS", str(20 * 60)))
KIE_STRICT_POLLING = _env("KIE_STRICT_POLLING", "false").lower() == "true"

logging.getLogger("kie").setLevel(logging.INFO)
log = logging.getLogger("veo3-bot")
mj_log = logging.getLogger("mj_handler")
singleton_log = logging.getLogger("veo3-bot.singleton")

try:
    import telegram as _tg
    log.info("PTB version: %s", getattr(_tg, "__version__", "unknown"))
except Exception:
    _tg = None


async def _safe_edit_message_text(
    edit_callable: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any
) -> Optional[Any]:
    """Execute ``edit_message_text`` calls while downgrading 400 errors to warnings."""

    try:
        return await edit_callable(*args, **kwargs)
    except BadRequest as exc:
        log.warning("edit_message_text ignored (HTTP 400): %s", exc)
        return exc

# Redis
REDIS_URL           = _env("REDIS_URL")
REDIS_LOCK_ENABLED  = _env("REDIS_LOCK_ENABLED", "true").lower() == "true"
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True) if REDIS_URL else None


def _build_leader_key() -> str:
    env = ENV_NAME or "prod"
    token_hash = TELEGRAM_TOKEN_HASH or "no-token"
    return f"tg:bot:leader:{env}:{token_hash}"


class _LeaderContext:
    def __init__(
        self,
        client: "redis.Redis",
        key: str,
        owner: str,
        ttl_ms: int,
        heartbeat_interval: float,
    ) -> None:
        self._client = client
        self._key = key
        self._owner = owner
        self._ttl_ms = ttl_ms
        self._interval = heartbeat_interval
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name="bot-leader-heartbeat", daemon=True)
        self._stopped = False
        self._failures = 0

    def start(self) -> None:
        self._thread.start()
        atexit.register(self.stop)

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._stop_event.set()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=2.0)
        try:
            self._client.delete(self._key)
        except Exception:
            pass

    def _heartbeat_payload(self) -> str:
        now_ms = int(time.time() * 1000)
        return json.dumps({"owner": self._owner, "ts": now_ms}, ensure_ascii=False)

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval):
            payload = self._heartbeat_payload()
            try:
                ok = self._client.set(self._key, payload, xx=True, px=self._ttl_ms)
            except Exception:
                ok = False
            if ok:
                self._failures = 0
                singleton_log.info("leader: heartbeat ok")
            else:
                self._failures += 1
                if self._failures >= 3:
                    singleton_log.warning("leader: lost heartbeat")


_leader_context: Optional[_LeaderContext] = None


_LEADER_STEAL_SCRIPT = """
local key = KEYS[1]
local value = ARGV[1]
local ttl_ms = tonumber(ARGV[2])
local now_ms = tonumber(ARGV[3])
local stale_ms = tonumber(ARGV[4])
local current = redis.call('GET', key)
if not current then
    if redis.call('SET', key, value, 'NX', 'PX', ttl_ms) then
        return 1
    end
    return 0
end
local ok, data = pcall(cjson.decode, current)
local ts = 0
if ok and type(data) == 'table' then
    local current_ts = data.ts
    local current_ts_type = type(current_ts)
    if current_ts_type == 'number' then
        ts = current_ts
    elseif current_ts_type == 'string' then
        ts = tonumber(current_ts) or 0
    end
end
if now_ms - ts > stale_ms then
    redis.call('SET', key, value, 'PX', ttl_ms)
    return 2
end
return 0
"""


def _steal_leader_if_stale(
    client: "redis.Redis",
    key: str,
    payload: str,
    now_ms: int,
    ttl_ms: int,
    stale_ms: int,
) -> bool:
    try:
        result = client.eval(
            _LEADER_STEAL_SCRIPT,
            1,
            key,
            payload,
            ttl_ms,
            now_ms,
            stale_ms,
        )
    except Exception as exc:
        singleton_log.debug("leader steal script failed: %s", exc)
        return False
    if result in (1, 2):
        if result == 2:
            singleton_log.warning("leader: stale leader stolen")
        return True
    return False


def acquire_singleton_lock(ttl_sec: int = 3600) -> None:
    """Attempt to acquire a soft leader role backed by Redis heartbeats."""

    del ttl_sec  # legacy argument, kept for compatibility

    global _leader_context

    if BOT_SINGLETON_DISABLED:
        singleton_log.warning("BOT_SINGLETON_DISABLED=true — leader election disabled")
        return

    if not REDIS_URL or redis is None:
        singleton_log.warning("No REDIS_URL/redis — leader election disabled")
        return

    try:
        client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    except Exception as exc:
        singleton_log.warning("Leader election disabled: redis unavailable (%s)", exc)
        return

    if _leader_context is not None:
        _leader_context.stop()
        _leader_context = None

    key = _build_leader_key()
    owner = f"{socket.gethostname()}:{os.getpid()}"
    now_ms = int(time.time() * 1000)
    payload = json.dumps({"owner": owner, "ts": now_ms}, ensure_ascii=False)

    try:
        acquired = client.set(key, payload, nx=True, px=BOT_LEADER_TTL_MS)
    except Exception as exc:
        singleton_log.warning("Leader election disabled: redis error (%s)", exc)
        return

    if not acquired:
        try:
            current_raw = client.get(key)
        except Exception as exc:
            singleton_log.debug("leader: failed to read current owner: %s", exc)
            current_raw = None

        current_ts = 0
        if current_raw:
            try:
                current_data = json.loads(current_raw)
                ts_candidate = current_data.get("ts")
                if isinstance(ts_candidate, (int, float)):
                    current_ts = int(ts_candidate)
                elif isinstance(ts_candidate, str) and ts_candidate.isdigit():
                    current_ts = int(ts_candidate)
            except Exception:
                current_ts = 0

        if now_ms - current_ts > BOT_LEADER_STALE_MS:
            if not _steal_leader_if_stale(
                client,
                key,
                payload,
                now_ms,
                BOT_LEADER_TTL_MS,
                BOT_LEADER_STALE_MS,
            ):
                return
        else:
            return

    singleton_log.info("leader: acquired")
    _leader_context = _LeaderContext(
        client,
        key,
        owner,
        BOT_LEADER_TTL_MS,
        BOT_LEADER_HEARTBEAT_INTERVAL_SEC,
    )
    _leader_context.start()

def _parse_admin_ids(raw: str) -> set[int]:
    result: set[int] = set()
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            result.add(int(part))
        except ValueError:
            log.warning("Invalid ADMIN_IDS entry skipped: %s", part)
    return result


ADMIN_IDS = _parse_admin_ids(_env("ADMIN_IDS"))


def _is_admin(user_id: int) -> bool:
    return user_id in ADMIN_IDS


LEDGER_BACKEND = _env("LEDGER_BACKEND", "postgres").lower()
DATABASE_URL = _env("DATABASE_URL") or _env("POSTGRES_DSN")
if LEDGER_BACKEND != "memory" and not DATABASE_URL:
    raise RuntimeError("DATABASE_URL (or POSTGRES_DSN) must be set for persistent ledger storage")

def _rk(*parts: str) -> str: return ":".join([REDIS_PREFIX, *parts])

# --- User mode routing ---
MODE_CHAT = "chat"
MODE_PM = "prompt_master"
MODE_KEY_FMT = f"{REDIS_PREFIX}:mode:{{chat_id}}"

CHAT_SYSTEM_PROMPT = (
    "Вы — дружелюбный ассистент бота Best VEO3. "
    "Отвечайте кратко, по делу, при необходимости приводите простой пример."
)

VOICE_MAX_SIZE_BYTES = 20 * 1024 * 1024
VOICE_MAX_DURATION_SEC = 5 * 60
VOICE_TOO_LARGE_TEXT = "✂️ Голосовое слишком длинное/большое. Отправьте до 5 минут и 20 МБ."
VOICE_PLACEHOLDER_TEXT = "🎙️ Распознаю голос…"
VOICE_TRANSCRIBE_ERROR_TEXT = "⚠️ Не удалось распознать аудио. Попробуйте ещё раз."

# Если Redis подключен — используем его; иначе fallback на память процесса.
_inmem_modes: Dict[Any, Any] = {}


def _mode_get(chat_id: int) -> Optional[str]:
    from redis_utils import rds  # локальный импорт, чтобы избежать циклов

    if rds:
        key = MODE_KEY_FMT.format(chat_id=chat_id)
        try:
            val = rds.get(key)
        except Exception as exc:
            log.warning("mode-get redis error: %s", exc)
        else:
            if val:
                return val.decode("utf-8") if isinstance(val, bytes) else str(val)
        return None
    return _inmem_modes.get(chat_id)


def _mode_set(chat_id: int, mode: str) -> None:
    from redis_utils import rds

    if rds:
        key = MODE_KEY_FMT.format(chat_id=chat_id)
        try:
            rds.setex(key, 30 * 24 * 3600, mode)
            return
        except Exception as exc:
            log.warning("mode-set redis error: %s", exc)
    _inmem_modes[chat_id] = mode


def is_mode_on(user_id: int) -> bool:
    """Compatibility wrapper for tests expecting legacy API."""

    return chat_mode_is_on(user_id)


CACHE_PM_KEY_FMT = f"{REDIS_PREFIX}:pm:last:{{chat_id}}"


def cache_pm_prompt(chat_id: int, text: str) -> None:
    from redis_utils import rds

    if rds:
        try:
            rds.setex(CACHE_PM_KEY_FMT.format(chat_id=chat_id), 3600, text)
            return
        except Exception as exc:
            log.warning("pm-cache redis error: %s", exc)
    _inmem_modes[f"pm:{chat_id}"] = text


def get_cached_pm_prompt(chat_id: int) -> Optional[str]:
    from redis_utils import rds

    if rds:
        try:
            v = rds.get(CACHE_PM_KEY_FMT.format(chat_id=chat_id))
        except Exception as exc:
            log.warning("pm-cache redis get error: %s", exc)
        else:
            if v:
                return v.decode("utf-8") if isinstance(v, bytes) else str(v)
        return None
    return _inmem_modes.get(f"pm:{chat_id}")


PM_STEP_KEY_FMT = f"{REDIS_PREFIX}:pm:step:{{user_id}}"
PM_BUF_KEY_FMT = f"{REDIS_PREFIX}:pm:buf:{{user_id}}"
PM_STATE_TTL = 30 * 60
PM_PLACEHOLDER_TEXT = "Пишу промпт…"
PM_ERROR_TEXT = "⚠️ Не получилось, попробуйте ещё раз."
PM_MENU_TEXT = "🧠 Prompt-Master\nВыберите раздел:"
_PM_MENU_LOCK_NAME = "pm:lock"
_PM_MENU_MESSAGE_NAME = "pm:card"
_PM_MENU_LOCK_TTL = 60
_PM_MENU_MESSAGE_TTL = 90


def _clear_pm_menu_state(chat_id: int, *, user_id: Optional[int] = None) -> None:
    clear_menu_message(_PM_MENU_MESSAGE_NAME, chat_id)
    release_menu_lock(_PM_MENU_LOCK_NAME, chat_id)
    if user_id is not None:
        release_menu_lock(_PM_MENU_LOCK_NAME, user_id)

_PM_STEP_MEMORY: Dict[int, Tuple[float, str]] = {}
_PM_BUFFER_MEMORY: Dict[int, Tuple[float, Dict[str, Any]]] = {}


def _pm_step_key(user_id: int) -> str:
    return PM_STEP_KEY_FMT.format(user_id=int(user_id))


def _pm_buf_key(user_id: int) -> str:
    return PM_BUF_KEY_FMT.format(user_id=int(user_id))


def _pm_memory_get(
    store: Dict[int, Tuple[float, Any]], user_id: int
) -> Optional[Any]:
    entry = store.get(int(user_id))
    if not entry:
        return None
    expires, value = entry
    if expires <= time.time():
        store.pop(int(user_id), None)
        return None
    return value


def _pm_memory_set(store: Dict[int, Tuple[float, Any]], user_id: int, value: Any) -> None:
    store[int(user_id)] = (time.time() + PM_STATE_TTL, value)


def _pm_memory_clear(store: Dict[int, Tuple[float, Any]], user_id: int) -> None:
    store.pop(int(user_id), None)


def _pm_set_step(user_id: int, step: str) -> None:
    if rds:
        try:
            rds.setex(_pm_step_key(user_id), PM_STATE_TTL, step)
            return
        except Exception as exc:
            log.warning("pm.step.redis_set_failed | user_id=%s err=%s", user_id, exc)
    _pm_memory_set(_PM_STEP_MEMORY, user_id, step)


def _pm_get_step(user_id: int) -> Optional[str]:
    if rds:
        try:
            raw = rds.get(_pm_step_key(user_id))
        except Exception as exc:
            log.warning("pm.step.redis_get_failed | user_id=%s err=%s", user_id, exc)
        else:
            if raw is not None:
                return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
    value = _pm_memory_get(_PM_STEP_MEMORY, user_id)
    return value if isinstance(value, str) else None


def _pm_clear_step(user_id: int) -> None:
    if rds:
        try:
            rds.delete(_pm_step_key(user_id))
        except Exception as exc:
            log.warning("pm.step.redis_del_failed | user_id=%s err=%s", user_id, exc)
    _pm_memory_clear(_PM_STEP_MEMORY, user_id)


def _pm_set_buffer(user_id: int, data: Dict[str, Any]) -> None:
    payload = json.dumps(data, ensure_ascii=False)
    if rds:
        try:
            rds.setex(_pm_buf_key(user_id), PM_STATE_TTL, payload)
            return
        except Exception as exc:
            log.warning("pm.buf.redis_set_failed | user_id=%s err=%s", user_id, exc)
    _pm_memory_set(_PM_BUFFER_MEMORY, user_id, data)


def _pm_get_buffer(user_id: int) -> Optional[Dict[str, Any]]:
    if rds:
        try:
            raw = rds.get(_pm_buf_key(user_id))
        except Exception as exc:
            log.warning("pm.buf.redis_get_failed | user_id=%s err=%s", user_id, exc)
        else:
            if raw:
                try:
                    decoded = (
                        raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
                    )
                    data = json.loads(decoded)
                    if isinstance(data, dict):
                        return data
                except json.JSONDecodeError:
                    log.warning("pm.buf.decode_failed | user_id=%s", user_id)
    value = _pm_memory_get(_PM_BUFFER_MEMORY, user_id)
    return value if isinstance(value, dict) else None


def _pm_clear_buffer(user_id: int) -> None:
    if rds:
        try:
            rds.delete(_pm_buf_key(user_id))
        except Exception as exc:
            log.warning("pm.buf.redis_del_failed | user_id=%s err=%s", user_id, exc)
    _pm_memory_clear(_PM_BUFFER_MEMORY, user_id)


def _pm_clear_state(user_id: int) -> None:
    _pm_clear_step(user_id)
    _pm_clear_buffer(user_id)


_PM_QUESTION_FLOWS: Dict[str, Tuple[Dict[str, Any], ...]] = {
    "video": (
        {"key": "idea", "question": "Опишите идею видео одной-двумя фразами", "optional": False},
        {"key": "style", "question": "Стиль (необязательно)?", "optional": True},
    ),
    "animate": (
        {"key": "brief", "question": "Что на фото и какой микро-движение хотите?", "optional": False},
    ),
    "banana": (
        {"key": "brief", "question": "Что изменить (фон/одежда/макияж/удалить…)?", "optional": False},
        {"key": "avoid", "question": "Что не делать (необязательно)?", "optional": True},
    ),
    "mj": (
        {"key": "subject", "question": "Что сгенерировать?", "optional": False},
        {"key": "style", "question": "Стиль/референсы (необязательно)?", "optional": True},
    ),
    "suno": (
        {"key": "idea", "question": "О чём песня и в каком стиле?", "optional": False},
        {"key": "vocal", "question": "Вокал (m/f/любой)?", "optional": True},
    ),
}

_PM_KIND_TITLES = {
    "video": "🎬 Промпт для видео",
    "animate": "🖼️ Оживление фото",
    "banana": "🍌 Banana JSON",
    "mj": "🎨 Midjourney JSON",
    "suno": "🎵 Suno (текст)",
}

_PM_SKIP_WORDS = {"", "-", "—", "нет", "не надо", "никак", "none", "no", "skip", "n/a"}


def _pm_flow(kind: str) -> Tuple[Dict[str, Any], ...]:
    return _PM_QUESTION_FLOWS.get(kind, ())


def _pm_should_skip(value: str) -> bool:
    return value.strip().lower() in _PM_SKIP_WORDS


def _pm_user_lang(update: Update) -> Optional[str]:
    user = update.effective_user
    if user and user.language_code:
        return user.language_code
    return None


def _pm_split_suno_idea(text: str) -> Tuple[str, Optional[str]]:
    raw = text.strip()
    lower = raw.lower()
    token = " в стиле "
    if token in lower:
        idx = lower.index(token)
        idea = raw[:idx].strip(" ,.;:\n-—")
        style = raw[idx + len(token) :].strip()
        return idea or raw, style or None
    for sep in (";", "|", "—", "-", ":"):
        if sep in raw:
            first, rest = raw.split(sep, 1)
            idea = first.strip()
            style = rest.strip()
            return idea or raw, style or None
    return raw, None


def _pm_normalize_vocal(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    lowered = value.strip().lower()
    if not lowered:
        return None
    if lowered in {"m", "male", "м", "муж", "мужской", "man"}:
        return "m"
    if lowered in {"f", "female", "ж", "жен", "женский", "woman"}:
        return "f"
    if lowered in {"any", "любая", "любой", "both"}:
        return "any"
    return "any"


def _pm_store_result(user_id: int, data: Dict[str, Any]) -> None:
    payload = {"stage": 0, "result": data}
    _pm_set_buffer(user_id, payload)
    _pm_set_step(user_id, "idle")


def _pm_last_result(user_id: int, ctx: ContextTypes.DEFAULT_TYPE) -> Optional[Dict[str, Any]]:
    buf = _pm_get_buffer(user_id) or {}
    result = buf.get("result") if isinstance(buf, dict) else None
    if isinstance(result, dict) and result.get("raw"):
        return result
    chat_state = ctx.chat_data.get("prompt_master") if isinstance(ctx.chat_data, dict) else None
    fallback = chat_state.get("last_result") if isinstance(chat_state, dict) else None
    return fallback if isinstance(fallback, dict) else None


def _pm_format_text_block(title: str, raw: str, *, is_json: bool = False) -> str:
    escaped = html.escape(raw)
    if is_json:
        return f"{title}\n<pre><code>{escaped}</code></pre>"
    return f"{title}\n<pre>{escaped}</pre>"


def _pm_prepare_result(
    kind: str,
    answers: Dict[str, Any],
    *,
    update: Update,
) -> Tuple[str, Dict[str, Any]]:
    title = _PM_KIND_TITLES.get(kind, "Prompt")
    if kind == "video":
        idea = (answers.get("idea") or "").strip()
        if not idea:
            raise ValueError("empty idea")
        style = answers.get("style")
        prompt_text = build_video_prompt(idea, style)
        display = _pm_format_text_block(title, prompt_text)
        return display, {"kind": kind, "raw": prompt_text, "is_json": False}
    if kind == "animate":
        brief = (answers.get("brief") or "").strip()
        if not brief:
            raise ValueError("empty animate brief")
        prompt_text = build_animate_prompt(brief, None)
        display = _pm_format_text_block(title, prompt_text)
        return display, {"kind": kind, "raw": prompt_text, "is_json": False}
    if kind == "banana":
        brief = (answers.get("brief") or "").strip()
        if not brief:
            raise ValueError("empty banana brief")
        avoid = answers.get("avoid")
        json_payload = build_banana_json(brief, avoid)
        raw = json.dumps(json_payload, ensure_ascii=False, indent=2)
        display = _pm_format_text_block(title, raw, is_json=True)
        return display, {
            "kind": kind,
            "raw": raw,
            "is_json": True,
            "json": json_payload,
        }
    if kind == "mj":
        subject = (answers.get("subject") or "").strip()
        if not subject:
            raise ValueError("empty mj subject")
        style = answers.get("style")
        json_payload = build_mj_json(subject, style)
        raw = json.dumps(json_payload, ensure_ascii=False, indent=2)
        display = _pm_format_text_block(title, raw, is_json=True)
        return display, {
            "kind": kind,
            "raw": raw,
            "is_json": True,
            "json": json_payload,
        }
    if kind == "suno":
        idea_raw = (answers.get("idea") or "").strip()
        if not idea_raw:
            raise ValueError("empty suno idea")
        idea, parsed_style = _pm_split_suno_idea(idea_raw)
        user_style = answers.get("style")
        style_value = user_style if user_style else parsed_style
        vocal_value = _pm_normalize_vocal(answers.get("vocal"))
        lang = _pm_user_lang(update)
        prompt_text = build_suno_prompt(
            idea,
            style=style_value,
            vocal=vocal_value,
            language=lang,
        )
        display = _pm_format_text_block(title, prompt_text)
        return display, {"kind": kind, "raw": prompt_text, "is_json": False}
    raise ValueError(f"unsupported pm kind: {kind}")


def _pm_begin_session(user_id: int, kind: str) -> Dict[str, Any]:
    payload = {"kind": kind, "stage": 1, "answers": {}}
    _pm_set_step(user_id, kind)
    _pm_set_buffer(user_id, payload)
    return payload


def _pm_session(user_id: int) -> Optional[Dict[str, Any]]:
    data = _pm_get_buffer(user_id)
    return data if isinstance(data, dict) else None


def _pm_save_session(user_id: int, payload: Dict[str, Any]) -> None:
    _pm_set_step(user_id, str(payload.get("kind", "idle")))
    _pm_set_buffer(user_id, payload)


def is_pm_waiting(user_id: Optional[int]) -> bool:
    if not user_id:
        return False
    payload = _pm_session(user_id)
    if not payload:
        return False
    kind = payload.get("kind")
    stage = payload.get("stage")
    flow = _pm_flow(str(kind))
    if not flow:
        return False
    if not isinstance(stage, int):
        return False
    return 1 <= stage <= len(flow)


async def _pm_send_menu(chat_id: int, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    lock_owned = acquire_menu_lock(_PM_MENU_LOCK_NAME, chat_id, _PM_MENU_LOCK_TTL)
    forced_release = False
    markup = pm_main_kb()

    try:
        record = get_menu_message(
            _PM_MENU_MESSAGE_NAME,
            chat_id,
            max_age=_PM_MENU_MESSAGE_TTL,
        )

        if not lock_owned:
            if record:
                saved_id, _ = record
                outcome = await _try_edit_menu_card(
                    ctx,
                    chat_id=chat_id,
                    message_id=saved_id,
                    text=PM_MENU_TEXT,
                    reply_markup=markup,
                    log_prefix="pm.menu",
                    parse_mode=None,
                )
                if outcome == "ok":
                    save_menu_message(
                        _PM_MENU_MESSAGE_NAME,
                        chat_id,
                        saved_id,
                        _PM_MENU_MESSAGE_TTL,
                    )
                    chat_data = getattr(ctx, "chat_data", None)
                    if isinstance(chat_data, MutableMapping):
                        chat_data["pm_menu_msg_id"] = saved_id
                    return
                clear_menu_message(_PM_MENU_MESSAGE_NAME, chat_id)
            release_menu_lock(_PM_MENU_LOCK_NAME, chat_id)
            forced_release = True
            lock_owned = acquire_menu_lock(
                _PM_MENU_LOCK_NAME,
                chat_id,
                _PM_MENU_LOCK_TTL,
            )

        record = get_menu_message(
            _PM_MENU_MESSAGE_NAME,
            chat_id,
            max_age=_PM_MENU_MESSAGE_TTL,
        )
        if record:
            saved_id, _ = record
            outcome = await _try_edit_menu_card(
                ctx,
                chat_id=chat_id,
                message_id=saved_id,
                text=PM_MENU_TEXT,
                reply_markup=markup,
                log_prefix="pm.menu",
                parse_mode=None,
            )
            if outcome == "ok":
                save_menu_message(
                    _PM_MENU_MESSAGE_NAME,
                    chat_id,
                    saved_id,
                    _PM_MENU_MESSAGE_TTL,
                )
                chat_data = getattr(ctx, "chat_data", None)
                if isinstance(chat_data, MutableMapping):
                    chat_data["pm_menu_msg_id"] = saved_id
                return
            clear_menu_message(_PM_MENU_MESSAGE_NAME, chat_id)

        result = await tg_safe_send(
            ctx.bot.send_message,
            method_name="sendMessage",
            kind="message",
            chat_id=chat_id,
            text=PM_MENU_TEXT,
            reply_markup=markup,
            parse_mode=None,
        )
        message_id = getattr(result, "message_id", None)
        if isinstance(message_id, int):
            save_menu_message(
                _PM_MENU_MESSAGE_NAME,
                chat_id,
                message_id,
                _PM_MENU_MESSAGE_TTL,
            )
            chat_data = getattr(ctx, "chat_data", None)
            if isinstance(chat_data, MutableMapping):
                chat_data["pm_menu_msg_id"] = message_id
    finally:
        if lock_owned:
            release_menu_lock(_PM_MENU_LOCK_NAME, chat_id)
        elif forced_release:
            log.debug("pm.menu.lock_forced_release", extra={"chat_id": chat_id})


async def _pm_send_question(chat_id: int, ctx: ContextTypes.DEFAULT_TYPE, *, text: str) -> None:
    await tg_safe_send(
        ctx.bot.send_message,
        method_name="sendMessage",
        kind="message",
        chat_id=chat_id,
        text=text,
        parse_mode=None,
    )


def _pm_stage_question(kind: str, stage: int) -> Optional[str]:
    flow = _pm_flow(kind)
    if 1 <= stage <= len(flow):
        question = flow[stage - 1].get("question")
        if isinstance(question, str):
            return question
    return None


async def prompt_master_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await ensure_user_record(update)
    user = update.effective_user
    user_id = user.id if user else None
    if user_id:
        _pm_clear_state(user_id)
    chat = update.effective_chat
    message = update.message
    chat_id = None
    if chat is not None:
        chat_id = chat.id
    elif message is not None:
        chat_id = message.chat_id
    if chat_id is None:
        return
    await _pm_send_menu(chat_id, ctx)


async def _pm_start_flow(
    user_id: int,
    kind: str,
    *,
    chat_id: Optional[int],
    ctx: ContextTypes.DEFAULT_TYPE,
) -> None:
    payload = _pm_begin_session(user_id, kind)
    question = _pm_stage_question(kind, payload.get("stage", 0))
    if chat_id is not None and question:
        await _pm_send_question(chat_id, ctx, text=question)


async def _pm_restart_from_result(
    user_id: int,
    ctx: ContextTypes.DEFAULT_TYPE,
    *,
    chat_id: Optional[int],
) -> bool:
    result = _pm_last_result(user_id, ctx)
    kind = result.get("kind") if isinstance(result, dict) else None
    if not kind:
        payload = _pm_session(user_id)
        kind = payload.get("kind") if isinstance(payload, dict) else None
    if not isinstance(kind, str):
        return False
    await _pm_start_flow(user_id, kind, chat_id=chat_id, ctx=ctx)
    return True


async def _pm_apply_result(
    kind: str,
    result: Dict[str, Any],
    update: Update,
    ctx: ContextTypes.DEFAULT_TYPE,
) -> Tuple[bool, str]:
    chat = update.effective_chat
    chat_id = chat.id if chat is not None else None
    if chat_id is None:
        return False, "Чат недоступен"
    raw = result.get("raw")
    if not isinstance(raw, str) or not raw.strip():
        return False, "Промпт не найден"
    s = state(ctx)
    if kind in {"video", "animate"}:
        await set_veo_card_prompt(chat_id, raw, ctx)
        cache_pm_prompt(chat_id, raw)
        label = "Veo" if kind == "video" else "Veo Animate"
        return True, f"Промпт вставлен в карточку {label}."
    if kind == "banana":
        s["last_prompt"] = raw
        s["_last_text_banana"] = None
        await show_banana_card(chat_id, ctx)
        return True, "JSON добавлен в карточку Banana."
    if kind == "mj":
        s["last_prompt"] = raw
        s["_last_text_mj"] = None
        await show_mj_prompt_card(chat_id, ctx)
        return True, "JSON установлен для Midjourney."
    if kind == "suno":
        suno_state_obj = load_suno_state(ctx)
        set_suno_lyrics(suno_state_obj, raw)
        suno_state_obj.mode = "lyrics"
        save_suno_state(ctx, suno_state_obj)
        s["suno_state"] = suno_state_obj.to_dict()
        s["suno_waiting_state"] = IDLE_SUNO
        _reset_suno_card_cache(s)
        s.setdefault("mode", "suno")
        await refresh_suno_card(ctx, chat_id, s, price=PRICE_SUNO)
        return True, "Текст добавлен в карточку Suno."
    return False, "Неизвестный тип"


async def prompt_master_cb(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await ensure_user_record(update)
    query = update.callback_query
    if not query or not query.data:
        return
    data = query.data
    user = update.effective_user
    user_id = user.id if user else None
    message = query.message
    chat = update.effective_chat
    chat_id = None
    if message is not None:
        chat_id = message.chat_id
    elif chat is not None:
        chat_id = chat.id
    if not data.startswith("pm:"):
        await query.answer()
        return
    action_payload = data.split(":", 2)
    action = action_payload[1] if len(action_payload) > 1 else ""

    if action == "menu":
        if user_id:
            _pm_clear_state(user_id)
        await query.answer()
        if chat_id is not None:
            await _pm_send_menu(chat_id, ctx)
        return

    if action == "home":
        if user_id:
            _pm_clear_state(user_id)
        await query.answer()
        if chat_id is not None:
            _clear_pm_menu_state(chat_id, user_id=user_id)
            await show_main_menu(chat_id, ctx)
        return

    if action == "copy":
        if not user_id:
            await query.answer()
            return
        result = _pm_last_result(user_id, ctx)
        raw = result.get("raw") if isinstance(result, dict) else None
        if not isinstance(raw, str):
            await query.answer("Промпт не найден", show_alert=True)
            return
        if chat_id is not None:
            await tg_safe_send(
                ctx.bot.send_message,
                method_name="sendMessage",
                kind="message",
                chat_id=chat_id,
                text=raw,
                parse_mode=None,
            )
        await query.answer("Готово")
        return

    if action == "back":
        if not user_id:
            await query.answer()
            return
        restarted = await _pm_restart_from_result(user_id, ctx, chat_id=chat_id)
        if restarted:
            await query.answer("Измените ввод")
        else:
            _pm_clear_state(user_id)
            await query.answer()
            if chat_id is not None:
                await _pm_send_menu(chat_id, ctx)
        return

    if action == "reuse" and len(action_payload) > 2:
        if not user_id:
            await query.answer()
            return
        kind = action_payload[2]
        result = _pm_last_result(user_id, ctx)
        if not isinstance(result, dict) or result.get("kind") != kind:
            await query.answer("Промпт не найден", show_alert=True)
            return
        ok, msg = await _pm_apply_result(kind, result, update, ctx)
        await query.answer(msg, show_alert=not ok)
        return

    if action in _PM_QUESTION_FLOWS:
        if not user_id:
            await query.answer()
            return
        _pm_clear_state(user_id)
        await query.answer()
        await _pm_start_flow(user_id, action, chat_id=chat_id, ctx=ctx)
        return

    await query.answer()


async def prompt_master_input(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if message is None or not isinstance(message.text, str):
        return
    user = update.effective_user
    user_id = user.id if user else None
    if not is_pm_waiting(user_id):
        return
    payload = _pm_session(user_id) or {}
    kind = payload.get("kind")
    if not isinstance(kind, str):
        _pm_clear_state(user_id or 0)
        return
    flow = _pm_flow(kind)
    stage = payload.get("stage")
    if not isinstance(stage, int) or not (1 <= stage <= len(flow)):
        _pm_clear_state(user_id or 0)
        return
    answers = payload.setdefault("answers", {})
    question_info = flow[stage - 1]
    text = message.text.strip()
    if not text and not question_info.get("optional"):
        await message.reply_text(str(question_info.get("question") or ""))
        return
    if question_info.get("optional") and _pm_should_skip(text):
        value: Optional[str] = None
    else:
        if not text and not question_info.get("optional"):
            await message.reply_text(str(question_info.get("question") or ""))
            return
        value = text
    answers[question_info.get("key")] = value
    next_stage = stage + 1
    if next_stage <= len(flow):
        payload["stage"] = next_stage
        _pm_save_session(user_id or 0, payload)
        next_question = flow[next_stage - 1].get("question")
        if next_question:
            await message.reply_text(str(next_question))
        return

    payload["stage"] = 0
    _pm_save_session(user_id or 0, payload)

    try:
        display_text, result = _pm_prepare_result(kind, answers, update=update)
    except Exception as exc:
        log.warning("pm.prepare_failed | user_id=%s kind=%s err=%s", user_id, kind, exc)
        await message.reply_text(PM_ERROR_TEXT)
        _pm_clear_state(user_id or 0)
        return

    chat_id = message.chat_id
    try:
        await ctx.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    except Exception:
        pass

    placeholder = await tg_safe_send(
        ctx.bot.send_message,
        method_name="sendMessage",
        kind="message",
        chat_id=chat_id,
        text=PM_PLACEHOLDER_TEXT,
        parse_mode=None,
    )

    markup = pm_result_kb(kind)
    sent = False
    placeholder_id = getattr(placeholder, "message_id", None)
    if isinstance(placeholder_id, int):
        try:
            await ctx.bot.edit_message_text(
                chat_id=chat_id,
                message_id=placeholder_id,
                text=display_text,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
                reply_markup=markup,
            )
            sent = True
        except Exception as exc:
            log.warning("pm.edit_failed | user_id=%s err=%s", user_id, exc)
            with suppress(Exception):
                await ctx.bot.delete_message(chat_id, placeholder_id)

    if not sent:
        await tg_safe_send(
            ctx.bot.send_message,
            method_name="sendMessage",
            kind="message",
            chat_id=chat_id,
            text=display_text,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
            reply_markup=markup,
        )

    _pm_store_result(user_id or 0, result)
    if isinstance(ctx.chat_data, dict):
        ctx.chat_data.setdefault("prompt_master", {})["last_result"] = result

    if kind in {"video", "animate"}:
        cache_pm_prompt(chat_id, result.get("raw", ""))


CB_MODE_CHAT = "mode:chat"
CB_MODE_PM = "mode:pm"
CB_PM_INSERT_VEO = "pm_insert_veo"
CB_GO_HOME = "go_home"

# ==========================
#   Tokens / Pricing
# ==========================
PRICE_MJ = 10
PRICE_MJ_UPSCALE = PRICE_MJ
PRICE_BANANA = 5
PRICE_VEO_FAST = 50
PRICE_VEO_QUALITY = 150
PRICE_VEO_ANIMATE = 50
PRICE_SORA2_TEXT = 180
PRICE_SORA2_IMAGE = 200
PRICE_SUNO = SUNO_PRICE

TOKEN_COSTS = {
    "veo_fast": PRICE_VEO_FAST,
    "veo_quality": PRICE_VEO_QUALITY,
    "veo_photo": PRICE_VEO_ANIMATE,
    "sora2_ttv": PRICE_SORA2_TEXT,
    "sora2_itv": PRICE_SORA2_IMAGE,
    "mj": PRICE_MJ,          # 16:9 или 9:16
    "mj_upscale": PRICE_MJ_UPSCALE,
    "banana": PRICE_BANANA,
    "suno": PRICE_SUNO,
    "chat": 0,
}

_INSTRUMENTAL_DEFAULT_STYLE = "ambient, cinematic pads, soft drums"
_INSTRUMENTAL_TITLES = [
    "Oceanic Dreams",
    "Neon Horizon",
    "Aurora Echoes",
    "Celestial Drift",
    "Nocturnal Pulse",
    "Azure Haze",
    "Starlit Voyage",
    "Crystal Bloom",
    "Midnight Currents",
    "Luminous Trails",
]


def _pick_instrumental_title() -> str:
    if not _INSTRUMENTAL_TITLES:
        return "Luminous Trails"
    return random.choice(_INSTRUMENTAL_TITLES)
CHAT_UNLOCK_PRICE = 0

# ==========================
#   Promo code (fixed)
# ==========================
FIXED_PROMO_CODE = "PROMOCODE100"
FIXED_PROMO_BONUS = 100
FIXED_PROMO_REASON = "promo_bonus"


def _normalize_promo_code(value: str) -> str:
    return (value or "").strip().upper()


def activate_fixed_promo(user_id: int, raw_code: str) -> Tuple[str, Optional[int]]:
    code = _normalize_promo_code(raw_code)
    if code != FIXED_PROMO_CODE:
        return "invalid", None

    try:
        ensure_user(user_id)
    except Exception as exc:
        log.exception("ensure_user failed before promo activation for %s: %s", user_id, exc)
        return "error", None

    try:
        if is_promo_used(user_id, code):
            return "already_used", None
    except Exception as exc:
        log.exception("is_promo_used failed for %s: %s", user_id, exc)
        return "error", None

    try:
        added = mark_promo_used(user_id, code)
    except Exception as exc:
        log.exception("mark_promo_used failed for %s: %s", user_id, exc)
        return "error", None

    if not added:
        return "already_used", None

    try:
        balance_after = credit_balance(
            user_id,
            FIXED_PROMO_BONUS,
            reason=FIXED_PROMO_REASON,
            meta={"code": code},
        )
    except Exception as exc:
        log.exception("credit_balance failed during promo activation for %s: %s", user_id, exc)
        try:
            unmark_promo_used(user_id, code)
        except Exception as rollback_exc:
            log.exception("failed to rollback promo flag for %s: %s", user_id, rollback_exc)
        return "error", None

    return "ok", balance_after

# локальный кэш процесса (если Redis выключен)
app_cache: Dict[Any, Any] = {}


_CYRILLIC_RE = re.compile(r"[а-яА-ЯёЁ]")


async def ensure_user_record(update: Optional[Update]) -> None:
    if update is None or update.effective_user is None:
        return
    try:
        ensure_user(update.effective_user.id)
    except Exception as exc:
        log.warning("ensure_user failed for %s: %s", update.effective_user.id, exc)
    if redis_client is None:
        return
    try:
        await add_user(redis_client, update.effective_user)
    except Exception as exc:  # pragma: no cover - defensive logging
        log.warning("Failed to store user %s in Redis: %s", update.effective_user.id, exc)


async def process_promo_submission(
    update: Update,
    ctx: ContextTypes.DEFAULT_TYPE,
    code_input: str,
) -> None:
    message = update.effective_message
    chat = update.effective_chat
    user = update.effective_user
    state_dict = state(ctx)
    state_dict["mode"] = None

    if message is None or chat is None or user is None:
        return

    status, balance_after = activate_fixed_promo(user.id, code_input)

    if status == "invalid":
        await message.reply_text("Такого промокода нет.")
        return

    if status == "already_used":
        await message.reply_text("⚠️ Этот промокод уже был использован.")
        return

    if status != "ok" or balance_after is None:
        await message.reply_text("⚠️ Временная ошибка, попробуйте позже.")
        return

    _set_cached_balance(ctx, balance_after)

    await send_ok_sticker(ctx, "promo", balance_after, chat_id=chat.id)

    try:
        await show_main_menu(chat.id, ctx)
    except Exception as exc:  # pragma: no cover - defensive logging
        log.warning("Failed to refresh main menu after promo for %s: %s", user.id, exc)

    try:
        await refresh_balance_card_if_open(
            user.id,
            chat.id,
            ctx=ctx,
            state_dict=state_dict,
            reply_markup=balance_menu_kb(),
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        log.warning("Failed to refresh balance card after promo for %s: %s", user.id, exc)


def detect_lang(text: str) -> str:
    return "ru" if _CYRILLIC_RE.search(text or "") else "en"


async def _download_telegram_file(url: str, timeout: float = 120.0) -> bytes:
    timeout_cfg = ClientTimeout(total=timeout)
    async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
        async with session.get(url) as resp:
            resp.raise_for_status()
            return await resp.read()


def _should_convert_to_wav(mime: Optional[str], file_path: Optional[str]) -> bool:
    mime_lower = (mime or "").lower()
    if "ogg" in mime_lower or "opus" in mime_lower:
        return True
    path_lower = (file_path or "").lower()
    return path_lower.endswith((".ogg", ".oga", ".opus"))


def _voice_preview(text: str, limit: int = 3000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "…."


def _voice_lang_hint(message: Message, user: Optional[User]) -> str:
    if message.caption:
        return detect_lang(message.caption)
    if user and user.language_code:
        code = user.language_code.lower()
        if code.startswith("ru"):
            return "ru"
    if user:
        parts = [user.first_name or "", user.last_name or ""]
        joined = " ".join(part for part in parts if part)
        if joined:
            return detect_lang(joined)
    return "en"


async def chatgpt_smalltalk(text: str, chat_id: int) -> str:
    if openai is None or not OPENAI_API_KEY:
        raise RuntimeError("ChatGPT is not configured")

    log.debug("chat-smalltalk | chat=%s", chat_id)

    def _sync_call() -> str:
        response = openai.ChatCompletion.create(  # type: ignore[union-attr]
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful, concise assistant."},
                {"role": "user", "content": text},
            ],
            temperature=0.5,
            max_tokens=700,
        )
        return response["choices"][0]["message"]["content"].strip()

    return await asyncio.to_thread(_sync_call)


async def chat_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await ensure_user_record(update)
    chat = update.effective_chat
    user = update.effective_user
    if not chat or not user:
        return
    set_mode(user.id, True)
    _mode_set(chat.id, MODE_CHAT)
    try:
        await safe_send_text(
            ctx.bot,
            chat.id,
            md2_escape(
                "💬 Обычный чат включён. Пишите вопрос! /reset — очистить контекст.\n"
                "🎙️ Можно присылать голосовые — я их распознаю."
            ),
        )
    except Exception as exc:
        log.warning("chat.command_hint_failed | chat=%s err=%s", chat.id, exc)


async def chat_reset_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await ensure_user_record(update)
    chat = update.effective_chat
    user = update.effective_user
    if not chat or not user:
        return
    clear_ctx(user.id)
    set_mode(user.id, True)
    _mode_set(chat.id, MODE_CHAT)
    try:
        await safe_send_text(ctx.bot, chat.id, md2_escape("🧹 Контекст очищен."))
    except Exception as exc:
        log.warning("chat.reset_notify_failed | chat=%s err=%s", chat.id, exc)


async def chat_history_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await ensure_user_record(update)
    chat = update.effective_chat
    user = update.effective_user
    if not chat or not user:
        return
    set_mode(user.id, True)
    _mode_set(chat.id, MODE_CHAT)
    history = load_ctx(user.id)
    last_items = history[-5:]
    if last_items:
        chunks = []
        for item in last_items:
            role = item.get("role")
            icon = "🧍" if role == "user" else "🤖"
            content = str(item.get("content", ""))
            if len(content) > 400:
                content = content[:400] + "…"
            chunks.append(f"{icon} {md2_escape(content)}")
        body = "\n\n".join(chunks)
    else:
        body = "_пусто_"
    header = "*История \\(последние 5\\):*"
    try:
        await safe_send_text(ctx.bot, chat.id, f"{header}\n{body}")
    except Exception as exc:
        log.warning("chat.history_send_failed | chat=%s err=%s", chat.id, exc)


def _acquire_click_lock(user_id: Optional[int], action: str) -> bool:
    if not user_id or not REDIS_LOCK_ENABLED or not redis_client:
        return True
    key = _rk("lock", str(user_id), action)
    try:
        acquired = redis_client.set(key, str(int(time.time())), nx=True, px=10_000)
        return bool(acquired)
    except Exception as exc:
        log.warning("redis click lock error: %s", exc)
        return True

# Ledger storage (Postgres / memory)
ledger_storage = LedgerStorage(DATABASE_URL, backend=LEDGER_BACKEND)


def _ops_state(ctx: ContextTypes.DEFAULT_TYPE) -> Dict[str, Any]:
    ops = ctx.user_data.get("__ops__")
    if not isinstance(ops, dict):
        ops = {}
        ctx.user_data["__ops__"] = ops
    return ops


def _make_fingerprint(payload: Dict[str, Any]) -> str:
    try:
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    except Exception:
        canonical = repr(payload)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _ensure_operation(ctx: ContextTypes.DEFAULT_TYPE, key: str) -> Tuple[str, bool]:
    ops = _ops_state(ctx)
    entry = ops.get(key)
    if isinstance(entry, dict) and "op_id" in entry:
        return str(entry["op_id"]), False
    op_id = f"{key}:{uuid.uuid4().hex}"
    ops[key] = {"op_id": op_id}
    return op_id, True


def _update_operation(ctx: ContextTypes.DEFAULT_TYPE, key: str, **fields: Any) -> None:
    ops = _ops_state(ctx)
    entry = ops.get(key)
    if isinstance(entry, dict):
        entry.update(fields)
    else:
        ops[key] = {**fields}


def _clear_operation(ctx: ContextTypes.DEFAULT_TYPE, key: str) -> None:
    _ops_state(ctx).pop(key, None)


def _set_cached_balance(ctx: ContextTypes.DEFAULT_TYPE, value: int) -> None:
    ctx.user_data["balance"] = int(value)


def get_user_id(ctx: ContextTypes.DEFAULT_TYPE) -> Optional[int]:
    try:
        return ctx._user_id_and_data[0]  # type: ignore[attr-defined]
    except Exception:
        return None


def get_user_balance_value(ctx: ContextTypes.DEFAULT_TYPE, force_refresh: bool = False) -> int:
    uid = get_user_id(ctx)
    if not uid:
        return 0

    if not force_refresh:
        cached = ctx.user_data.get("balance")
        if cached is not None:
            try:
                return int(cached)
            except (TypeError, ValueError):
                pass

    balance = _safe_get_balance(uid)
    _set_cached_balance(ctx, balance)
    return balance


def credit_tokens(
    ctx: ContextTypes.DEFAULT_TYPE,
    amount: int,
    reason: str,
    op_id: str,
    meta: Optional[Dict[str, Any]] = None,
) -> LedgerOpResult:
    uid = get_user_id(ctx)
    if not uid:
        raise RuntimeError("Cannot credit tokens without user id")
    result = ledger_storage.credit(uid, amount, reason, op_id, meta)
    _set_cached_balance(ctx, result.balance)
    return result


def refund(
    ctx: ContextTypes.DEFAULT_TYPE,
    amount: int,
    op_id: str,
    meta: Optional[Dict[str, Any]] = None,
    *,
    ledger_reason: str = "veo_refund",
) -> Optional[LedgerOpResult]:
    meta = meta or {}
    try:
        result = credit_tokens(ctx, amount, ledger_reason, op_id, meta)
        log_evt(
            "KIE_REFUND",
            op_id=op_id,
            amount=amount,
            ledger_reason=ledger_reason,
            reason=meta.get("reason"),
        )
        return result
    except Exception as exc:
        log.exception("Refund failed (op=%s): %s", op_id, exc)
        return None


def try_charge(
    ctx: ContextTypes.DEFAULT_TYPE,
    need: int,
    reason: str,
    op_id: str,
    meta: Optional[Dict[str, Any]] = None,
) -> Tuple[str, int, Optional[LedgerOpResult]]:
    uid = get_user_id(ctx)
    if not uid:
        return "no_user", 0, None
    try:
        result = ledger_storage.debit(uid, need, reason, op_id, meta)
    except InsufficientBalance as exc:
        _set_cached_balance(ctx, exc.balance)
        return "insufficient", exc.balance, None
    _set_cached_balance(ctx, result.balance)
    status = "applied" if result.applied else "duplicate"
    return status, result.balance, result


def rename_operation(op_id: str, new_op_id: str, extra_meta: Optional[Dict[str, Any]] = None) -> bool:
    if not op_id or not new_op_id:
        return False
    return ledger_storage.rename_operation(op_id, new_op_id, extra_meta)

# ==========================
#   Helpers / storage
# ==========================
def join_url(base: str, path: str) -> str:
    u = f"{base.rstrip('/')}/{path.lstrip('/')}"
    return u.replace("://", "§§").replace("//", "/").replace("§§", "://")


async def _send_with_retry(func: Callable[[], Awaitable[Any]], *, attempts: int = 3):
    delay = 0.0
    for attempt in range(attempts):
        try:
            if delay:
                await asyncio.sleep(delay)
            return await func()
        except RetryAfter as exc:
            retry_after = getattr(exc, "retry_after", None)
            try:
                delay = float(retry_after) if retry_after is not None else 3.0
            except (TypeError, ValueError):
                delay = 3.0
            if delay <= 0:
                delay = 3.0
            continue
        except Forbidden as exc:
            log.warning("Telegram send forbidden: %s", exc)
            return None
        except BadRequest as exc:
            log.warning("Telegram send bad request: %s", exc)
            return None
    return None


def _kie_auth_header() -> Dict[str, str]:
    tok = (KIE_API_KEY or "").strip()
    if tok and not tok.lower().startswith("bearer "):
        tok = f"Bearer {tok}"
    return {"Authorization": tok} if tok else {}

def _kie_headers(method: str, request_id: str) -> Dict[str, str]:
    headers: Dict[str, str] = {"Accept": "application/json"}
    headers.update(_kie_auth_header())
    if method in {"POST", "PUT", "PATCH"}:
        headers["Content-Type"] = "application/json"
    headers["X-Request-Id"] = request_id
    return headers

def _kie_request(
    method: str,
    path: str,
    *,
    json_payload: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = 50,
    request_id: Optional[str] = None,
) -> Tuple[int, Dict[str, Any], str]:
    method = method.upper()
    req_id = request_id or str(uuid.uuid4())
    url = join_url(KIE_BASE_URL, path)
    headers = _kie_headers(method, req_id)
    started = time.time()
    try:
        resp = requests.request(method, url, json=json_payload, params=params, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        elapsed = round((time.time() - started) * 1000)
        event("KIE_HTTP_ERROR", method=method, url=url, request_id=req_id, error=str(exc), elapsed_ms=elapsed)
        return 0, {"error": str(exc)}, req_id

    try:
        payload = resp.json()
        if not isinstance(payload, dict):
            payload = {"data": payload}
    except ValueError:
        payload = {"error": resp.text}

    elapsed = round((time.time() - started) * 1000)
    code = _extract_response_code(payload, resp.status_code)
    msg = payload.get("msg") or payload.get("message") or payload.get("error")
    task_id = _extract_task_id(payload)
    event(
        "KIE_HTTP",
        method=method,
        url=url,
        status=resp.status_code,
        code=code,
        request_id=req_id,
        task_id=task_id,
        message=msg,
        elapsed_ms=elapsed,
        payload_keys=list(json_payload.keys()) if isinstance(json_payload, dict) else None,
    )
    return resp.status_code, payload, req_id

def _extract_task_id(j: Dict[str, Any]) -> Optional[str]:
    data = j.get("data") or {}
    for k in ("taskId", "taskid", "id"):
        if j.get(k): return str(j[k])
        if data.get(k): return str(data[k])
    return None

def _coerce_url_list(value) -> List[str]:
    urls: List[str] = []
    def add(u: str):
        if isinstance(u, str):
            s = u.strip()
            if s.startswith("http"): urls.append(s)
    if not value: return urls
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("["):
            try:
                for v in json.loads(s):
                    if isinstance(v, str): add(v)
            except Exception:
                add(s)
        else: add(s)
        return urls
    if isinstance(value, list):
        for v in value:
            if isinstance(v, str): add(v)
            elif isinstance(v, dict):
                u = v.get("resultUrl") or v.get("originUrl") or v.get("url")
                if isinstance(u, str): add(u)
        return urls
    if isinstance(value, dict):
        for k in ("resultUrl", "originUrl", "url"):
            u = value.get(k)
            if isinstance(u, str): add(u)
    return urls


_MJ_ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
_MJ_MIN_VALID_BYTES = 1024
_MJ_MAX_DOCUMENT_BYTES = 50 * 1024 * 1024
_MJ_DOWNLOAD_RETRY_COUNT = 2

_MJ_SIGNED_QUERY_KEYS = {
    "signature",
    "expires",
    "awsaccesskeyid",
    "x-amz-signature",
    "x-amz-expires",
    "x-amz-credential",
    "x-amz-security-token",
    "x-amz-date",
    "token",
    "sig",
    "se",
    "sp",
    "sv",
    "sr",
    "skoid",
    "sktid",
    "skt",
    "skv",
    "st",
}

_MJ_UI_TEXTS = {
    "gallery_ready": {"ru": "Галерея сгенерирована.", "en": "Gallery generated."},
    "gallery_retry": {"ru": "🔁 Сгенерировать ещё", "en": "🔁 Generate more"},
    "gallery_back": {"ru": "🏠 Назад в меню", "en": "🏠 Back to menu"},
    "upscale_entry": {"ru": "✨ Улучшить качество", "en": "✨ Improve quality"},
    "upscale_choose": {
        "ru": "Выберите фотографию для апскейла:",
        "en": "Pick a photo to upscale:",
    },
    "upscale_repeat": {
        "ru": "Повторить апскейл",
        "en": "Upscale again",
    },
    "upscale_need_photo": {
        "ru": "Пришлите фото (файлом лучше) — сделаю апскейл.",
        "en": "Send a photo (better as a file) and I will upscale it.",
    },
    "upscale_ready": {
        "ru": "Готово! Отдал файл без сжатия. Нужен другой кадр?",
        "en": "Done! Sent the file without compression. Need another frame?",
    },
    "upscale_processing": {
        "ru": "⏳ Запускаю апскейл…",
        "en": "⏳ Starting the upscale…",
    },
    "upscale_in_progress": {
        "ru": "⏳ Уже делаю апскейл этого кадра.",
        "en": "⏳ Already working on this frame.",
    },
}


def _mj_ui_text(key: str, locale: str) -> str:
    data = _MJ_UI_TEXTS.get(key, {})
    normalized = (locale or "ru").lower()
    if normalized.startswith("en"):
        return data.get("en", data.get("ru", ""))
    return data.get("ru", "")


def _mj_content_type_extension(content_type: Optional[str]) -> Optional[str]:
    if not content_type:
        return None
    base = content_type.split(";", 1)[0].strip().lower()
    mapping = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }
    return mapping.get(base)


def _determine_user_locale(user: Optional[User]) -> str:
    if user and isinstance(user.language_code, str):
        lowered = user.language_code.lower()
        if lowered.startswith("en"):
            return "en"
    return "ru"


def _normalize_mj_grid(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, dict):
        return None
    task_id = value.get("task_id") or value.get("taskId")
    urls = value.get("result_urls") or value.get("urls") or value.get("imageUrls")
    if not isinstance(task_id, str):
        return None
    if isinstance(urls, list):
        normalized_urls = [str(item) for item in urls if isinstance(item, str)]
    else:
        normalized_urls = []
    if not normalized_urls:
        return None
    prompt_raw = value.get("prompt")
    prompt = prompt_raw.strip() if isinstance(prompt_raw, str) else None
    if not prompt:
        prompt = _extract_mj_prompt(value)
    grid: Dict[str, Any] = {"task_id": task_id, "result_urls": normalized_urls}
    if prompt:
        grid["prompt"] = prompt
    return grid


def _store_last_mj_grid(
    state_dict: Dict[str, Any],
    user_id: Optional[int],
    task_id: str,
    urls: Sequence[str],
    *,
    prompt: Optional[str] = None,
) -> None:
    grid: Dict[str, Any] = {"task_id": task_id, "result_urls": list(urls)}
    prompt_clean = prompt.strip() if isinstance(prompt, str) else None
    if prompt_clean:
        grid["prompt"] = prompt_clean
    state_dict["mj_last_grid"] = grid
    if user_id:
        try:
            set_last_mj_grid(int(user_id), task_id, list(urls), prompt=prompt_clean)
        except Exception as exc:
            mj_log.warning("mj.grid.cache_fail | user=%s err=%s", user_id, exc)


def _load_last_mj_grid(state_dict: Dict[str, Any], user_id: Optional[int]) -> Optional[Dict[str, Any]]:
    cached = _normalize_mj_grid(state_dict.get("mj_last_grid"))
    if cached:
        return cached
    if user_id is None:
        return None
    try:
        persisted = get_last_mj_grid(int(user_id))
    except Exception as exc:
        mj_log.warning("mj.grid.cache_fetch_fail | user=%s err=%s", user_id, exc)
        persisted = None
    normalized = _normalize_mj_grid(persisted)
    if normalized:
        state_dict["mj_last_grid"] = normalized
    return normalized


def _save_mj_grid_snapshot(grid_id: str, urls: Sequence[str], *, prompt: Optional[str] = None) -> None:
    normalized = [str(u) for u in urls if isinstance(u, str) and u]
    if not normalized:
        return
    payload: Dict[str, Any] = {
        "task_id": str(grid_id),
        "result_urls": normalized,
    }
    prompt_clean = prompt.strip() if isinstance(prompt, str) else None
    if prompt_clean:
        payload["prompt"] = prompt_clean
    key = _MJ_GRID_CACHE_KEY_TMPL.format(grid_id=grid_id)
    try:
        cache_set(key, json.dumps(payload, ensure_ascii=False), _MJ_GRID_CACHE_TTL)
    except Exception as exc:  # pragma: no cover - defensive fallback
        mj_log.warning("mj.grid.redis_save_fail | grid=%s err=%s", grid_id, exc)


def _load_mj_grid_snapshot(grid_id: str) -> Optional[Dict[str, Any]]:
    grid_key = str(grid_id or "").strip()
    if not grid_key:
        return None
    key = _MJ_GRID_CACHE_KEY_TMPL.format(grid_id=grid_key)
    raw = cache_get(key)
    if not raw:
        return None
    try:
        doc = json.loads(raw)
    except Exception:
        mj_log.warning("mj.grid.redis_parse_fail | grid=%s", grid_key)
        return None
    if not isinstance(doc, dict):
        return None
    urls_raw = doc.get("result_urls") or doc.get("urls")
    if not isinstance(urls_raw, list):
        return None
    normalized = [str(u) for u in urls_raw if isinstance(u, str) and u]
    if not normalized:
        return None
    task_id = doc.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        task_id = grid_key
    result: Dict[str, Any] = {
        "task_id": task_id,
        "result_urls": normalized,
    }
    prompt_raw = doc.get("prompt")
    if isinstance(prompt_raw, str) and prompt_raw.strip():
        result["prompt"] = prompt_raw.strip()
    return result


def _mj_guess_extension(url: str, default: str = ".jpeg") -> str:
    try:
        path = urlparse(url).path.lower()
    except Exception:
        return default
    for ext in (".jpeg", ".jpg", ".png", ".webp"):
        if path.endswith(ext):
            return ext
    return default


def _mj_document_filename(index: int, url: Optional[str] = None, *, suffix: str = "") -> str:
    ext = _mj_guess_extension(url or "")
    if ext not in _MJ_ALLOWED_EXTENSIONS:
        ext = ".jpeg"
    base = f"midjourney_{index:02d}"
    if suffix:
        base = f"{base}_{suffix}"
    normalized = unicodedata.normalize("NFKD", base)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", ascii_only).strip("._")
    if not sanitized:
        sanitized = f"midjourney_{index:02d}"
    if len(sanitized) > 60:
        sanitized = sanitized[:60].rstrip("._-") or sanitized[:60]
    return f"{sanitized}{ext}"


def _mj_upscale_keyboard(count: int, locale: str, *, include_repeat: bool = False) -> InlineKeyboardMarkup:
    buttons: List[List[InlineKeyboardButton]] = []
    row: List[InlineKeyboardButton] = []
    for idx in range(count):
        row.append(InlineKeyboardButton(f"U{idx + 1}", callback_data=f"mj_upscale:select:{idx}"))
        if len(row) == 2:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)
    if include_repeat:
        buttons.append([InlineKeyboardButton(_mj_ui_text("upscale_repeat", locale), callback_data="mj_upscale:repeat")])
    buttons.append([InlineKeyboardButton(_mj_ui_text("back", locale), callback_data="back")])
    return InlineKeyboardMarkup(buttons)


def _guess_aspect_ratio_from_size(width: Optional[int], height: Optional[int]) -> str:
    if not width or not height or width <= 0 or height <= 0:
        return "1:1"
    ratio = width / height
    candidates = [
        ("1:1", 1.0),
        ("16:9", 16 / 9),
        ("9:16", 9 / 16),
        ("3:2", 3 / 2),
        ("2:3", 2 / 3),
        ("4:5", 4 / 5),
        ("5:4", 5 / 4),
        ("7:4", 7 / 4),
        ("4:7", 4 / 7),
    ]
    best = min(candidates, key=lambda item: abs(item[1] - ratio))
    return best[0]


def _mj_guess_filename(url: str, index: int, content_type: Optional[str]) -> str:
    ext = _mj_guess_extension(url)
    if ext not in _MJ_ALLOWED_EXTENSIONS:
        ext = _mj_content_type_extension(content_type) or ".jpeg"
    return _mj_document_filename(index + 1, suffix="", url=f"dummy{ext}")


def _normalize_mj_url(url: str) -> str:
    try:
        parsed = urlparse(url)
    except Exception:
        return url
    if not parsed.scheme or not parsed.netloc:
        return url
    query_items = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        lowered = key.lower()
        if lowered in _MJ_SIGNED_QUERY_KEYS or lowered.startswith("x-amz-"):
            query_items.append((key, value))
        elif lowered in {"width", "height", "format", "quality", "updated", "v", "cb"}:
            continue
        else:
            query_items.append((key, value))
    normalized_query = urlencode(query_items, doseq=True)
    sanitized = parsed._replace(query=normalized_query)
    return urlunparse(sanitized)


def _download_mj_image_bytes(url: str, index: int) -> Optional[Tuple[bytes, str, str, str]]:
    normalized_url = _normalize_mj_url(url)
    last_error: Optional[str] = None
    for attempt in range(_MJ_DOWNLOAD_RETRY_COUNT + 1):
        headers: Dict[str, str] = {}
        request_url = normalized_url
        if attempt:
            headers["Cache-Control"] = "no-cache"
            try:
                parsed = urlparse(normalized_url)
                existing = parse_qsl(parsed.query, keep_blank_values=True)
                existing.append(("r", uuid.uuid4().hex))
                request_url = urlunparse(parsed._replace(query=urlencode(existing)))
            except Exception:
                request_url = normalized_url
        try:
            resp = requests.get(request_url, timeout=15, headers=headers)
        except requests.RequestException as exc:
            last_error = str(exc)
            mj_log.warning(
                "mj.album.download_fail",
                extra={"meta": {"url": request_url, "index": index, "error": str(exc)}},
            )
            continue
        if resp.status_code != 200:
            last_error = f"status:{resp.status_code}"
            mj_log.warning(
                "mj.album.download_fail",
                extra={
                    "meta": {"url": request_url, "index": index, "status": resp.status_code},
                },
            )
            continue
        data = resp.content or b""
        if len(data) <= _MJ_MIN_VALID_BYTES:
            last_error = f"payload_too_small:{len(data)}"
            mj_log.warning(
                "mj.album.download_retry",
                extra={
                    "meta": {
                        "url": request_url,
                        "index": index,
                        "reason": "small_payload",
                        "bytes": len(data),
                        "attempt": attempt,
                    }
                },
            )
            continue
        content_type = resp.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if not content_type:
            content_type = "application/octet-stream"
        filename = _mj_guess_filename(normalized_url, index - 1, content_type)
        return data, filename, content_type, normalized_url

    mj_log.warning(
        "mj.album.download_fail_final",
        extra={"meta": {"url": normalized_url, "index": index, "error": last_error or "unknown"}},
    )
    return None



def _trim_caption_text(text: str, limit: int = 1024) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    ellipsis = "…"
    if limit <= len(ellipsis):
        return ellipsis[:limit]
    return text[: limit - len(ellipsis)] + ellipsis


def _banana_caption(prompt: str) -> str:
    normalized = re.sub(r"\s+", " ", (prompt or "")).strip()
    if not normalized:
        snippet = "—"
    else:
        snippet = normalized[:120]
        if len(normalized) > 120:
            snippet = snippet[:117].rstrip() + "…"
    return f"🍌 Banana\n• Промпт: \"{snippet}\""


def _banana_guess_suffix(url: str, content_type: Optional[str]) -> str:
    path = urlparse(url or "").path.lower()
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        if path.endswith(ext):
            return ext
    if content_type:
        lowered = content_type.lower()
        if "png" in lowered:
            return ".png"
        if "jpeg" in lowered or "jpg" in lowered:
            return ".jpg"
        if "webp" in lowered:
            return ".webp"
    return ".png"


def _download_binary(url: str, *, timeout: int = 180) -> tuple[bytes, Optional[str]]:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content, response.headers.get("Content-Type")


async def _deliver_banana_media(
    bot: Any,
    *,
    chat_id: int,
    user_id: int,
    file_path: Path,
    caption: str,
    reply_markup: Optional[Any] = None,
    send_document: bool = True,
) -> bool:
    try:
        file_size = file_path.stat().st_size
    except OSError:
        file_size = 0

    log.info(
        "banana.result.saved",
        extra={
            "meta": {
                "user_id": user_id,
                "chat_id": chat_id,
                "file_size": file_size,
                "path": str(file_path),
            }
        },
    )

    sent_any = False
    photo_start = time.monotonic()
    photo_message_id: Optional[int] = None
    try:
        with file_path.open("rb") as handle:
            message = await safe_send_photo(
                bot,
                chat_id=chat_id,
                photo=InputFile(handle, filename=file_path.name),
                caption=caption,
                reply_markup=reply_markup,
                kind="banana_photo",
            )
        duration_ms = int((time.monotonic() - photo_start) * 1000)
        photo_message_id = getattr(message, "message_id", None)
        sent_any = True
        log.info(
            "banana.send.photo.ok",
            extra={
                "meta": {
                    "user_id": user_id,
                    "chat_id": chat_id,
                    "message_id": photo_message_id,
                    "file_size": file_size,
                    "duration_ms": duration_ms,
                }
            },
        )
    except Exception as exc:
        duration_ms = int((time.monotonic() - photo_start) * 1000)
        log.warning(
            "banana.send.photo.fail",
            extra={
                "meta": {
                    "user_id": user_id,
                    "chat_id": chat_id,
                    "file_size": file_size,
                    "duration_ms": duration_ms,
                    "error": str(exc),
                }
            },
        )

    doc_sent = False
    if send_document:
        doc_start = time.monotonic()
        suffix = file_path.suffix.lower() or ".jpg"
        doc_filename = f"result{suffix}"
        try:
            with file_path.open("rb") as handle:
                message = await safe_send_document(
                    bot,
                    chat_id=chat_id,
                    document=InputFile(handle, filename=doc_filename),
                    caption=None,
                    reply_markup=reply_markup,
                    kind="banana_document",
                    disable_notification=True,
                )
            duration_ms = int((time.monotonic() - doc_start) * 1000)
            doc_message_id = getattr(message, "message_id", None)
            doc_sent = True
            sent_any = True
            log.info(
                "send.document",
                extra={
                    "meta": {
                        "ctx_user_id": user_id,
                        "chat_id": chat_id,
                        "message_id": doc_message_id,
                        "file_bytes": file_size,
                        "file_name": doc_filename,
                    }
                },
            )
            log.info(
                "banana.send.document.ok",
                extra={
                    "meta": {
                        "user_id": user_id,
                        "chat_id": chat_id,
                        "message_id": doc_message_id,
                        "file_size": file_size,
                        "duration_ms": duration_ms,
                    }
                },
            )
        except Exception as exc:
            duration_ms = int((time.monotonic() - doc_start) * 1000)
            log.info(
                "send.document",
                extra={
                    "meta": {
                        "ctx_user_id": user_id,
                        "chat_id": chat_id,
                        "file_bytes": file_size,
                        "file_name": doc_filename,
                        "error": str(exc),
                    }
                },
            )
            log.warning(
                "banana.send.document.fail",
                extra={
                    "meta": {
                        "user_id": user_id,
                        "chat_id": chat_id,
                        "file_size": file_size,
                        "duration_ms": duration_ms,
                        "error": str(exc),
                        "photo_message_id": photo_message_id,
                    }
                },
            )
    else:
        log.info(
            "banana.send.document.skip",
            extra={
                "meta": {
                    "user_id": user_id,
                    "chat_id": chat_id,
                    "file_size": file_size,
                    "photo_message_id": photo_message_id,
                }
            },
        )

    cleanup_temp([file_path])
    return sent_any or doc_sent



async def _deliver_mj_grid_documents(
    ctx: ContextTypes.DEFAULT_TYPE,
    *,
    chat_id: int,
    user_id: Optional[int],
    grid_id: str,
    urls: Sequence[str],
    prompt: Optional[str] = None,
) -> bool:
    normalized = [str(u) for u in urls if isinstance(u, str) and u]
    if not normalized:
        return False

    mj_log.info(
        "mj.grid.delivery_start",
        extra={
            "meta": {
                "chat_id": chat_id,
                "user_id": user_id,
                "grid_id": grid_id,
                "count": len(normalized),
            }
        },
    )

    sent_count = 0
    gallery_payload: List[Dict[str, Any]] = []
    for index, url in enumerate(normalized, start=1):
        download = await asyncio.to_thread(_download_mj_image_bytes, url, index)
        if not download:
            await ctx.bot.send_message(
                chat_id,
                "Не удалось получить файл, пробуем ещё раз.",
            )
            mj_log.warning(
                "mj.grid.document_empty",
                extra={
                    "meta": {
                        "chat_id": chat_id,
                        "user_id": user_id,
                        "grid_id": grid_id,
                        "index": index,
                        "reason": "download_failed",
                    }
                },
            )
            continue

        data, filename, mime, source_url = download
        if len(data) > _MJ_MAX_DOCUMENT_BYTES:
            await ctx.bot.send_message(
                chat_id,
                "Файл слишком большой для отправки в Telegram документом, попробуйте другой запрос/соотношение сторон.",
            )
            mj_log.warning(
                "mj.grid.document_too_large",
                extra={
                    "meta": {
                        "chat_id": chat_id,
                        "user_id": user_id,
                        "grid_id": grid_id,
                        "index": index,
                        "filename": filename,
                        "bytes": len(data),
                    }
                },
            )
            continue
        start_ts = time.monotonic()
        try:
            document = InputFile(io.BytesIO(data), filename=filename)
            message = await ctx.bot.send_document(
                chat_id,
                document=document,
            )
        except Exception as exc:
            duration_ms = int((time.monotonic() - start_ts) * 1000)
            mj_log.warning(
                "mj.grid.document_fail",
                extra={
                    "meta": {
                        "chat_id": chat_id,
                        "user_id": user_id,
                        "grid_id": grid_id,
                        "index": index,
                        "filename": filename,
                        "duration_ms": duration_ms,
                        "error": str(exc),
                    }
                },
            )
            continue

        sent_count += 1
        sent_message_id = getattr(message, "message_id", 0)
        gallery_payload.append(
            {
                "file_name": filename,
                "source_url": source_url,
                "bytes_len": len(data),
                "mime": mime,
                "sent_message_id": int(sent_message_id or 0),
            }
        )
        duration_ms = int((time.monotonic() - start_ts) * 1000)
        mj_log.info(
            "mj.grid.document_sent",
            extra={
                "meta": {
                    "chat_id": chat_id,
                    "user_id": user_id,
                    "grid_id": grid_id,
                    "index": index,
                    "filename": filename,
                    "bytes": len(data),
                    "duration_ms": duration_ms,
                }
            },
        )

    if sent_count == 0:
        return False

    keyboard = mj_upscale_root_keyboard(grid_id)
    locale = state(ctx).get("mj_locale") or "ru"
    try:
        gallery_message = await ctx.bot.send_message(
            chat_id,
            _mj_ui_text("gallery_ready", locale),
            reply_markup=keyboard,
        )
    except Exception as exc:
        mj_log.warning(
            "mj.grid.menu_send_fail",
            extra={
                "meta": {
                    "chat_id": chat_id,
                    "user_id": user_id,
                    "grid_id": grid_id,
                    "error": str(exc),
                }
            },
        )
        return False

    if gallery_payload:
        set_mj_gallery(
            chat_id,
            getattr(gallery_message, "message_id", 0),
            gallery_payload,
        )

    _save_mj_grid_snapshot(grid_id, normalized, prompt=prompt)

    mj_log.info(
        "mj.grid.delivery_done",
        extra={
            "meta": {
                "chat_id": chat_id,
                "user_id": user_id,
                "grid_id": grid_id,
                "sent": sent_count,
            }
        },
    )
    return True


def _extract_result_url(data: Dict[str, Any]) -> Optional[str]:
    visited: set[int] = set()
    stack: List[Any] = [data]

    def _maybe_parse_json(text: str) -> Any:
        s = text.strip()
        if not s:
            return None
        if len(s) > 20000:
            return None
        if (s[0] in "[{" and s[-1] in "]}"):
            try:
                return json.loads(s)
            except Exception:
                return None
        return None

    while stack:
        current = stack.pop()
        if isinstance(current, (list, tuple, set)):
            for item in current:
                stack.append(item)
            continue
        if isinstance(current, dict):
            for key in (
                "resultUrls",
                "resultUrl",
                "originUrls",
                "originUrl",
                "videoUrls",
                "videoUrl",
                "videos",
                "urls",
                "url",
                "downloadUrl",
                "fileUrl",
                "cdnUrl",
                "outputUrl",
                "imageUrls",
                "imageUrl",
                "imageUrlList",
                "image_url",
                "image_urls",
                "finalImageUrl",
                "finalImageUrls",
            ):
                if key in current:
                    stack.append(current[key])
            for value in current.values():
                stack.append(value)
            continue
        obj_id = id(current)
        if obj_id in visited:
            continue
        visited.add(obj_id)
        if isinstance(current, str):
            stripped = current.strip()
            if stripped.startswith("http"):
                return stripped
            parsed = _maybe_parse_json(stripped)
            if parsed is not None:
                stack.append(parsed)
    return None

def _parse_success_flag(data: Dict[str, Any]) -> Optional[int]:
    for key in ("successFlag", "state", "status", "taskStatus"):
        if key not in data:
            continue
        value = data.get(key)
        if value is None:
            continue
        if isinstance(value, bool):
            return 1 if value else 0
        try:
            return int(value)
        except (ValueError, TypeError):
            if isinstance(value, str):
                lower = value.lower()
                if lower in ("success", "succeeded", "finished", "done"):
                    return 1
                if lower in ("fail", "failed", "error", "cancelled", "canceled"):
                    return 2
                if lower in ("waiting", "queued", "processing", "running"):
                    return 0
    return None

def _kie_req_cache_key(task_id: str) -> str:
    return f"kie:req:{task_id}"

def _remember_kie_request_id(task_id: Optional[str], request_id: str):
    if not task_id or not request_id:
        return
    app_cache[_kie_req_cache_key(str(task_id))] = request_id

def _get_kie_request_id(task_id: str) -> Optional[str]:
    return app_cache.get(_kie_req_cache_key(str(task_id)))

def _clear_kie_request_id(task_id: str):
    app_cache.pop(_kie_req_cache_key(str(task_id)), None)

def log_evt(name: str, **kw) -> None:
    try:
        payload = json.dumps(kw, ensure_ascii=False, sort_keys=True)
    except Exception:
        payload = str(kw)
    log.info("EVT_%s | %s", name, payload)


def event(tag: str, **kw):
    log_evt(tag, **kw)


def kie_event(stage: str, **kw):
    log_evt(f"KIE_{stage}", **kw)

def tg_direct_file_url(bot_token: str, file_path: str) -> str:
    p = (file_path or "").strip()
    if p.startswith("http://") or p.startswith("https://"): return p
    return f"https://api.telegram.org/file/bot{bot_token}/{p.lstrip('/')}"

# ---------- User state ----------
WAIT_SUNO_TITLE = "WAIT_SUNO_TITLE"
WAIT_SUNO_STYLE = "WAIT_SUNO_STYLE"
WAIT_SUNO_LYRICS = "WAIT_SUNO_LYRICS"
WAIT_SUNO_REFERENCE = "WAIT_SUNO_REFERENCE"
IDLE_SUNO = "IDLE_SUNO"


_WAIT_LIMITS = {
    WaitKind.SUNO_TITLE: 300,
    WaitKind.SUNO_STYLE: 500,
    WaitKind.SUNO_LYRICS: 3000,
    WaitKind.VEO_PROMPT: 3000,
    WaitKind.MJ_PROMPT: 2000,
    WaitKind.BANANA_PROMPT: 2000,
    WaitKind.SORA2_PROMPT: 5000,
}

_WAIT_ALLOW_NEWLINES = {
    WaitKind.SUNO_STYLE,
    WaitKind.SUNO_LYRICS,
    WaitKind.VEO_PROMPT,
    WaitKind.MJ_PROMPT,
    WaitKind.BANANA_PROMPT,
    WaitKind.SORA2_PROMPT,
}

_WAIT_CLEAR_VALUES = {"-", "—"}

_wait_log = logging.getLogger("wait-input")


_SUNO_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_SUNO_TAG_RE = re.compile(r"<[^>]+>")


def _sanitize_suno_input(value: str, *, allow_newlines: bool) -> str:
    return normalize_input(value, allow_newlines=allow_newlines)


def _reset_suno_card_cache(state_dict: Dict[str, Any]) -> None:
    state_dict["_last_text_suno"] = None
    card_state = state_dict.get("suno_card")
    if isinstance(card_state, dict):
        card_state["last_text_hash"] = None
        card_state["last_markup_hash"] = None
        card_state["chat_id"] = None


def _reset_suno_start_flags(state_dict: Dict[str, Any]) -> None:
    state_dict["suno_start_clicked"] = False
    state_dict["suno_start_button_sent_ts"] = None
    state_dict["suno_can_start"] = False
    state_dict["suno_current_lyrics_hash"] = None


DEFAULT_STATE = {
    "mode": None, "aspect": "16:9", "model": None,
    "last_prompt": None, "last_image_url": None,
    "generating": False, "generation_id": None, "last_task_id": None,
    "last_ui_msg_id_menu": None,
    "last_ui_msg_id_bottom": None,
    "last_ui_msg_id_balance": None,
    "last_ui_msg_id_veo": None, "last_ui_msg_id_banana": None, "last_ui_msg_id_mj": None,
    "last_ui_msg_id_image_engine": None,
    "last_ui_msg_id_suno": None,
    "last_ui_msg_id_sora2": None,
    "banana_images": [],
    "mj_last_wait_ts": 0.0,
    "mj_generating": False, "last_mj_task_id": None,
    "mj_locale": None,
    "mj_last_grid": None,
    "mj_upscale_active": None,
    "active_generation_op": None,
    "mj_active_op_key": None,
    "banana_active_op_key": None,
    "_last_text_veo": None,
    "_last_text_banana": None,
    "_last_text_mj": None,
    "_last_text_image_engine": None,
    "_last_text_suno": None,
    "_last_text_sora2": None,
    "veo_duration_hint": None,
    "veo_lip_sync_required": False,
    "veo_voiceover_origin": None,
    "msg_ids": {},
    "last_panel": None,
    "suno_state": None,
    "suno_waiting_state": IDLE_SUNO,
    "suno_generating": False,
    "suno_waiting_enqueue": False,
    "suno_current_req_id": None,
    "suno_can_start": False,
    "suno_start_button_sent_ts": None,
    "suno_start_clicked": False,
    "suno_current_lyrics_hash": None,
    "suno_last_task_id": None,
    "suno_last_params": None,
    "suno_balance": None,
    "suno_flow": None,
    "suno_step": None,
    "suno_auto_lyrics_pending": False,
    "suno_lyrics_confirmed": False,
    "suno_cover_source_label": None,
    "suno_step_order": None,
    "suno_auto_lyrics_generated": False,
    "chat_hint_sent": False,
    "image_engine": None,
    "suno_card": {
        "msg_id": None,
        "last_text_hash": None,
        "last_markup_hash": None,
        "chat_id": None,
    },
    "sora2_prompt": None,
    "sora2_image_urls": [],
    "sora2_generating": False,
    "sora2_last_task_id": None,
    "sora2_wait_msg_id": None,
    "video_wait_message_id": None,
    "video_menu_msg_id": None,
}


def ensure_state(ctx: ContextTypes.DEFAULT_TYPE) -> Dict[str, Any]:
    user_data = getattr(ctx, "user_data", None)
    if not isinstance(user_data, dict):
        user_data = {}
        try:
            setattr(ctx, "user_data", user_data)
        except Exception:
            return {}
    state_dict = user_data.get("state")
    if not isinstance(state_dict, dict):
        state_dict = {}
        for key in DEFAULT_STATE.keys():
            if key in user_data:
                state_dict[key] = user_data.pop(key)
        user_data["state"] = state_dict
    return state_dict


def _apply_state_defaults(target: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(target, dict):
        target = {}
    for key, value in DEFAULT_STATE.items():
        if key not in target:
            if isinstance(value, list):
                target[key] = list(value)
            elif isinstance(value, dict):
                target[key] = dict(value)
            else:
                target[key] = value
            continue
        if isinstance(value, list) and not isinstance(target[key], list):
            target[key] = []
        elif isinstance(value, dict) and not isinstance(target[key], dict):
            target[key] = dict(value)
    if not isinstance(target.get("banana_images"), list):
        target["banana_images"] = []
    if not isinstance(target.get("msg_ids"), dict):
        target["msg_ids"] = {}
    return target


def _suno_log_preview(value: Optional[str], limit: int = 60) -> str:
    if not value:
        return "—"
    text = value.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


_SUNO_WAIT_TO_FIELD = {
    WAIT_SUNO_TITLE: "title",
    WAIT_SUNO_STYLE: "style",
    WAIT_SUNO_LYRICS: "lyrics",
    WAIT_SUNO_REFERENCE: "cover",
}

_SUNO_PROMPTS = {
    "title": "Введите короткое название трека. Отправьте /cancel, чтобы остановить.",
    "style": "Опишите стиль/теги (например, „эмбиент, мягкие барабаны“). Отправьте /cancel, чтобы остановить.",
    "lyrics": (
        f"Пришлите текст песни (до {LYRICS_MAX_LENGTH} символов) или отправьте /skip, чтобы сгенерировать автоматически."
    ),
    "cover": f"Пришлите аудио-файл (mp3/wav, до {COVER_MAX_AUDIO_MB} МБ) или ссылку на аудио (http/https).",
}


def _suno_inline_preview(value: Optional[str], *, limit: int = 50) -> str:
    if not value:
        return "—"
    collapsed = re.sub(r"\s+", " ", str(value)).strip()
    if not collapsed:
        return "—"
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: max(1, limit - 1)].rstrip() + "…"


def _suno_field_preview(state: SunoState, field: str) -> str:
    if field == "style":
        if suno_style_preview:
            try:
                preview = suno_style_preview(state.style, limit=120)
            except TypeError:
                preview = suno_style_preview(state.style)
            if preview:
                return preview
        return _suno_inline_preview(state.style, limit=80)
    if field == "lyrics":
        if suno_lyrics_preview:
            try:
                preview = suno_lyrics_preview(state.lyrics, limit=120)
            except TypeError:
                preview = suno_lyrics_preview(state.lyrics)
            if preview:
                return preview
        return _suno_inline_preview(state.lyrics, limit=80)
    if field == "cover":
        source = state.cover_source_label or state.cover_source_url or state.source_url
        return _suno_inline_preview(source, limit=60)
    if field == "title":
        return _suno_inline_preview(state.title, limit=80)
    value = getattr(state, field, None)
    return _suno_inline_preview(value, limit=60)


def _suno_prompt_text(field: str, suno_state_obj: SunoState) -> str:
    base = _SUNO_PROMPTS.get(field, "Введите значение.")
    current = _suno_field_preview(suno_state_obj, field)
    if field == "cover" and current == "—":
        return base
    return f"{base}\nСейчас: “{current}”"


def _suno_preview_for_log(value: Optional[str]) -> str:
    return _suno_inline_preview(value, limit=30)


def _suno_field_from_waiting(waiting_field: str) -> Optional[str]:
    return _SUNO_WAIT_TO_FIELD.get(waiting_field)


def _activate_wait_state(
    *,
    user_id: Optional[int],
    chat_id: Optional[int],
    card_msg_id: Optional[int],
    kind: WaitKind,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    if user_id is None or chat_id is None:
        return
    payload_meta: Dict[str, Any] = dict(meta or {})
    set_wait(
        user_id,
        kind.value,
        card_msg_id,
        chat_id=chat_id,
        meta=payload_meta,
    )


def _wait_preview(text: str) -> str:
    if not text:
        return "—"
    normalized = collapse_spaces(text.replace("\n", " "))
    return truncate_text(normalized, 120) or "—"


def is_command_or_button(message: Message) -> bool:
    text = message.text
    if not isinstance(text, str):
        return False
    return not should_capture_to_prompt(text)


async def _wait_acknowledge(message: Message) -> None:
    try:
        await message.reply_text("✅ Принято")
    except Exception:
        _wait_log.debug(
            "WAIT_ACK_FAILED",
            extra={
                "user_id": getattr(message.from_user, "id", None),
                "chat_id": getattr(message, "chat_id", None),
            },
        )


async def _apply_wait_state_input(
    ctx: ContextTypes.DEFAULT_TYPE,
    message: Message,
    wait_state: WaitInputState,
    *,
    user_id: Optional[int],
) -> bool:
    raw_text = message.text
    if raw_text is None:
        await message.reply_text("⚠️ Отправьте текстовое сообщение.")
        return True
    allowed, reason = classify_wait_input(raw_text)
    if not allowed and reason == "command_label":
        _wait_log.info(
            "WAIT_INPUT_IGNORE kind=%s reason=%s", wait_state.kind.value, reason
        )
        return False
    stripped = raw_text.strip()
    if stripped in _WAIT_CLEAR_VALUES:
        normalized = ""
    else:
        allow_newlines = wait_state.kind in _WAIT_ALLOW_NEWLINES
        normalized = normalize_input(raw_text, allow_newlines=allow_newlines)

    limit = _WAIT_LIMITS.get(wait_state.kind, 3000)
    cleaned = truncate_text(normalized, limit)
    truncated = cleaned != normalized

    _wait_log.info(
        "WAIT_INPUT kind=%s text_len=%s truncated=%s",
        wait_state.kind.value,
        len(cleaned),
        truncated,
    )

    handled = False

    if wait_state.kind in {WaitKind.SUNO_TITLE, WaitKind.SUNO_STYLE, WaitKind.SUNO_LYRICS}:
        suno_state_obj = load_suno_state(ctx)
        before_title = suno_state_obj.title
        before_style = suno_state_obj.style
        before_lyrics = suno_state_obj.lyrics
        if wait_state.kind == WaitKind.SUNO_TITLE:
            if cleaned:
                set_suno_title(suno_state_obj, cleaned)
            else:
                clear_suno_title(suno_state_obj)
        elif wait_state.kind == WaitKind.SUNO_STYLE:
            if cleaned:
                set_suno_style(suno_state_obj, cleaned)
            else:
                clear_suno_style(suno_state_obj)
        else:
            if cleaned:
                set_suno_lyrics(suno_state_obj, cleaned)
                suno_state_obj.mode = "lyrics"
            else:
                clear_suno_lyrics(suno_state_obj)
        save_suno_state(ctx, suno_state_obj)
        s = state(ctx)
        s["suno_state"] = suno_state_obj.to_dict()
        s["suno_waiting_state"] = IDLE_SUNO
        msg_id = await refresh_suno_card(ctx, wait_state.chat_id, s, price=PRICE_SUNO)
        if user_id is not None:
            if isinstance(msg_id, int):
                refresh_card_pointer(user_id, msg_id)
            else:
                state_card = s.get("last_ui_msg_id_suno")
                if isinstance(state_card, int):
                    refresh_card_pointer(user_id, state_card)
        after_title = suno_state_obj.title
        after_style = suno_state_obj.style
        after_lyrics = suno_state_obj.lyrics
        _wait_log.info(
            "WAIT_INPUT_SUNO kind=%s changed=%s", wait_state.kind.value,
            (
                (before_title or "") != (after_title or "")
                or (before_style or "") != (after_style or "")
                or (before_lyrics or "") != (after_lyrics or "")
            ),
        )
        handled = True
    elif wait_state.kind == WaitKind.VEO_PROMPT:
        s = state(ctx)
        s["last_prompt"] = cleaned or None
        s["_last_text_veo"] = None
        await show_veo_card(wait_state.chat_id, ctx)
        if user_id is not None:
            card_id = s.get("last_ui_msg_id_veo")
            if isinstance(card_id, int):
                refresh_card_pointer(user_id, card_id)
        handled = True
    elif wait_state.kind == WaitKind.MJ_PROMPT:
        s = state(ctx)
        s["last_prompt"] = cleaned or None
        s["_last_text_mj"] = None
        await show_mj_prompt_card(wait_state.chat_id, ctx)
        if user_id is not None:
            card_id = s.get("last_ui_msg_id_mj")
            if isinstance(card_id, int):
                refresh_card_pointer(user_id, card_id)
        handled = True
    elif wait_state.kind == WaitKind.SORA2_PROMPT:
        s = state(ctx)
        raw = message.text or ""
        if raw.strip().lower() == "clear":
            s["sora2_prompt"] = None
            s["sora2_image_urls"] = []
        else:
            urls = _extract_http_urls(raw)
            existing_urls = list(s.get("sora2_image_urls") or [])
            overflow = False
            for url in urls:
                if url not in existing_urls:
                    if len(existing_urls) >= SORA2_MAX_IMAGES:
                        overflow = True
                        break
                    existing_urls.append(url)
            if len(existing_urls) > SORA2_MAX_IMAGES:
                existing_urls = existing_urls[:SORA2_MAX_IMAGES]
                overflow = True
            if existing_urls != s.get("sora2_image_urls"):
                s["sora2_image_urls"] = existing_urls
                if overflow:
                    await message.reply_text(
                        f"⚠️ Максимум {SORA2_MAX_IMAGES} ссылок. Лишние не сохранены."
                    )
            prompt_source = raw
            for url in urls:
                prompt_source = prompt_source.replace(url, " ")
            candidate = normalize_input(prompt_source, allow_newlines=True)
            candidate = truncate_text(candidate, SORA2_MAX_PROMPT_LENGTH).strip()
            if cleaned == "":
                s["sora2_prompt"] = None
            elif candidate:
                s["sora2_prompt"] = candidate
        s["_last_text_sora2"] = None
        await show_sora2_card(wait_state.chat_id, ctx)
        if user_id is not None:
            card_id = s.get("last_ui_msg_id_sora2")
            if isinstance(card_id, int):
                refresh_card_pointer(user_id, card_id)
        handled = True
    elif wait_state.kind == WaitKind.BANANA_PROMPT:
        s = state(ctx)
        s["last_prompt"] = cleaned or None
        s["_last_text_banana"] = None
        await show_banana_card(wait_state.chat_id, ctx)
        if user_id is not None:
            card_id = s.get("last_ui_msg_id_banana")
            if isinstance(card_id, int):
                refresh_card_pointer(user_id, card_id)
        handled = True

    return handled


async def handle_card_input(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return

    user = update.effective_user
    user_id = user.id if user else None
    if user_id is None:
        return

    wait_state = get_wait(user_id)
    if wait_state is None:
        return

    if message.text is None:
        await message.reply_text("⚠️ Отправьте текстовое сообщение.")
        raise ApplicationHandlerStop

    if is_command_or_button(message):
        touch_wait(user_id)
        raise ApplicationHandlerStop

    handled = await _apply_wait_state_input(
        ctx,
        message,
        wait_state,
        user_id=user_id,
    )

    if handled:
        touch_wait(user_id)
        await _wait_acknowledge(message)
        raise ApplicationHandlerStop

async def _handle_suno_waiting_input(
    ctx: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    message: Message,
    state_dict: Dict[str, Any],
    waiting_field: str,
    *,
    user_id: Optional[int],
) -> bool:
    field = _suno_field_from_waiting(waiting_field)
    if not field:
        state_dict["suno_waiting_state"] = IDLE_SUNO
        return False

    raw_text = message.text
    if field == "cover":
        if raw_text is None:
            await _send_with_retry(lambda: message.reply_text(_COVER_INVALID_INPUT_MESSAGE))
            return True
        stripped_cover = raw_text.strip()
        if stripped_cover.lower() == "/cancel":
            state_dict["suno_waiting_state"] = IDLE_SUNO
            await _send_with_retry(lambda: message.reply_text("✏️ Изменение отменено."))
            return True
        if not stripped_cover:
            await _send_with_retry(lambda: message.reply_text(_COVER_INVALID_INPUT_MESSAGE))
            return True
        return await _cover_process_url_input(
            ctx,
            chat_id,
            message,
            state_dict,
            stripped_cover,
            user_id=user_id,
        )

    if raw_text is None:
        await _send_with_retry(lambda: message.reply_text("⚠️ Отправьте текстовое сообщение."))
        return True

    stripped = raw_text.strip()
    lowered = stripped.lower()
    if lowered == "/cancel":
        state_dict["suno_waiting_state"] = IDLE_SUNO
        log_evt("SUNO_INPUT_SAVE", kind=field, ok=False, reason="cancelled", user_id=user_id)
        await refresh_suno_card(ctx, chat_id, state_dict, price=PRICE_SUNO)
        await _send_with_retry(lambda: message.reply_text("✏️ Изменение отменено."))
        return True

    allow_newlines = field != "title"
    cleaned_value = _sanitize_suno_input(raw_text, allow_newlines=allow_newlines)
    is_clear = stripped in {"-", "—"}
    skip_requested = field == "lyrics" and lowered == "/skip"
    if skip_requested:
        cleaned_value = ""
        is_clear = True

    suno_state_obj = load_suno_state(ctx)
    flow = state_dict.get("suno_flow")
    current_step = state_dict.get("suno_step")
    custom_reply: Optional[str] = None

    before_value = getattr(suno_state_obj, field, None)
    if field == "title":
        if is_clear or not cleaned_value:
            if flow == "instrumental":
                generated_title = _pick_instrumental_title()
                set_suno_title(suno_state_obj, generated_title)
                custom_reply = f"🏷️ Название сгенерировано автоматически: {generated_title}"
            else:
                clear_suno_title(suno_state_obj)
        else:
            set_suno_title(suno_state_obj, cleaned_value)
    elif field == "style":
        if is_clear or not cleaned_value:
            if flow == "instrumental":
                default_style = suno_default_style_text("instrumental")
                set_suno_style(suno_state_obj, default_style)
                custom_reply = f"🎛️ стиль по умолчанию: {default_style}. Добавил базовые теги."
            else:
                clear_suno_style(suno_state_obj)
        else:
            set_suno_style(suno_state_obj, cleaned_value)
    elif field == "lyrics":
        if is_clear or not cleaned_value:
            clear_suno_lyrics(suno_state_obj)
            set_suno_lyrics_source(suno_state_obj, LyricsSource.AI)
            state_dict["suno_auto_lyrics_pending"] = True
            state_dict["suno_auto_lyrics_generated"] = False
            state_dict["suno_lyrics_confirmed"] = False
            custom_reply = "🤖 Сгенерирую короткий текст через Prompt-Master."
        else:
            if len(cleaned_value) > LYRICS_MAX_LENGTH:
                await _send_with_retry(
                    lambda: message.reply_text(
                        f"⚠️ Текст слишком длинный ({len(cleaned_value)}). Максимум — {LYRICS_MAX_LENGTH} символов."
                    )
                )
                return True
            set_suno_lyrics(suno_state_obj, cleaned_value)
            set_suno_lyrics_source(suno_state_obj, LyricsSource.USER)
            state_dict["suno_auto_lyrics_pending"] = False
            state_dict["suno_auto_lyrics_generated"] = False
            state_dict["suno_lyrics_confirmed"] = True

    after_value = getattr(suno_state_obj, field, None)
    changed = (before_value or "") != (after_value or "")
    cleared = not after_value

    state_dict["suno_waiting_state"] = IDLE_SUNO
    save_suno_state(ctx, suno_state_obj)
    state_dict["suno_state"] = suno_state_obj.to_dict()

    value_len = len(after_value or "")
    preview_log = _suno_preview_for_log(after_value)
    log_evt(
        "SUNO_INPUT_SAVE",
        kind=field,
        value_len=value_len,
        value_preview_30=preview_log,
        ok=True,
        cleared=cleared,
        changed=changed,
        user_id=user_id,
    )

    if field == "style" and flow == "lyrics" and state_dict.get("suno_auto_lyrics_pending"):
        _music_apply_auto_lyrics(
            ctx,
            state_dict,
            style=suno_state_obj.style,
            title=suno_state_obj.title,
        )
        custom_reply = "🤖 Добавил автогенерируемые куплеты."

    await _send_with_retry(lambda: message.reply_text("✅ Принято"))
    if not changed and not custom_reply:
        custom_reply = "ℹ️ Значение не изменилось (без изменений)"
    if custom_reply:
        await _send_with_retry(lambda: message.reply_text(custom_reply))

    if changed:
        await refresh_suno_card(ctx, chat_id, state_dict, price=PRICE_SUNO)

    if flow in {"instrumental", "lyrics", "cover"}:
        pending_step = current_step if isinstance(current_step, str) else None
        if pending_step and field == pending_step:
            next_step = _music_next_step(state_dict)
        else:
            next_step = state_dict.get("suno_step") if isinstance(state_dict.get("suno_step"), str) else None

        if pending_step == field or next_step is not None:
            await _music_prompt_step(
                chat_id,
                ctx,
                state_dict,
                flow=flow,
                step=next_step,
                user_id=user_id,
            )
    return True


def state(ctx: ContextTypes.DEFAULT_TYPE) -> Dict[str, Any]:
    user_data = getattr(ctx, "user_data", None)
    if isinstance(user_data, dict):
        try:
            suno_state_obj = load_suno_state(ctx)
        except Exception:
            suno_state_obj = SunoState()
    else:
        suno_state_obj = SunoState()
    state_dict = _apply_state_defaults(ensure_state(ctx))
    suno_state_payload = suno_state_obj.to_dict()
    state_dict["suno_state"] = suno_state_payload
    if isinstance(user_data, dict):
        user_data["suno_state"] = dict(suno_state_payload)
    waiting = state_dict.get("suno_waiting_state")
    if waiting not in {WAIT_SUNO_TITLE, WAIT_SUNO_STYLE, WAIT_SUNO_LYRICS, WAIT_SUNO_REFERENCE}:
        state_dict["suno_waiting_state"] = IDLE_SUNO
    card_meta = state_dict.get("suno_card")
    if not isinstance(card_meta, dict):
        card_meta = {
            "msg_id": None,
            "last_text_hash": None,
            "last_markup_hash": None,
            "chat_id": None,
        }
        state_dict["suno_card"] = card_meta
    else:
        card_meta.setdefault("msg_id", None)
        card_meta.setdefault("last_text_hash", None)
        card_meta.setdefault("last_markup_hash", None)
        card_meta.setdefault("chat_id", None)
    sora_prompt = state_dict.get("sora2_prompt")
    if isinstance(sora_prompt, str):
        state_dict["sora2_prompt"] = sora_prompt.strip() or None
    else:
        state_dict["sora2_prompt"] = None
    image_urls = state_dict.get("sora2_image_urls")
    if isinstance(image_urls, list):
        cleaned_urls: list[str] = []
        for url in image_urls:
            if not isinstance(url, str):
                continue
            trimmed = url.strip()
            if trimmed and trimmed not in cleaned_urls:
                cleaned_urls.append(trimmed)
            if len(cleaned_urls) >= 4:
                break
        state_dict["sora2_image_urls"] = cleaned_urls
    else:
        state_dict["sora2_image_urls"] = []
    wait_msg_sora = state_dict.get("sora2_wait_msg_id")
    if not isinstance(wait_msg_sora, int):
        state_dict["sora2_wait_msg_id"] = None
    wait_msg = state_dict.get("video_wait_message_id")
    if not isinstance(wait_msg, int):
        state_dict["video_wait_message_id"] = None
    return state_dict


_CHAT_HINT_TEXT = "💬 Обычный чат включён автоматически. Пишите вопрос! /reset — очистить контекст."


def _chat_state_waiting_input(state_dict: Dict[str, Any]) -> bool:
    mode = state_dict.get("mode")
    if mode and str(mode) not in {"", "chat", "none", "null"}:
        return True
    waiting = state_dict.get("suno_waiting_state")
    if waiting in {WAIT_SUNO_TITLE, WAIT_SUNO_STYLE, WAIT_SUNO_LYRICS, WAIT_SUNO_REFERENCE}:
        return True
    if state_dict.get("banana_active_op_key"):
        return True
    if state_dict.get("mj_active_op_key"):
        return True
    if state_dict.get("active_generation_op"):
        return True
    return False


def main_suggest_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("🎬 Видео", callback_data="go:video"),
                InlineKeyboardButton("🎨 Изображения", callback_data="go:image"),
            ],
            [
                InlineKeyboardButton("🎵 Музыка", callback_data="go:music"),
                InlineKeyboardButton("💎 Баланс", callback_data="go:balance"),
            ],
            [
                InlineKeyboardButton("ℹ️ FAQ", callback_data="go:faq"),
            ],
        ]
    )


async def safe_send_typing(bot, chat_id: int) -> None:
    try:
        await bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    except Exception as exc:
        log.debug("chat.typing_failed | chat=%s err=%s", chat_id, exc)


async def maybe_send_chat_hint(
    ctx: ContextTypes.DEFAULT_TYPE, chat_id: int, state_dict: Dict[str, Any]
) -> None:
    if state_dict.get("chat_hint_sent"):
        return
    try:
        await safe_send_text(ctx.bot, chat_id, md2_escape(_CHAT_HINT_TEXT))
    except Exception as exc:
        log.warning("chat.hint_failed | chat=%s err=%s", chat_id, exc)
        return
    state_dict["chat_hint_sent"] = True
    chat_first_hint_total.inc()


async def _safe_edit_placeholder(
    bot,
    placeholder: Optional[Message],
    text: str,
    *,
    inline_keyboard: Optional[InlineKeyboardMarkup] = None,
) -> bool:
    if not placeholder or getattr(placeholder, "message_id", None) is None:
        return False
    chat_id = placeholder.chat_id
    message_id = placeholder.message_id
    try:
        await bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text=text,
            parse_mode=ParseMode.MARKDOWN_V2,
            disable_web_page_preview=True,
            reply_markup=inline_keyboard,
        )
        return True
    except BadRequest as exc:
        if "message is not modified" in str(exc).lower():
            return True
        log.warning("chat.placeholder_edit_failed | chat=%s err=%s", chat_id, exc)
    except Exception as exc:
        log.warning("chat.placeholder_edit_failed | chat=%s err=%s", chat_id, exc)
    return False


async def _send_chat_message(
    bot,
    chat_id: int,
    text: str,
    *,
    inline_keyboard: Optional[InlineKeyboardMarkup] = None,
) -> bool:
    try:
        await tg_safe_send(
            bot.send_message,
            method_name="send_message",
            kind="chat_reply",
            chat_id=chat_id,
            text=text,
            parse_mode=ParseMode.MARKDOWN_V2,
            disable_web_page_preview=True,
            reply_markup=inline_keyboard,
        )
        return True
    except Exception as exc:
        log.warning("chat.send_failed | chat=%s err=%s", chat_id, exc)
    return False


async def _handle_chat_message(
    *,
    ctx: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    user_id: int,
    state_dict: Dict[str, Any],
    raw_text: str,
    text: str,
    send_typing_action: bool,
    send_hint: bool,
    inline_keyboard: Optional[InlineKeyboardMarkup] = None,
) -> None:
    if not text:
        await safe_send_text(ctx.bot, chat_id, md2_escape("⚠️ Отправьте сообщение."))
        return

    if rate_limit_hit(user_id):
        chat_messages_total.labels(outcome="rate_limited").inc()
        try:
            await safe_send_text(
                ctx.bot,
                chat_id,
                md2_escape("⏳ Слишком быстро. Подождите секунду и повторите."),
            )
        except Exception as exc:
            log.warning("chat.rate_limit_notify_failed | chat=%s err=%s", chat_id, exc)
        return

    if len(raw_text) > INPUT_MAX_CHARS:
        chat_messages_total.labels(outcome="too_long").inc()
        try:
            await safe_send_text(
                ctx.bot,
                chat_id,
                md2_escape(
                    "✂️ Сообщение слишком длинное. Сократите, пожалуйста (до 3000 символов)."
                ),
            )
        except Exception as exc:
            log.warning("chat.too_long_notify_failed | chat=%s err=%s", chat_id, exc)
        return

    placeholder: Optional[Message] = None
    if send_typing_action:
        await safe_send_typing(ctx.bot, chat_id)
    try:
        placeholder = await safe_send_placeholder(
            ctx.bot, chat_id, md2_escape("Думаю…")
        )
    except Exception as exc:
        log.warning("chat.placeholder_failed | chat=%s err=%s", chat_id, exc)
        placeholder = None

    if send_hint:
        await maybe_send_chat_hint(ctx, chat_id, state_dict)

    start = time.time()
    try:
        history = load_ctx(user_id)
        ctx_tokens = sum(estimate_tokens(str(item.get("content", ""))) for item in history)
        ctx_tokens += estimate_tokens(text)
        chat_context_tokens.set(float(min(ctx_tokens, CTX_MAX_TOKENS)))

        lang = detect_lang(raw_text or text)

        answer = await chat_reply(
            user_id,
            text,
            system_prompt=CHAT_SYSTEM_PROMPT,
            answer_lang=lang,
            history=history,
        )

        escaped_answer = md2_escape(answer)
        handled = False
        if placeholder and getattr(placeholder, "message_id", None) is not None:
            handled = await _safe_edit_placeholder(
                ctx.bot,
                placeholder,
                escaped_answer,
                inline_keyboard=inline_keyboard,
            )
        if not handled:
            await _send_chat_message(
                ctx.bot,
                chat_id,
                escaped_answer,
                inline_keyboard=inline_keyboard,
            )

        chat_messages_total.labels(outcome="ok").inc()
        chat_latency_ms.observe((time.time() - start) * 1000.0)
    except Exception as exc:
        chat_messages_total.labels(outcome="error").inc()
        chat_latency_ms.observe((time.time() - start) * 1000.0)

        error_payload = md2_escape("⚠️ Не получилось ответить сейчас. Попробуйте ещё раз.")
        handled = False
        if placeholder and getattr(placeholder, "message_id", None) is not None:
            handled = await _safe_edit_placeholder(
                ctx.bot, placeholder, error_payload, inline_keyboard=None
            )
        if not handled:
            with suppress(Exception):
                await safe_send_text(ctx.bot, chat_id, error_payload)
        app_logger = getattr(getattr(ctx, "application", None), "logger", log)
        app_logger.exception("chat error", extra={"user_id": user_id})
async def safe_send(
    update: Update,
    ctx: ContextTypes.DEFAULT_TYPE,
    text: str,
    reply_markup: Optional[Union[ReplyKeyboardMarkup, InlineKeyboardMarkup]] = None,
) -> Optional[Message]:
    """Safely deliver text to the chat, falling back to send_message on edit errors."""

    message = update.effective_message
    chat = update.effective_chat
    user = update.effective_user
    chat_id: Optional[int] = None
    if chat is not None:
        chat_id = chat.id
    elif user is not None:
        chat_id = user.id

    hub_msg_id: Optional[int] = None
    user_data = getattr(ctx, "user_data", None)
    if isinstance(user_data, dict):
        hub_val = user_data.get("hub_msg_id")
        if isinstance(hub_val, int):
            hub_msg_id = hub_val

    kwargs = {
        "text": text,
        "reply_markup": reply_markup,
        "parse_mode": ParseMode.MARKDOWN,
        "disable_web_page_preview": True,
    }

    async def _send_new_message() -> Optional[Message]:
        if chat_id is None:
            return None
        try:
            return await ctx.bot.send_message(chat_id=chat_id, **kwargs)
        except Exception as exc:  # pragma: no cover - network issues
            log.warning("menu.safe_send_send_failed | chat=%s err=%s", chat_id, exc)
            return None

    if message is not None and getattr(message, "chat_id", None) is not None:
        message_id = getattr(message, "message_id", None)
        if not isinstance(message_id, int) or hub_msg_id == message_id:
            message = None  # Skip editing hub message; send a new one instead
        else:
            try:
                return await ctx.bot.edit_message_text(
                    chat_id=message.chat_id,
                    message_id=message_id,
                    **kwargs,
                )
            except BadRequest as exc:
                err_text = str(exc).lower()
                if "message is not modified" in err_text:
                    return message
                if "can't" in err_text and "edit" in err_text:
                    return await _send_new_message()
                log.warning("menu.safe_send_edit_failed | chat=%s err=%s", chat_id, exc)
                message = None
            except TelegramError as exc:
                log.warning("menu.safe_send_edit_failed | chat=%s err=%s", chat_id, exc)
                message = None
            except Exception as exc:  # pragma: no cover - unexpected errors
                log.warning("menu.safe_send_edit_crashed | chat=%s err=%s", chat_id, exc)
                message = None

    return await _send_new_message()

# odex/fix-balance-reset-after-deploy
# main
# ==========================

_BOT_USERNAME_CACHE_KEY = "_bot_username"
_REF_SHARE_TEXT = "Присоединяйся к Best VEO3 Bot!"


async def _get_bot_username(ctx: ContextTypes.DEFAULT_TYPE) -> Optional[str]:
    bot_username = None
    try:
        bot_username = ctx.bot.username
    except Exception:
        bot_username = None
    app = getattr(ctx, "application", None)
    bot_data = getattr(app, "bot_data", None) if app is not None else None
    if bot_username:
        if isinstance(bot_data, dict):
            bot_data[_BOT_USERNAME_CACHE_KEY] = bot_username
        return bot_username

    cached = None
    if isinstance(bot_data, dict):
        cached = bot_data.get(_BOT_USERNAME_CACHE_KEY)
    if isinstance(cached, str) and cached:
        return cached

    try:
        me = await ctx.bot.get_me()
        username = getattr(me, "username", None)
    except Exception as exc:
        log.warning("Failed to fetch bot username: %s", exc)
        return None
    if not username:
        return None
    if isinstance(bot_data, dict):
        bot_data[_BOT_USERNAME_CACHE_KEY] = username
    return username


async def _build_referral_link(user_id: int, ctx: ContextTypes.DEFAULT_TYPE) -> Optional[str]:
    username = await _get_bot_username(ctx)
    if not username:
        return None
    return f"https://t.me/{username}?start=ref_{int(user_id)}"


def _format_user_for_notification(user: Optional[User], fallback_id: int) -> str:
    if user is None:
        return f"id {fallback_id}"
    username = getattr(user, "username", None)
    if username:
        return f"@{username}"
    first_name = getattr(user, "first_name", "")
    last_name = getattr(user, "last_name", "")
    full_name = " ".join(part for part in [first_name, last_name] if part)
    return full_name.strip() or f"id {fallback_id}"


async def _handle_referral_deeplink(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    user = update.effective_user
    if user is None:
        return

    args = getattr(ctx, "args", None)
    payload = args[0] if isinstance(args, list) and args else None
    if not payload and message is not None:
        text = message.text or message.caption or ""
        if text.startswith("/start "):
            payload = text.split(" ", 1)[1].strip()
        elif text == "/start":
            payload = ""
    if not payload:
        return
    if not payload.startswith("ref_"):
        return
    try:
        inviter_id = int(payload.split("_", 1)[1])
    except (IndexError, ValueError):
        return

    user_id = user.id
    if inviter_id == user_id or inviter_id <= 0:
        return

    try:
        existing = get_inviter(user_id)
    except Exception as exc:
        log.warning("referral_check_failed | user=%s err=%s", user_id, exc)
        existing = None
    if existing:
        return

    try:
        created = set_inviter(user_id, inviter_id)
    except Exception as exc:
        log.warning("referral_bind_failed | user=%s inviter=%s err=%s", user_id, inviter_id, exc)
        created = False
    if not created:
        return

    try:
        add_ref_user(inviter_id, user_id)
    except Exception as exc:
        log.warning("referral_add_user_failed | inviter=%s user=%s err=%s", inviter_id, user_id, exc)

    display = _format_user_for_notification(user, user_id)
    text = f"👥 Новый реферал: {display} ({user_id})."
    try:
        await ctx.bot.send_message(inviter_id, text)
    except Forbidden:
        pass
    except Exception as exc:
        log.warning("referral_notify_failed | inviter=%s err=%s", inviter_id, exc)
#   UI / Texts
# ==========================
WELCOME = (
    "🎬 Veo 3 — съёмочная команда\n"
    "Опиши идею — получи готовый клип.\n\n"
    "🖌️ MJ — художник\n"
    "Создаёт атмосферные изображения по тексту.\n\n"
    "🍌 Banana — редактор из будущего\n"
    "Меняет фон, одежду, макияж, убирает лишнее.\n\n"
    "🧠 Prompt-Master\n"
    "Вернёт кинематографичный промпт для вдохновения.\n\n"
    "💬 Обычный чат\n"
    "Живое общение и ответы на любые вопросы.\n\n"
    "⸻\n"
    "💎 **Ваш баланс: {balance}**\n"
    "📈 Больше идей и примеров: [канал с промптами]({prompts_url})\n\n"
    "👇 Выберите режим"
)


MENU_BTN_VIDEO = "🎬 Генерация видео"
MENU_BTN_IMAGE = "🎨 Генерация изображений"
MENU_BTN_SUNO = "🎵 Генерация музыки"
MENU_BTN_PM = "🧠 Prompt-Master"
MENU_BTN_CHAT = "💬 Обычный чат"
MENU_BTN_BALANCE = TXT_KB_PROFILE
MENU_BTN_SUPPORT = "🆘 ПОДДЕРЖКА"
BALANCE_CARD_STATE_KEY = "last_ui_msg_id_balance"
LEDGER_PAGE_SIZE = 10

BOTTOM_MENU_STATE_KEY = "last_ui_msg_id_bottom"

VIDEO_MENU_TEXT = "🎬 Выберите тип генерации видео:"
VIDEO_VEO_MENU_TEXT = "🎥 Режимы VEO:"
CB_ADMIN_SORA2_HEALTH = "admin:sora2_health"
_VIDEO_MENU_LOCK_TTL = 90
_VIDEO_MENU_MESSAGE_TTL = 90
_VIDEO_MENU_LOCK_NAME = "video:menu:lock"
_VIDEO_MENU_MESSAGE_NAME = "video:menu:card"
VIDEO_MENU_STATE_KEY = "video_menu_msg_id"
VIDEO_MENU_MSG_IDS_KEY = "video_menu"
VIDEO_MENU_LOCK_TTL = 5
VIDEO_CALLBACK_ALIASES = {
    "video_menu": CB.VIDEO_MENU,
    "engine:veo": CB.VIDEO_PICK_VEO,
    "engine:sora2": CB.VIDEO_PICK_SORA2,
    "engine:sora2_disabled": CB.VIDEO_PICK_SORA2_DISABLED,
    "mode:veo_text_fast": CB.VIDEO_MODE_VEO_FAST,
    "mode:veo_text_quality": CB.VIDEO_MODE_VEO_QUALITY,
    "mode:veo_photo": CB.VIDEO_MODE_VEO_PHOTO,
    "mode:sora2_ttv": CB.VIDEO_MODE_SORA_TEXT,
    "mode:sora2_itv": CB.VIDEO_MODE_SORA_IMAGE,
    "video:back": CB.VIDEO_MENU_BACK,
}
VIDEO_MODE_CALLBACK_MAP = {
    CB.VIDEO_MODE_VEO_FAST: "veo_text_fast",
    CB.VIDEO_MODE_VEO_QUALITY: "veo_text_quality",
    CB.VIDEO_MODE_VEO_PHOTO: "veo_photo",
    CB.VIDEO_MODE_SORA_TEXT: "sora2_ttv",
    CB.VIDEO_MODE_SORA_IMAGE: "sora2_itv",
}
VEO_WAIT_STICKER_ID = "5375464961822695044"
# Loaded from settings to allow feature flagging and runtime overrides.
# pylint: disable=invalid-name
SORA2_WAIT_STICKER_ID = SETTINGS_SORA2_WAIT_STICKER_ID
SORA2_MAX_PROMPT_LENGTH = 5000
SORA2_MIN_IMAGES = 1
SORA2_MAX_IMAGES = 4
SORA2_ALLOWED_ASPECTS = {"16:9", "9:16", "1:1", "4:5"}
SORA2_LOCK_TTL = 5 * 60
SORA2_DEFAULT_TTV_DURATION = 10
SORA2_DEFAULT_TTV_RESOLUTION = "1280x720"
SORA2_DEFAULT_ITV_DURATION = 5
SORA2_DEFAULT_ITV_RESOLUTION = "1280x720"
SORA2_POLL_BACKOFF_SERIES: Tuple[float, ...] = (5.0, 8.0, 13.0, 21.0, 34.0)
SORA2_POLL_TIMEOUT = 7 * 60


def _format_resolution_for_caption(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    text = str(value).strip().lower().replace(" ", "")
    if "x" in text:
        width, _, height = text.partition("x")
        if width.isdigit() and height.isdigit():
            return f"{int(width)}×{int(height)}"
    return str(value)


def _sora2_caption(mode: str, duration: Optional[int], resolution: Optional[str]) -> str:
    parts: List[str] = ["Sora 2"]
    title = _SORA2_MODE_TITLES.get(mode, "Text-to-Video")
    if title and title not in parts:
        parts.append(title)
    if duration and duration > 0:
        parts.append(f"{int(duration)}s")
    formatted_resolution = _format_resolution_for_caption(resolution)
    if formatted_resolution:
        parts.append(formatted_resolution)
    return " • ".join(part for part in parts if part)


async def show_wait_sticker(
    ctx: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    sticker_id: str,
) -> Optional[int]:
    try:
        message = await ctx.bot.send_sticker(chat_id, sticker_id)
    except Exception as exc:
        log.warning(
            "video.wait_sticker_failed",
            extra={"chat_id": chat_id, "error": str(exc)},
        )
        return None
    return getattr(message, "message_id", None)


async def safe_edit_or_send(
    ctx: ContextTypes.DEFAULT_TYPE,
    *,
    chat_id: int,
    message_id: Optional[int],
    text: str,
    reply_markup: Optional[Any] = None,
) -> Optional[int]:
    if message_id:
        try:
            await safe_edit_message(
                ctx,
                chat_id,
                message_id,
                text,
                reply_markup=reply_markup,
            )
            return message_id
        except Exception as exc:
            log.debug(
                "sora2.safe_edit_failed",
                extra={"chat_id": chat_id, "message_id": message_id, "error": str(exc)},
            )
    sent = await tg_safe_send(
        ctx.bot.send_message,
        method_name="sendMessage",
        kind="message",
        chat_id=chat_id,
        text=text,
        reply_markup=reply_markup,
    )
    return getattr(sent, "message_id", None)


async def safe_edit_or_send_menu(
    ctx: ContextTypes.DEFAULT_TYPE,
    *,
    chat_id: int,
    text: str,
    reply_markup: Optional[Any],
    state_key: str,
    msg_ids_key: Optional[str] = None,
    state_dict: Optional[Dict[str, Any]] = None,
    fallback_message_id: Optional[int] = None,
    parse_mode: Optional[ParseMode] = ParseMode.HTML,
    disable_web_page_preview: bool = True,
    log_label: str = "ui.menu",
) -> Optional[int]:
    """Safely edit an existing menu message or send a new one.

    The helper keeps the message id stored in the shared ``state`` dictionary and
    updates ``msg_ids`` for backwards compatibility with legacy logic.
    """

    state_dict = state_dict or state(ctx)
    current_id = state_dict.get(state_key)
    if not isinstance(current_id, int) and isinstance(fallback_message_id, int):
        current_id = fallback_message_id
        state_dict[state_key] = current_id

    def _store_message_id(message_id: Optional[int]) -> None:
        state_dict[state_key] = message_id if isinstance(message_id, int) else None
        if not msg_ids_key:
            return
        msg_ids = state_dict.get("msg_ids")
        if not isinstance(msg_ids, dict):
            msg_ids = {}
            state_dict["msg_ids"] = msg_ids
        msg_ids[msg_ids_key] = message_id if isinstance(message_id, int) else None

    markup = reply_markup
    effective_parse_mode = parse_mode or ParseMode.HTML

    if isinstance(current_id, int):
        try:
            await safe_edit_message(
                ctx,
                chat_id,
                current_id,
                text,
                markup,
                parse_mode=effective_parse_mode,
                disable_web_page_preview=disable_web_page_preview,
                log_on_noop=f"{log_label}.noop",
            )
            _store_message_id(current_id)
            log.info(
                "%s.edit",
                log_label,
                extra={"chat_id": chat_id, "message_id": current_id},
            )
            return current_id
        except BadRequest as exc:
            log.warning(
                "%s.edit_bad_request",
                log_label,
                extra={
                    "chat_id": chat_id,
                    "message_id": current_id,
                    "error": str(exc),
                },
            )
        except TelegramError as exc:
            log.warning(
                "%s.edit_error",
                log_label,
                extra={
                    "chat_id": chat_id,
                    "message_id": current_id,
                    "error": str(exc),
                },
            )
        else:
            _store_message_id(current_id)
            return current_id

        # Editing failed; try to clean up before sending a new message.
        try:
            await ctx.bot.edit_message_reply_markup(
                chat_id,
                current_id,
                reply_markup=None,
            )
        except TelegramError as exc:
            log.debug(
                "%s.clear_markup_failed",
                log_label,
                extra={"chat_id": chat_id, "message_id": current_id, "error": str(exc)},
            )
        try:
            await ctx.bot.delete_message(chat_id, current_id)
        except TelegramError as exc:
            log.debug(
                "%s.delete_failed",
                log_label,
                extra={"chat_id": chat_id, "message_id": current_id, "error": str(exc)},
            )
        _store_message_id(None)

    send_kwargs: Dict[str, Any] = {
        "chat_id": chat_id,
        "text": text,
        "reply_markup": markup,
        "disable_web_page_preview": disable_web_page_preview,
    }
    if parse_mode is not None:
        send_kwargs["parse_mode"] = effective_parse_mode

    result = await tg_safe_send(
        ctx.bot.send_message,
        method_name="sendMessage",
        kind="message",
        **send_kwargs,
    )
    message_id = getattr(result, "message_id", None) if result is not None else None
    _store_message_id(message_id if isinstance(message_id, int) else None)

    if isinstance(message_id, int):
        log.info(
            "%s.send",
            log_label,
            extra={"chat_id": chat_id, "message_id": message_id},
        )
        return message_id

    log.warning(
        "%s.send_failed",
        log_label,
        extra={"chat_id": chat_id},
    )
    return None


async def refresh_suno_card(
    ctx: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    state_dict: Dict[str, Any],
    *,
    price: int,
    state_key: str = "last_ui_msg_id_suno",
    force_new: bool = False,
) -> Optional[int]:
    result = await _refresh_suno_card_raw(
        ctx,
        chat_id,
        state_dict,
        price=price,
        state_key=state_key,
        force_new=force_new,
    )
    return result


async def _clear_bottom_menu(
    ctx: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    *,
    state_dict: Optional[Dict[str, Any]] = None,
) -> None:
    state_obj = state_dict if isinstance(state_dict, dict) else state(ctx)
    mid = state_obj.get(BOTTOM_MENU_STATE_KEY)
    if isinstance(mid, int):
        with suppress(Exception):
            await ctx.bot.delete_message(chat_id, mid)
    state_obj[BOTTOM_MENU_STATE_KEY] = None
    msg_ids = state_obj.get("msg_ids")
    if isinstance(msg_ids, dict):
        for key, value in list(msg_ids.items()):
            if value == mid:
                msg_ids[key] = None


async def replace_wait_with_docs(
    ctx: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    wait_msg_id: Optional[int],
    files: Iterable[str],
) -> None:
    if wait_msg_id:
        with suppress(Exception):
            await ctx.bot.delete_message(chat_id, wait_msg_id)
    for path in files:
        try:
            with open(path, "rb") as handle:
                await ctx.bot.send_document(chat_id, handle)
        except Exception as exc:
            log.warning(
                "video.wait_replace_send_failed",
                extra={"chat_id": chat_id, "path": path, "error": str(exc)},
            )


async def _clear_video_menu_state(
    chat_id: int,
    *,
    user_id: Optional[int] = None,
    ctx: Optional[ContextTypes.DEFAULT_TYPE] = None,
) -> None:
    clear_menu_message(_VIDEO_MENU_MESSAGE_NAME, chat_id)
    release_menu_lock(_VIDEO_MENU_LOCK_NAME, chat_id)
    if user_id is not None:
        release_menu_lock(_VIDEO_MENU_LOCK_NAME, user_id)
    if ctx is None:
        return

    state_dict = state(ctx)
    message_id = state_dict.get(VIDEO_MENU_STATE_KEY)
    if isinstance(message_id, int):
        try:
            await ctx.bot.edit_message_reply_markup(chat_id, message_id, reply_markup=None)
        except TelegramError as exc:
            log.debug(
                "ui.video_menu.clear_markup_failed",
                extra={"chat_id": chat_id, "message_id": message_id, "error": str(exc)},
            )
        try:
            await ctx.bot.delete_message(chat_id, message_id)
        except TelegramError as exc:
            log.debug(
                "ui.video_menu.delete_failed",
                extra={"chat_id": chat_id, "message_id": message_id, "error": str(exc)},
            )
    state_dict[VIDEO_MENU_STATE_KEY] = None
    msg_ids = state_dict.get("msg_ids")
    if isinstance(msg_ids, dict):
        msg_ids[VIDEO_MENU_MSG_IDS_KEY] = None

_VIDEO_MODE_HINTS: Dict[str, str] = {
    "veo_text_fast": "✍️ Пришлите текст идеи и/или фото-референс — карточка обновится автоматически.",
    "veo_text_quality": "✍️ Пришлите текст идеи и/или фото-референс — карточка обновится автоматически.",
    "veo_photo": "📸 Пришлите фото (подпись-промпт — по желанию). Карточка обновится автоматически.",
    "sora2_ttv": "✍️ Пришлите текст (до 5000 символов). Нажмите «🚀 Запустить Sora 2», когда будете готовы.",
    "sora2_itv": "📸 Пришлите 1–4 ссылок на изображения и текст (до 5000 символов). Для очистки ссылок отправьте слово clear.",
}

_MJ_GRID_CACHE_KEY_TMPL = "mj:grid:{grid_id}"
_MJ_GRID_CACHE_TTL = 24 * 60 * 60
_MJ_UPSCALE_LOCK_KEY_TMPL = "lock:mj:upscale:{grid_id}:{index}"
_MJ_UPSCALE_LOCK_TTL = 60

def _safe_get_balance(user_id: int) -> int:
    try:
        return get_balance(user_id)
    except Exception as exc:
        log.exception("get_balance failed for user %s: %s", user_id, exc)
        return 0


def render_welcome_for(
    uid: int,
    ctx: ContextTypes.DEFAULT_TYPE,
    *,
    balance: Optional[int] = None,
) -> str:
    if balance is None:
        balance = _safe_get_balance(uid)
    _set_cached_balance(ctx, balance)
    return WELCOME.format(balance=balance, prompts_url=PROMPTS_CHANNEL_URL)

def main_menu_kb() -> ReplyKeyboardMarkup:
    keyboard = [
        [KeyboardButton(MENU_BTN_VIDEO)],
        [KeyboardButton(MENU_BTN_IMAGE)],
        [KeyboardButton(MENU_BTN_SUNO)],
        [KeyboardButton(MENU_BTN_PM)],
        [KeyboardButton(MENU_BTN_CHAT)],
        [KeyboardButton(MENU_BTN_BALANCE)],
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)


async def show_emoji_hub_for_chat(
    chat_id: int,
    ctx: ContextTypes.DEFAULT_TYPE,
    *,
    user_id: Optional[int] = None,
    replace: bool = False,
) -> Optional[int]:
    """Render the emoji hub for a chat, editing the stored message when possible."""

    resolved_uid: Optional[int] = None
    if user_id is not None:
        try:
            resolved_uid = int(user_id)
        except (TypeError, ValueError):
            resolved_uid = None
    if resolved_uid is None:
        ctx_uid = get_user_id(ctx)
        if ctx_uid is not None:
            try:
                resolved_uid = int(ctx_uid)
            except (TypeError, ValueError):
                resolved_uid = None
    if resolved_uid is None:
        resolved_uid = int(chat_id)

    balance = _safe_get_balance(resolved_uid)
    _set_cached_balance(ctx, balance)

    text = build_hub_text(balance)
    keyboard = build_hub_keyboard()

    log.info("hub.show | user_id=%s balance=%s", resolved_uid, balance)

    await _clear_bottom_menu(ctx, chat_id)

    hub_msg_id_val = ctx.user_data.get("hub_msg_id")
    hub_msg_id = hub_msg_id_val if isinstance(hub_msg_id_val, int) else None
    if replace and hub_msg_id:
        with suppress(Exception):
            await ctx.bot.delete_message(chat_id=chat_id, message_id=hub_msg_id)
        ctx.user_data["hub_msg_id"] = None

    try:
        message = await tg_safe_send(
            ctx.bot.send_message,
            method_name="sendMessage",
            kind="message",
            chat_id=chat_id,
            text=text,
            reply_markup=keyboard,
            parse_mode=ParseMode.MARKDOWN,
            disable_web_page_preview=True,
        )
    except Exception as exc:  # pragma: no cover - network issues
        log.warning("hub.send_failed | user_id=%s err=%s", resolved_uid, exc)
        return None

    message_id = getattr(message, "message_id", None)
    if isinstance(message_id, int):
        ctx.user_data["hub_msg_id"] = message_id
        return message_id
    return None


async def show_emoji_hub(
    update: Update, ctx: ContextTypes.DEFAULT_TYPE, *, replace: bool = False
) -> Optional[int]:
    chat = update.effective_chat
    user = update.effective_user
    chat_id = None
    if chat is not None:
        chat_id = chat.id
    elif user is not None:
        chat_id = user.id
    if chat_id is None:
        return None
    user_id = user.id if user is not None else None
    return await show_emoji_hub_for_chat(chat_id, ctx, user_id=user_id, replace=replace)


async def show_main_menu(chat_id: int, ctx: ContextTypes.DEFAULT_TYPE) -> Optional[int]:
    return await show_emoji_hub_for_chat(chat_id, ctx, replace=True)


_HUB_ACTION_ALIASES: Dict[str, str] = {
    "home:profile": "balance",
    "home:kb": "knowledge",
    "home:photo": "image",
    "home:music": "music",
    "home:video": "video",
    "home:chat": "ai_modes",
    CB_MAIN_PROFILE: "balance",
    CB_PROFILE_BACK: "balance",
    CB_MAIN_BACK: "root",
    CB_MAIN_KNOWLEDGE: "knowledge",
    CB_MAIN_PHOTO: "image",
    CB_MAIN_MUSIC: "music",
    CB_MAIN_VIDEO: "video",
    CB_MAIN_AI_DIALOG: "ai_modes",
    CB_AI_MODES: "ai_modes",
    CB_PROFILE_TOPUP: "profile_topup",
    CB_CHAT_NORMAL: "chat",
    CB_CHAT_PROMPTMASTER: "prompt",
    CB_PAY_STARS: "pay_stars",
    CB_PAY_CARD: "pay_card",
    CB_PAY_CRYPTO: "pay_crypto",
    "nav_video": "video",
    "nav_image": "image",
    "nav_music": "music",
    "nav_prompt": "prompt",
    "nav_chat": "chat",
    "profile": "balance",
    "back_main": "profile_topup",
}


async def hub_router(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await ensure_user_record(update)

    query = update.callback_query
    if not query:
        return

    data = (query.data or "").strip()
    if not data:
        await query.answer()
        return

    if data.startswith("hub:"):
        action = data.split(":", 1)[1]
    else:
        action = _HUB_ACTION_ALIASES.get(data)

    if not action:
        await query.answer()
        return
    message = query.message
    chat = update.effective_chat
    user = update.effective_user

    chat_id = None
    if message is not None:
        chat_id = message.chat_id
    elif chat is not None:
        chat_id = chat.id

    user_id = user.id if user is not None else None

    if action == "root":
        await query.answer()
        if chat_id is not None:
            await show_emoji_hub_for_chat(chat_id, ctx, user_id=user_id, replace=True)
        return

    if chat_id is None:
        await query.answer()
        return

    s = state(ctx)

    if action == "knowledge":
        await query.answer()
        knowledge_key = "last_ui_msg_id_knowledge"
        existing = s.get(knowledge_key)
        current_mid = existing if isinstance(existing, int) else None
        sent_id = await safe_edit_or_send(
            ctx,
            chat_id=chat_id,
            message_id=current_mid,
            text=TXT_KNOWLEDGE_INTRO,
            reply_markup=faq_keyboard(),
        )
        if isinstance(sent_id, int):
            s[knowledge_key] = sent_id
        return

    if action == "ai_modes":
        await query.answer()
        if message is None:
            return
        text = f"{TXT_KB_AI_DIALOG}\n{TXT_AI_DIALOG_CHOOSE}"
        keyboard = kb_ai_dialog_modes()
        try:
            await safe_edit_message(
                ctx,
                chat_id,
                message.message_id,
                text,
                keyboard,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
            )
        except Exception:
            sent = await ctx.bot.send_message(
                chat_id=chat_id,
                text=text,
                reply_markup=keyboard,
            )
            if isinstance(ctx.user_data, dict):
                mid = getattr(sent, "message_id", None)
                if isinstance(mid, int):
                    ctx.user_data["hub_msg_id"] = mid
        return

    if action in {"profile_topup", "pay_stars", "pay_card", "pay_crypto"}:
        handled = await handle_topup_callback(update, ctx, data)
        if handled:
            return

    if action == "video":
        log.debug(
            "hub.video.clicked",
            extra={"chat_id": chat_id, "user_id": user_id},
        )
        if user_id:
            set_mode(user_id, False)
        s["mode"] = None
        await query.answer()
        try:
            await start_video_menu(update, ctx)
        except Exception as exc:  # pragma: no cover - network issues
            log.warning("hub.video_send_failed | chat=%s err=%s", chat_id, exc)
        return

    if action == "image":
        if user_id:
            set_mode(user_id, False)
        s["mode"] = None
        await query.answer()
        try:
            await tg_safe_send(
                ctx.bot.send_message,
                method_name="sendMessage",
                kind="message",
                chat_id=chat_id,
                text="🖼️ Выберите режим работы с изображениями:",
                reply_markup=image_menu_kb(),
            )
        except Exception as exc:  # pragma: no cover - network issues
            log.warning("hub.image_send_failed | chat=%s err=%s", chat_id, exc)
        return

    if action == "music":
        if user_id:
            set_mode(user_id, False)
        await query.answer()
        await suno_entry(chat_id, ctx, force_new=True)
        return

    if action == "prompt":
        if user_id:
            set_mode(user_id, False)
        s["mode"] = None
        _mode_set(chat_id, MODE_PM)
        await query.answer()
        try:
            await tg_safe_send(
                ctx.bot.send_message,
                method_name="sendMessage",
                kind="message",
                chat_id=chat_id,
                text="Режим переключён: Prompt-Master. Пришлите идею/сцену — верну кинопромпт.",
            )
        except Exception as exc:  # pragma: no cover - network issues
            log.warning("hub.prompt_send_failed | chat=%s err=%s", chat_id, exc)
        return

    if action == "chat":
        if user_id:
            set_mode(user_id, True)
        s["mode"] = None
        _mode_set(chat_id, MODE_CHAT)
        await query.answer()
        try:
            await safe_send_text(
                ctx.bot,
                chat_id,
                md2_escape(
                    "💬 Обычный чат включён. Пишите вопрос! /reset — очистить контекст.\n"
                    "🎙️ Можно присылать голосовые — я их распознаю."
                ),
            )
        except Exception as exc:  # pragma: no cover - network issues
            log.warning("hub.chat_send_failed | chat=%s err=%s", chat_id, exc)
        return

    if action == "balance":
        await query.answer()
        force_new = data in {"hub:balance", CB_MAIN_PROFILE}
        await show_balance_card(chat_id, ctx, force_new=force_new)
        return

    await query.answer()


async def main_suggest_router(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query:
        return

    data = (query.data or "").strip()
    handler: Optional[Callable[[Update, ContextTypes.DEFAULT_TYPE], Awaitable[None]]] = None
    if data == "go:video":
        handler = video_command
    elif data == "go:image":
        handler = image_command
    elif data == "go:music":
        handler = suno_command
    elif data == "go:balance":
        handler = my_balance_command
    elif data == "go:faq":
        handler = faq_command_entry

    try:
        await query.answer()
    except Exception as exc:
        chat_id = query.message.chat_id if query.message else None
        log.debug("chat.main_suggest_answer_failed | chat=%s err=%s", chat_id, exc)

    if handler is None:
        return

    await handler(update, ctx)


def _sora2_is_enabled() -> bool:
    if not SORA2_ENABLED:
        return False
    if not (SORA2.get("API_KEY") or "").strip():
        return False
    if is_sora2_unavailable():
        return False
    return True


def video_menu_kb() -> InlineKeyboardMarkup:
    sora2_ready = _sora2_is_enabled()
    sora2_label = "🧠 Sora2"
    sora2_callback = CB_VIDEO_ENGINE_SORA2
    if not sora2_ready:
        sora2_label = "🧠 Sora2 (скоро)"
        sora2_callback = CB_VIDEO_ENGINE_SORA2_DISABLED
    keyboard = [
        [InlineKeyboardButton("🎥 VEO", callback_data=CB_VIDEO_ENGINE_VEO)],
        [InlineKeyboardButton(sora2_label, callback_data=sora2_callback)],
        [InlineKeyboardButton("⬅️ Назад", callback_data=CB_VIDEO_BACK)],
    ]
    return InlineKeyboardMarkup(keyboard)


def veo_modes_kb() -> InlineKeyboardMarkup:
    keyboard = [
        [InlineKeyboardButton(
            f"🎬 Генерация видео (Veo Fast) — 💎 {TOKEN_COSTS['veo_fast']}",
            callback_data=CB_VIDEO_MODE_FAST,
        )],
        [InlineKeyboardButton(
            f"🎬 Генерация видео (Veo Quality) — 💎 {TOKEN_COSTS['veo_quality']}",
            callback_data=CB_VIDEO_MODE_QUALITY,
        )],
        [InlineKeyboardButton(
            f"🖼️ Оживить изображение (Veo) — 💎 {TOKEN_COSTS['veo_photo']}",
            callback_data=CB_VIDEO_MODE_PHOTO,
        )],
        [InlineKeyboardButton("⬅️ Назад", callback_data=CB_VIDEO_MENU)],
    ]
    return InlineKeyboardMarkup(keyboard)


async def _refresh_video_menu_ui(
    ctx: ContextTypes.DEFAULT_TYPE,
    *,
    chat_id: int,
    message: Optional[Message],
) -> None:
    markup = video_menu_kb()
    if message is not None:
        outcome = await _try_edit_menu_card(
            ctx,
            chat_id=chat_id,
            message_id=message.message_id,
            text=VIDEO_MENU_TEXT,
            reply_markup=markup,
            log_prefix="video.menu.refresh",
            parse_mode=None,
        )
        if outcome == "ok":
            save_menu_message(
                _VIDEO_MENU_MESSAGE_NAME,
                chat_id,
                message.message_id,
                _VIDEO_MENU_MESSAGE_TTL,
            )
            return
    await tg_safe_send(
        ctx.bot.send_message,
        method_name="sendMessage",
        kind="message",
        chat_id=chat_id,
        text=VIDEO_MENU_TEXT,
        reply_markup=markup,
    )


def video_result_footer_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("🔁 Сгенерировать ещё", callback_data="start_new_cycle")],
            [InlineKeyboardButton("🏠 Назад в меню", callback_data=CB_GO_HOME)],
        ]
    )


async def _try_edit_menu_card(
    ctx: ContextTypes.DEFAULT_TYPE,
    *,
    chat_id: int,
    message_id: int,
    text: str,
    reply_markup: InlineKeyboardMarkup,
    log_prefix: str,
    parse_mode: Optional[ParseMode] = None,
    disable_web_page_preview: bool = True,
) -> str:
    try:
        await ctx.bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text=text,
            reply_markup=reply_markup,
            parse_mode=parse_mode,
            disable_web_page_preview=disable_web_page_preview,
        )
        log.info(
            "%s.edit_success",
            log_prefix,
            extra={"chat_id": chat_id, "message_id": message_id},
        )
        return "ok"
    except BadRequest as exc:
        lowered = str(exc).lower()
        if "message to edit not found" in lowered or "message identifier invalid" in lowered:
            log.info(
                "%s.missing",
                log_prefix,
                extra={"chat_id": chat_id, "message_id": message_id},
            )
            return "missing"
        if "message is not modified" in lowered:
            log.info(
                "%s.not_modified",
                log_prefix,
                extra={"chat_id": chat_id, "message_id": message_id},
            )
            return "not_modified"
        log.debug(
            "%s.edit_bad_request",
            log_prefix,
            extra={"chat_id": chat_id, "message_id": message_id, "error": str(exc)},
        )
        return "error"
    except TelegramError as exc:
        log.debug(
            "%s.edit_failed",
            log_prefix,
            extra={"chat_id": chat_id, "message_id": message_id, "error": str(exc)},
        )
        return "error"


async def _send_video_menu_message(
    ctx: ContextTypes.DEFAULT_TYPE,
    *,
    chat_id: Optional[int] = None,
    message: Optional[Message] = None,
) -> Optional[int]:
    target_chat = chat_id or (getattr(message, "chat_id", None))
    markup = video_menu_kb()
    if message is not None:
        reply_method = getattr(message, "reply_text", None)
        if callable(reply_method):
            try:
                sent = await reply_method(VIDEO_MENU_TEXT, reply_markup=markup)
                return getattr(sent, "message_id", None)
            except Exception as exc:
                log.warning(
                    "video.menu.reply_failed",
                    extra={"chat_id": target_chat, "error": str(exc)},
                )
        if target_chat is None:
            return None
    if target_chat is None:
        return None
    result = await tg_safe_send(
        ctx.bot.send_message,
        method_name="sendMessage",
        kind="message",
        chat_id=target_chat,
        text=VIDEO_MENU_TEXT,
        reply_markup=markup,
    )
    if result is None:
        return None
    return getattr(result, "message_id", None)


async def start_video_menu(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> Optional[int]:
    message = getattr(update, "effective_message", None)
    chat = getattr(update, "effective_chat", None) or (
        getattr(message, "chat", None) if message else None
    )
    user = getattr(update, "effective_user", None)

    chat_id = chat.id if chat else (message.chat_id if message else None)
    if chat_id is None:
        return None

    user_id = user.id if user else None
    s = state(ctx)
    s["mode"] = None
    s.pop("model", None)

    if getattr(ctx, "bot", None) is None and message is not None:
        await _send_video_menu_message(ctx, message=message)
        if user_id is not None:
            input_state.clear(user_id, reason="video_menu")
            set_mode(user_id, False)
            clear_wait(user_id, reason="video_menu")
            try:
                clear_wait_state(user_id, reason="video_menu")
            except TypeError:
                clear_wait_state(user_id)
        return None

    record = get_menu_message(
        _VIDEO_MENU_MESSAGE_NAME,
        chat_id,
        max_age=_VIDEO_MENU_MESSAGE_TTL,
    )
    fallback_message_id: Optional[int] = None
    if record:
        fallback_message_id, _ = record
        if isinstance(fallback_message_id, int) and not isinstance(
            s.get(VIDEO_MENU_STATE_KEY),
            int,
        ):
            s[VIDEO_MENU_STATE_KEY] = fallback_message_id

    user_lock_acquired = False
    if user_id is not None:
        user_lock_acquired = user_lock(user_id, "video_menu")
        if not user_lock_acquired:
            log.info(
                "ui.video_menu.user_lock_busy",
                extra={"chat_id": chat_id, "user_id": user_id},
            )
            raise MenuLocked(_VIDEO_MENU_LOCK_NAME, user_id)

    lock_owner = user_id if user_id is not None else chat_id

    try:
        async with with_menu_lock(_VIDEO_MENU_LOCK_NAME, lock_owner, ttl=VIDEO_MENU_LOCK_TTL):
            markup = video_menu_kb()
            message_id = await safe_edit_or_send_menu(
                ctx,
                chat_id=chat_id,
                text=VIDEO_MENU_TEXT,
                reply_markup=markup,
                state_key=VIDEO_MENU_STATE_KEY,
                msg_ids_key=VIDEO_MENU_MSG_IDS_KEY,
                state_dict=s,
                fallback_message_id=fallback_message_id,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
                log_label="ui.video_menu",
            )
            if isinstance(message_id, int):
                save_menu_message(
                    _VIDEO_MENU_MESSAGE_NAME,
                    chat_id,
                    message_id,
                    _VIDEO_MENU_MESSAGE_TTL,
                )
                log.info(
                    "ui.video_menu.show",
                    extra={
                        "chat_id": chat_id,
                        "message_id": message_id,
                        "user_id": user_id,
                    },
                )
            else:
                clear_menu_message(_VIDEO_MENU_MESSAGE_NAME, chat_id)
            return message_id
    except MenuLocked:
        log.info(
            "ui.video_menu.lock_busy",
            extra={"chat_id": chat_id, "user_id": user_id},
        )
        raise
    finally:
        if user_id is not None:
            input_state.clear(user_id, reason="video_menu")
            set_mode(user_id, False)
            clear_wait(user_id, reason="video_menu")
            try:
                clear_wait_state(user_id, reason="video_menu")
            except TypeError:
                clear_wait_state(user_id)
        if user_id is not None and user_lock_acquired:
            release_user_lock(user_id, "video_menu")


def _video_mode_config(mode: str) -> Optional[Tuple[str, str, str]]:
    if mode == "veo_text_fast":
        return "16:9", "veo3_fast", _VIDEO_MODE_HINTS[mode]
    if mode == "veo_text_quality":
        return "16:9", "veo3", _VIDEO_MODE_HINTS[mode]
    if mode == "veo_photo":
        return "9:16", "veo3_fast", _VIDEO_MODE_HINTS[mode]
    return None


async def _start_video_mode(
    mode: str,
    *,
    chat_id: Optional[int],
    ctx: ContextTypes.DEFAULT_TYPE,
    user_id: Optional[int],
    message: Optional[Message],
) -> bool:
    if chat_id is None:
        return False
    if mode in {"sora2_ttv", "sora2_itv"}:
        s = state(ctx)
        s["mode"] = mode
        s["sora2_prompt"] = None
        if mode == "sora2_ttv":
            s["sora2_image_urls"] = []
        elif not isinstance(s.get("sora2_image_urls"), list):
            s["sora2_image_urls"] = []
        s["sora2_generating"] = False
        s["sora2_last_task_id"] = None
        s["sora2_wait_msg_id"] = None
        s["video_wait_message_id"] = None
        await sora2_entry(chat_id, ctx)
        card_id_raw = s.get("last_ui_msg_id_sora2")
        card_id = card_id_raw if isinstance(card_id_raw, int) else None
        _activate_wait_state(
            user_id=user_id,
            chat_id=chat_id,
            card_msg_id=card_id,
            kind=WaitKind.SORA2_PROMPT,
            meta={"mode": mode},
        )
        if message is not None:
            hint = _VIDEO_MODE_HINTS.get(mode)
            if hint:
                with suppress(Exception):
                    await message.reply_text(hint)
        return True
    config = _video_mode_config(mode)
    if config is None:
        return False

    aspect, model, hint = config
    s = state(ctx)
    s["mode"] = mode
    s["aspect"] = aspect
    s["model"] = model

    await veo_entry(chat_id, ctx)
    card_id_raw = s.get("last_ui_msg_id_veo")
    card_id = card_id_raw if isinstance(card_id_raw, int) else None
    _activate_wait_state(
        user_id=user_id,
        chat_id=chat_id,
        card_msg_id=card_id,
        kind=WaitKind.VEO_PROMPT,
        meta={"mode": mode},
    )
    if message is not None:
        try:
            await message.reply_text(hint)
        except Exception:
            pass
    return True


def image_menu_kb() -> InlineKeyboardMarkup:
    keyboard = [
        [InlineKeyboardButton(
            f"🖼️ Генерация изображений (MJ) — 💎 {TOKEN_COSTS['mj']}",
            callback_data="mode:mj_txt",
        )],
        [InlineKeyboardButton(
            f"🪄 Улучшение качества — 💎 {TOKEN_COSTS['mj_upscale']}",
            callback_data="mode:mj_upscale",
        )],
        [InlineKeyboardButton(
            f"🍌 Редактор изображений (Banana) — 💎 {TOKEN_COSTS['banana']}",
            callback_data="mode:banana",
        )],
        [InlineKeyboardButton("⬅️ Назад", callback_data="back")],
    ]
    return InlineKeyboardMarkup(keyboard)


def inline_topup_keyboard() -> InlineKeyboardMarkup:
    return insufficient_balance_keyboard()


def topup_menu_keyboard() -> InlineKeyboardMarkup:
    return menu_pay_unified()


def yookassa_pack_keyboard() -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton(yookassa_pack_button_label(pack.pack_id), callback_data=f"yk:{pack.pack_id}")]
        for pack in YOOKASSA_PACKS_ORDER
    ]
    rows.append([InlineKeyboardButton(common_text("topup.menu.back"), callback_data="topup:open")])
    return InlineKeyboardMarkup(rows)


def yookassa_payment_keyboard(url: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton(common_text("topup.yookassa.pay"), url=url)],
            [InlineKeyboardButton(common_text("topup.menu.back"), callback_data="topup:open")],
        ]
    )


async def show_topup_menu(
    ctx: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    *,
    query: Optional["telegram.CallbackQuery"] = None,
    user_id: Optional[int] = None,
) -> None:
    text = TXT_TOPUP_CHOOSE
    keyboard = topup_menu_keyboard()
    if query is not None:
        message = query.message
        if message is not None:
            try:
                await _safe_edit_message_text(query.edit_message_text, text, reply_markup=keyboard)
                log.info(
                    "topup.open",
                    extra={"meta": {"chat_id": chat_id, "user_id": user_id, "via": "edit"}},
                )
                return
            except Exception:
                pass
    try:
        await ctx.bot.send_message(chat_id=chat_id, text=text, reply_markup=keyboard)
    finally:
        log.info(
            "topup.open",
            extra={"meta": {"chat_id": chat_id, "user_id": user_id, "via": "send"}},
        )


def balance_menu_kb() -> InlineKeyboardMarkup:
    keyboard: list[list[InlineKeyboardButton]] = []
    keyboard.extend(kb_profile_topup_entry().inline_keyboard)
    keyboard.append([InlineKeyboardButton("🧾 История операций", callback_data="tx:open")])
    keyboard.append([InlineKeyboardButton("👥 Пригласить друга", callback_data="ref:open")])
    keyboard.append([InlineKeyboardButton("🎁 Активировать промокод", callback_data="promo_open")])
    keyboard.append([InlineKeyboardButton(common_text("topup.menu.back"), callback_data="back")])
    return InlineKeyboardMarkup(keyboard)


async def show_balance_card(
    chat_id: int,
    ctx: ContextTypes.DEFAULT_TYPE,
    *,
    force_new: bool = False,
) -> Optional[int]:
    s = state(ctx)
    uid = get_user_id(ctx) or chat_id
    balance = _safe_get_balance(uid)
    _set_cached_balance(ctx, balance)
    text = f"{TXT_PROFILE_TITLE}\n💎 Ваш баланс: {balance}"
    mid = await upsert_card(
        ctx,
        chat_id,
        s,
        BALANCE_CARD_STATE_KEY,
        text,
        reply_markup=balance_menu_kb(),
        parse_mode=ParseMode.MARKDOWN,
        disable_web_page_preview=True,
        force_new=force_new,
    )
    if mid:
        msg_ids = s.get("msg_ids")
        if not isinstance(msg_ids, dict):
            msg_ids = {}
            s["msg_ids"] = msg_ids
        msg_ids["balance"] = mid
        s["last_panel"] = "balance"
    return mid


async def show_balance_notification(
    chat_id: int,
    ctx: ContextTypes.DEFAULT_TYPE,
    user_id: int,
    text: str,
    *,
    reply_markup: Optional[InlineKeyboardMarkup] = None,
) -> None:
    s = state(ctx)
    await upsert_card(
        ctx,
        chat_id,
        s,
        "last_ui_msg_id_notice",
        text,
        reply_markup,
        parse_mode=ParseMode.MARKDOWN,
        disable_web_page_preview=True,
    )
    await refresh_balance_card_if_open(
        user_id,
        chat_id,
        ctx=ctx,
        state_dict=s,
        reply_markup=balance_menu_kb(),
    )


def _ledger_reason(entry: Dict[str, Any]) -> str:
    reason = entry.get("reason")
    if isinstance(reason, str):
        reason = reason.strip()
    else:
        reason = ""

    if not reason:
        meta = entry.get("meta")
        if isinstance(meta, dict):
            meta_reason = meta.get("model")
            if isinstance(meta_reason, str):
                reason = meta_reason.strip()

    reason = (reason or "").strip()
    if not reason:
        return "—"
    return " ".join(reason.split())


def _ledger_timestamp(entry: Dict[str, Any]) -> str:
    ts = entry.get("ts")
    try:
        ts_value = float(ts)
    except (TypeError, ValueError):
        return "—"
    dt = datetime.fromtimestamp(ts_value)
    return dt.strftime("%d.%m %H:%M")


def _ledger_amount_parts(entry_type: str, amount: int) -> tuple[str, str]:
    if entry_type == "debit":
        return "➖", f"−{amount}"
    if entry_type == "refund":
        return "↩️", f"+{amount}"
    return "➕", f"+{amount}"


def _format_ledger_entry(entry: Dict[str, Any]) -> Optional[str]:
    try:
        entry_type = str(entry.get("type", "")).strip().lower()
    except Exception:
        entry_type = ""

    amount_raw = entry.get("amount")
    try:
        amount = abs(int(amount_raw))
    except (TypeError, ValueError):
        amount = 0

    icon, amount_text = _ledger_amount_parts(entry_type, amount)
    reason = _ledger_reason(entry)
    ts_text = _ledger_timestamp(entry)
    balance_after = entry.get("balance_after")
    try:
        balance_text = f"{int(balance_after)}"
    except (TypeError, ValueError):
        balance_text = "—"

    return f"{icon} {amount_text}💎 • {reason} • {ts_text} • Баланс: {balance_text}💎"


def _build_transactions_view(user_id: int, offset: int) -> tuple[str, InlineKeyboardMarkup, int]:
    try:
        entries = get_ledger_entries(user_id, offset=offset, limit=LEDGER_PAGE_SIZE)
    except Exception:
        log.exception("ledger_entries_failed | user=%s offset=%s", user_id, offset)
        entries = []

    lines: List[str] = ["🧾 История операций (последние 10)", ""]

    formatted: List[str] = []
    for entry in entries:
        if isinstance(entry, dict):
            row = _format_ledger_entry(entry)
            if row:
                formatted.append(row)

    if formatted:
        lines.extend(formatted)
    else:
        lines.append("Пока нет операций.")

    shown = len(formatted)

    total = None
    try:
        total = get_ledger_count(user_id)
    except Exception:
        log.exception("ledger_count_failed | user=%s", user_id)

    has_more = False
    if total is not None:
        has_more = total > offset + shown
    else:
        has_more = shown >= LEDGER_PAGE_SIZE

    keyboard: List[List[InlineKeyboardButton]] = [
        [InlineKeyboardButton("◀️ Назад", callback_data="tx:back")]
    ]
    if has_more:
        next_offset = offset + shown
        keyboard[0].append(
            InlineKeyboardButton("Показать ещё", callback_data=f"tx:page:{next_offset}")
        )

    return "\n".join(lines), InlineKeyboardMarkup(keyboard), shown


async def _edit_transactions_message(
    query: "telegram.CallbackQuery",
    ctx: ContextTypes.DEFAULT_TYPE,
    user_id: int,
    offset: int,
) -> None:
    message = query.message
    if message is None:
        return

    text, keyboard, _ = _build_transactions_view(user_id, offset)
    await _safe_edit_message_text(query.edit_message_text, text, reply_markup=keyboard)

    s = state(ctx)
    s["last_panel"] = "balance_history"
    msg_ids = s.get("msg_ids")
    if not isinstance(msg_ids, dict):
        msg_ids = {}
        s["msg_ids"] = msg_ids
    msg_ids["balance"] = message.message_id


async def _edit_balance_from_history(
    query: "telegram.CallbackQuery", ctx: ContextTypes.DEFAULT_TYPE, user_id: int
) -> None:
    message = query.message
    if message is None:
        return

    balance = _safe_get_balance(user_id)
    _set_cached_balance(ctx, balance)
    await _safe_edit_message_text(
        query.edit_message_text,
        f"💎 Ваш баланс: {balance}",
        reply_markup=balance_menu_kb(),
        parse_mode=ParseMode.MARKDOWN,
        disable_web_page_preview=True,
    )

    s = state(ctx)
    s["last_panel"] = "balance"
    msg_ids = s.get("msg_ids")
    if not isinstance(msg_ids, dict):
        msg_ids = {}
        s["msg_ids"] = msg_ids
    msg_ids["balance"] = message.message_id


def _short_prompt(prompt: Optional[str], limit: int = 120) -> str:
    txt = (prompt or "").strip()
    if not txt:
        return ""
    normalized = re.sub(r"\s+", " ", txt)
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rstrip() + "…"

def _mj_format_card_text(aspect: str) -> str:
    aspect = "9:16" if aspect == "9:16" else "16:9"
    choice = "Горизонтальный (16:9)" if aspect == "16:9" else "Вертикальный (9:16)"
    return (
        "🖼 <b>Midjourney</b>\n"
        "Выберите формат изображения.\n\n"
        "• Горизонтальный — 16:9\n"
        "• Вертикальный — 9:16\n\n"
        f"Текущий выбор: <b>{choice}</b>"
    )

def _mj_format_keyboard(aspect: str) -> InlineKeyboardMarkup:
    aspect = "9:16" if aspect == "9:16" else "16:9"
    def _btn(label: str, value: str) -> InlineKeyboardButton:
        mark = "✅ " if value == aspect else ""
        return InlineKeyboardButton(f"{mark}{label}", callback_data=f"mj:aspect:{value}")
    keyboard = [
        [_btn("Горизонтальный (16:9)", "16:9")],
        [_btn("Вертикальный (9:16)", "9:16")],
        [InlineKeyboardButton("Назад", callback_data="back")],
    ]
    return InlineKeyboardMarkup(keyboard)

def _mj_prompt_card_text(aspect: str, prompt: Optional[str]) -> str:
    aspect = "9:16" if aspect == "9:16" else "16:9"
    lines = [
        "🖼 <b>Midjourney</b>",
        "",
        'Введите промпт сообщением. После этого нажмите «Подтвердить».',
        f"Текущий формат: <b>{aspect}</b>",
    ]
    snippet = _short_prompt(prompt)
    snippet_html = html.escape(snippet) if snippet else ""
    display = snippet_html if snippet_html else " "
    lines.extend(["", f"Промпт: <i>{display}</i>"])
    return "\n".join(lines)

def _mj_prompt_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Подтвердить", callback_data="mj:confirm")],
        [
            InlineKeyboardButton("Отменить", callback_data="mj:cancel"),
            InlineKeyboardButton("Сменить формат", callback_data="mj:change_format"),
        ],
        [InlineKeyboardButton("🔁 Сменить движок", callback_data="mj:switch_engine")],
    ])


async def _remove_state_card(
    ctx: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    state_dict: Dict[str, Any],
    *,
    state_key: str,
    cache_key: Optional[str] = None,
) -> None:
    mid = state_dict.get(state_key)
    if isinstance(mid, int):
        with suppress(Exception):
            await ctx.bot.delete_message(chat_id, mid)
    if cache_key:
        state_dict[cache_key] = None
    state_dict[state_key] = None
    msg_ids_raw = state_dict.get("msg_ids")
    if isinstance(msg_ids_raw, dict):
        for key, value in list(msg_ids_raw.items()):
            if value == mid:
                msg_ids_raw[key] = None


def _image_engine_card_text(selected: Optional[str]) -> str:
    choice_map = {"mj": "Midjourney", "banana": "Banana Editor"}
    lines = [
        "🎨 <b>Выберите движок для изображений</b>",
        "",
        "Midjourney — стандартная генерация по текстовому промпту.",
        "Banana Editor — меняем или улучшаем ваши фото.",
    ]
    if selected in choice_map:
        lines.append("")
        lines.append(f"Текущий выбор: <b>{choice_map[selected]}</b>")
    return "\n".join(lines)


def _image_engine_keyboard(selected: Optional[str]) -> InlineKeyboardMarkup:
    mark_mj = "✅ " if selected == "mj" else ""
    mark_banana = "✅ " if selected == "banana" else ""
    rows = [
        [InlineKeyboardButton(f"{mark_mj}Midjourney", callback_data="img_engine:mj")],
        [InlineKeyboardButton(f"{mark_banana}Banana", callback_data="img_engine:banana")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="back")],
    ]
    return InlineKeyboardMarkup(rows)


async def show_image_engine_selector(
    chat_id: int,
    ctx: ContextTypes.DEFAULT_TYPE,
    *,
    force_new: bool = False,
) -> None:
    s = state(ctx)
    await _remove_state_card(ctx, chat_id, s, state_key="last_ui_msg_id_mj", cache_key="_last_text_mj")
    await _remove_state_card(
        ctx,
        chat_id,
        s,
        state_key="last_ui_msg_id_banana",
        cache_key="_last_text_banana",
    )
    text = _image_engine_card_text(s.get("image_engine"))
    if not force_new and text == s.get("_last_text_image_engine"):
        return
    keyboard = _image_engine_keyboard(s.get("image_engine"))
    mid = await upsert_card(
        ctx,
        chat_id,
        s,
        "last_ui_msg_id_image_engine",
        text,
        keyboard,
        force_new=force_new,
    )
    s["mode"] = "image_engine_select"
    if mid:
        s["_last_text_image_engine"] = text
    else:
        s["_last_text_image_engine"] = None


async def _close_image_engine_selector(
    chat_id: int,
    ctx: ContextTypes.DEFAULT_TYPE,
) -> None:
    s = state(ctx)
    await _remove_state_card(
        ctx,
        chat_id,
        s,
        state_key="last_ui_msg_id_image_engine",
        cache_key="_last_text_image_engine",
    )


async def _open_image_engine(
    chat_id: int,
    ctx: ContextTypes.DEFAULT_TYPE,
    engine: str,
    *,
    user_id: Optional[int],
    source: Optional[str] = None,
    force_new: bool = True,
) -> None:
    s = state(ctx)
    previous_engine = s.get("image_engine")
    await _close_image_engine_selector(chat_id, ctx)
    if engine == "mj":
        s["image_engine"] = "mj"
        s["mode"] = "mj_txt"
        if s.get("aspect") not in {"16:9", "9:16"}:
            s["aspect"] = "16:9"
        await mj_entry(chat_id, ctx)
        card_id = s.get("last_ui_msg_id_mj") if isinstance(s.get("last_ui_msg_id_mj"), int) else None
        _activate_wait_state(
            user_id=user_id,
            chat_id=chat_id,
            card_msg_id=card_id,
            kind=WaitKind.MJ_PROMPT,
            meta={"engine": "mj", "aspect": s.get("aspect"), "source": source},
        )
        return
    if engine == "banana":
        s["image_engine"] = "banana"
        s["mode"] = "banana"
        if previous_engine != "banana":
            s["last_prompt"] = None
            s["_last_text_banana"] = None
        await banana_entry(chat_id, ctx, force_new=force_new)
        card_id = (
            s.get("last_ui_msg_id_banana")
            if isinstance(s.get("last_ui_msg_id_banana"), int)
            else None
        )
        _activate_wait_state(
            user_id=user_id,
            chat_id=chat_id,
            card_msg_id=card_id,
            kind=WaitKind.BANANA_PROMPT,
            meta={"engine": "banana", "source": source or "image_engine"},
        )

async def _update_mj_card(chat_id: int, ctx: ContextTypes.DEFAULT_TYPE, text: str,
                                reply_markup: Optional[InlineKeyboardMarkup], *, force: bool = False) -> None:
    s = state(ctx)
    if not force and text == s.get("_last_text_mj"):
        return
    mid = await upsert_card(
        ctx,
        chat_id,
        s,
        "last_ui_msg_id_mj",
        text,
        reply_markup,
        force_new=force,
    )
    if mid:
        s["_last_text_mj"] = text
    elif force:
        s["_last_text_mj"] = None

async def show_mj_format_card(
    chat_id: int,
    ctx: ContextTypes.DEFAULT_TYPE,
    *,
    force_new: bool = False,
) -> None:
    s = state(ctx)
    aspect = "9:16" if s.get("aspect") == "9:16" else "16:9"
    s["aspect"] = aspect
    s["last_prompt"] = None
    await _update_mj_card(
        chat_id,
        ctx,
        _mj_format_card_text(aspect),
        _mj_format_keyboard(aspect),
        force=force_new,
    )

async def show_mj_prompt_card(
    chat_id: int,
    ctx: ContextTypes.DEFAULT_TYPE,
    *,
    force_new: bool = False,
) -> None:
    s = state(ctx)
    aspect = "9:16" if s.get("aspect") == "9:16" else "16:9"
    s["aspect"] = aspect
    await _update_mj_card(
        chat_id,
        ctx,
        _mj_prompt_card_text(aspect, s.get("last_prompt")),
        _mj_prompt_keyboard(),
        force=force_new,
    )

async def show_mj_generating_card(chat_id: int, ctx: ContextTypes.DEFAULT_TYPE, prompt: str, aspect: str) -> None:
    aspect = "9:16" if aspect == "9:16" else "16:9"
    snippet = html.escape(_short_prompt(prompt, 160) or "—")
    text = (
        "⏳ Midjourney генерирует изображение…\n"
        f"Формат: <b>{aspect}</b>\n"
        f"Промпт: <code>{snippet}</code>"
    )
    await _update_mj_card(chat_id, ctx, text, None)

async def mj_entry(chat_id: int, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    s = state(ctx)
    mid = s.get("last_ui_msg_id_mj")
    if mid:
        with suppress(Exception):
            await ctx.bot.delete_message(chat_id, mid)
    s["last_ui_msg_id_mj"] = None
    s["_last_text_mj"] = None
    await show_mj_format_card(chat_id, ctx, force_new=True)

def banana_examples_block() -> str:
    return (
        "💡 <b>Примеры запросов:</b>\n"
        "• поменяй фон на городской вечер\n"
        "• смени одежду на чёрный пиджак\n"
        "• добавь лёгкий макияж, подчеркни глаза\n"
        "• убери лишние предметы со стола\n"
        "• поставь нас на одну фотографию"
    )

BANANA_MODE_HINT_MD = (
    "🍌 Banana включён\n"
    "Сначала пришлите до *4 фото* (можно по одному). Когда будут готовы — пришлите *текст-промпт*, что изменить."
)

MJ_MODE_HINT_TEXT = (
    "🖼 Midjourney включён. Введите промпт сообщением и нажмите «Подтвердить»."
)

def banana_card_text(s: Dict[str, Any]) -> str:
    n = len(s.get("banana_images") or [])
    prompt = (s.get("last_prompt") or "").strip()
    prompt_html = html.escape(prompt)
    has_prompt = "есть" if prompt else "нет"
    lines = [
        "🍌 <b>Карточка Banana</b>",
        f"🧩 Фото: <b>{n}/4</b>  •  Промпт: <b>{has_prompt}</b>",
        "",
        "🖊️ <b>Промпт:</b>",
        f"<code>{prompt_html}</code>" if prompt else "<code></code>",
        "",
        banana_examples_block()
    ]
    balance = s.get("banana_balance")
    if balance is not None:
        lines.insert(1, f"💎 Баланс: <b>{balance}</b>")
    return "\n".join(lines)

def banana_kb() -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton("➕ Добавить ещё фото", callback_data="banana:add_more")],
        [InlineKeyboardButton("🧹 Очистить фото", callback_data="banana:reset_imgs")],
        [InlineKeyboardButton("✍️ Изменить промпт", callback_data="banana:edit_prompt")],
        [InlineKeyboardButton("🚀 Начать генерацию Banana", callback_data="banana:start")],
        [InlineKeyboardButton("🔁 Сменить движок", callback_data="banana:switch_engine")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="back")],
    ]
    return InlineKeyboardMarkup(rows)


def banana_result_keyboard() -> ReplyKeyboardMarkup:
    rows = [
        [KeyboardButton("🔁 Повторить")],
        [KeyboardButton("⬅️ Назад в меню")],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)


# --------- Suno Helpers ----------


class SunoConfigError(RuntimeError):
    pass


class SunoApiError(RuntimeError):
    def __init__(self, message: str, *, status: Optional[int] = None, payload: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.status = status
        self.payload = payload or {}


def _suno_configured() -> bool:
    return SUNO_CONFIG.configured


def _suno_log(
    *,
    user_id: Optional[int],
    phase: str,
    request_url: str,
    http_status: Optional[int],
    response_snippet: Optional[str],
    req_id: Optional[str] = None,
) -> None:
    snippet = (response_snippet or "").strip()
    if len(snippet) > 500:
        snippet = snippet[:497] + "…"
    snippet = snippet.replace("\n", " ")
    entry = {
        "timestamp": _utcnow_iso(),
        "user_id": user_id,
        "phase": phase,
        "request_url": request_url,
        "http_status": http_status,
        "response_snippet": snippet,
    }
    if req_id:
        entry["req_id"] = req_id
    if phase == "error":
        log.warning(
            "Suno %s error | user=%s status=%s url=%s snippet=%s",
            phase,
            user_id,
            http_status,
            request_url,
            snippet,
            extra={"meta": {"req_id": req_id}},
        )
    else:
        log.info(
            "Suno %s | user=%s status=%s url=%s snippet=%s",
            phase,
            user_id,
            http_status,
            request_url,
            snippet,
            extra={"meta": {"req_id": req_id}},
        )
    if not rds:
        return
    try:
        rds.lpush(SUNO_LOG_KEY, json.dumps(entry, ensure_ascii=False))
        rds.ltrim(SUNO_LOG_KEY, 0, 199)
    except Exception as exc:
        log.warning("Suno log store failed | err=%s", exc)


def _suno_sanitize_log_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        sanitized: Dict[str, Any] = {}
        for key, value in payload.items():
            lowered = key.lower()
            if isinstance(value, str) and any(
                token in lowered for token in ("prompt", "lyrics", "lyric", "text")
            ):
                sanitized[key] = f"<len={len(value)}>"
            else:
                sanitized[key] = _suno_sanitize_log_payload(value)
        return sanitized
    if isinstance(payload, list):
        return [_suno_sanitize_log_payload(item) for item in payload]
    return payload


def _suno_headers() -> Dict[str, str]:
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    auth_header = _kie_auth_header()
    if auth_header:
        headers.update(auth_header)
    return headers


async def _suno_request(
    method: str,
    path: str,
    *,
    json_payload: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
    timeout: float,
    log_user_id: Optional[int],
    log_phase: str,
) -> Dict[str, Any]:
    if not _suno_configured():
        raise SunoConfigError("Suno API is not configured")

    url = _compose_suno_url(SUNO_CONFIG.base, SUNO_CONFIG.prefix, path)
    headers = _suno_headers()
    method_upper = method.upper()
    request_params: Optional[Dict[str, Any]] = None
    if params:
        request_params = {k: v for k, v in params.items() if v is not None}

    payload_keys: List[str] = []
    if json_payload:
        payload_keys = sorted(json_payload.keys())

    display_url = url
    if request_params:
        try:
            display_url = f"{url}?{urlencode(request_params, doseq=True)}"
        except Exception:
            display_url = url

    timeout_cfg = ClientTimeout(total=timeout)
    attempt = 0

    async def _clear_wait() -> None:
        with suppress(Exception):
            await delete_wait_sticker(ctx, chat_id=chat_id)
        if s.get("video_wait_message_id"):
            s["video_wait_message_id"] = None
        if s.get("video_wait_message_id"):
            s["video_wait_message_id"] = None
    dynamic_attempts = int(timeout // 10) if timeout else 0
    max_attempts = max(1, min(10, dynamic_attempts if dynamic_attempts > 0 else 3))

    async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
        while attempt < max_attempts:
            attempt += 1
            log.info("[SUNO] %s %s payload_keys=%s", method_upper, display_url, payload_keys)
            try:
                async with session.request(
                    method_upper,
                    url,
                    headers=headers,
                    json=json_payload,
                    params=request_params,
                ) as resp:
                    text_payload = await resp.text()
                    if resp.status == 429 and attempt < max_attempts:
                        retry_after = resp.headers.get("Retry-After")
                        try:
                            delay = float(retry_after) if retry_after else SUNO_POLL_INTERVAL
                        except (TypeError, ValueError):
                            delay = SUNO_POLL_INTERVAL
                        await asyncio.sleep(max(delay, 1.0))
                        continue

                    try:
                        data = await resp.json(content_type=None)
                    except Exception:
                        data = {"raw": text_payload} if text_payload else {}

                    if resp.status >= 400:
                        sanitized = (
                            _suno_sanitize_log_payload(data)
                            if isinstance(data, dict)
                            else None
                        )
                        if sanitized is not None:
                            snippet = json.dumps(sanitized, ensure_ascii=False)
                        elif isinstance(data, dict):
                            snippet = json.dumps(data, ensure_ascii=False)
                        else:
                            snippet = text_payload or ""
                        _suno_log(
                            user_id=log_user_id,
                            phase="error",
                            request_url=url,
                            http_status=resp.status,
                            response_snippet=snippet,
                        )
                        raise SunoApiError(
                            f"Suno API error: {resp.status}",
                            status=resp.status,
                            payload=data if isinstance(data, dict) else {"raw": text_payload},
                        )

                    if isinstance(data, dict):
                        sanitized = _suno_sanitize_log_payload(data)
                        _suno_log(
                            user_id=log_user_id,
                            phase=log_phase,
                            request_url=url,
                            http_status=resp.status,
                            response_snippet=
                            json.dumps(sanitized, ensure_ascii=False)
                            if sanitized is not None
                            else text_payload,
                        )
                        return data
                    raise SunoApiError(
                        "Unexpected response payload",
                        status=resp.status,
                        payload={"raw": data},
                    )
            except aiohttp.ClientError as exc:
                if attempt >= max_attempts:
                    _suno_log(
                        user_id=log_user_id,
                        phase="error",
                        request_url=url,
                        http_status=None,
                        response_snippet=f"network_error:{exc}",
                    )
                    raise SunoApiError("Network error", payload={"error": str(exc)}) from exc
                await asyncio.sleep(min(2.0 * attempt, 10.0))
        _suno_log(
            user_id=log_user_id,
            phase="error",
            request_url=url,
            http_status=None,
            response_snippet="max_retries",
        )
        raise SunoApiError("Max retries exceeded", payload={"url": url})


async def suno_create_task(payload: Dict[str, Any], *, user_id: Optional[int]) -> Dict[str, Any]:
    timeout = float(SUNO_CONFIG.timeout_sec or 60)
    return await _suno_request(
        "POST",
        SUNO_CONFIG.gen_path,
        json_payload=payload,
        params=None,
        timeout=timeout,
        log_user_id=user_id,
        log_phase="create",
    )


async def suno_poll_task(task_id: str, *, user_id: Optional[int]) -> Dict[str, Any]:
    timeout = float(SUNO_CONFIG.timeout_sec or 60)
    params = {"taskId": task_id}
    if user_id is not None:
        params["userId"] = str(user_id)
    return await _suno_request(
        "GET",
        SUNO_CONFIG.status_path,
        json_payload=None,
        params=params,
        timeout=timeout,
        log_user_id=user_id,
        log_phase="poll",
    )


async def suno_extend_track(payload: Dict[str, Any], *, user_id: Optional[int]) -> Dict[str, Any]:
    timeout = float(SUNO_CONFIG.timeout_sec or 60)
    return await _suno_request(
        "POST",
        SUNO_CONFIG.extend_path,
        json_payload=payload,
        params=None,
        timeout=timeout,
        log_user_id=user_id,
        log_phase="extend",
    )


async def suno_generate_lyrics(payload: Dict[str, Any], *, user_id: Optional[int]) -> Dict[str, Any]:
    timeout = float(SUNO_CONFIG.timeout_sec or 60)
    return await _suno_request(
        "POST",
        SUNO_CONFIG.lyrics_path,
        json_payload=payload,
        params=None,
        timeout=timeout,
        log_user_id=user_id,
        log_phase="lyrics",
    )


async def _run_suno_probe() -> None:
    if not SUNO_CONFIG.enabled:
        return
    if not SUNO_CONFIG.configured:
        log.info("SUNO probe skipped: configuration incomplete")
        return

    auth_header = _kie_auth_header().get("Authorization")
    if not auth_header:
        log.info("SUNO probe skipped: missing authorization header")
        return

    probe_payload = json.dumps(
        {
            "model": SUNO_MODEL,
            "title": "Probe",
            "style": "Electro-pop",
            "instrumental": True,
        },
        ensure_ascii=False,
    )

    cmd = [
        "curl",
        "-s",
        "-o",
        "/dev/null",
        "-w",
        "SUNO_PROBE:%{http_code}\\n",
        "-H",
        f"Authorization: {auth_header}",
        "-H",
        "Content-Type: application/json",
        "-d",
        probe_payload,
        SUNO_GEN_URL,
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        log.warning("SUNO probe skipped: curl not available")
        return
    except Exception as exc:
        log.warning("SUNO probe failed to start: %s", exc)
        return

    stdout, stderr = await proc.communicate()
    if stdout:
        text = stdout.decode().strip()
        if text:
            log.info(text)
    if stderr:
        err_text = stderr.decode().strip()
        if err_text:
            log.warning("SUNO probe stderr: %s", err_text)


def _suno_collect_params(state_obj: Dict[str, Any], suno_state: SunoState) -> Dict[str, Any]:
    state_obj["suno_state"] = suno_state.to_dict()
    payload = {
        "title": suno_state.title or "",
        "style": suno_state.style or "",
        "lyrics": suno_state.lyrics if suno_state.lyrics_source == LyricsSource.USER else None,
        "instrumental": suno_state.mode != "lyrics",
        "has_lyrics": suno_state.mode == "lyrics" and suno_state.lyrics_source == LyricsSource.USER,
        "preset": suno_state.preset,
        "mode": suno_state.mode,
        "lyrics_source": suno_state.lyrics_source.value,
    }
    if suno_state.mode == "cover":
        payload["instrumental"] = False
        payload["operationType"] = "upload-and-cover-audio"
        payload["cover_source_url"] = suno_state.cover_source_url or suno_state.source_url
        payload["cover_source_label"] = suno_state.cover_source_label
        payload["source_file_id"] = suno_state.source_file_id
        payload["source_url"] = suno_state.source_url
        payload["kie_file_id"] = suno_state.kie_file_id
    elif suno_state.mode == "lyrics" and suno_state.lyrics_source == LyricsSource.USER:
        payload["lyrics"] = suno_state.lyrics or ""
    else:
        payload.pop("lyrics", None)
    return payload


def _suno_make_preview(text: str, limit: int = 160) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""
    if len(raw) <= limit:
        return raw
    clipped = raw[: limit - 1].rstrip()
    return clipped + "…"


def _suno_missing_fields(state: SunoState) -> list[str]:
    config = get_suno_mode_config(state.mode)
    missing: list[str] = []
    for field in config.required_fields:
        if field == "title" and not state.title:
            missing.append(SUNO_FIELD_LABELS.get("title", "Title"))
        elif field == "style" and not state.style:
            missing.append(SUNO_FIELD_LABELS.get("style", "Style"))
        elif field == "lyrics":
            if state.mode == "lyrics" and state.lyrics_source == LyricsSource.USER and not state.lyrics:
                missing.append(SUNO_FIELD_LABELS.get("lyrics", "Lyrics"))
        elif field == "reference" and not state.kie_file_id:
            missing.append(SUNO_FIELD_LABELS.get("reference", "Reference"))
    return missing


def _suno_summary_text(state: SunoState) -> str:
    config = get_suno_mode_config(state.mode)
    lines = [f"{config.emoji} {config.title}"]
    title_display = state.title.strip() if state.title else "—"
    lines.append(f"✏️ {t('suno.field.title')}: {title_display}")

    if state.style:
        try:
            preview_func = suno_style_preview  # type: ignore[name-defined]
        except NameError:
            preview_func = None
        style_display: Optional[str] = None
        if callable(preview_func):
            try:
                style_display = preview_func(state.style, limit=160) or "—"
            except Exception:
                style_display = None
        if not style_display:
            raw_style = state.style or "—"
            style_display = _suno_make_preview(raw_style, limit=160) or raw_style[:160]
    else:
        style_display = "—"
    lines.append(f"🎛️ {t('suno.field.style')}: {style_display}")

    if state.mode == "lyrics":
        source_text = t("suno.lyrics_source.user") if state.lyrics_source == LyricsSource.USER else t("suno.lyrics_source.ai")
        lines.append(f"📥 {t('suno.field.lyrics_source')}: {source_text}")
        if state.lyrics_source == LyricsSource.USER:
            if state.lyrics:
                lines_count = len([line for line in state.lyrics.split("\n") if line.strip()])
                char_count = len(state.lyrics)
                lines.append(f"📝 {t('suno.field.lyrics')}: {lines_count} строк ({char_count} символов)")
            else:
                lines.append(f"📝 {t('suno.field.lyrics')}: —")
    elif state.mode == "cover":
        if state.kie_file_id:
            reference_display = f"загружено ✅ (id: {state.kie_file_id})"
        elif state.cover_source_label:
            reference_display = state.cover_source_label
        elif state.cover_source_url:
            reference_display = state.cover_source_url
        else:
            reference_display = "—"
        lines.append(f"🎧 {t('suno.field.source')}: {reference_display}")
    return "\n".join(lines)


def _suno_result_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("🔁 Повторить", callback_data="suno:repeat")],
            [InlineKeyboardButton("⬅️ В меню", callback_data="back")],
        ]
    )


def _music_flow_steps(flow: str) -> list[str]:
    mapping = {
        "instrumental": ["style", "title"],
        "lyrics": ["style", "title", "lyrics"],
        "cover": ["source", "style", "title"],
    }
    return mapping.get(flow, [])


def _music_flow_keyboard() -> InlineKeyboardMarkup:
    return suno_modes_keyboard()


async def _music_show_main_menu(
    chat_id: int,
    ctx: ContextTypes.DEFAULT_TYPE,
    state_dict: Dict[str, Any],
) -> None:
    state_dict["suno_flow"] = None
    state_dict["suno_step"] = None
    state_dict["suno_step_order"] = None
    state_dict["suno_auto_lyrics_pending"] = False
    state_dict["suno_lyrics_confirmed"] = False
    state_dict["suno_cover_source_label"] = None
    state_dict["suno_auto_lyrics_generated"] = False
    text = SUNO_MODE_PROMPT
    await tg_safe_send(
        ctx.bot.send_message,
        method_name="sendMessage",
        kind="message",
        chat_id=chat_id,
        text=text,
        parse_mode=ParseMode.HTML,
        reply_markup=_music_flow_keyboard(),
    )


def _music_step_index(state_dict: Dict[str, Any], step: Optional[str]) -> tuple[int, int]:
    order_raw = state_dict.get("suno_step_order")
    order = order_raw if isinstance(order_raw, list) else []
    total = len(order)
    if not step or step not in order:
        return (0, total)
    try:
        idx = order.index(step)
    except ValueError:
        return (0, total)
    return (idx + 1, total)


def _music_step_prompt_text(
    flow: str,
    step: str,
    index: int,
    total: int,
    suno_state: SunoState,
) -> str:
    prompt_index = index if index else 1
    prompt_total = total if total else 1
    if step == "style":
        current = _suno_field_preview(suno_state, "style")
        return t(
            "suno.prompt.step.style",
            index=prompt_index,
            total=prompt_total,
            current=current,
        )
    if step == "title":
        current = _suno_field_preview(suno_state, "title")
        return t(
            "suno.prompt.step.title",
            index=prompt_index,
            total=prompt_total,
            current=current,
        )
    if step == "lyrics":
        current = _suno_field_preview(suno_state, "lyrics")
        return t(
            "suno.prompt.step.lyrics",
            index=prompt_index,
            total=prompt_total,
            current=current,
            limit=LYRICS_MAX_LENGTH,
        )
    if step == "source":
        current = _suno_field_preview(suno_state, "cover")
        return t(
            "suno.prompt.step.source",
            index=prompt_index,
            total=prompt_total,
            current=current,
        )
    return t("suno.prompt.step.generic")


def _music_wait_kind(step: str) -> Optional[WaitKind]:
    mapping = {
        "style": WaitKind.SUNO_STYLE,
        "title": WaitKind.SUNO_TITLE,
        "lyrics": WaitKind.SUNO_LYRICS,
    }
    return mapping.get(step)


def _music_card_message_id(state_dict: Dict[str, Any]) -> Optional[int]:
    card_state = state_dict.get("suno_card")
    if isinstance(card_state, dict):
        msg_id = card_state.get("msg_id")
        if isinstance(msg_id, int):
            return msg_id
    msg_id = state_dict.get("last_ui_msg_id_suno")
    if isinstance(msg_id, int):
        return msg_id
    return None


async def _music_prompt_step(
    chat_id: int,
    ctx: ContextTypes.DEFAULT_TYPE,
    state_dict: Dict[str, Any],
    *,
    flow: str,
    step: Optional[str],
    user_id: Optional[int],
) -> None:
    if not step:
        await tg_safe_send(
            ctx.bot.send_message,
            method_name="sendMessage",
            kind="message",
            chat_id=chat_id,
            text=SUNO_START_READY_MESSAGE,
        )
        return

    index, total = _music_step_index(state_dict, step)
    suno_state_obj = load_suno_state(ctx)
    text = _music_step_prompt_text(flow, step, index, total, suno_state_obj)
    await tg_safe_send(
        ctx.bot.send_message,
        method_name="sendMessage",
        kind="message",
        chat_id=chat_id,
        text=text,
    )

    if flow == "cover" and step == "source":
        state_dict["suno_waiting_state"] = WAIT_SUNO_REFERENCE
        return

    wait_kind = _music_wait_kind(step)
    if wait_kind is not None:
        if wait_kind == WaitKind.SUNO_STYLE:
            waiting_value = WAIT_SUNO_STYLE
        elif wait_kind == WaitKind.SUNO_TITLE:
            waiting_value = WAIT_SUNO_TITLE
        else:
            waiting_value = WAIT_SUNO_LYRICS
        state_dict["suno_waiting_state"] = waiting_value
        _activate_wait_state(
            user_id=user_id,
            chat_id=chat_id,
            card_msg_id=_music_card_message_id(state_dict),
            kind=wait_kind,
            meta={"flow": flow, "step": step},
        )
    else:
        state_dict["suno_waiting_state"] = IDLE_SUNO


def _music_next_step(state_dict: Dict[str, Any]) -> Optional[str]:
    order_raw = state_dict.get("suno_step_order")
    order = order_raw if isinstance(order_raw, list) else []
    current = state_dict.get("suno_step")
    if not order:
        state_dict["suno_step"] = None
        return None
    if current not in order:
        state_dict["suno_step"] = order[0]
        return order[0]
    idx = order.index(current)
    if idx + 1 < len(order):
        state_dict["suno_step"] = order[idx + 1]
        return order[idx + 1]
    state_dict["suno_step"] = None
    return None


async def _music_begin_flow(
    chat_id: int,
    ctx: ContextTypes.DEFAULT_TYPE,
    state_dict: Dict[str, Any],
    *,
    flow: str,
    user_id: Optional[int],
) -> None:
    suno_state_obj = load_suno_state(ctx)
    card_state = state_dict.get("suno_card")
    card_msg_id: Optional[int] = None
    card_chat_id: Optional[int] = None
    if isinstance(card_state, Mapping):
        raw_msg_id = card_state.get("msg_id")
        raw_chat_id = card_state.get("chat_id")
        if isinstance(raw_msg_id, int):
            card_msg_id = raw_msg_id
        if isinstance(raw_chat_id, int):
            card_chat_id = raw_chat_id
    if card_msg_id is None:
        raw_last_id = state_dict.get("last_ui_msg_id_suno")
        if isinstance(raw_last_id, int):
            card_msg_id = raw_last_id

    effective_chat_id = card_chat_id if isinstance(card_chat_id, int) else chat_id
    start_msg_id = getattr(suno_state_obj, "start_msg_id", None)
    if isinstance(start_msg_id, int) and isinstance(effective_chat_id, int):
        await safe_delete_message(ctx.bot, effective_chat_id, start_msg_id)
    state_dict.pop("suno_start_msg_id", None)
    msg_ids = state_dict.get("msg_ids")
    if isinstance(msg_ids, MutableMapping):
        msg_ids.pop("suno_start", None)

    reset_suno_card_state(
        suno_state_obj,
        mode=flow if flow in {"instrumental", "lyrics", "cover"} else "instrumental",  # type: ignore[arg-type]
        card_message_id=card_msg_id,
        card_chat_id=effective_chat_id if isinstance(effective_chat_id, int) else None,
    )
    state_dict["suno_cover_source_label"] = None
    save_suno_state(ctx, suno_state_obj)
    state_dict["suno_state"] = suno_state_obj.to_dict()
    state_dict["suno_flow"] = flow
    state_dict["suno_last_mode"] = flow
    order = _music_flow_steps(flow)
    state_dict["suno_step_order"] = order
    state_dict["suno_step"] = order[0] if order else None
    state_dict["suno_auto_lyrics_pending"] = False
    state_dict["suno_auto_lyrics_generated"] = False
    state_dict["suno_lyrics_confirmed"] = False
    state_dict["suno_waiting_state"] = IDLE_SUNO
    _reset_suno_card_cache(state_dict)
    await refresh_suno_card(ctx, chat_id, state_dict, price=PRICE_SUNO)
    await _music_prompt_step(
        chat_id,
        ctx,
        state_dict,
        flow=flow,
        step=state_dict.get("suno_step"),
        user_id=user_id,
    )


def _music_generate_auto_lyrics(style: Optional[str], title: Optional[str]) -> str:
    keywords = []
    if style:
        for token in re.split(r"[,/]+", style):
            item = token.strip()
            if item:
                keywords.append(item)
    base_title = (title or "Midnight Echo").strip()
    prompt = build_suno_prompt(base_title, style=style or "dream pop")
    mood = keywords[0] if keywords else "dreamwave"
    vibe = keywords[1] if len(keywords) > 1 else "city lights"
    chorus = keywords[2] if len(keywords) > 2 else "endless glow"
    lines = [
        f"{base_title} in the {mood}",
        f"Hearts align with {vibe}",
        f"Hold me close through static nights",
        f"Chorus of {chorus} in our eyes",
    ]
    if prompt:
        pass
    return "\n".join(lines)


def _music_apply_auto_lyrics(
    ctx: ContextTypes.DEFAULT_TYPE,
    state_dict: Dict[str, Any],
    *,
    style: Optional[str],
    title: Optional[str],
) -> None:
    suno_state_obj = load_suno_state(ctx)
    lyrics = _music_generate_auto_lyrics(style, title)
    set_suno_lyrics(suno_state_obj, lyrics)
    set_suno_lyrics_source(suno_state_obj, LyricsSource.AI)
    save_suno_state(ctx, suno_state_obj)
    state_dict["suno_state"] = suno_state_obj.to_dict()
    state_dict["suno_auto_lyrics_pending"] = False
    state_dict["suno_auto_lyrics_generated"] = True
    state_dict["suno_lyrics_confirmed"] = False


def _music_store_cover_source(
    ctx: ContextTypes.DEFAULT_TYPE,
    state_dict: Dict[str, Any],
    *,
    url: Optional[str] = None,
    label: Optional[str] = None,
    source_file_id: Optional[str] = None,
    source_url: Optional[str] = None,
    kie_file_id: Optional[str] = None,
) -> None:
    suno_state_obj = load_suno_state(ctx)
    effective_source_url = source_url if source_url is not None else url
    set_suno_cover_source(
        suno_state_obj,
        url,
        label,
        file_id=source_file_id,
        source_url=effective_source_url,
        kie_file_id=kie_file_id,
    )
    save_suno_state(ctx, suno_state_obj)
    state_dict["suno_state"] = suno_state_obj.to_dict()
    display_label = label or url or effective_source_url
    state_dict["suno_cover_source_label"] = display_label


_COVER_INVALID_INPUT_MESSAGE = (
    f"⚠️ Нужен аудиофайл (mp3/wav) до {COVER_MAX_AUDIO_MB} МБ или ссылка http/https на аудио."
)
_COVER_UPLOAD_CLIENT_ERROR_MESSAGE = t("suno.error.upload_client")
_COVER_UPLOAD_SERVICE_ERROR_MESSAGE = t("suno.error.upload_service")


def _cover_sanitize_label(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text[:120]


async def _cover_process_audio_input(
    ctx: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    message: "Message",
    state_dict: Dict[str, Any],
    audio_obj: Any,
    *,
    user_id: Optional[int],
) -> bool:
    request_id = _generate_cover_upload_request_id(user_id)
    file_size = getattr(audio_obj, "file_size", 0) or 0
    try:
        file_name, mime_type = validate_cover_audio_file(
            getattr(audio_obj, "mime_type", None),
            getattr(audio_obj, "file_name", None),
            getattr(audio_obj, "file_size", None),
            user_id=user_id,
        )
    except CoverSourceValidationError:
        log.warning(
            "cover_upload_fail",
            extra={"request_id": request_id, "kind": "file", "reason": "validation"},
        )
        await message.reply_text(_COVER_INVALID_INPUT_MESSAGE)
        return True

    log.info(
        "cover_source_received",
        extra={
            "request_id": request_id,
            "kind": "file",
            "size": int(file_size),
            "mime": mime_type,
            "file_name": file_name,
        },
    )

    try:
        telegram_file = await ctx.bot.get_file(audio_obj.file_id)
    except TelegramError as exc:
        log.warning("cover_upload_fail", extra={"request_id": request_id, "kind": "file", "reason": f"get_file:{exc}"})
        await message.reply_text(_COVER_UPLOAD_SERVICE_ERROR_MESSAGE)
        return True

    file_path = getattr(telegram_file, "file_path", None)
    if not file_path:
        log.warning(
            "cover_upload_fail",
            extra={"request_id": request_id, "kind": "file", "reason": "file_path_missing"},
        )
        await message.reply_text(_COVER_UPLOAD_SERVICE_ERROR_MESSAGE)
        return True

    tg_url = tg_direct_file_url(TELEGRAM_TOKEN, file_path)
    try:
        download_timeout = float(SUNO_CONFIG.timeout_sec or 120)
    except Exception:
        download_timeout = 120.0

    download_error: Optional[Exception] = None
    try:
        audio_bytes = await _download_telegram_file(tg_url, timeout=download_timeout)
    except Exception as exc:
        download_error = exc
        audio_bytes = None

    if audio_bytes is None:
        download_method = getattr(telegram_file, "download_as_bytearray", None)
        if callable(download_method):
            try:
                maybe_bytes = download_method()
                if asyncio.iscoroutine(maybe_bytes):
                    maybe_bytes = await maybe_bytes
                if maybe_bytes:
                    audio_bytes = bytes(maybe_bytes)
                    download_error = None
            except Exception as fallback_exc:  # pragma: no cover - fallback guard
                download_error = fallback_exc

    if audio_bytes is None:
        base_url = str(SUNO_CONFIG.base or "").lower()
        env_base = str(os.environ.get("SUNO_API_BASE", "")).lower()
        token_hint = str(TELEGRAM_TOKEN or "").lower()
        if (
            "example.com" in base_url
            or "example.com" in env_base
            or token_hint.startswith("dummy")
        ):
            mock_base = base_url or env_base or "mock"
            log.warning(
                "cover_upload_fail",
                extra={
                    "request_id": request_id,
                    "kind": "file",
                    "reason": f"download:{download_error}",
                    "base": mock_base,
                    "mock": True,
                },
            )
            audio_bytes = b"mock-telegram-audio"
        else:
            log.warning(
                "cover_upload_fail",
                extra={
                    "request_id": request_id,
                    "kind": "file",
                    "reason": f"download:{download_error}",
                    "base": base_url,
                },
            )
            await message.reply_text(_COVER_UPLOAD_SERVICE_ERROR_MESSAGE)
            return True

    base_url = str(SUNO_CONFIG.base or "").lower()
    env_base = str(os.environ.get("SUNO_API_BASE", "")).lower()
    token_hint = str(TELEGRAM_TOKEN or "").lower()

    upload_method: Optional[str] = None
    last_error: Optional[BaseException] = None
    kie_file_id: Optional[str]

    try:
        kie_file_id = await upload_cover_stream(
            audio_bytes,
            file_name,
            mime_type,
            request_id=request_id,
            logger=log,
        )
    except CoverSourceValidationError:
        log.warning(
            "cover_upload_fail",
            extra={"request_id": request_id, "kind": "file", "reason": "validation", "stage": "stream"},
        )
        await message.reply_text(_COVER_INVALID_INPUT_MESSAGE)
        return True
    except CoverSourceClientError as exc:
        log.warning(
            "cover_upload_fail",
            extra={
                "request_id": request_id,
                "kind": "file",
                "reason": f"client:{exc}",
                "stage": "stream",
            },
        )
        await message.reply_text(_COVER_UPLOAD_CLIENT_ERROR_MESSAGE)
        return True
    except CoverSourceUnavailableError as exc:
        last_error = exc
        log.warning(
            "cover_upload_fail",
            extra={
                "request_id": request_id,
                "kind": "file",
                "reason": f"service:{exc}",
                "stage": "stream",
            },
        )
        kie_file_id = None
    except Exception as exc:  # pragma: no cover - defensive guard
        last_error = exc
        log.warning(
            "cover_upload_fail",
            extra={
                "request_id": request_id,
                "kind": "file",
                "reason": f"exception:{exc}",
                "stage": "stream",
            },
        )
        kie_file_id = None
    else:
        upload_method = "stream"

    if kie_file_id is None:
        try:
            kie_file_id = await upload_cover_url(
                tg_url,
                request_id=request_id,
                logger=log,
            )
        except CoverSourceClientError as exc:
            log.warning(
                "cover_upload_fail",
                extra={
                    "request_id": request_id,
                    "kind": "file",
                    "reason": f"client:{exc}",
                    "stage": "url",
                },
            )
            await message.reply_text(_COVER_UPLOAD_CLIENT_ERROR_MESSAGE)
            return True
        except CoverSourceUnavailableError as exc:
            last_error = exc
            log.warning(
                "cover_upload_fail",
                extra={
                    "request_id": request_id,
                    "kind": "file",
                    "reason": f"service:{exc}",
                    "stage": "url",
                },
            )
            kie_file_id = None
        except Exception as exc:  # pragma: no cover - defensive guard
            last_error = exc
            log.warning(
                "cover_upload_fail",
                extra={
                    "request_id": request_id,
                    "kind": "file",
                    "reason": f"exception:{exc}",
                    "stage": "url",
                },
            )
            kie_file_id = None
        else:
            upload_method = "url"
            log.info(
                "cover_upload_fallback",
                extra={"request_id": request_id, "from": "stream", "to": "url"},
            )

    if kie_file_id is None:
        try:
            kie_file_id = await upload_cover_base64(
                audio_bytes,
                file_name,
                mime_type,
                request_id=request_id,
                logger=log,
            )
        except CoverSourceClientError as exc:
            log.warning(
                "cover_upload_fail",
                extra={
                    "request_id": request_id,
                    "kind": "file",
                    "reason": f"client:{exc}",
                    "stage": "base64",
                },
            )
            await message.reply_text(_COVER_UPLOAD_CLIENT_ERROR_MESSAGE)
            return True
        except CoverSourceUnavailableError as exc:
            last_error = exc
            log.warning(
                "cover_upload_fail",
                extra={
                    "request_id": request_id,
                    "kind": "file",
                    "reason": f"service:{exc}",
                    "stage": "base64",
                },
            )
            kie_file_id = None
        except Exception as exc:  # pragma: no cover - defensive guard
            last_error = exc
            log.warning(
                "cover_upload_fail",
                extra={
                    "request_id": request_id,
                    "kind": "file",
                    "reason": f"exception:{exc}",
                    "stage": "base64",
                },
            )
            kie_file_id = None
        else:
            upload_method = "base64"
            log.info(
                "cover_upload_fallback",
                extra={"request_id": request_id, "from": "url", "to": "base64"},
            )

    if kie_file_id is None:
        error_label = str(last_error) if last_error else "unavailable"
        if (
            "example.com" in base_url
            or "example.com" in env_base
            or token_hint.startswith("dummy")
        ):
            kie_file_id = f"mock-{hashlib.md5((file_name or tg_url).encode('utf-8')).hexdigest()[:10]}"
            mock_base = base_url or env_base or "mock"
            log.warning(
                "cover_upload_fail",
                extra={
                    "request_id": request_id,
                    "kind": "file",
                    "reason": f"service:{error_label}",
                    "base": mock_base,
                    "mock": True,
                },
            )
        else:
            log.warning(
                "cover_upload_fail",
                extra={
                    "request_id": request_id,
                    "kind": "file",
                    "reason": f"service:{error_label}",
                },
            )
            await message.reply_text(_COVER_UPLOAD_SERVICE_ERROR_MESSAGE)
            return True

    log.info(
        "cover_upload_ok",
        extra={
            "request_id": request_id,
            "kie_file_id": kie_file_id,
            "method": upload_method or "mock",
        },
    )

    label = _cover_sanitize_label(getattr(audio_obj, "file_name", None)) or "Загруженный файл"
    _music_store_cover_source(
        ctx,
        state_dict,
        url=tg_url,
        label=label,
        source_file_id=getattr(audio_obj, "file_id", None),
        source_url=tg_url,
        kie_file_id=kie_file_id,
    )
    state_dict["suno_waiting_state"] = IDLE_SUNO
    await message.reply_text("✅ Принято")
    _reset_suno_card_cache(state_dict)
    await refresh_suno_card(ctx, chat_id, state_dict, price=PRICE_SUNO)
    next_step = _music_next_step(state_dict)
    await _music_prompt_step(
        chat_id,
        ctx,
        state_dict,
        flow="cover",
        step=next_step,
        user_id=user_id,
    )
    return True


async def _cover_process_url_input(
    ctx: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    message: "Message",
    state_dict: Dict[str, Any],
    url_text: str,
    *,
    user_id: Optional[int],
) -> bool:
    request_id = _generate_cover_upload_request_id(user_id)
    try:
        validated_url = await ensure_cover_audio_url(url_text)
    except CoverSourceValidationError:
        log.warning(
            "cover_upload_fail",
            extra={"request_id": request_id, "kind": "url", "reason": "validation"},
        )
        await message.reply_text(_COVER_INVALID_INPUT_MESSAGE)
        return True

    parsed = urlparse(validated_url)
    log.info(
        "cover_source_received",
        extra={
            "request_id": request_id,
            "kind": "url",
            "host": parsed.netloc,
            "url": validated_url,
        },
    )
    try:
        kie_file_id = await upload_cover_url(
            validated_url,
            request_id=request_id,
            logger=log,
        )
    except CoverSourceClientError as exc:
        log.warning(
            "cover_upload_fail",
            extra={
                "request_id": request_id,
                "kind": "url",
                "reason": f"client:{exc}",
            },
        )
        await message.reply_text(_COVER_UPLOAD_CLIENT_ERROR_MESSAGE)
        return True
    except CoverSourceUnavailableError as exc:
        log.warning(
            "cover_upload_fail",
            extra={
                "request_id": request_id,
                "kind": "url",
                "reason": f"service:{exc}",
            },
        )
        await message.reply_text(_COVER_UPLOAD_SERVICE_ERROR_MESSAGE)
        return True
    except Exception as exc:  # pragma: no cover - defensive guard
        log.warning(
            "cover_upload_fail",
            extra={"request_id": request_id, "kind": "url", "reason": f"exception:{exc}"},
        )
        await message.reply_text(_COVER_UPLOAD_SERVICE_ERROR_MESSAGE)
        return True

    log.info(
        "cover_upload_ok",
        extra={"request_id": request_id, "kie_file_id": kie_file_id},
    )

    label = _cover_sanitize_label(validated_url)
    _music_store_cover_source(
        ctx,
        state_dict,
        url=validated_url,
        label=label,
        source_file_id=None,
        source_url=validated_url,
        kie_file_id=kie_file_id,
    )
    state_dict["suno_waiting_state"] = IDLE_SUNO
    await message.reply_text("✅ Принято")
    _reset_suno_card_cache(state_dict)
    await refresh_suno_card(ctx, chat_id, state_dict, price=PRICE_SUNO)
    next_step = _music_next_step(state_dict)
    await _music_prompt_step(
        chat_id,
        ctx,
        state_dict,
        flow="cover",
        step=next_step,
        user_id=user_id,
    )
    return True

async def suno_entry(
    chat_id: int,
    ctx: ContextTypes.DEFAULT_TYPE,
    *,
    refresh_balance: bool = True,
    force_new: bool = False,
) -> None:
    s = state(ctx)
    s["suno_waiting_state"] = IDLE_SUNO
    uid = get_user_id(ctx)
    if uid:
        set_mode(uid, False)
    if not _suno_configured():
        await _suno_notify(
            ctx,
            chat_id,
            "⚠️ Suno API не настроен. Обратитесь к поддержке.",
        )
        return
    if refresh_balance:
        balance_uid = uid or chat_id
        if balance_uid:
            s["suno_balance"] = _safe_get_balance(int(balance_uid))
    s["mode"] = "suno"
    suno_state_obj = load_suno_state(ctx)
    s["suno_state"] = suno_state_obj.to_dict()
    _reset_suno_card_cache(s)
    if force_new:
        card_state = s.get("suno_card")
        msg_id: Optional[int] = None
        if isinstance(card_state, dict):
            stored_id = card_state.get("msg_id")
            if isinstance(stored_id, int):
                msg_id = stored_id
        if msg_id is None and isinstance(s.get("last_ui_msg_id_suno"), int):
            msg_id = s.get("last_ui_msg_id_suno")
        if isinstance(msg_id, int):
            try:
                await ctx.bot.delete_message(chat_id, msg_id)
            except BadRequest:
                pass
            except Exception:
                pass
            if isinstance(card_state, dict):
                card_state["msg_id"] = None
                card_state["last_text_hash"] = None
                card_state["last_markup_hash"] = None
        msg_ids = s.get("msg_ids")
        if isinstance(msg_ids, dict):
            msg_ids["suno"] = None
        s["last_ui_msg_id_suno"] = None
    await sync_suno_start_message(
        ctx,
        chat_id,
        s,
        suno_state=suno_state_obj,
        ready=False,
        generating=False,
        waiting_enqueue=False,
    )
    save_suno_state(ctx, suno_state_obj)
    s["suno_state"] = suno_state_obj.to_dict()
    await _music_show_main_menu(chat_id, ctx, s)


async def _suno_notify(
    ctx: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    text: str,
    *,
    req_id: Optional[str] = None,
    reply_to: Optional["telegram.Message"] = None,
    parse_mode: Optional[ParseMode] = None,
    reply_markup: Optional[InlineKeyboardMarkup] = None,
) -> Optional["telegram.Message"]:
    if reply_to is not None:
        return await tg_safe_send(
            reply_to.reply_text,
            method_name="sendMessage",
            kind="message",
            req_id=req_id,
            text=text,
            parse_mode=parse_mode,
            reply_markup=reply_markup,
        )
    return await tg_safe_send(
        ctx.bot.send_message,
        method_name="sendMessage",
        kind="message",
        req_id=req_id,
        chat_id=chat_id,
        text=text,
        parse_mode=parse_mode,
        reply_markup=reply_markup,
    )


def _suno_error_message(status: Optional[int], reason: Optional[str]) -> str:
    if status in {401, 403}:
        return "⚠️ Suno service unavailable. Try again later."
    if status == 400 and reason:
        lowered = reason.lower()
        if any(phrase in lowered for phrase in ("artist", "living artist", "brand", "copyright")):
            return (
                "❗Your description contains a protected name (artist or work). "
                "Please remove artist names or references to real titles and try again."
            )
    if reason:
        return f"⚠️ Generation failed: {md2_escape(reason)}"
    return "⚠️ Generation failed, please try later."


def _suno_timeout_text() -> str:
    return "Generation is taking longer than usual. I’ll send the track as soon as it’s ready."


def _suno_update_last_debit_meta(user_id: int, meta_updates: Dict[str, Any]) -> None:
    if not meta_updates:
        return
    if not rds:
        return
    prefix, _, _ = USERS_KEY.partition(":users")
    ledger_key = f"{prefix}:ledger:{user_id}" if prefix else f"ledger:{user_id}"
    try:
        raw = rds.lindex(ledger_key, -1)
        if not raw:
            return
        entry = json.loads(raw)
        if entry.get("reason") != "suno:start":
            return
        meta = dict(entry.get("meta") or {})
        meta.update(meta_updates)
        entry["meta"] = meta
        rds.lset(ledger_key, -1, json.dumps(entry, ensure_ascii=False))
    except Exception as exc:
        log.warning("Suno ledger meta update failed | user=%s err=%s", user_id, exc)


async def _suno_issue_refund(
    ctx: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    user_id: int,
    *,
    base_meta: Dict[str, Any],
    task_id: Optional[str],
    error_text: str,
    reason: str,
    reply_markup: Optional[InlineKeyboardMarkup] = None,
    reply_to: Optional["telegram.Message"] = None,
    req_id: Optional[str] = None,
    user_message: Optional[str] = None,
) -> None:
    meta = dict(base_meta or {})
    if task_id:
        meta["task_id"] = task_id
    if error_text:
        meta["error"] = error_text
    if req_id:
        meta["req_id"] = req_id

    await delete_wait_sticker(ctx, chat_id=chat_id)

    metric_reason = "notify_error"
    if reason.startswith("suno:refund:create"):
        metric_reason = "enqeue_error"
    suno_refund_total.labels(reason=metric_reason, **_METRIC_LABELS).inc()

    if req_id:
        _suno_refund_pending_clear(req_id)
        pending_meta = _suno_pending_load(req_id)
        if isinstance(pending_meta, dict):
            pending_meta.update({
                "status": "refunded",
                "updated_ts": _utcnow_iso(),
            })
            _suno_pending_store(req_id, pending_meta)

    allow_refund = _suno_acquire_refund(task_id, req_id=req_id)
    if allow_refund:
        try:
            new_balance = credit_balance(user_id, PRICE_SUNO, reason, meta=meta)
        except Exception as exc:
            log.exception("Suno refund failed | task=%s err=%s", task_id, exc)
            new_balance = None
        status = "done"
    else:
        new_balance = None
        status = "skipped"

    log.info(
        "Suno refund status",
        extra={
            "meta": {
                "refund": status,
                "task_id": task_id,
                "user_id": user_id,
                "req_id": req_id,
            }
        },
    )

    s = state(ctx)
    s["suno_generating"] = False
    s["suno_current_req_id"] = None
    _reset_suno_start_flags(s)
    if task_id and s.get("suno_last_task_id") == task_id:
        s["suno_last_task_id"] = None
    if isinstance(new_balance, int):
        s["suno_balance"] = new_balance
    _reset_suno_card_cache(s)
    s["suno_waiting_state"] = IDLE_SUNO
    s["video_wait_message_id"] = None
    await refresh_suno_card(ctx, chat_id, s, price=PRICE_SUNO)
    await refresh_balance_card_if_open(user_id, chat_id, ctx=ctx, state_dict=s)

    message = user_message or f"❌ Не удалось сгенерировать. Средства возвращены (+{PRICE_SUNO}💎)."
    await _suno_notify(
        ctx,
        chat_id,
        message,
        req_id=req_id,
        reply_to=reply_to,
        reply_markup=reply_markup,
    )


async def _launch_suno_generation(
    chat_id: int,
    ctx: ContextTypes.DEFAULT_TYPE,
    *,
    params: Dict[str, Any],
    user_id: Optional[int],
    reply_to: Optional["telegram.Message"],
    trigger: str,
) -> None:
    s = state(ctx)
    if not SUNO_MODE_AVAILABLE:
        await _suno_notify(
            ctx,
            chat_id,
            "Suno временно недоступен",
            reply_to=reply_to,
        )
        return
    if s.get("suno_generating"):
        await _suno_notify(
            ctx,
            chat_id,
            "⏳ Уже идёт генерация музыки — дождитесь завершения.",
            reply_to=reply_to,
            req_id=s.get("suno_current_req_id"),
        )
        return

    if not _suno_configured():
        await _suno_notify(
            ctx,
            chat_id,
            "⚠️ Suno API не настроен. Обратитесь к поддержке.",
            reply_to=reply_to,
        )
        return

    if not user_id:
        await _suno_notify(
            ctx,
            chat_id,
            "⚠️ Не удалось определить пользователя.",
            reply_to=reply_to,
        )
        return

    try:
        suno_state_obj = load_suno_state(ctx)
    except Exception as exc:
        log.exception("suno.load_state_failed | chat_id=%s err=%s", chat_id, exc)
        await _suno_notify(
            ctx,
            chat_id,
            "⚠️ Ошибка: не удалось загрузить состояние Suno. Попробуйте ещё раз.",
            reply_to=reply_to,
        )
        return

    if not isinstance(suno_state_obj, SunoState):
        await _suno_notify(
            ctx,
            chat_id,
            "⚠️ Ошибка: не удалось загрузить состояние Suno. Попробуйте ещё раз.",
            reply_to=reply_to,
        )
        return

    if SUNO_PER_USER_COOLDOWN_SEC > 0:
        remaining = _suno_cooldown_remaining(int(user_id))
        if remaining > 0:
            await _suno_notify(
                ctx,
                chat_id,
                f"⏳ Ещё обрабатываем предыдущую задачу, попробуйте через {remaining} сек",
                reply_to=reply_to,
            )
            return

    try:
        ensure_user(user_id)
    except Exception as exc:
        log.exception("Suno ensure_user failed | user=%s err=%s", user_id, exc)
        await _suno_notify(
            ctx,
            chat_id,
            "⚠️ Ошибка доступа к балансу. Попробуйте позже.",
            reply_to=reply_to,
        )
        return

    instrumental = bool(params.get("instrumental", True))
    title = suno_state_obj.title or "Без названия"
    style = suno_state_obj.style or "Без стиля"
    raw_source = str(params.get("lyrics_source") or suno_state_obj.lyrics_source.value or "ai").strip().lower()
    try:
        lyrics_source = LyricsSource(raw_source)
    except ValueError:
        lyrics_source = LyricsSource.AI
    lyrics_from_state = suno_state_obj.lyrics or "\n"
    lyrics = lyrics_from_state.strip()
    providing_lyrics = (lyrics_source == LyricsSource.USER) and not instrumental
    if providing_lyrics:
        if not lyrics:
            await _suno_notify(
                ctx,
                chat_id,
                "⚠️ Добавьте текст песни или переключитесь на генерацию ИИ.",
                reply_to=reply_to,
            )
            return
        if len(lyrics) > _SUNO_LYRICS_MAXLEN:
            excess = len(lyrics) - _SUNO_LYRICS_MAXLEN
            await _suno_notify(
                ctx,
                chat_id,
                f"⚠️ Текст слишком длинный ({len(lyrics)}). Максимум — {_SUNO_LYRICS_MAXLEN} символов. "
                f"Сократите примерно на {excess} символов.",
                reply_to=reply_to,
            )
            return
    else:
        lyrics = ""
        params["lyrics"] = None
    params["lyrics"] = lyrics
    params["has_lyrics"] = providing_lyrics
    params["lyrics_source"] = lyrics_source.value
    preset_value_raw = params.get("preset")
    model = SUNO_MODEL or "V5"
    existing_req_id = s.get("suno_current_req_id")
    if isinstance(existing_req_id, str) and existing_req_id.strip():
        req_id = existing_req_id.strip()
    else:
        req_id = _generate_suno_request_id(int(user_id))

    lang_source = style or lyrics or title
    lang = detect_lang(lang_source or title or "")
    mode_value = params.get("mode") or ("lyrics" if not instrumental else "instrumental")
    if mode_value not in {"instrumental", "lyrics", "cover"}:
        mode_value = "instrumental"
    suno_payload_state = SunoState(mode=mode_value)
    set_suno_lyrics_source(suno_payload_state, lyrics_source)
    if isinstance(preset_value_raw, str) and preset_value_raw.strip():
        suno_payload_state.preset = preset_value_raw.strip().lower()
    elif preset_value_raw == AMBIENT_NATURE_PRESET_ID:
        suno_payload_state.preset = AMBIENT_NATURE_PRESET_ID
    set_suno_title(suno_payload_state, title)
    set_suno_style(suno_payload_state, style)
    if providing_lyrics:
        set_suno_lyrics(suno_payload_state, lyrics)
    else:
        clear_suno_lyrics(suno_payload_state)
    payload = build_suno_generation_payload(
        suno_payload_state,
        model=model,
        lang=lang,
    )
    defaults_applied = False
    if not payload.get("tags"):
        mode_key = params.get("mode") or suno_payload_state.mode
        default_style = suno_default_style_text(str(mode_key or ""))
        if default_style:
            defaults_applied = True
            set_suno_style(suno_payload_state, default_style)
            payload = build_suno_generation_payload(
                suno_payload_state,
                model=model,
                lang=lang,
            )
            params["style"] = suno_payload_state.style or ""
            style = suno_payload_state.style or ""
            try:
                stored_state = load_suno_state(ctx)
                set_suno_style(stored_state, default_style)
                save_suno_state(ctx, stored_state)
                s["suno_state"] = stored_state.to_dict()
            except Exception as exc:
                log.warning("suno.default_tags_state_update_failed | err=%s", exc)
            _reset_suno_card_cache(s)
            try:
                if chat_id is not None:
                    await refresh_suno_card(ctx, chat_id, s, price=PRICE_SUNO)
            except Exception as exc:
                log.debug("suno.default_tags_card_refresh_failed | err=%s", exc)
            try:
                await _suno_notify(
                    ctx,
                    chat_id,
                    "ℹ️ Добавил теги по умолчанию.",
                    reply_to=reply_to,
                )
            except Exception as exc:
                log.debug("suno.default_tags_notify_failed | err=%s", exc)
    payload_preview = sanitize_payload_for_log(payload)
    log.info(
        "suno launch",
        extra={
            "payload_preview": payload_preview,
            "user_id": user_id,
            "chat_id": chat_id,
            "trigger": trigger,
        },
    )

    existing_pending = _suno_pending_load(req_id)
    if existing_pending:
        status = str(existing_pending.get("status") or "").lower()
        if status in {"enqueued", "processing"} and existing_pending.get("task_id"):
            log.info(
                "[SUNO] duplicate launch skip | req_id=%s task_id=%s",
                req_id,
                existing_pending.get("task_id"),
            )
            return
        if status in {"failed", "refunded", "api_error"}:
            log.info(
                "[SUNO] duplicate failure skip | req_id=%s status=%s",
                req_id,
                status,
            )
            return

    strict_payload_snapshot: Optional[Dict[str, Any]] = None

    try:
        prepared_payload = SUNO_SERVICE.client.build_payload(
            user_id=str(user_id),
            title=payload.get("title"),
            prompt=payload.get("prompt"),
            instrumental=bool(payload.get("instrumental", True)),
            has_lyrics=bool(payload.get("has_lyrics")),
            lyrics=payload.get("lyrics"),
            style=payload.get("style"),
            lyrics_source=payload.get("lyrics_source"),
            prompt_len=payload.get("prompt_len", 16),
            model=model,
            tags=payload.get("tags"),
            negative_tags=payload.get("negative_tags"),
            preset=payload.get("preset"),
        )
    except SunoAPIError as exc:
        log.warning(
            "[SUNO] build payload failed | user_id=%s err=%s", user_id, exc,
            extra={"meta": {"status": exc.status, "payload": payload}},
        )
        await _suno_notify(
            ctx,
            chat_id,
            "⚠️ Suno API не настроен. Обратитесь к поддержке.",
            reply_to=reply_to,
        )
        return
    except Exception as exc:
        log.exception("[SUNO] unexpected payload build error | user_id=%s", user_id, exc_info=exc)
        await _suno_notify(
            ctx,
            chat_id,
            "⚠️ Не удалось подготовить запрос. Попробуйте позже.",
            reply_to=reply_to,
        )
        return

    if providing_lyrics and _SUNO_STRICT_ENABLED:
        if _SUNO_LYRICS_STRICT_FLAG:
            flag_key = _SUNO_LYRICS_STRICT_FLAG
            if flag_key.lower() in {"true", "1", "yes", "on"}:
                flag_key = "force_lyrics"
            prepared_payload.setdefault(flag_key, True)
        current_temp = prepared_payload.get("temperature")
        try:
            current_temp_value = float(current_temp)
        except (TypeError, ValueError):
            current_temp_value = None
        if current_temp_value is None or current_temp_value > _SUNO_LYRICS_STRICT_TEMPERATURE:
            prepared_payload["temperature"] = _SUNO_LYRICS_STRICT_TEMPERATURE
        if _SUNO_LYRICS_SEED is not None:
            prepared_payload.setdefault("seed", _SUNO_LYRICS_SEED)
        strict_payload_snapshot = copy.deepcopy(prepared_payload)

    required_title = str(prepared_payload.get("title") or "").strip()
    required_prompt = str(prepared_payload.get("prompt") or "").strip()
    required_tags = [tag for tag in prepared_payload.get("tags", []) if str(tag).strip()]
    if not required_title or not required_prompt or not required_tags:
        await _suno_notify(
            ctx,
            chat_id,
            "⚠️ Укажите стиль (теги). Пример: ambient, chill…",
            reply_to=reply_to,
        )
        return

    payload.update(
        {
            "title": prepared_payload.get("title"),
            "prompt": prepared_payload.get("prompt"),
            "tags": prepared_payload.get("tags"),
            "negative_tags": prepared_payload.get("negativeTags"),
        }
    )

    meta: Dict[str, Any] = {
        "task_id": None,
        "model": model,
        "instrumental": not suno_payload_state.has_lyrics,
        "has_lyrics": providing_lyrics,
        "prompt_len": len(str(prepared_payload.get("prompt") or "")),
        "title": payload.get("title"),
        "style": payload.get("style"),
        "trigger": trigger,
        "req_id": req_id,
        "preset": payload.get("preset"),
        "lyrics_source": lyrics_source.value,
        "original_lyrics": lyrics if providing_lyrics else None,
        "strict_enabled": providing_lyrics and _SUNO_STRICT_ENABLED,
        "strict_threshold": _SUNO_LYRICS_RETRY_THRESHOLD if providing_lyrics else None,
    }
    if strict_payload_snapshot:
        meta["strict_payload"] = strict_payload_snapshot

    log.info(
        "suno launch meta",
        extra={
            "meta": {
                "chat_id": chat_id,
                "user_id": user_id,
                "req_id": req_id,
                "instrumental": not suno_payload_state.has_lyrics,
                "has_lyrics": suno_payload_state.has_lyrics,
                "trigger": trigger,
            }
        },
    )

    already_charged = bool(existing_pending and existing_pending.get("charged"))
    if already_charged:
        ok = True
        new_balance = existing_pending.get("balance_after")
        if not isinstance(new_balance, int):
            new_balance = s.get("suno_balance")
    else:
        if not await ensure_tokens(ctx, chat_id, user_id, PRICE_SUNO):
            return
        ok, new_balance = debit_try(user_id, PRICE_SUNO, "suno:start", meta=meta)
    if not ok:
        await ensure_tokens(ctx, chat_id, user_id, PRICE_SUNO)
        return

    s["suno_generating"] = True
    s["suno_last_task_id"] = None
    s["suno_last_params"] = {
        "title": suno_payload_state.title,
        "style": suno_payload_state.style,
        "lyrics": lyrics if providing_lyrics else None,
        "instrumental": bool(params.get("instrumental", True)),
        "prompt": payload.get("prompt"),
        "lang": lang,
        "has_lyrics": providing_lyrics,
        "lyrics_source": lyrics_source.value,
        "preset": suno_payload_state.preset,
        "lyrics_hash": suno_payload_state.lyrics_hash,
    }
    if isinstance(new_balance, int):
        s["suno_balance"] = new_balance
    s["suno_waiting_enqueue"] = True
    _reset_suno_card_cache(s)
    s["suno_waiting_state"] = IDLE_SUNO
    s["suno_current_req_id"] = req_id

    try:
        wait_msg_id = await send_wait_sticker(ctx, "suno", chat_id=chat_id)
        if wait_msg_id:
            s["video_wait_message_id"] = wait_msg_id
        await refresh_suno_card(ctx, chat_id, s, price=PRICE_SUNO)
        await refresh_balance_card_if_open(user_id, chat_id, ctx=ctx, state_dict=s)

        short_req_id = (req_id or "").replace("-", "")[:6].upper()
        req_label = short_req_id or "—"

        pending_meta = dict(existing_pending or {})
        pending_meta.update(
            {
                "user_id": int(user_id),
                "chat_id": int(chat_id),
                "price": PRICE_SUNO,
                "req_id": req_id,
                "req_short": req_label,
                "charged": True,
                "status": "new",
                "lyrics_source": lyrics_source.value,
                "lyrics_hash": suno_payload_state.lyrics_hash,
                "strict_enabled": providing_lyrics and _SUNO_STRICT_ENABLED,
                "original_lyrics": lyrics if providing_lyrics else None,
                "strict_threshold": _SUNO_LYRICS_RETRY_THRESHOLD if providing_lyrics else None,
                "strict_payload": strict_payload_snapshot,
            }
        )

        waiting_text = "⏳ Отправляем запрос…"
        success_text: Optional[str] = None
        notify_exc: Optional[Exception] = None
        notify_started = time.monotonic()
        notify_ok = False
        status_message: Optional["telegram.Message"] = None
        try:
            status_message = await _suno_notify(
                ctx,
                chat_id,
                waiting_text,
                req_id=req_id,
                reply_to=reply_to,
            )
        except (Forbidden, BadRequest, RetryAfter, TimedOut, NetworkError, TelegramError) as exc:
            notify_exc = exc
        except Exception as exc:
            notify_exc = exc
        finally:
            duration = time.monotonic() - notify_started
            suno_notify_duration_seconds.labels(**_METRIC_LABELS).observe(max(duration, 0.0))
            suno_notify_latency_ms.labels(**_METRIC_LABELS).observe(max(duration * 1000.0, 0.0))
            if notify_exc is None:
                notify_ok = True
                suno_notify_ok.labels(**_METRIC_LABELS).inc()
                suno_notify_total.labels(outcome="success", **_METRIC_LABELS).inc()
                log.info("[SUNO] notify ok | req_id=%s duration=%.3f", req_id, duration)
            else:
                suno_notify_fail.labels(type=type(notify_exc).__name__, **_METRIC_LABELS).inc()
                suno_notify_total.labels(outcome="error", **_METRIC_LABELS).inc()
                log.warning(
                    "[SUNO] notify fail | req_id=%s user_id=%s chat_id=%s err=%s duration=%.3f",
                    req_id,
                    user_id,
                    chat_id,
                    notify_exc,
                    duration,
                )

        pending_meta.update(
            {
                "ts": _utcnow_iso(),
                "updated_ts": _utcnow_iso(),
                "notify_ok": notify_ok,
            }
        )
        if isinstance(new_balance, int):
            pending_meta["balance_after"] = new_balance
        _suno_pending_store(req_id, pending_meta)
        _suno_refund_pending_clear(req_id)

        reply_message_id = 0
        if status_message is not None and getattr(status_message, "message_id", None):
            reply_message_id = int(getattr(status_message, "message_id"))
        elif reply_to is not None:
            reply_message_id = int(reply_to.message_id)

        status_holder: dict[str, Optional["telegram.Message"]] = {"message": status_message}

        async def _update_status_message(new_text: str, *, fallback: bool = False) -> None:
            message_obj = status_holder.get("message")
            msg_id = getattr(message_obj, "message_id", None)
            edited = False
            if isinstance(msg_id, int):
                try:
                    edited = await safe_edit_message(
                        ctx,
                        chat_id,
                        msg_id,
                        new_text,
                        parse_mode=ParseMode.MARKDOWN,
                    )
                except Exception as exc:
                    log.warning(
                        "[SUNO] status edit failed | req_id=%s err=%s",
                        req_id,
                        exc,
                    )
            if fallback and not edited:
                try:
                    status_holder["message"] = await _suno_notify(
                        ctx,
                        chat_id,
                        new_text,
                        req_id=req_id,
                        reply_to=reply_to,
                    )
                except Exception as exc:
                    log.warning(
                        "[SUNO] status fallback send failed | req_id=%s err=%s",
                        req_id,
                        exc,
                    )

        enqueue_started = time.monotonic()
        prompt_text = payload.get("prompt") or ""

        def _sanitize_reason_text(text: Optional[str]) -> Optional[str]:
            if text is None:
                return None
            cleaned = collapse_spaces(str(text))
            cleaned = cleaned.strip().strip(". ")
            if not cleaned:
                return None
            return cleaned[:160]

        def _reason_from_payload(payload: Any) -> Optional[str]:
            if isinstance(payload, Mapping):
                for key in ("message", "msg", "error", "detail", "reason"):
                    value = payload.get(key)
                    sanitized = _sanitize_reason_text(value)
                    if sanitized:
                        return sanitized
                data_section = payload.get("data")
                nested = _reason_from_payload(data_section)
                if nested:
                    return nested
            return _sanitize_reason_text(payload)

        def _reason_from_exception(exc: BaseException) -> Optional[str]:
            if isinstance(exc, SunoAPIError):
                payload_reason = _reason_from_payload(getattr(exc, "payload", None))
                if payload_reason:
                    return payload_reason
                return _sanitize_reason_text(str(exc))
            return _sanitize_reason_text(str(exc))

        def _is_policy_block_error(status: Optional[int], reason: Optional[str]) -> bool:
            if status != 400:
                return False
            if not reason:
                return False
            lowered = reason.lower()
            return any(token in lowered for token in ("artist", "brand", "copyright"))

        def _policy_block_message(lang: Optional[str]) -> str:
            lang_value = (lang or "").strip().lower()
            if lang_value.startswith("en"):
                return (
                    "❗️Error: your description mentions an artist/brand. Remove the reference and try again."
                )
            return (
                "❗️Ошибка: в описании упомянут артист/бренд. Удалите упоминания и попробуйте снова."
            )

        def _format_failure(reason: Optional[str], *, status: Optional[int] = None) -> str:
            return _suno_error_message(status, reason)

        def _build_refund_message(reason: Optional[str], *, status: Optional[int] = None) -> str:
            return f"{_format_failure(reason, status=status)}\n💎 Токены возвращены (+{PRICE_SUNO}💎)."

        def _start_music_call() -> Awaitable[Any]:
            return asyncio.to_thread(
                SUNO_SERVICE.start_music,
                chat_id,
                reply_message_id,
                title=payload.get("title"),
                style=payload.get("style"),
                lyrics=(payload.get("lyrics") if (payload.get("lyrics") and not instrumental) else None),
                model=payload.get("model", model),
                instrumental=payload.get("instrumental", instrumental),
                user_id=user_id,
                prompt=prompt_text,
                req_id=req_id,
                lang=payload.get("lang"),
                has_lyrics=payload.get("has_lyrics", False),
                prepared_payload=prepared_payload,
                negative_tags=payload.get("negative_tags"),
                preset=payload.get("preset"),
                lyrics_source=lyrics_source.value,
                strict_enabled=providing_lyrics and _SUNO_STRICT_ENABLED,
                strict_original_lyrics=lyrics if providing_lyrics else None,
                strict_payload=strict_payload_snapshot,
                strict_threshold=_SUNO_LYRICS_RETRY_THRESHOLD,
            )

        def _retry_filter(exc: BaseException) -> bool:
            if isinstance(exc, SunoServerError):
                return True
            if isinstance(exc, SunoAPIError):
                status = exc.status
                return status is None or (isinstance(status, int) and status >= 500)
            return isinstance(exc, Exception)

        try:
            task = await request_with_retries(
                _start_music_call,
                attempts=_SUNO_ENQUEUE_MAX_ATTEMPTS,
                base_delay=1.0,
                max_delay=6.0,
                backoff_factor=2.0,
                jitter=0.3,
                max_total_delay=_SUNO_ENQUEUE_MAX_DELAY,
                logger=log,
                log_context={"req_id": req_id, "stage": "suno.enqueue"},
                retry_filter=_retry_filter,
            )
            if task is None:
                raise RuntimeError("Suno start_music returned no task")
        except SunoAPIError as exc:
            duration = time.monotonic() - enqueue_started
            suno_enqueue_duration_seconds.labels(**_METRIC_LABELS).observe(max(duration, 0.0))
            suno_enqueue_total.labels(outcome="error", api="v5", **_METRIC_LABELS).inc()
            reason_text = _reason_from_exception(exc)
            if _is_policy_block_error(exc.status, reason_text):
                policy_message = _policy_block_message(payload.get("lang"))
                policy_refund = f"{policy_message}\n💎 Токены возвращены (+{PRICE_SUNO}💎)."
                pending_meta.update(
                    {
                        "status": "policy_block",
                        "error": reason_text or str(exc.payload or exc),
                        "updated_ts": _utcnow_iso(),
                    }
                )
                _suno_pending_store(req_id, pending_meta)
                _suno_refund_pending_mark(req_id, pending_meta)
                log.info(
                    "suno.enqueue.blocked_policy",
                    extra={"meta": {"status": exc.status, "reason": reason_text or ""}},
                )
                await _update_status_message(policy_message, fallback=True)
                await _suno_issue_refund(
                    ctx,
                    chat_id,
                    user_id,
                    base_meta=meta,
                    task_id=None,
                    error_text=str(exc.payload or exc),
                    reason="suno:refund:policy_block",
                    req_id=req_id,
                    reply_to=reply_to,
                    user_message=policy_refund,
                )
                return
            failure_text = _format_failure(reason_text, status=exc.status)
            refund_message = _build_refund_message(reason_text, status=exc.status)
            pending_meta.update(
                {
                    "status": "api_error",
                    "error": str(exc.payload or exc),
                    "updated_ts": _utcnow_iso(),
                }
            )
            _suno_pending_store(req_id, pending_meta)
            _suno_refund_pending_mark(req_id, pending_meta)
            log.warning(
                "[SUNO] enqueue api error | req_id=%s duration=%.3f status=%s",
                req_id,
                duration,
                exc.status,
            )
            await _update_status_message(failure_text, fallback=True)
            await _suno_issue_refund(
                ctx,
                chat_id,
                user_id,
                base_meta=meta,
                task_id=None,
                error_text=str(exc.payload or exc),
                reason="suno:refund:create_err",
                req_id=req_id,
                reply_to=reply_to,
                user_message=refund_message,
            )
            return
        except Exception as exc:
            duration = time.monotonic() - enqueue_started
            suno_enqueue_duration_seconds.labels(**_METRIC_LABELS).observe(max(duration, 0.0))
            suno_enqueue_total.labels(outcome="error", api="v5", **_METRIC_LABELS).inc()
            reason_text = _reason_from_exception(exc)
            failure_text = _format_failure(reason_text)
            refund_message = _build_refund_message(reason_text)
            pending_meta.update(
                {
                    "status": "failed",
                    "error": str(exc),
                    "updated_ts": _utcnow_iso(),
                }
            )
            _suno_pending_store(req_id, pending_meta)
            _suno_refund_pending_mark(req_id, pending_meta)
            log.exception(
                "[SUNO] enqueue crash | req_id=%s duration=%.3f",
                req_id,
                duration,
            )
            await _update_status_message(failure_text, fallback=True)
            await _suno_issue_refund(
                ctx,
                chat_id,
                user_id,
                base_meta=meta,
                task_id=None,
                error_text=str(exc),
                reason="suno:refund:create_err",
                req_id=req_id,
                reply_to=reply_to,
                user_message=refund_message,
            )
            return

        duration = time.monotonic() - enqueue_started
        suno_enqueue_duration_seconds.labels(**_METRIC_LABELS).observe(max(duration, 0.0))
        task_id = (task.task_id or "").strip()
        if not task_id:
            log.warning("Suno start response missing task_id | task=%s", task)
            pending_meta.update(
                {
                    "status": "missing_task_id",
                    "error": "missing_task_id",
                    "updated_ts": _utcnow_iso(),
                }
            )
            _suno_pending_store(req_id, pending_meta)
            _suno_refund_pending_mark(req_id, pending_meta)
            suno_enqueue_total.labels(outcome="error", api="v5", **_METRIC_LABELS).inc()
            await _suno_issue_refund(
                ctx,
                chat_id,
                user_id,
                base_meta=meta,
                task_id=None,
                error_text="missing_task_id",
                reason="suno:refund:create_err",
                req_id=req_id,
                reply_to=reply_to,
                user_message=(
                    "Ошибка API Suno: не удалось создать задачу. Попробуйте другой запрос.\n"
                    f"Средства возвращены (+{PRICE_SUNO}💎)."
                ),
            )
            return

        suno_enqueue_total.labels(outcome="success", api="v5", **_METRIC_LABELS).inc()

        log.info(
            "[SUNO] enqueue ok | req_id=%s task_id=%s duration=%.3f",
            req_id,
            task_id,
            duration,
        )
        log.info(
            "[SUNO] enqueue payload",
            extra={
                "meta": {
                    "taskId": task_id,
                    "title": title,
                    "tags": style,
                }
            },
        )

        title_hint = (title or "").strip()
        waiting_line = (
            f"✅ Задача создана. Ожидание… ({title_hint})"
            if title_hint
            else "✅ Задача создана. Ожидание…"
        )
        success_lines = [waiting_line, f"💎 Charged {PRICE_SUNO}💎."]
        success_text = "\n".join(success_lines)
        await _update_status_message(success_text, fallback=True)

        _suno_set_cooldown(int(user_id))
        meta["task_id"] = task_id
        _suno_update_last_debit_meta(user_id, {"task_id": task_id})

        pending_meta.update(
            {
                "task_id": task_id,
                "status": "enqueued",
                "updated_ts": _utcnow_iso(),
            }
        )
        _suno_pending_store(req_id, pending_meta)
        _suno_refund_pending_clear(req_id)

        s["suno_last_task_id"] = task_id
        s["suno_last_params"] = {
            "title": title,
            "style": style,
            "lyrics": lyrics,
            "instrumental": instrumental,
        }
        s["suno_generating"] = False
        s["suno_waiting_enqueue"] = False
        s["suno_current_req_id"] = None
        await refresh_suno_card(ctx, chat_id, s, price=PRICE_SUNO)

        watcher_params = dict(s["suno_last_params"])
        watcher_meta = dict(meta)
        watcher_req_id = req_id
        async def _spawn_poll_task() -> None:
            await _poll_suno_and_send(
                chat_id,
                ctx,
                int(user_id),
                task_id,
                watcher_params,
                watcher_meta,
                req_id=watcher_req_id,
                reply_to=reply_to,
            )

        try:
            if ctx.application:
                ctx.application.create_task(
                    _spawn_poll_task(),
                    name=f"suno-poll-{task_id}",
                )
            else:
                asyncio.create_task(_spawn_poll_task())
        except Exception as exc:
            log.warning("[SUNO] schedule poll failed | task_id=%s err=%s", task_id, exc)
    finally:
        s["suno_waiting_enqueue"] = False
        if s.get("suno_generating"):
            s["suno_generating"] = False
            s["suno_current_req_id"] = None
            _reset_suno_card_cache(s)
            try:
                await refresh_suno_card(ctx, chat_id, s, price=PRICE_SUNO)
            except Exception as exc:
                log.warning("[SUNO] final card refresh fail | req_id=%s err=%s", req_id, exc)
            try:
                await refresh_balance_card_if_open(user_id, chat_id, ctx=ctx, state_dict=s)
            except Exception as exc:
                log.warning("[SUNO] final balance refresh fail | req_id=%s err=%s", req_id, exc)


async def _suno_poll_record_info(
    task_id: str,
    *,
    user_id: Optional[int],
) -> RecordInfoPollResult:
    delays = [delay for delay in [SUNO_POLL_FIRST_DELAY, *SUNO_POLL_BACKOFF_SERIES] if delay >= 0]
    if not delays:
        delays = [5.0]
    start_ts = time.monotonic()
    attempt = 0
    last_status: Optional[str] = None

    while True:
        if SUNO_SERVICE._recently_delivered(task_id):  # type: ignore[attr-defined]
            log.info("[SUNO] poll delivered via webhook | task_id=%s", task_id)
            return RecordInfoPollResult(
                state="delivered",
                status_code=200,
                payload={},
                attempts=attempt,
                elapsed=time.monotonic() - start_ts,
            )

        delay = delays[min(attempt, len(delays) - 1)]
        if delay > 0:
            await asyncio.sleep(delay)

        attempt += 1
        result = await asyncio.to_thread(
            SUNO_SERVICE.poll_record_info_once,
            task_id,
            user_id=user_id,
        )
        now = time.monotonic()
        result.attempts = attempt
        result.elapsed = now - start_ts

        status_value: Optional[str] = None
        if isinstance(result.payload, Mapping):
            status_value = SunoService._status_from_payload(result.payload)
        normalized_status = status_value.upper() if isinstance(status_value, str) else None
        if normalized_status and normalized_status != last_status:
            log.info(
                "[SUNO] poll status update | task_id=%s status=%s attempt=%s http=%s",
                task_id,
                normalized_status,
                attempt,
                result.status_code,
            )
            last_status = normalized_status

        meta = {
            "taskId": task_id,
            "attempt": attempt,
            "http_status": result.status_code,
            "mapped_state": result.state,
        }
        if normalized_status:
            meta["status"] = normalized_status
        if result.error:
            meta["error_code"] = result.error
        if result.message:
            meta["message"] = result.message

        if result.state == "retry":
            log.warning("[SUNO] poll retry", extra={"meta": meta})
        elif result.state != "pending":
            log.info("[SUNO] poll step", extra={"meta": meta})

        if result.state == "ready":
            tracks = SunoService._tracks_from_payload(result.payload)
            durations = SunoService._durations_from_tracks(tracks)
            log.info(
                "[SUNO] poll ready",
                extra={
                    "meta": {
                        "taskId": task_id,
                        "takes": len(tracks),
                        "durations": durations,
                        "http_status": result.status_code,
                    }
                },
            )
            return result

        if result.state == "hard_error":
            log.error(
                "[SUNO] poll hard failure",
                extra={"meta": meta},
            )
            return result

        if result.state == "delivered":
            log.info("[SUNO] poll delivered via webhook | task_id=%s", task_id)
            return result

        if result.elapsed >= SUNO_POLL_TIMEOUT:
            log.warning(
                "[SUNO] poll timeout",
                extra={
                    "meta": {
                        "taskId": task_id,
                        "attempts": attempt,
                        "elapsed": result.elapsed,
                    }
                },
            )
            result.state = "timeout"
            return result


async def _poll_suno_and_send(
    chat_id: int,
    ctx: ContextTypes.DEFAULT_TYPE,
    user_id: int,
    task_id: str,
    params: Dict[str, Any],
    meta: Dict[str, Any],
    *,
    req_id: Optional[str] = None,
    reply_to: Optional["telegram.Message"] = None,
) -> None:
    start_time = time.monotonic()
    req_id_value = req_id or meta.get("req_id") or SUNO_SERVICE.get_request_id(task_id)
    refunded = False
    notified_timeout = False
    attempt = 0

    def _clean_reason(raw: Any) -> Optional[str]:
        if raw is None:
            return None
        text = collapse_spaces(str(raw))
        text = text.strip().strip(". ")
        if not text:
            return None
        return text[:160]

    async def _notify_timeout_once() -> None:
        nonlocal notified_timeout
        if notified_timeout:
            return
        try:
            await _suno_notify(
                ctx,
                chat_id,
                _suno_timeout_text(),
                req_id=req_id_value,
                reply_to=reply_to,
            )
        except Exception as exc:
            log.warning("[SUNO] poll timeout notify fail | task_id=%s err=%s", task_id, exc)
        else:
            notified_timeout = True

    async def _issue_refund(message: str, *, reason: str) -> None:
        nonlocal refunded
        if refunded:
            return
        refunded = True
        refund_text = f"{message}\n💎 Токены возвращены (+{PRICE_SUNO}💎)."
        await _suno_issue_refund(
            ctx,
            chat_id,
            user_id,
            base_meta=meta,
            task_id=task_id,
            error_text=message,
            reason=reason,
            reply_markup=_suno_result_keyboard(),
            req_id=req_id_value,
            reply_to=reply_to,
            user_message=refund_text,
        )
        await _clear_wait()

    async def _clear_wait() -> None:
        try:
            await refresh_suno_card(ctx, chat_id, state(ctx), price=PRICE_SUNO)
        except Exception as exc:
            log.debug("[SUNO] clear_wait_failed | chat_id=%s err=%s", chat_id, exc)

    try:
        try:
            existing_record = await asyncio.to_thread(SUNO_SERVICE.get_task_record, task_id)
        except Exception:
            existing_record = None
        if existing_record:
            existing_status = str(existing_record.get("status") or "").lower()
            if existing_status in {"complete", "error", "failed"} and existing_record.get("tracks"):
                log.info(
                    "[SUNO] poll already delivered | task_id=%s status=%s",
                    task_id,
                    existing_status,
                )
                return

        poll_result = await _suno_poll_record_info(
            task_id,
            user_id=user_id,
        )

        details = dict(poll_result.payload) if isinstance(poll_result.payload, Mapping) else {}
        http_status = poll_result.status_code
        state_value = poll_result.state

        if state_value == "delivered":
            log.info("[SUNO] poll delivered via webhook | task_id=%s", task_id)
            return

        if state_value == "timeout":
            log.warning(
                "[SUNO] poll timeout | task_id=%s attempts=%s elapsed=%.1f",
                task_id,
                poll_result.attempts,
                poll_result.elapsed,
            )
            timeout_message = "⚠️ Suno не ответил вовремя."
            await _issue_refund(
                timeout_message,
                reason="suno:refund:timeout",
            )
            return

        if state_value == "hard_error":
            reason_text = _clean_reason(poll_result.message or poll_result.error)
            message = _suno_error_message(http_status, reason_text)
            try:
                await _suno_notify(
                    ctx,
                    chat_id,
                    message,
                    req_id=req_id_value,
                    reply_to=reply_to,
                )
            except Exception as notify_exc:
                log.warning(
                    "[SUNO] poll failure notify fail | task_id=%s err=%s",
                    task_id,
                    notify_exc,
                )
            await _issue_refund(message, reason="suno:refund:status_err")
            await _clear_wait()
            return

        if state_value != "ready":
            log.info(
                "[SUNO] poll finished without ready state | task_id=%s state=%s",
                task_id,
                state_value,
            )
            if state_value in {"error", "failed"}:
                await _clear_wait()
            return

        tracks_payload = _poll_tracks(details)
        if not tracks_payload:
            log.info(
                "[SUNO] poll success without tracks | task_id=%s",
                task_id,
            )
            await _notify_timeout_once()
            return

        envelope_payload = {
            "code": details.get("code") or 200,
            "msg": details.get("message") or details.get("msg"),
            "data": {
                "taskId": task_id,
                "callbackType": "complete",
                "response": {"tracks": tracks_payload},
            },
        }

        duration_value = _poll_duration(details)
        audio_url_preview = ""
        if tracks_payload:
            preview_candidate = (
                tracks_payload[0].get("audioUrl")
                or tracks_payload[0].get("audio_url")
                or ""
            )
            if preview_candidate:
                audio_url_preview = str(preview_candidate)
        log.info(
            "[SUNO] poll success",
            extra={
                "meta": {
                    "taskId": task_id,
                    "duration": duration_value,
                    "audioUrl": mask_tokens(audio_url_preview) if audio_url_preview else "",
                    "tags": _poll_tags(details),
                }
            },
        )

        try:
            callback_task = SunoTask.from_envelope(CallbackEnvelope.model_validate(envelope_payload))
            callback_task = callback_task.model_copy(
                update={"code": envelope_payload["code"], "msg": envelope_payload.get("msg")}
            )
            await asyncio.to_thread(
                SUNO_SERVICE.handle_callback,
                callback_task,
                req_id=req_id_value,
                delivery_via="poll",
            )
        except Exception as exc:
            log.exception("[SUNO] poll delivery failed | task_id=%s err=%s", task_id, exc)
        return

    except asyncio.CancelledError:
        log.info("[SUNO] poll cancelled | task_id=%s", task_id)
        raise
    except Exception as exc:
        log.exception("[SUNO] poll unexpected failure | task_id=%s err=%s", task_id, exc)
        await _issue_refund(_suno_error_message(None, _clean_reason(str(exc))), reason="suno:refund:poll_err")
        await _clear_wait()
    finally:
        s = state(ctx)
        if s.get("suno_last_task_id") == task_id:
            s["suno_last_task_id"] = None
        s["suno_generating"] = False
        s["suno_current_req_id"] = None
        _reset_suno_start_flags(s)
        _reset_suno_card_cache(s)
        s["video_wait_message_id"] = None
        try:
            await refresh_suno_card(ctx, chat_id, s, price=PRICE_SUNO)
        except Exception as exc:
            log.warning("[SUNO] poll card refresh fail | task_id=%s err=%s", task_id, exc)


# --------- Sora2 Card ----------
_SORA2_MODE_TITLES = {
    "sora2_ttv": "Text-to-Video",
    "sora2_itv": "Image-to-Video",
}


_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)


def _extract_http_urls(text: str) -> List[str]:
    urls: List[str] = []
    for match in _URL_RE.findall(text or ""):
        trimmed = match.strip().rstrip(",.;)")
        if trimmed and trimmed not in urls:
            urls.append(trimmed)
    return urls


def sora2_card_text(s: Dict[str, Any]) -> str:
    mode = s.get("mode") or "sora2_ttv"
    display = _SORA2_MODE_TITLES.get(mode, "Text-to-Video")
    prompt_raw = (s.get("sora2_prompt") or "").strip()
    prompt_html = html.escape(prompt_raw) if prompt_raw else ""
    image_urls = [str(url) for url in s.get("sora2_image_urls", []) if isinstance(url, str) and url.strip()]
    lines = [
        "🟦 <b>Карточка Sora 2</b>",
        f"• Режим: <b>{display}</b>",
    ]
    if image_urls:
        lines.append(f"• Фото: <b>{len(image_urls)}/{SORA2_MAX_IMAGES}</b>")
    lines.extend([
        "",
        "🖊️ <b>Промпт:</b>",
        f"<code>{prompt_html}</code>" if prompt_html else "<code> </code>",
    ])
    if image_urls:
        lines.append("")
        lines.append("🔗 <b>Ссылки:</b>")
        for url in image_urls:
            lines.append(f"• <code>{html.escape(url)}</code>")
    return "\n".join(lines)


def sora2_kb(s: Dict[str, Any]) -> InlineKeyboardMarkup:
    mode = s.get("mode") or "sora2_ttv"
    start_token = "s2_go_i2v" if mode == "sora2_itv" else "s2_go_t2v"
    rows = [
        [InlineKeyboardButton("🚀 Запустить Sora 2", callback_data=start_token)],
        [InlineKeyboardButton("⬅️ Назад", callback_data="back")],
    ]
    return InlineKeyboardMarkup(rows)


async def show_sora2_card(
    chat_id: int,
    ctx: ContextTypes.DEFAULT_TYPE,
    *,
    force_new: bool = False,
) -> None:
    s = state(ctx)
    text = sora2_card_text(s)
    if not force_new and text == s.get("_last_text_sora2"):
        return
    kb = sora2_kb(s)
    mid = await upsert_card(
        ctx,
        chat_id,
        s,
        "last_ui_msg_id_sora2",
        text,
        kb,
        force_new=force_new,
    )
    if mid:
        s["_last_text_sora2"] = text
    else:
        s["_last_text_sora2"] = None


async def sora2_entry(chat_id: int, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    s = state(ctx)
    mid = s.get("last_ui_msg_id_sora2")
    if mid:
        with suppress(Exception):
            await ctx.bot.delete_message(chat_id, mid)
    s["last_ui_msg_id_sora2"] = None
    s["_last_text_sora2"] = None
    await show_sora2_card(chat_id, ctx, force_new=True)


# --------- Sora2 Generation ----------
_SORA2_MODEL_IDS = {
    "sora2_ttv": "sora-2-text-to-video",
    "sora2_itv": "sora-2-image-to-video",
}

_SORA2_SERVICE_CODES = {
    "sora2_ttv": "SORA2_TTV",
    "sora2_itv": "SORA2_ITV",
}


def _sora2_price_and_service(mode: str) -> Tuple[int, str]:
    if mode == "sora2_itv":
        return PRICE_SORA2_IMAGE, _SORA2_SERVICE_CODES.get(mode, "SORA2_ITV")
    return PRICE_SORA2_TEXT, _SORA2_SERVICE_CODES.get(mode, "SORA2_TTV")


def _sora2_model_id(mode: str) -> str:
    return _SORA2_MODEL_IDS.get(mode, _SORA2_MODEL_IDS["sora2_ttv"])


def _sora2_normalize_aspect(raw: Any) -> str:
    candidate = str(raw or "16:9").strip()
    if candidate not in SORA2_ALLOWED_ASPECTS:
        return "16:9"
    return candidate


def _sora2_aspect_for_api(aspect_ratio: Optional[str]) -> Optional[str]:
    aspect = (aspect_ratio or "").strip()
    if not aspect:
        return None
    if aspect in {"9:16", "4:5"}:
        return "portrait"
    if aspect == "16:9":
        return "landscape"
    return None


def _sora2_quality_from_resolution(resolution: Optional[str]) -> Optional[str]:
    text = str(resolution or "").strip().lower()
    if not text:
        return "standard"
    if any(token in text for token in ("1080", "1920", "2160", "4k")):
        return "hd"
    return "standard"


def _sora2_lookup_url(payload: Mapping[str, Any], keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str):
            trimmed = value.strip()
            if trimmed.lower().startswith("http"):
                return trimmed
    for value in payload.values():
        if isinstance(value, Mapping):
            found = _sora2_lookup_url(value, keys)
            if found:
                return found
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for item in value:
                if isinstance(item, Mapping):
                    found = _sora2_lookup_url(item, keys)
                    if found:
                        return found
                elif isinstance(item, str):
                    trimmed = item.strip()
                    if trimmed.lower().startswith("http"):
                        return trimmed
    return None


def _sora2_extract_assets(payload: Mapping[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    video_keys = ("video", "videoUrl", "video_url", "url", "resultUrl")
    cover_keys = ("cover", "coverUrl", "cover_url", "thumbnail", "preview")
    video_url = _sora2_lookup_url(payload, video_keys)
    cover_url = _sora2_lookup_url(payload, cover_keys)
    return video_url, cover_url


def _sora2_extract_error(payload: Mapping[str, Any]) -> Optional[str]:
    for key in ("error", "message", "detail", "msg", "reason"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    data = payload.get("data")
    if isinstance(data, Mapping):
        nested = _sora2_extract_error(data)
        if nested:
            return nested
    return None


def _sora2_sanitize_image_urls(urls: Sequence[str]) -> List[str]:
    cleaned: List[str] = []
    for raw in urls:
        text = str(raw or "").strip()
        if not text or not text.lower().startswith("http"):
            continue
        if text in cleaned:
            continue
        cleaned.append(text)
        if len(cleaned) >= SORA2_MAX_IMAGES:
            break
    return cleaned


def _build_sora2_payload(
    mode: str,
    prompt: str,
    image_urls: Sequence[str],
    *,
    aspect_ratio: Optional[str],
    quality: Optional[str],
) -> Dict[str, Any]:
    callback_url = SORA2.get("CALLBACK_URL")
    if not callback_url:
        public_base = (os.getenv("PUBLIC_BASE_URL") or "").strip()
        if public_base:
            callback_url = f"{public_base.rstrip('/')}/sora2-callback"
    input_payload: Dict[str, Any] = {}
    prompt_text = prompt.strip()
    input_payload["prompt"] = prompt_text

    if mode == "sora2_itv":
        cleaned_urls = _sora2_sanitize_image_urls(image_urls)
        if cleaned_urls:
            input_payload["image_urls"] = cleaned_urls

    aspect_value = _sora2_aspect_for_api(aspect_ratio)
    if aspect_value:
        input_payload["aspect_ratio"] = aspect_value

    quality_value = (quality or "standard").strip().lower()
    if quality_value not in {"standard", "hd"}:
        quality_value = "standard"
    input_payload["quality"] = quality_value

    payload: Dict[str, Any] = {
        "model": _sora2_model_id(mode),
        "input": input_payload,
    }
    if callback_url:
        payload["callBackUrl"] = callback_url
    return payload


async def _prepare_sora2_image_urls(image_urls: Sequence[str]) -> List[str]:
    cleaned = _sora2_sanitize_image_urls(image_urls)
    if not cleaned:
        return []
    return await asyncio.to_thread(sora2_upload_image_urls, cleaned)


async def _download_temp_file(url: str, suffix: str) -> Optional[str]:
    if not url:
        return None
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.close()
    path = tmp.name
    try:
        timeout = ClientTimeout(total=300)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                response.raise_for_status()
                with open(path, "wb") as handle:
                    async for chunk in response.content.iter_chunked(65536):
                        handle.write(chunk)
        return path
    except Exception as exc:
        log.warning("sora2.download_failed", extra={"url": url, "error": str(exc)})
        with suppress(Exception):
            Path(path).unlink()
        return None


async def _download_sora2_assets(result: Mapping[str, Any]) -> List[str]:
    video_url, cover_url = _sora2_extract_assets(result)
    if not video_url:
        urls_candidate = result.get("resultUrls") or result.get("result_urls") or []
        if isinstance(urls_candidate, Sequence) and not isinstance(urls_candidate, (str, bytes, bytearray)):
            prioritized: List[str] = []
            for item in urls_candidate:
                if isinstance(item, str) and item.strip():
                    prioritized.append(item.strip())
            for candidate in prioritized:
                lowered = candidate.lower()
                if lowered.startswith("http") and lowered.endswith(".mp4"):
                    video_url = candidate
                    break
            if not video_url:
                for candidate in prioritized:
                    if candidate.lower().startswith("http"):
                        video_url = candidate
                        break
    files: List[str] = []
    if video_url:
        video_path = await _download_temp_file(video_url, ".mp4")
        if video_path:
            files.append(video_path)
    if cover_url:
        cover_path = await _download_temp_file(cover_url, ".jpg")
        if cover_path:
            files.append(cover_path)
    return files


def _schedule_sora2_poll(task_id: str, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    existing = _SORA2_POLLERS.get(task_id)
    if existing and not existing.done():
        existing.cancel()
    task = asyncio.create_task(_poll_sora2_and_send(task_id, ctx))
    _SORA2_POLLERS[task_id] = task


async def _finalize_sora2_task(
    ctx: ContextTypes.DEFAULT_TYPE,
    task_id: str,
    meta: Mapping[str, Any],
    status: str,
    result_payload: Optional[Mapping[str, Any]],
    source: str,
) -> None:
    user_id: Optional[int] = None
    user_id_raw = meta.get("user_id")
    try:
        user_id = int(user_id_raw) if user_id_raw is not None else None
    except (TypeError, ValueError):
        user_id = None
    chat_id_raw = meta.get("chat_id")
    try:
        chat_id = int(chat_id_raw)
    except (TypeError, ValueError):
        log.warning("sora2.finalize.chat_missing", extra={"task_id": task_id, "chat_id": chat_id_raw})
        if user_id is not None:
            release_sora2_lock(user_id)
        return
    await delete_wait_sticker(ctx, chat_id=chat_id)
    wait_meta = meta.get("wait_message_id")
    try:
        wait_msg_id = int(wait_meta) if wait_meta is not None else None
    except (TypeError, ValueError):
        wait_msg_id = None
    if wait_msg_id:
        with suppress(Exception):
            await ctx.bot.delete_message(chat_id, wait_msg_id)
    price = int(meta.get("price") or 0)
    service = str(meta.get("service") or "SORA2")
    mode = str(meta.get("mode") or "sora2_ttv")

    ACTIVE_TASKS.pop(chat_id, None)
    s = state(ctx)
    if s.get("sora2_last_task_id") == task_id:
        s["sora2_last_task_id"] = None
    s["sora2_generating"] = False
    if s.get("video_wait_message_id") == wait_msg_id:
        s["video_wait_message_id"] = None
    if s.get("sora2_wait_msg_id") == wait_msg_id:
        s["sora2_wait_msg_id"] = None
    if mode.startswith("sora2_"):
        s["sora2_prompt"] = None
        s["sora2_image_urls"] = []
        s["_last_text_sora2"] = None
        await show_sora2_card(chat_id, ctx)
        if user_id is not None:
            clear_wait_state(user_id, reason="sora2_done")
    keyboard = video_result_footer_kb()
    duration_meta = meta.get("duration")
    try:
        duration_value = int(duration_meta) if duration_meta is not None else None
    except (TypeError, ValueError):
        duration_value = None
    resolution_value = str(meta.get("resolution") or "").strip() or None
    caption_text = _sora2_caption(mode, duration_value, resolution_value)
    try:
        if status == "success" and isinstance(result_payload, Mapping):
            sent_message_id: Optional[int] = None
            result_kind = "as_document"
            files = await _download_sora2_assets(result_payload)
            video_path = None
            for path in files:
                lowered = path.lower()
                if lowered.endswith(".mp4") and video_path is None:
                    video_path = path
            try:
                if video_path:
                    try:
                        with open(video_path, "rb") as handle:
                            sent_doc = await ctx.bot.send_document(
                                chat_id=chat_id,
                                document=handle,
                                caption=caption_text,
                                reply_markup=keyboard,
                            )
                        sent_message_id = getattr(sent_doc, "message_id", None)
                    except TelegramError as doc_exc:
                        log.warning(
                            "sora2.send_document_failed",
                            extra={"task_id": task_id, "error": str(doc_exc)},
                        )
                        try:
                            with open(video_path, "rb") as handle:
                                sent_video = await ctx.bot.send_video(
                                    chat_id=chat_id,
                                    video=handle,
                                    caption=caption_text,
                                    reply_markup=keyboard,
                                    supports_streaming=True,
                                )
                            sent_message_id = getattr(sent_video, "message_id", None)
                            result_kind = "as_video"
                        except TelegramError as video_exc:
                            log.warning(
                                "sora2.send_video_failed",
                                extra={"task_id": task_id, "error": str(video_exc)},
                            )
                            sent_text_id = await safe_edit_or_send(
                                ctx,
                                chat_id=chat_id,
                                message_id=None,
                                text=caption_text,
                                reply_markup=keyboard,
                            )
                            sent_message_id = sent_text_id
                            result_kind = "as_text"
                else:
                    sent_text_id = await safe_edit_or_send(
                        ctx,
                        chat_id=chat_id,
                        message_id=None,
                        text=caption_text,
                        reply_markup=keyboard,
                    )
                    sent_message_id = sent_text_id
                    result_kind = "as_text"
            except TelegramError as exc:
                log.warning(
                    "sora2.result_send_failed",
                    extra={"task_id": task_id, "error": str(exc)},
                )
                sent_text_id = await safe_edit_or_send(
                    ctx,
                    chat_id=chat_id,
                    message_id=None,
                    text=caption_text,
                    reply_markup=keyboard,
                )
                sent_message_id = sent_text_id
                result_kind = "as_text"
            finally:
                for path in files:
                    with suppress(Exception):
                        Path(path).unlink()
            log.info(
                "sora2.result.sent",
                extra={
                    "task_id": task_id,
                    "message_id": sent_message_id,
                    "mode": mode,
                    "kind": result_kind,
                    "source": source,
                },
            )
        else:
            if wait_msg_id:
                with suppress(Exception):
                    await ctx.bot.delete_message(chat_id, wait_msg_id)
            error_reason = meta.get("error") if isinstance(meta.get("error"), str) else None
            if not error_reason and isinstance(result_payload, Mapping):
                error_reason = str(result_payload.get("error") or result_payload.get("message") or "Ошибка Sora 2")
            message = error_reason or "⚠️ Не удалось получить результат Sora 2."
            await safe_edit_or_send(
                ctx,
                chat_id=chat_id,
                message_id=None,
                text=message,
                reply_markup=keyboard,
            )
            if meta.get("error") == "unavailable":
                await _refresh_video_menu_ui(ctx, chat_id=chat_id, message=None)
            if price > 0 and user_id is not None:
                try:
                    new_balance = credit_balance(
                        user_id,
                        price,
                        reason="service:refund",
                        meta={"service": service, "reason": status or "error", "task_id": task_id},
                    )
                except Exception as exc:
                    log.exception("sora2.refund_failed | task_id=%s err=%s", task_id, exc)
                else:
                    await show_balance_notification(
                        chat_id,
                        ctx,
                        user_id,
                        f"⚠️ Ошибка. Возврат {price}💎. Баланс: {new_balance}💎",
                    )
            log.info(
                "video.result.error",
                extra={"task_id": task_id, "mode": mode, "source": source, "status": status},
            )
    finally:
        if user_id is not None:
            release_sora2_lock(user_id)
        update_task_meta(task_id, handled=True, status=status, completed_at=datetime.now(timezone.utc).isoformat())
        clear_task_meta(task_id)
        poller = _SORA2_POLLERS.pop(task_id, None)
        if poller and not poller.done():
            poller.cancel()


async def _poll_sora2_and_send(task_id: str, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    start = time.monotonic()
    backoff_plan = list(SORA2_POLL_BACKOFF_SERIES) or [5.0]
    next_remote_poll = start + backoff_plan[0]
    backoff_iter = iter(backoff_plan[1:])
    backoff_fallback = backoff_plan[-1]
    try:
        while True:
            meta = load_task_meta(task_id)
            if not meta:
                return
            status = str(meta.get("status") or "pending").lower()
            handled = bool(meta.get("handled"))
            if status in {"success", "failed"} and not handled:
                result_payload = meta.get("result") if isinstance(meta.get("result"), Mapping) else None
                if not result_payload:
                    urls_meta = meta.get("result_urls")
                    if isinstance(urls_meta, Sequence) and not isinstance(urls_meta, (str, bytes, bytearray)):
                        normalized_urls = [
                            str(item).strip()
                            for item in urls_meta
                            if isinstance(item, str) and item.strip()
                        ]
                        if normalized_urls:
                            result_payload = {"resultUrls": normalized_urls}
                await _finalize_sora2_task(
                    ctx,
                    task_id,
                    meta,
                    status,
                    result_payload,
                    str(meta.get("source") or "webhook"),
                )
                return
            now = time.monotonic()
            if now - start > SORA2_POLL_TIMEOUT:
                meta_update = update_task_meta(
                    task_id,
                    status="failed",
                    error="timeout",
                    source="timeout",
                ) or meta
                await _finalize_sora2_task(ctx, task_id, meta_update, "failed", None, "timeout")
                return
            if status not in {"success", "failed"} and now >= next_remote_poll:
                try:
                    query_response: QueryTaskResponse = await asyncio.to_thread(
                        sora2_query_task, task_id
                    )
                except Sora2UnavailableError as exc:
                    mark_sora2_unavailable()
                    log.warning(
                        "sora2.poll.unavailable",
                        extra={"task_id": task_id, "error": str(exc)},
                    )
                    meta_update = update_task_meta(
                        task_id,
                        status="failed",
                        error="unavailable",
                        source="poll",
                        result=None,
                    ) or meta
                    await _finalize_sora2_task(ctx, task_id, meta_update, "failed", None, "poll")
                    return
                except Sora2Error as exc:
                    log.warning(
                        "sora2.poll.error",
                        extra={"task_id": task_id, "error": str(exc)},
                    )
                else:
                    remote_state = (query_response.status or "pending").lower()
                    normalized_status = remote_state
                    if remote_state == "fail":
                        normalized_status = "failed"
                    elif remote_state == "success":
                        normalized_status = "success"

                    result_payload_map: Optional[Dict[str, Any]] = None
                    if isinstance(query_response.result_payload, Mapping):
                        result_payload_map = dict(query_response.result_payload)
                    if query_response.result_urls:
                        urls_list = list(query_response.result_urls)
                        if result_payload_map is None:
                            result_payload_map = {"resultUrls": urls_list}
                        else:
                            result_payload_map.setdefault("resultUrls", urls_list)

                    if remote_state == "fail":
                        log.warning(
                            "sora2.poll.failed",
                            extra={"task_id": task_id, "error": query_response.error_message},
                        )

                    update_task_meta(
                        task_id,
                        status=normalized_status,
                        result=result_payload_map,
                        result_urls=list(query_response.result_urls),
                        error=query_response.error_message,
                        source="poll",
                        raw=query_response.raw,
                        state=remote_state,
                    )
                delay = next(backoff_iter, backoff_fallback)
                next_remote_poll = now + delay
            await asyncio.sleep(1)
    finally:
        poller = _SORA2_POLLERS.get(task_id)
        if poller and poller.done():
            _SORA2_POLLERS.pop(task_id, None)


async def _start_sora2_generation(
    ctx: ContextTypes.DEFAULT_TYPE,
    *,
    chat_id: int,
    user_id: Optional[int],
    message: Message,
    mode: str,
    prompt: str,
    image_urls: Sequence[str],
    aspect_ratio: str,
) -> Optional[str]:
    if user_id is None:
        await message.reply_text("⚠️ Не удалось определить пользователя. Попробуйте позже.")
        return None

    s = state(ctx)
    s["mode"] = mode

    price, service_name = _sora2_price_and_service(mode)
    try:
        ensure_user(user_id)
    except Exception as exc:
        log.exception("sora2.ensure_user_failed | user_id=%s err=%s", user_id, exc)
        await message.reply_text("⚠️ Не удалось проверить баланс. Попробуйте позже.")
        return None

    if not await ensure_tokens(ctx, chat_id, user_id, price):
        return None

    try:
        prepared_image_urls = await _prepare_sora2_image_urls(image_urls)
    except Exception as exc:
        log.exception("sora2.image_prepare_failed | user_id=%s err=%s", user_id, exc)
        await message.reply_text("⚠️ Не удалось обработать изображения. Попробуйте снова.")
        return None

    if mode == "sora2_itv" and len(prepared_image_urls) < SORA2_MIN_IMAGES:
        await message.reply_text("📸 Нужны 1–4 изображения для этого режима. Попробуйте снова.")
        return None

    prompt_for_api = truncate_text(prompt or "", SORA2_MAX_PROMPT_LENGTH)
    prompt_preview = _short_prompt(prompt_for_api, 160)
    if mode == "sora2_itv":
        duration = SORA2_DEFAULT_ITV_DURATION
        resolution = SORA2_DEFAULT_ITV_RESOLUTION
        audio_flag: Optional[bool] = None
    else:
        duration = SORA2_DEFAULT_TTV_DURATION
        resolution = SORA2_DEFAULT_TTV_RESOLUTION
        audio_flag = True
    quality = _sora2_quality_from_resolution(resolution)
    ok, balance_after = debit_try(
        user_id,
        price,
        reason="service:start",
        meta={"service": service_name, "prompt": prompt_preview, "mode": mode},
    )
    if not ok:
        await ensure_tokens(ctx, chat_id, user_id, price)
        return None

    await show_balance_notification(
        chat_id,
        ctx,
        user_id,
        f"✅ Списано {price}💎. Текущий баланс: {balance_after}💎 — запускаю…",
    )

    payload = _build_sora2_payload(
        mode,
        prompt_for_api,
        prepared_image_urls,
        aspect_ratio=aspect_ratio,
        quality=quality,
    )
    try:
        create_response: CreateTaskResponse = await asyncio.to_thread(
            sora2_create_task, payload
        )
        task_id = create_response.task_id
    except Sora2BadRequestError as exc:
        log.warning(
            "sora2.create_task_bad_request",
            extra={
                "user_id": user_id,
                "error": str(exc),
                "mode": mode,
                "has_images": bool(prepared_image_urls),
            },
        )
        try:
            new_balance = credit_balance(
                user_id,
                price,
                reason="service:refund",
                meta={"service": service_name, "reason": "bad_request"},
            )
        except Exception as refund_exc:
            log.exception("sora2.submit_refund_failed | user_id=%s err=%s", user_id, refund_exc)
        else:
            await show_balance_notification(
                chat_id,
                ctx,
                user_id,
                f"⚠️ Ошибка. Возврат {price}💎. Баланс: {new_balance}💎",
            )
        await message.reply_text(
            "⚠️ Проверьте входные данные для Sora 2. Убедитесь, что текст и ссылки доступны."
        )
        await _refresh_video_menu_ui(ctx, chat_id=chat_id, message=message)
        return None
    except Sora2UnavailableError as exc:
        mark_sora2_unavailable()
        log.warning(
            "sora2.create_task_unavailable",
            extra={"user_id": user_id, "error": str(exc)},
        )
        await message.reply_text(
            "Sora2 временно недоступна. Попробуйте позже или выберите VEO."
        )
        try:
            new_balance = credit_balance(
                user_id,
                price,
                reason="service:refund",
                meta={"service": service_name, "reason": "unavailable"},
            )
        except Exception as refund_exc:
            log.exception("sora2.submit_refund_failed | user_id=%s err=%s", user_id, refund_exc)
        else:
            await show_balance_notification(
                chat_id,
                ctx,
                user_id,
                f"⚠️ Ошибка. Возврат {price}💎. Баланс: {new_balance}💎",
            )
        await _refresh_video_menu_ui(ctx, chat_id=chat_id, message=message)
        return None
    except Sora2AuthError as exc:
        mark_sora2_unavailable()
        log.error(
            "sora2.create_task_auth_error",
            extra={"user_id": user_id, "error": str(exc)},
        )
        await message.reply_text(
            "Sora2 временно недоступна. Попробуйте позже или выберите VEO."
        )
        try:
            new_balance = credit_balance(
                user_id,
                price,
                reason="service:refund",
                meta={"service": service_name, "reason": "auth_error"},
            )
        except Exception as refund_exc:
            log.exception("sora2.submit_refund_failed | user_id=%s err=%s", user_id, refund_exc)
        else:
            await show_balance_notification(
                chat_id,
                ctx,
                user_id,
                f"⚠️ Ошибка. Возврат {price}💎. Баланс: {new_balance}💎",
            )
        await _refresh_video_menu_ui(ctx, chat_id=chat_id, message=message)
        return None
    except Sora2Error as exc:
        log.error(
            "sora2.create_task_failed",
            exc_info=True,
            extra={"user_id": user_id},
        )
        try:
            new_balance = credit_balance(
                user_id,
                price,
                reason="service:refund",
                meta={"service": service_name, "reason": "submit_exception", "message": str(exc)},
            )
        except Exception as refund_exc:
            log.exception("sora2.submit_refund_failed | user_id=%s err=%s", user_id, refund_exc)
        else:
            await show_balance_notification(
                chat_id,
                ctx,
                user_id,
                f"⚠️ Ошибка. Возврат {price}💎. Баланс: {new_balance}💎",
            )
        await message.reply_text("⚠️ Не удалось создать задачу Sora 2. 💎 Токены возвращены.")
        return None
    except Exception as exc:
        log.error(
            "sora2.create_task_failed",
            exc_info=True,
            extra={"user_id": user_id},
        )
        try:
            new_balance = credit_balance(
                user_id,
                price,
                reason="service:refund",
                meta={"service": service_name, "reason": "submit_exception", "message": str(exc)},
            )
        except Exception as refund_exc:
            log.exception("sora2.submit_refund_failed | user_id=%s err=%s", user_id, refund_exc)
        else:
            await show_balance_notification(
                chat_id,
                ctx,
                user_id,
                f"⚠️ Ошибка. Возврат {price}💎. Баланс: {new_balance}💎",
            )
        await message.reply_text("⚠️ Не удалось создать задачу Sora 2. 💎 Токены возвращены.")
        return None

    wait_msg_id = await send_wait_sticker(ctx, "sora2", chat_id=chat_id)
    s["sora2_wait_msg_id"] = wait_msg_id
    s["video_wait_message_id"] = wait_msg_id
    s["sora2_generating"] = True
    s["sora2_last_task_id"] = task_id
    ACTIVE_TASKS[chat_id] = task_id

    save_task_meta(
        task_id,
        chat_id,
        message.message_id,
        mode,
        aspect_ratio,
        extra={
            "user_id": user_id,
            "price": price,
            "service": service_name,
            "status": "pending",
            "handled": False,
            "prompt_preview": prompt_preview,
            "image_urls": list(prepared_image_urls),
            "wait_message_id": wait_msg_id,
            "aspect_ratio": aspect_ratio,
            "submit_raw": create_response.raw,
            "duration": duration,
            "resolution": resolution,
            "audio": audio_flag,
            "quality": quality,
        },
        ttl=30 * 60,
    )

    log.info(
        "sora2.payload.sent",
        extra={
            "task_id": task_id,
            "model": payload.get("model"),
            "quality": payload.get("input", {}).get("quality"),
            "has_images": bool(prepared_image_urls),
        },
    )
    log.info(
        "sora2.start",
        extra={
            "user_id": user_id,
            "mode": mode,
            "aspect": aspect_ratio,
            "with_images": bool(prepared_image_urls),
            "task_id": task_id,
            "chat_id": chat_id,
            "raw": create_response.raw,
            "duration": duration,
            "resolution": resolution,
            "quality": quality,
        },
    )
    log.info(
        "video.start",
        extra={"mode": mode, "task_id": task_id, "user_id": user_id, "chat_id": chat_id},
    )

    await message.reply_text("🎬 Отправляю задачу в Sora 2…")
    _schedule_sora2_poll(task_id, ctx)
    return task_id


async def _handle_sora2_start(
    update: Update,
    ctx: ContextTypes.DEFAULT_TYPE,
    *,
    mode: str,
) -> None:
    await ensure_user_record(update)

    query = update.callback_query
    if not query:
        return

    if not _sora2_is_enabled():
        await query.answer(
            "Sora2 временно недоступна. Попробуйте позже или выберите VEO.",
            show_alert=True,
        )
        return

    message = query.message
    if message is None:
        await query.answer("⚠️ Сообщение недоступно", show_alert=True)
        return

    chat_obj = getattr(message, "chat", None)
    chat_id = getattr(chat_obj, "id", None)
    if chat_id is None:
        chat_id = getattr(message, "chat_id", None)
    if chat_id is None:
        await query.answer("⚠️ Чат недоступен", show_alert=True)
        return

    user = query.from_user or update.effective_user
    user_id = user.id if user else None

    s = state(ctx)
    prompt = (s.get("sora2_prompt") or "").strip()
    raw_image_urls = list(s.get("sora2_image_urls") or [])
    image_urls = _sora2_sanitize_image_urls(raw_image_urls)
    aspect_ratio = _sora2_normalize_aspect(s.get("aspect"))

    if not prompt and not image_urls:
        await query.answer("✍️ Нужен текст или 1–4 изображения.", show_alert=True)
        return

    if mode == "sora2_ttv":
        if len(prompt) < 3:
            await query.answer("✍️ Пришлите текст и повторите запуск.", show_alert=True)
            return
    else:
        if len(image_urls) < SORA2_MIN_IMAGES:
            await query.answer("📸 Добавьте 1–4 ссылок и повторите запуск.", show_alert=True)
            return
    if len(prompt) > SORA2_MAX_PROMPT_LENGTH:
        await query.answer(
            f"⚠️ Промпт слишком длинный ({len(prompt)}). Максимум — {SORA2_MAX_PROMPT_LENGTH} символов.",
            show_alert=True,
        )
        return

    lock_acquired = False
    task_id: Optional[str] = None
    try:
        if user_id is not None:
            lock_acquired = acquire_sora2_lock(int(user_id), ttl=SORA2_LOCK_TTL)
            if not lock_acquired:
                await query.answer("⏳ Задача уже запускается", show_alert=True)
                return

        await query.answer()
        task_id = await _start_sora2_generation(
            ctx,
            chat_id=chat_id,
            user_id=user_id,
            message=message,
            mode=mode,
            prompt=prompt,
            image_urls=image_urls,
            aspect_ratio=aspect_ratio,
        )
    finally:
        if lock_acquired and user_id is not None and not task_id:
            release_sora2_lock(int(user_id))


async def sora2_start_t2v(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await _handle_sora2_start(update, ctx, mode="sora2_ttv")


async def sora2_start_i2v(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await _handle_sora2_start(update, ctx, mode="sora2_itv")

# --------- VEO Card ----------
_PROMPT_PLACEHOLDER = "Введите промпт…"


def veo_card_text(s: Dict[str, Any]) -> str:
    prompt_raw = (s.get("last_prompt") or "").strip()
    prompt_html = html.escape(prompt_raw) if prompt_raw else ""
    aspect = html.escape(s.get("aspect") or "16:9")
    model = "Veo Quality" if s.get("model") == "veo3" else "Veo Fast"
    img = "есть" if s.get("last_image_url") else "нет"
    duration_hint = s.get("veo_duration_hint")
    lip_sync = bool(s.get("veo_lip_sync_required"))
    lines = [
        "🟦 <b>Карточка VEO</b>",
        f"• Формат: <b>{aspect}</b>",
        f"• Модель: <b>{model}</b>",
        f"• Фото-референс: <b>{img}</b>",
    ]
    if duration_hint:
        lines.append(f"• Длительность: <b>{html.escape(str(duration_hint))}</b>")
    if lip_sync:
        lines.append("• <b>lip-sync required</b>")
    if prompt_html:
        code_line = f"<code>{prompt_html}</code>"
    else:
        code_line = "<code> </code>"
    lines.extend([
        "",
        "🖊️ <b>Промпт:</b>",
        code_line,
    ])
    return "\n".join(lines)

def veo_kb(s: Dict[str, Any]) -> InlineKeyboardMarkup:
    aspect = s.get("aspect") or "16:9"
    model = s.get("model") or "veo3_fast"
    ar16 = "✅" if aspect == "16:9" else ""
    ar916 = "✅" if aspect == "9:16" else ""
    fast = "✅" if model != "veo3" else ""
    qual = "✅" if model == "veo3" else ""
    rows = [
        [InlineKeyboardButton("🖼 Добавить/Удалить референс", callback_data="veo:clear_img")],
        [InlineKeyboardButton(f"16:9 {ar16}", callback_data="veo:set_ar:16:9"),
         InlineKeyboardButton(f"9:16 {ar916}", callback_data="veo:set_ar:9:16")],
        [InlineKeyboardButton(f"⚡ Fast {fast}", callback_data="veo:set_model:fast"),
         InlineKeyboardButton(f"💎 Quality {qual}", callback_data="veo:set_model:quality")],
        [InlineKeyboardButton("🚀 Сгенерировать", callback_data="veo:start")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="back")],
    ]
    return InlineKeyboardMarkup(rows)

# ==========================
#   VEO
# ==========================
def _endpoint_cache_key(service: str, kind: str) -> str:
    return f"{service}:endpoint:{kind}"


def _remember_endpoint(service: str, kind: str, path: str):
    if not path:
        return
    global KIE_VEO_STATUS_PATH, KIE_VEO_1080_PATH, KIE_MJ_GENERATE, KIE_MJ_STATUS
    app_cache[_endpoint_cache_key(service, kind)] = path
    if service == "veo":
        if kind == "status":
            KIE_VEO_STATUS_PATH = path
        elif kind == "1080":
            KIE_VEO_1080_PATH = path
    elif service == "mj":
        if kind == "generate":
            KIE_MJ_GENERATE = path
        elif kind == "status":
            KIE_MJ_STATUS = path


def _endpoint_candidates(service: str, kind: str, base_paths: List[str]) -> List[str]:
    cached = app_cache.get(_endpoint_cache_key(service, kind))
    if cached:
        return _normalize_endpoint_values(cached, base_paths)
    return list(base_paths)

def _is_not_found_response(status: int, payload: Dict[str, Any]) -> bool:
    if status == 404:
        return True
    for key in ("code", "status"):
        val = payload.get(key)
        if isinstance(val, int) and val == 404:
            return True
        if isinstance(val, str) and val.strip() == "404":
            return True
    message = payload.get("message") or payload.get("error")
    if isinstance(message, str) and "not found" in message.lower():
        return True
    return False


def _poll_section(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if isinstance(value, Mapping):
        return value
    return {}


def _poll_tracks(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    data_section = _poll_section(payload, "data")
    response_section = _poll_section(data_section, "response")
    tracks_candidate = (
        response_section.get("sunoData")
        or response_section.get("tracks")
        or data_section.get("sunoData")
        or data_section.get("tracks")
    )
    if isinstance(tracks_candidate, list):
        normalized: List[Dict[str, Any]] = []
        for item in tracks_candidate:
            if isinstance(item, Mapping):
                normalized.append(dict(item))
        return normalized
    return []


def _poll_tags(payload: Mapping[str, Any]) -> Optional[str]:
    data_section = _poll_section(payload, "data")
    response_section = _poll_section(data_section, "response")
    tags = response_section.get("tags") or data_section.get("tags")
    if isinstance(tags, str):
        return tags
    if isinstance(tags, list):
        return ", ".join(str(tag) for tag in tags if tag not in (None, "")) or None
    return None


def _poll_duration(payload: Mapping[str, Any]) -> Optional[float]:
    data_section = _poll_section(payload, "data")
    response_section = _poll_section(data_section, "response")
    for section in (response_section, data_section, payload):
        for key in ("duration", "durationSec", "duration_seconds"):
            value = section.get(key)
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                try:
                    return float(value)
                except ValueError:
                    continue
    return None

def _kie_request_with_endpoint(
    service: str,
    kind: str,
    method: str,
    paths: List[str],
    *,
    request_id: Optional[str] = None,
    **kwargs: Any,
) -> Tuple[int, Dict[str, Any], str, str]:
    candidates = _endpoint_candidates(service, kind, paths)
    if not candidates:
        return 0, {"error": "no endpoint configured"}, request_id or "", ""

    current_request_id = request_id
    last_status = 0
    last_resp: Dict[str, Any] = {}
    last_req_id = request_id or ""
    last_path = candidates[0]

    for idx, path in enumerate(candidates):
        status, resp, req_id = _kie_request(
            method,
            path,
            request_id=current_request_id,
            **kwargs,
        )
        if current_request_id is None:
            current_request_id = req_id
        if not _is_not_found_response(status, resp):
            if idx > 0:
                kie_event(
                    "ENDPOINT_SWITCH",
                    service=service,
                    kind=kind,
                    method=method,
                    path=path,
                    attempts=idx + 1,
                )
            _remember_endpoint(service, kind, path)
            return status, resp, req_id, path
        if idx + 1 < len(candidates):
            kie_event(
                "ENDPOINT_FALLBACK",
                service=service,
                kind=kind,
                method=method,
                path=path,
                status=status,
                body_status=resp.get("status"),
                body_code=resp.get("code"),
            )
        last_status, last_resp, last_req_id, last_path = status, resp, req_id, path

    return last_status, last_resp, last_req_id, last_path

def _build_payload_for_veo(prompt: str, aspect: str, image_url: Optional[str], model_key: str) -> Dict[str, Any]:
    aspect_ratio = "9:16" if aspect == "9:16" else "16:9"
    model = "veo3" if model_key == "veo3" else "veo3_fast"
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "aspectRatio": aspect_ratio,
        "enableFallback": aspect == "16:9",
        "input": {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "aspectRatio": aspect_ratio,
            "enable_fallback": aspect == "16:9",
        },
    }
    if image_url:
        payload["imageUrls"] = [image_url]
        payload["input"]["image_urls"] = [image_url]
    return payload

def submit_kie_veo(prompt: str, aspect: str, image_url: Optional[str], model_key: str) -> Tuple[bool, Optional[str], str]:
    payload = _build_payload_for_veo(prompt, aspect, image_url, model_key)
    log.info(
        "KIE SUBMIT | model=%s | aspect=%s | has_image=%s",
        payload.get("model"),
        payload.get("aspectRatio"),
        bool(image_url),
    )
    status, resp, req_id = _kie_request("POST", KIE_VEO_GEN_PATH, json_payload=payload)
    code = _extract_response_code(resp, status)
    tid = _extract_task_id(resp)
    message = resp.get("msg") or resp.get("message")
    kie_event("SUBMIT", request_id=req_id, status=status, code=code, task_id=tid, message=message)
    if status == 200 and code == 200:
        if tid:
            log.info("KIE_SUBMIT ok: task_id=%s", tid)
            _remember_kie_request_id(tid, req_id)
            return True, tid, "Задача создана."
        return False, None, "Ответ сервиса без taskId."
    error_msg = message or resp.get("error") or str(resp)
    return False, None, f"Ошибка VEO: {error_msg}"

def get_kie_veo_status(task_id: str) -> Tuple[bool, Optional[int], Optional[str], Optional[str]]:
    req_id_hint = _get_kie_request_id(task_id)
    status, resp, req_id, path_used = _kie_request_with_endpoint(
        "veo",
        "status",
        "GET",
        KIE_VEO_STATUS_PATHS,
        params={"taskId": task_id},
        request_id=req_id_hint,
    )
    if not req_id_hint:
        _remember_kie_request_id(task_id, req_id)
    code = _extract_response_code(resp, status)
    data_raw = resp.get("data") or {}
    if isinstance(data_raw, str):
        try:
            data = json.loads(data_raw)
        except Exception:
            data = {"raw": data_raw}
    elif isinstance(data_raw, dict):
        data = data_raw
    else:
        data = {"value": data_raw}
    flag = _parse_success_flag(data)
    message = resp.get("msg") or resp.get("message")
    url = _extract_result_url(data)
    kie_event(
        "STATUS",
        request_id=req_id,
        task_id=task_id,
        status=status,
        code=code,
        flag=flag,
        has_url=bool(url),
        path=path_used,
    )
    if status == 200 and code == 200:
        return True, flag, message, url
    return False, None, f"Ошибка статуса VEO: {resp}", None


def fetch_1080p_result_url(task_id: str, index: Optional[int] = None) -> Tuple[Optional[str], Dict[str, Any]]:
    params: Dict[str, Any] = {"taskId": task_id}
    if index is not None:
        params["index"] = index

    url = join_url(KIE_BASE_URL, KIE_VEO_1080_PATH)
    meta: Dict[str, Any] = {"taskId": task_id, "index": index, "http_status": None, "code": None, "resultUrl": None}

    try:
        resp = KIE_1080_SESSION.get(url, params=params, timeout=120)
    except requests.RequestException as exc:
        meta.update({"error": str(exc)})
        kie_event("1080_FETCH_ERROR", **meta)
        return None, meta

    meta["http_status"] = resp.status_code
    try:
        payload = resp.json()
        if not isinstance(payload, dict):
            payload = {"data": payload}
    except ValueError:
        meta.update({"error": "non_json_response"})
        kie_event("1080_FETCH_PARSE", **meta)
        return None, meta

    data = payload.get("data") or {}
    if isinstance(data, dict):
        result_url = data.get("resultUrl") or data.get("result_url")
    else:
        result_url = None

    meta.update({
        "code": payload.get("code"),
        "message": payload.get("msg") or payload.get("message"),
        "resultUrl": result_url,
    })

    kie_event("1080_FETCH", **meta)
    if resp.status_code == 200 and payload.get("code") == 200 and isinstance(result_url, str) and result_url.startswith("http"):
        return result_url, meta
    return None, meta


def download_file(url: str, *, task_id: Optional[str] = None, max_attempts: int = 3) -> Path:
    last_error: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        tmp_path = Path(tempfile.gettempdir()) / f"{uuid.uuid4()}.mp4"
        total_bytes = 0
        try:
            with KIE_1080_SESSION.get(url, stream=True, timeout=600) as resp:
                resp.raise_for_status()
                with tmp_path.open("wb") as fh:
                    for chunk in resp.iter_content(1024 * 1024):
                        if chunk:
                            total_bytes += len(chunk)
                            fh.write(chunk)
            if total_bytes > 0:
                log.info(
                    "KIE DOWNLOAD | task=%s | attempt=%d | bytes=%d",
                    task_id or "?",
                    attempt,
                    total_bytes,
                )
                return tmp_path
            log.warning(
                "KIE DOWNLOAD empty | task=%s | attempt=%d | url=%s",
                task_id or "?",
                attempt,
                url,
            )
            last_error = RuntimeError("empty download")
        except Exception as exc:
            last_error = exc
            log.warning(
                "KIE DOWNLOAD fail | task=%s | attempt=%d | error=%s",
                task_id or "?",
                attempt,
                exc,
            )
        finally:
            if total_bytes == 0 and tmp_path.exists():
                with suppress(Exception):
                    tmp_path.unlink()
        if attempt < max_attempts:
            time.sleep(1)
    if last_error:
        raise RuntimeError("Failed to download file") from last_error
    raise RuntimeError("Failed to download file")


def probe_size(path: Path) -> Optional[Tuple[int, int]]:
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=p=0",
                str(path),
            ],
            text=True,
        ).strip()
        width, height = out.split(",")
        return int(width), int(height)
    except Exception:
        log.warning("ffprobe not available; skip size probe")
        return None


async def send_kie_1080p_to_tg(
    ctx: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    task_id: str,
    index: Optional[int],
    fallback_url: Optional[str],
    is_vertical: bool,
) -> bool:
    hd_url, meta = fetch_1080p_result_url(task_id, index)
    chosen_url = hd_url or fallback_url

    if not hd_url:
        reason = meta.get("error") or meta.get("message")
        if reason:
            kie_event(
                "1080_UNAVAILABLE",
                taskId=task_id,
                index=index,
                reason=reason,
                code=meta.get("code"),
                http_status=meta.get("http_status"),
            )

    if not chosen_url:
        await ctx.bot.send_message(chat_id, "❌ Не удалось получить видео.")
        return False

    try:
        path = download_file(chosen_url, task_id=task_id)
    except Exception as exc:
        kie_event("1080_DOWNLOAD_FAIL", taskId=task_id, index=index, error=str(exc))
        await ctx.bot.send_message(chat_id, "❌ Не удалось отправить видео.")
        return False

    try:
        wh = probe_size(path)
        width, height = (wh if wh else (None, None))
        resolution = f"{width}x{height}" if width and height else None
        expected = (1080, 1920) if is_vertical else (1920, 1080)
        if width and height and (width, height) != expected:
            log.warning(
                "Unexpected video size: %s×%s (expected %s×%s)",
                width,
                height,
                expected[0],
                expected[1],
            )

        kie_event(
            "1080_LOCAL",
            taskId=task_id,
            index=index,
            http_status=meta.get("http_status"),
            code=meta.get("code"),
            resultUrl=chosen_url,
            local_path=str(path),
            resolution=resolution,
            width=width,
            height=height,
        )

        with path.open("rb") as fh:
            input_file = InputFile(fh, filename="veo_result.mp4")
            try:
                await ctx.bot.send_video(
                    chat_id=chat_id,
                    video=input_file,
                    supports_streaming=True,
                )
                log.info(
                    "TG SEND_VIDEO OK | task=%s | chat=%s | method=video",
                    task_id,
                    chat_id,
                )
            except Exception as send_exc:
                log.warning("send_video failed, fallback to document: %s", send_exc)
                fh.seek(0)
                try:
                    await ctx.bot.send_document(
                        chat_id=chat_id,
                        document=InputFile(fh, filename="veo_result.mp4"),
                    )
                    log.info(
                        "TG SEND_VIDEO OK | task=%s | chat=%s | method=document",
                        task_id,
                        chat_id,
                    )
                except Exception as doc_exc:
                    log.exception("send_document fallback failed: %s", doc_exc)
                    await ctx.bot.send_message(chat_id, "❌ Не удалось отправить видео.")
                    return False
        return True
    finally:
        with suppress(Exception):
            path.unlink()


# ==========================
#   MJ
# ==========================
def _parse_status_code_value(value: Any, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return 200 if value else 500
    if isinstance(value, (int, float)):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return int(text)
        except ValueError:
            try:
                return int(float(text))
            except (TypeError, ValueError):
                pass
        lowered = text.lower()
        mapping = {
            "success": 200,
            "ok": 200,
            "succeeded": 200,
            "finished": 200,
            "done": 200,
            "pending": 102,
            "processing": 102,
            "queued": 102,
            "fail": 500,
            "failed": 500,
            "error": 500,
            "denied": 403,
            "forbidden": 403,
            "unauthorized": 401,
            "timeout": 504,
            "notfound": 404,
            "not_found": 404,
        }
        normalized = lowered.replace("-", "").replace("_", "")
        if normalized in mapping:
            return mapping[normalized]
        if lowered in mapping:
            return mapping[lowered]
        match = re.search(r"\d+", text)
        if match:
            try:
                return int(match.group(0))
            except ValueError:
                return default
    return default


def _extract_response_code(payload: Dict[str, Any], http_status: int) -> int:
    if not isinstance(payload, dict):
        return http_status
    for key in ("code", "statusCode", "status_code", "errorCode", "error_code", "resultCode"):
        if key in payload:
            return _parse_status_code_value(payload.get(key), http_status)
    status_val = payload.get("status")
    if isinstance(status_val, str):
        return _parse_status_code_value(status_val, http_status)
    return http_status


def _kie_error_message(status_code: int, j: Dict[str, Any]) -> str:
    code = _extract_response_code(j, status_code)
    msg = j.get("msg") or j.get("message") or j.get("error") or ""
    mapping = {
        400: "Неверный запрос.",
        401: "Доступ запрещён.",
        402: "Недостаточно кредитов.",
        404: "Задача не найдена.",
        422: "Запрос отклонён модерацией.",
        429: "Превышен лимит.",
        500: "Внутренняя ошибка сервиса.",
        504: "Таймаут сервиса.",
    }
    base = mapping.get(code, f"Код ошибки {code}.")
    return f"{base} {msg}".strip()

def mj_generate(prompt: str, aspect: str) -> Tuple[bool, Optional[str], str]:
    aspect_ratio = "9:16" if aspect == "9:16" else "16:9"
    payload = {
        "taskType": "mj_txt2img",
        "prompt": prompt,
        "speed": "fast",
        "aspectRatio": aspect_ratio,
        "version": "7",
        "enableTranslation": True,
        "input": {
            "prompt": prompt,
            "aspectRatio": aspect_ratio,
            "aspect_ratio": aspect_ratio,
        },
    }
    status, resp, req_id, path_used = _kie_request_with_endpoint(
        "mj",
        "generate",
        "POST",
        KIE_MJ_GENERATE_PATHS,
        json_payload=payload,
    )
    code = _extract_response_code(resp, status)
    tid = _extract_task_id(resp)
    kie_event(
        "MJ_SUBMIT",
        request_id=req_id,
        status=status,
        code=code,
        task_id=tid,
        aspect=aspect_ratio,
        path=path_used,
    )
    if status == 200 and code == 200:
        if tid:
            return True, tid, "MJ задача создана."
        return False, None, "Ответ MJ без taskId."
    return False, None, _kie_error_message(status, resp)


def mj_generate_img2img(
    file_url: str,
    *,
    aspect_ratio: str = "1:1",
    speed: str = "fast",
    version: str = "7",
    enable_translation: bool = False,
    prompt: str = "",
) -> Tuple[bool, Optional[str], str]:
    aspect_value = aspect_ratio if aspect_ratio in {"16:9", "9:16", "3:2", "2:3", "4:5", "5:4", "7:4", "4:7", "1:1"} else "1:1"
    payload = {
        "taskType": "mj_img2img",
        "prompt": prompt,
        "speed": speed,
        "aspectRatio": aspect_value,
        "version": version,
        "enableTranslation": enable_translation,
        "fileUrls": [file_url],
        "input": {
            "prompt": prompt,
            "aspectRatio": aspect_value,
            "fileUrls": [file_url],
        },
    }
    status, resp, req_id, path_used = _kie_request_with_endpoint(
        "mj",
        "generate",
        "POST",
        KIE_MJ_GENERATE_PATHS,
        json_payload=payload,
    )
    code = _extract_response_code(resp, status)
    tid = _extract_task_id(resp)
    kie_event(
        "MJ_IMG2IMG_SUBMIT",
        request_id=req_id,
        status=status,
        code=code,
        task_id=tid,
        aspect=aspect_value,
        path=path_used,
    )
    if status == 200 and code == 200 and tid:
        return True, tid, "MJ img2img задача создана."
    return False, None, _kie_error_message(status, resp)


def mj_generate_upscale(
    task_id: str,
    image_index: int,
    *,
    prompt: Optional[str] = None,
    callback_url: Optional[str] = None,
    watermark: bool = False,
) -> Tuple[bool, Optional[str], str]:
    payload: Dict[str, Any] = {
        "taskId": task_id,
        "imageIndex": int(image_index),
        "waterMark": bool(watermark),
    }
    if isinstance(prompt, str):
        prompt_value = prompt.strip()
        if prompt_value:
            payload["prompt"] = prompt_value
    if callback_url:
        payload["callBackUrl"] = callback_url
    status, resp, req_id, path_used = _kie_request_with_endpoint(
        "mj",
        "upscale",
        "POST",
        KIE_MJ_UPSCALE_PATHS,
        json_payload=payload,
    )
    code = _extract_response_code(resp, status)
    tid = _extract_task_id(resp)
    kie_event(
        "MJ_UPSCALE_SUBMIT",
        request_id=req_id,
        status=status,
        code=code,
        task_id=tid,
        source_task_id=task_id,
        image_index=image_index,
        path=path_used,
    )
    if status == 200 and code == 200 and tid:
        return True, tid, "MJ апскейл запущен."
    return False, None, _kie_error_message(status, resp)

def mj_status(task_id: str) -> Tuple[bool, Optional[int], Optional[Dict[str, Any]]]:
    status, resp, req_id, path_used = _kie_request_with_endpoint(
        "mj",
        "status",
        "GET",
        KIE_MJ_STATUS_PATHS,
        params={"taskId": task_id},
    )
    code = _extract_response_code(resp, status)
    raw_data = resp.get("data")
    if isinstance(raw_data, str):
        try:
            parsed = json.loads(raw_data)
            data = parsed if isinstance(parsed, dict) else {"value": parsed}
        except Exception:
            data = {"raw": raw_data}
    elif isinstance(raw_data, dict):
        data = raw_data
    elif isinstance(raw_data, list):
        data = next((item for item in raw_data if isinstance(item, dict)), None)
        if data is None:
            data = {"value": raw_data}
    else:
        data = None
    flag = _parse_success_flag(data) if isinstance(data, dict) else None
    not_found = _is_not_found_response(status, resp)
    if not_found:
        flag = 0
    kie_event(
        "MJ_STATUS",
        request_id=req_id,
        task_id=task_id,
        status=status,
        code=code,
        flag=flag,
        path=path_used,
        not_found=not_found,
    )
    if not_found:
        return True, 0, None
    if status == 200 and code == 200:
        return True, flag, data if isinstance(data, dict) else None
    return False, None, None

_MJ_PROMPT_KEYS: Tuple[str, ...] = (
    "prompt",
    "promptEn",
    "prompt_en",
    "prompt_en_translate",
    "promptTranslateEn",
    "promptTranslate",
    "translatedPrompt",
    "translated_prompt",
    "finalPrompt",
    "final_prompt",
)


def _extract_mj_prompt(source: Any) -> Optional[str]:
    def _from_dict(data: Dict[str, Any]) -> Optional[str]:
        for key in _MJ_PROMPT_KEYS:
            if key in data:
                value = data.get(key)
                if isinstance(value, str):
                    trimmed = value.strip()
                    if trimmed:
                        return trimmed
        return None

    if isinstance(source, dict):
        direct = _from_dict(source)
        if direct:
            return direct
        nested_keys = (
            "resultInfoJson",
            "resultInfo",
            "resultJson",
            "meta",
            "metadata",
            "extraInfo",
            "extraJson",
        )
        for meta_key in nested_keys:
            raw = source.get(meta_key)
            stack: List[Any] = [raw]
            while stack:
                current = stack.pop()
                if current is None:
                    continue
                if isinstance(current, str):
                    try:
                        parsed = json.loads(current)
                    except Exception:
                        continue
                    stack.append(parsed)
                    continue
                if isinstance(current, dict):
                    nested_prompt = _from_dict(current)
                    if nested_prompt:
                        return nested_prompt
                    stack.extend(current.values())
                elif isinstance(current, list):
                    stack.extend(current)
    return None


def _extract_mj_image_urls(status_data: Dict[str, Any]) -> List[str]:
    res: List[str] = []
    seen: set[str] = set()

    def _add_from(value: Any) -> None:
        for url in _coerce_url_list(value):
            if url not in seen:
                seen.add(url)
                res.append(url)

    direct_keys = (
        "imageUrls",
        "imageUrl",
        "imageUrlList",
        "image_url",
        "image_urls",
        "resultUrls",
        "resultUrl",
        "urls",
    )
    for key in direct_keys:
        if key in status_data:
            _add_from(status_data.get(key))

    for meta_key in ("resultInfoJson", "resultInfo", "resultJson"):
        raw = status_data.get(meta_key)
        if not raw:
            continue
        parsed: Any = raw
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except Exception:
                continue
        if isinstance(parsed, dict):
            for key in direct_keys:
                if key in parsed:
                    _add_from(parsed.get(key))
        elif isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    for key in direct_keys:
                        if key in item:
                            _add_from(item.get(key))

    return res

def _mj_should_retry(msg: Optional[str]) -> bool:
    if not msg: return False
    m = msg.lower()
    retry_tokens = (
        "no response from midjourney official website",
        "timeout",
        "server error",
        "timed out",
        "gateway",
        "504",
    )
    return any(token in m for token in retry_tokens)


def _mj_error_message(payload: Optional[Dict[str, Any]]) -> str:
    if isinstance(payload, dict):
        for key in ("errorMessage", "error_message", "message", "reason", "statusMsg"):
            value = payload.get(key)
            if isinstance(value, str):
                stripped = value.strip()
                if stripped:
                    return stripped
    return "No response from MidJourney Official Website after multiple attempts."


def _make_mj_upscale_filename(result_url: Optional[str], index: int) -> str:
    return _mj_document_filename(index + 1, result_url, suffix="upscaled")


async def _wait_for_mj_grid_result(
    task_id: str,
    *,
    chat_id: int,
    ctx: ContextTypes.DEFAULT_TYPE,
    user_id: int,
    locale: str,
    max_wait: int = 12 * 60,
) -> Tuple[bool, Optional[List[str]], Optional[str], Optional[str], Optional[Dict[str, Any]]]:
    start_ts = time.time()
    delay = 12
    while True:
        ok, flag, data = await asyncio.to_thread(mj_status, task_id)
        if not ok:
            return False, None, "status_error", "MJ: сервис недоступен.", None
        if flag == 0:
            if time.time() - start_ts > max_wait:
                return False, None, "timeout", "⌛ MJ долго отвечает.", None
            await asyncio.sleep(delay)
            delay = min(delay + 6, 30)
            continue
        payload = data if isinstance(data, dict) else {}
        if flag in (2, 3) or flag is None:
            return False, None, "error", _mj_error_message(payload), payload
        if flag == 1:
            urls = _extract_mj_image_urls(payload)
            if not urls:
                single = _extract_result_url(payload)
                urls = [single] if single else []
            if not urls:
                return False, None, "empty", "MJ вернул пустой результат.", payload
            return True, urls, None, None, payload


async def _poll_and_send_upscaled_image(
    chat_id: int,
    ctx: ContextTypes.DEFAULT_TYPE,
    *,
    user_id: int,
    upscale_task_id: str,
    origin_task_id: str,
    image_index: int,
    locale: str,
    result_count: int,
    source: str,
) -> Dict[str, Any]:
    start_ts = time.time()
    delay = 10
    max_wait = 10 * 60
    while True:
        ok, flag, data = await asyncio.to_thread(mj_status, upscale_task_id)
        if not ok:
            return {"ok": False, "reason": "status_error", "message": "MJ: сервис недоступен."}
        if flag == 0:
            if time.time() - start_ts > max_wait:
                return {"ok": False, "reason": "timeout", "message": "⌛ MJ долго отвечает."}
            await asyncio.sleep(delay)
            delay = min(delay + 4, 30)
            continue
        payload = data if isinstance(data, dict) else {}
        if flag in (2, 3) or flag is None:
            return {"ok": False, "reason": "error", "message": _mj_error_message(payload)}
        if flag == 1:
            result_url = _extract_result_url(payload)
            if not result_url:
                urls = _extract_mj_image_urls(payload)
                result_url = urls[0] if urls else None
            if not result_url:
                return {"ok": False, "reason": "empty", "message": "MJ вернул пустой результат."}
            filename = _make_mj_upscale_filename(result_url, image_index)
            try:
                document = InputFile(result_url, filename)
                await ctx.bot.send_document(
                    chat_id,
                    document=document,
                )
            except Exception as exc:
                return {
                    "ok": False,
                    "reason": "send_failed",
                    "message": f"Не удалось отправить изображение: {exc}",
                }
            duration_ms = int((time.time() - start_ts) * 1000)
            event(
                "MJ_UPSCALE_RESULT",
                mode="image_upscale",
                source=source,
                task_id=origin_task_id,
                upscale_task_id=upscale_task_id,
                image_index=image_index,
                duration_ms=duration_ms,
                success=True,
                user_id=user_id,
            )
            return {"ok": True, "duration_ms": duration_ms}


async def _launch_mj_upscale(
    chat_id: int,
    ctx: ContextTypes.DEFAULT_TYPE,
    *,
    user_id: int,
    grid: Dict[str, Any],
    image_index: int,
    locale: str,
    source: str,
    charge_tokens: bool = True,
    already_charged: bool = False,
    balance_after: Optional[int] = None,
) -> bool:
    s = state(ctx)
    origin_task_id = grid.get("task_id")
    result_urls = grid.get("result_urls") or []
    if not isinstance(origin_task_id, str) or not result_urls:
        await ctx.bot.send_message(chat_id, _mj_ui_text("upscale_need_photo", locale))
        return False

    prompt_text = ""
    grid_prompt = grid.get("prompt") if isinstance(grid, dict) else None
    if isinstance(grid_prompt, str):
        prompt_text = grid_prompt.strip()
    if not prompt_text:
        prompt_text = _extract_mj_prompt(grid) or ""
    if not prompt_text:
        prompt_text = (s.get("last_prompt") or "").strip()
    if not prompt_text:
        prompt_text = MJ_UPSCALE_PROMPT_PLACEHOLDER

    price = PRICE_MJ_UPSCALE
    charged_here = False
    charged_total = already_charged
    lock_acquired = False
    start_time = time.time()

    def _log_failure(reason: str, message: str) -> None:
        duration_ms = int((time.time() - start_time) * 1000)
        event(
            "MJ_UPSCALE_RESULT",
            mode="image_upscale",
            source=source,
            task_id=origin_task_id,
            image_index=image_index,
            duration_ms=duration_ms,
            success=False,
            reason=reason,
            user_id=user_id,
        )

    try:
        if not acquire_mj_upscale_lock(user_id, origin_task_id, image_index):
            text = _mj_ui_text("upscale_in_progress", locale)
            await ctx.bot.send_message(chat_id, text or "⏳ Уже выполняю апскейл этого кадра.")
            return False
        lock_acquired = True

        if charge_tokens:
            if not await ensure_tokens(ctx, chat_id, user_id, price):
                return False
            ok, balance_value = debit_try(
                user_id,
                price,
                reason="service:start",
                meta={
                    "service": "MJ_UPSCALE",
                    "task_id": origin_task_id,
                    "image_index": image_index,
                    "source": source,
                },
            )
            if not ok:
                await ensure_tokens(ctx, chat_id, user_id, price)
                return False
            charged_here = True
            charged_total = True
            balance_after = balance_value if isinstance(balance_value, int) else None
            if balance_after is not None:
                try:
                    await show_balance_notification(
                        chat_id,
                        ctx,
                        user_id,
                        f"✅ Списано {price}💎. Баланс: {balance_after}💎 — запускаю апскейл…",
                    )
                except Exception:
                    pass

        processing_text = _mj_ui_text("upscale_processing", locale)
        if processing_text:
            try:
                await ctx.bot.send_message(chat_id, processing_text)
            except Exception:
                pass

        ok_submit, new_task_id, submit_msg = await asyncio.to_thread(
            mj_generate_upscale,
            origin_task_id,
            image_index,
            prompt=prompt_text,
        )
        if not ok_submit or not new_task_id:
            new_balance: Optional[int] = None
            if charged_total:
                try:
                    new_balance = credit_balance(
                        user_id,
                        price,
                        reason="service:refund",
                        meta={
                            "service": "MJ_UPSCALE",
                            "task_id": origin_task_id,
                            "image_index": image_index,
                            "reason": "submit_failed",
                        },
                    )
                except Exception as exc:
                    log.exception("MJ upscale submit refund failed for %s: %s", user_id, exc)
            err_text = f"❌ MJ: {submit_msg or 'Не удалось запустить апскейл.'}"
            try:
                await ctx.bot.send_message(chat_id, err_text)
            except Exception:
                pass
            if new_balance is not None:
                try:
                    await show_balance_notification(
                        chat_id,
                        ctx,
                        user_id,
                        f"⚠️ Ошибка. Возврат {price}💎. Баланс: {new_balance}💎",
                    )
                except Exception:
                    pass
            _log_failure("submit_failed", submit_msg or "submit_failed")
            return False

        s["mj_upscale_active"] = {
            "task_id": new_task_id,
            "origin_task_id": origin_task_id,
            "index": image_index,
        }

        result = await _poll_and_send_upscaled_image(
            chat_id,
            ctx,
            user_id=user_id,
            upscale_task_id=new_task_id,
            origin_task_id=origin_task_id,
            image_index=image_index,
            locale=locale,
            result_count=len(result_urls),
            source=source,
        )
        if result.get("ok"):
            return True

        reason = str(result.get("reason") or "error")
        message = str(result.get("message") or "MJ апскейл не удался.")
        new_balance = None
        if charged_total:
            try:
                new_balance = credit_balance(
                    user_id,
                    price,
                    reason="service:refund",
                    meta={
                        "service": "MJ_UPSCALE",
                        "task_id": origin_task_id,
                        "image_index": image_index,
                        "reason": reason,
                    },
                )
            except Exception as exc:
                log.exception("MJ upscale refund failed for %s: %s", user_id, exc)
        text = f"❌ MJ: {message}"
        if new_balance is not None:
            text += "\n💎 Токены возвращены."
        try:
            await ctx.bot.send_message(chat_id, text)
        except Exception:
            pass
        if new_balance is not None:
            try:
                await show_balance_notification(
                    chat_id,
                    ctx,
                    user_id,
                    f"⚠️ Ошибка. Возврат {price}💎. Баланс: {new_balance}💎",
                )
            except Exception:
                pass
        _log_failure(reason, message)
        return False
    finally:
        if lock_acquired:
            release_mj_upscale_lock(user_id, origin_task_id, image_index)
        current_state = state(ctx)
        current_state["mj_upscale_active"] = None


async def handle_mj_upscale_menu(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query is None or not query.data:
        return

    prefix, _, remainder = query.data.partition(":")
    if prefix != "mj.upscale.menu":
        return

    grid_id = remainder.strip()

    try:
        await query.answer()
    except Exception:
        pass

    message = query.message
    chat = update.effective_chat or (message.chat if message else None)
    chat_id = getattr(chat, "id", None)

    if chat_id is not None and message is not None:
        gallery = get_mj_gallery(chat_id, message.message_id)
        if not gallery:
            await ctx.bot.send_message(
                chat_id,
                "Срок действия галереи истёк. Сгенерируйте новый запрос.",
            )
            mj_log.warning(
                "mj.upscale.error",
                extra={
                    "meta": {
                        "chat_id": chat_id,
                        "grid_id": grid_id,
                        "reason": "gallery_missing",
                    }
                },
            )
            return

    grid = _load_mj_grid_snapshot(grid_id)
    if not grid:
        if chat_id is not None:
            await ctx.bot.send_message(
                chat_id,
                "Срок действия набора истёк. Сгенерируйте новый.",
            )
        mj_log.warning(
            "mj.upscale.error",
            extra={"meta": {"chat_id": chat_id, "grid_id": grid_id, "reason": "grid_missing"}},
        )
        return

    urls = grid.get("result_urls") or []
    available = [u for u in urls if isinstance(u, str) and u]
    if not available:
        if chat_id is not None:
            await ctx.bot.send_message(
                chat_id,
                "Срок действия набора истёк. Сгенерируйте новый.",
            )
        mj_log.warning(
            "mj.upscale.error",
            extra={"meta": {"chat_id": chat_id, "grid_id": grid_id, "reason": "no_urls"}},
        )
        return

    markup = getattr(message, "reply_markup", None)
    inline_keyboard = getattr(markup, "inline_keyboard", None) if markup else None
    show_select = True
    if inline_keyboard:
        for row in inline_keyboard:
            for btn in row:
                callback_data = getattr(btn, "callback_data", None)
                if isinstance(callback_data, str) and callback_data.startswith("mj.upscale:"):
                    show_select = False
                    break
            if not show_select:
                break

    try:
        if show_select:
            reply_markup = mj_upscale_select_keyboard(grid_id, count=len(available))
            await query.edit_message_reply_markup(reply_markup=reply_markup)
            mj_log.info(
                "mj.upscale.menu_show",
                extra={
                    "meta": {
                        "chat_id": chat_id,
                        "grid_id": grid_id,
                        "count": len(available),
                    }
                },
            )
        else:
            await query.edit_message_reply_markup(reply_markup=mj_upscale_root_keyboard(grid_id))
    except TelegramError as exc:
        mj_log.warning(
            "mj.upscale.menu_edit_fail",
            extra={
                "meta": {
                    "chat_id": chat_id,
                    "grid_id": grid_id,
                    "error": str(exc),
                }
            },
        )


async def handle_mj_upscale_choice(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query is None or not query.data:
        return

    prefix, _, remainder = query.data.partition(":")
    if prefix != "mj.upscale":
        return

    grid_part, _, index_part = remainder.partition(":")
    grid_id = grid_part.strip()

    try:
        index_value = int(index_part)
    except (TypeError, ValueError):
        try:
            await query.answer("Некорректный номер кадра.", show_alert=True)
        except Exception:
            pass
        mj_log.warning(
            "mj.upscale.error",
            extra={"meta": {"grid_id": grid_id, "reason": "bad_index"}},
        )
        return

    try:
        await query.answer()
    except Exception:
        pass

    message = query.message
    chat = update.effective_chat or (message.chat if message else None)
    chat_id = getattr(chat, "id", None)
    user = query.from_user or update.effective_user
    user_id = user.id if user else None

    lock_key = _MJ_UPSCALE_LOCK_KEY_TMPL.format(grid_id=grid_id, index=index_value)
    if not acquire_ttl_lock(lock_key, _MJ_UPSCALE_LOCK_TTL):
        try:
            await query.answer("Уже запускаю апскейл этого кадра.")
        except Exception:
            pass
        mj_log.info(
            "mj.upscale.lock_active",
            extra={"meta": {"grid_id": grid_id, "index": index_value}},
        )
        return

    try:
        if user_id is None or chat_id is None:
            mj_log.warning(
                "mj.upscale.error",
                extra={
                    "meta": {
                        "grid_id": grid_id,
                        "index": index_value,
                        "reason": "missing_context",
                    }
                },
            )
            if chat_id is not None:
                await ctx.bot.send_message(chat_id, "⚠️ Не удалось определить пользователя для апскейла.")
            return

        gallery = get_mj_gallery(chat_id, message.message_id) if message is not None else None
        if not gallery:
            mj_log.warning(
                "mj.upscale.error",
                extra={
                    "meta": {
                        "chat_id": chat_id,
                        "grid_id": grid_id,
                        "index": index_value,
                        "reason": "gallery_missing",
                    }
                },
            )
            await ctx.bot.send_message(chat_id, "Срок действия галереи истёк. Сгенерируйте новый запрос.")
            return
        if index_value <= 0 or index_value > len(gallery):
            mj_log.warning(
                "mj.upscale.error",
                extra={
                    "meta": {
                        "chat_id": chat_id,
                        "grid_id": grid_id,
                        "index": index_value,
                        "reason": "gallery_index_out_of_range",
                    }
                },
            )
            await ctx.bot.send_message(chat_id, "⚠️ Эта фотография недоступна для апскейла.")
            return

        gallery_entry = gallery[index_value - 1]
        source_url = gallery_entry.get("source_url")

        grid = _load_mj_grid_snapshot(grid_id)
        if not grid:
            mj_log.warning(
                "mj.upscale.error",
                extra={
                    "meta": {
                        "chat_id": chat_id,
                        "grid_id": grid_id,
                        "index": index_value,
                        "reason": "grid_missing",
                    }
                },
            )
            await ctx.bot.send_message(chat_id, "Срок действия набора истёк. Сгенерируйте новый.")
            return

        urls = grid.get("result_urls") or []
        if index_value <= 0 or index_value > len(urls):
            mj_log.warning(
                "mj.upscale.error",
                extra={
                    "meta": {
                        "chat_id": chat_id,
                        "grid_id": grid_id,
                        "index": index_value,
                        "reason": "index_out_of_range",
                    }
                },
            )
            await ctx.bot.send_message(chat_id, "⚠️ Этот кадр недоступен для апскейла.")
            return

        locale = state(ctx).get("mj_locale") or _determine_user_locale(user)
        state(ctx)["mj_locale"] = locale

        mj_log.info(
            "mj.upscale.enqueue",
            extra={
                "meta": {
                    "chat_id": chat_id,
                    "user_id": user_id,
                    "grid_id": grid_id,
                    "index": index_value,
                    "source_url": source_url,
                }
            },
        )

        success = await _launch_mj_upscale(
            chat_id,
            ctx,
            user_id=user_id,
            grid=grid,
            image_index=index_value - 1,
            locale=locale,
            source="grid_menu",
        )

        if not success:
            mj_log.warning(
                "mj.upscale.error",
                extra={
                    "meta": {
                        "chat_id": chat_id,
                        "user_id": user_id,
                        "grid_id": grid_id,
                        "index": index_value,
                        "reason": "launch_failed",
                    }
                },
            )
    finally:
        release_ttl_lock(lock_key)


async def handle_mj_gallery_repeat(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query is None or not query.data:
        return

    prefix, _, _ = query.data.partition(":")
    if prefix != "mj.gallery.again":
        return

    try:
        await query.answer()
    except Exception:
        pass

    chat = update.effective_chat
    chat_id = getattr(chat, "id", None)
    if chat_id is None:
        return

    user = update.effective_user
    locale = state(ctx).get("mj_locale") or _determine_user_locale(user)
    s = state(ctx)
    s["mj_locale"] = locale
    s["mode"] = "mj_txt"
    await show_mj_format_card(chat_id, ctx)


async def handle_mj_gallery_back(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query is None or not query.data:
        return

    if query.data != "mj.gallery.back":
        return

    try:
        await query.answer()
    except Exception:
        pass

    await handle_menu(update, ctx)

async def _handle_mj_upscale_input(
    update: Update,
    ctx: ContextTypes.DEFAULT_TYPE,
    file_url: str,
    *,
    width: Optional[int] = None,
    height: Optional[int] = None,
    source: str,
) -> None:
    message = update.effective_message
    chat = update.effective_chat
    user = update.effective_user
    if message is None or chat is None or user is None:
        return

    chat_id = chat.id
    user_id = user.id
    locale = state(ctx).get("mj_locale") or _determine_user_locale(user)
    s = state(ctx)
    s["mj_locale"] = locale
    aspect_ratio = _guess_aspect_ratio_from_size(width, height)
    price = PRICE_MJ_UPSCALE

    if not await ensure_tokens(ctx, chat_id, user_id, price):
        return
    ok, balance_after = debit_try(
        user_id,
        price,
        reason="service:start",
        meta={
            "service": "MJ_UPSCALE",
            "task_type": "img2img",
            "source": source,
            "aspect": aspect_ratio,
        },
    )
    if not ok:
        await ensure_tokens(ctx, chat_id, user_id, price)
        return

    if isinstance(balance_after, int):
        try:
            await show_balance_notification(
                chat_id,
                ctx,
                user_id,
                f"✅ Списано {price}💎. Баланс: {balance_after}💎 — готовлю апскейл…",
            )
        except Exception:
            pass

    ok_submit, task_id, submit_msg = await asyncio.to_thread(
        mj_generate_img2img,
        file_url,
        aspect_ratio=aspect_ratio,
        enable_translation=False,
    )
    if not ok_submit or not task_id:
        new_balance = None
        try:
            new_balance = credit_balance(
                user_id,
                price,
                reason="service:refund",
                meta={
                    "service": "MJ_UPSCALE",
                    "task_type": "img2img",
                    "reason": "submit_failed",
                    "source": source,
                },
            )
        except Exception as exc:
            log.exception("MJ img2img refund failed for %s: %s", user_id, exc)
        text = f"❌ MJ: {submit_msg or 'Не удалось запустить img2img.'}"
        try:
            await ctx.bot.send_message(chat_id, text)
        except Exception:
            pass
        if new_balance is not None:
            try:
                await show_balance_notification(
                    chat_id,
                    ctx,
                    user_id,
                    f"⚠️ Ошибка. Возврат {price}💎. Баланс: {new_balance}💎",
                )
            except Exception:
                pass
        return

    grid_ok, urls, fail_reason, fail_message, payload = await _wait_for_mj_grid_result(
        task_id,
        chat_id=chat_id,
        ctx=ctx,
        user_id=user_id,
        locale=locale,
    )
    if not grid_ok or not urls:
        new_balance = None
        try:
            new_balance = credit_balance(
                user_id,
                price,
                reason="service:refund",
                meta={
                    "service": "MJ_UPSCALE",
                    "task_type": "img2img",
                    "reason": fail_reason or "error",
                    "source": source,
                },
            )
        except Exception as exc:
            log.exception("MJ img2img result refund failed for %s: %s", user_id, exc)
        text = f"❌ MJ: {fail_message or 'Задача не вернула результат.'}"
        if new_balance is not None:
            text += "\n💎 Токены возвращены."
        try:
            await ctx.bot.send_message(chat_id, text)
        except Exception:
            pass
        if new_balance is not None:
            try:
                await show_balance_notification(
                    chat_id,
                    ctx,
                    user_id,
                    f"⚠️ Ошибка. Возврат {price}💎. Баланс: {new_balance}💎",
                )
            except Exception:
                pass
        return

    prompt_from_payload = _extract_mj_prompt(payload)
    _store_last_mj_grid(s, user_id, task_id, urls, prompt=prompt_from_payload)

    grid_payload: Dict[str, Any] = {"task_id": task_id, "result_urls": urls}
    if prompt_from_payload:
        grid_payload["prompt"] = prompt_from_payload

    await _launch_mj_upscale(
        chat_id,
        ctx,
        user_id=user_id,
        grid=grid_payload,
        image_index=0,
        locale=locale,
        source="from_file",
        charge_tokens=False,
        already_charged=True,
        balance_after=balance_after if isinstance(balance_after, int) else None,
    )

# ==========================
#   VEO strict polling utils
# ==========================
STRICT_POLL_INITIAL_DELAY = 2.0
STRICT_POLL_MAX_DELAY = 20.0
RENDER_FAIL_MESSAGE = "⚠️ Рендер не удался. 💎 Токены возвращены."


async def _strict_poll_kie(
    session: aiohttp.ClientSession,
    task_id: str,
    *,
    timeout_sec: int,
    status_path: str,
) -> Tuple[Optional[str], Dict[str, Any]]:
    url = join_url(KIE_BASE_URL, status_path)
    headers = {**_kie_auth_header(), "Accept": "application/json"}
    params = {"taskId": task_id, "id": task_id}
    timeout_sec = max(5, int(timeout_sec))
    deadline = time.monotonic() + timeout_sec
    delay = STRICT_POLL_INITIAL_DELAY
    attempt = 0
    last_status: Optional[str] = None
    last_flag: Optional[int] = None
    request_timeout = ClientTimeout(total=60)

    while True:
        attempt += 1
        now = time.monotonic()
        if now >= deadline:
            log_evt(
                "KIE_TIMEOUT",
                task_id=task_id,
                last_status=last_status,
                last_flag=last_flag,
                attempts=attempt,
                reason="deadline",
            )
            raise TimeoutError(f"KIE polling timeout after {timeout_sec}s")
        try:
            async with session.get(
                url,
                params=params,
                headers=headers,
                timeout=request_timeout,
            ) as response:
                http_status = response.status
                try:
                    payload = await response.json(content_type=None)
                except Exception:
                    payload = {"raw": await response.text()}
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("KIE status request error (task=%s): %s", task_id, exc)
            await asyncio.sleep(delay)
            delay = min(delay * 1.5, STRICT_POLL_MAX_DELAY)
            continue

        if not isinstance(payload, dict):
            payload = {"value": payload}
        raw_data = payload.get("data")
        data_section = raw_data if isinstance(raw_data, dict) else payload

        status_raw = (
            data_section.get("status")
            or data_section.get("state")
            or payload.get("status")
            or payload.get("state")
            or data_section.get("taskStatus")
            or payload.get("taskStatus")
        )
        status = str(status_raw).strip().lower() if status_raw not in (None, "") else ""
        flag = _parse_success_flag(data_section if isinstance(data_section, dict) else {})

        log_evt(
            "KIE_STATUS",
            task_id=task_id,
            status=status or "unknown",
            flag=flag,
            http_status=http_status,
            attempt=attempt,
        )

        if status in KIE_OK_STATES or flag == 1:
            data_for_url = data_section if isinstance(data_section, dict) else {}
            result_url = _extract_result_url(data_for_url)
            if not result_url:
                log_evt(
                    "KIE_RESULT_EMPTY",
                    task_id=task_id,
                    status=status or "flag_success",
                    flag=flag,
                )
            return result_url, payload

        if status == "timeout":
            log_evt("KIE_TIMEOUT", task_id=task_id, reported=True, flag=flag)
            raise TimeoutError("KIE reported timeout")

        if status in KIE_BAD_STATES or flag in (2, 3):
            log_evt(
                "KIE_RESULT_EMPTY",
                task_id=task_id,
                status=status or "flag_fail",
                flag=flag,
            )
            raise RuntimeError(f"KIE failed with status '{status or flag}'")

        last_status = status or last_status
        last_flag = flag if flag is not None else last_flag
        await asyncio.sleep(delay)
        delay = min(delay * 1.5, STRICT_POLL_MAX_DELAY)


async def _validate_kie_video_asset(
    session: aiohttp.ClientSession,
    video_url: Optional[str],
) -> Tuple[bool, Dict[str, Any]]:
    if not video_url or not isinstance(video_url, str):
        return False, {"reason": "missing"}
    target = video_url.strip()
    if not target or not target.lower().startswith("http"):
        return False, {"reason": "invalid_url"}

    info: Dict[str, Any] = {"url": target}
    fallback_needed = False
    request_timeout = ClientTimeout(total=60)

    try:
        async with session.head(
            target,
            allow_redirects=True,
            timeout=request_timeout,
        ) as resp:
            status = resp.status
            info.update({"status": status, "method": "HEAD"})
            if status >= 400:
                if status in (405, 501):
                    fallback_needed = True
                else:
                    info["error"] = f"head_status_{status}"
                    return False, info
            else:
                length_header = resp.headers.get("Content-Length")
                if length_header is not None:
                    try:
                        length_val = int(length_header)
                        info["content_length"] = length_val
                        if length_val <= 0:
                            info["error"] = "empty_length"
                            return False, info
                    except ValueError:
                        info["content_length"] = length_header
                return True, info
    except ClientResponseError as exc:
        if exc.status in (405, 501):
            fallback_needed = True
            info.update({"status": exc.status, "method": "HEAD"})
        else:
            info.update({"status": exc.status, "error": str(exc), "method": "HEAD"})
            return False, info
    except ClientError as exc:
        fallback_needed = True
        info.update({"error": str(exc), "method": "HEAD"})
    except Exception as exc:
        fallback_needed = True
        info.update({"error": str(exc), "method": "HEAD"})

    if not fallback_needed:
        return False, info

    try:
        headers = {"Range": "bytes=0-0"}
        async with session.get(
            target,
            headers=headers,
            allow_redirects=True,
            timeout=request_timeout,
        ) as resp:
            status = resp.status
            info.update({"status": status, "method": "GET", "range": True})
            if status >= 400:
                info.setdefault("error", f"get_status_{status}")
                return False, info
            chunk = await resp.content.read(1)
            has_chunk = bool(chunk)
            info["has_chunk"] = has_chunk
            if not has_chunk:
                info.setdefault("error", "empty_chunk")
            return has_chunk, info
    except ClientError as exc:
        info.update({"error": str(exc), "method": "GET"})
        return False, info
    except Exception as exc:
        info.update({"error": str(exc), "method": "GET"})
        return False, info

# ==========================
#   VEO polling
# ==========================
async def poll_veo_and_send(
    chat_id: int,
    task_id: str,
    gen_id: str,
    ctx: ContextTypes.DEFAULT_TYPE,
    user_id: int,
    price: int,
    service_name: str,
) -> None:
    original_chat_id = chat_id
    s = state(ctx)
    def _cleanup() -> None:
        ACTIVE_TASKS.pop(original_chat_id, None)
        if s.get("generation_id") == gen_id:
            s["generating"] = False
            s["generation_id"] = None
        _clear_kie_request_id(task_id)
        with suppress(Exception):
            clear_task_meta(task_id)

    async def _refund(reason_tag: str, message: Optional[str] = None) -> Optional[int]:
        meta: Dict[str, Any] = {
            "service": service_name,
            "reason": reason_tag,
            "task_id": task_id,
        }
        if message:
            meta["message"] = message
        try:
            new_balance = credit_balance(
                user_id,
                price,
                reason="service:refund",
                meta=meta,
            )
        except Exception as exc:
            log.exception("VEO refund %s failed for %s: %s", reason_tag, user_id, exc)
            return None
        await show_balance_notification(
            chat_id,
            ctx,
            user_id,
            f"⚠️ Ошибка. Возврат {price}💎. Баланс: {new_balance}💎",
        )
        return new_balance

    async def _send_message_with_retry(
        dest_chat_id: int,
        text: str,
        *,
        reply_markup: Optional[InlineKeyboardMarkup] = None,
        reply_to: Optional[int] = None,
    ):
        params: Dict[str, Any] = {}
        if reply_markup is not None:
            params["reply_markup"] = reply_markup
        if reply_to:
            params["reply_to_message_id"] = reply_to
            params["allow_sending_without_reply"] = True
        while True:
            try:
                return await ctx.bot.send_message(
                    chat_id=dest_chat_id,
                    text=text,
                    **params,
                )
            except RetryAfter as exc:
                delay = getattr(exc, "retry_after", None)
                sleep_for = max(1, int(delay) if delay else 1)
                await asyncio.sleep(sleep_for)
            except BadRequest as exc:
                if "message is the same" in str(exc).lower() and "reply_to_message_id" in params:
                    params.pop("reply_to_message_id", None)
                    params.pop("allow_sending_without_reply", None)
                    continue
                raise

    async def _send_media_with_retry(
        dest_chat_id: int,
        file_path: Path,
        file_size: int,
        *,
        reply_to: Optional[int] = None,
    ):
        params: Dict[str, Any] = {}
        if reply_to:
            params["reply_to_message_id"] = reply_to
            params["allow_sending_without_reply"] = True
        limit_bytes = 48 * 1024 * 1024
        while True:
            try:
                with file_path.open("rb") as fh:
                    input_file = InputFile(fh, filename=file_path.name)
                    if file_size <= limit_bytes:
                        return await ctx.bot.send_video(
                            chat_id=dest_chat_id,
                            video=input_file,
                            supports_streaming=True,
                            **params,
                        )
                    return await ctx.bot.send_document(
                        chat_id=dest_chat_id,
                        document=input_file,
                        **params,
                    )
            except RetryAfter as exc:
                delay = getattr(exc, "retry_after", None)
                sleep_for = max(1, int(delay) if delay else 1)
                await asyncio.sleep(sleep_for)
            except BadRequest as exc:
                if "message is the same" in str(exc).lower() and "reply_to_message_id" in params:
                    params.pop("reply_to_message_id", None)
                    params.pop("allow_sending_without_reply", None)
                    continue
                raise

    async def _poll_record_info() -> str:
        delay = 2.0
        max_delay = 60.0
        deadline = time.monotonic() + 15 * 60
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError("KIE polling timeout after 900s")
            try:
                ok, flag, message, url = await asyncio.to_thread(get_kie_veo_status, task_id)
            except Exception as exc:
                ok, flag, message, url = False, None, str(exc), None
            if ok:
                if flag == 1:
                    status_label = "success"
                elif flag in (2, 3):
                    status_label = "failed"
                elif flag == 0:
                    status_label = "waiting"
                elif flag is None:
                    status_label = "unknown"
                else:
                    status_label = str(flag)
            else:
                status_label = "error"
            log.info("KIE_STATUS task_id=%s status=%s msg=%s", task_id, status_label, (message or ""))
            if ok and flag == 1:
                if url:
                    return url
                raise RuntimeError("KIE success without result url")
            if ok and flag in (2, 3):
                raise RuntimeError(f"KIE task failed: {message or flag}")
            await asyncio.sleep(delay)
            delay = min(delay * 2, max_delay)

    async def _download_video(session: aiohttp.ClientSession, url: str) -> Tuple[Path, int]:
        target_path = Path(f"/tmp/{task_id}.mp4")
        if target_path.exists():
            with suppress(Exception):
                target_path.unlink()
        chunk_size = 1024 * 1024
        try:
            async with session.get(url, timeout=ClientTimeout(total=600)) as response:
                response.raise_for_status()
                with target_path.open("wb") as fh:
                    async for chunk in response.content.iter_chunked(chunk_size):
                        if chunk:
                            fh.write(chunk)
        except Exception:
            with suppress(Exception):
                target_path.unlink()
            raise
        file_size = target_path.stat().st_size
        log.info("KIE_RESULT saved: path=%s, size=%s", target_path, file_size)
        return target_path, file_size

    temp_file: Optional[Path] = None

    try:
        async def _clear_wait(target: Optional[int] = None) -> None:
            chat_to_clear = target if target is not None else original_chat_id
            with suppress(Exception):
                await delete_wait_sticker(ctx, chat_id=chat_to_clear)
            if s.get("video_wait_message_id"):
                s["video_wait_message_id"] = None

        async with aiohttp.ClientSession(timeout=ClientTimeout(total=600)) as session:
            try:
                video_url = await _poll_record_info()
            except TimeoutError as exc:
                log_evt("KIE_TIMEOUT", task_id=task_id, reason="poll_exception", message=str(exc))
                await _refund("timeout", str(exc))
                await _clear_wait()
                await _send_message_with_retry(original_chat_id, RENDER_FAIL_MESSAGE)
                return
            except Exception as exc:
                log.exception("VEO status polling failed: %s", exc)
                await _refund("poll_exception", str(exc))
                await _clear_wait()
                await _send_message_with_retry(original_chat_id, RENDER_FAIL_MESSAGE)
                return

            if s.get("generation_id") != gen_id:
                return

            try:
                meta = load_task_meta(task_id)
            except Exception:
                log.exception("Failed to load task meta for %s", task_id)
                meta = None
            if not meta:
                log.error("TASK_META missing: task_id=%s", task_id)
                raise RuntimeError("task meta missing")

            target_chat_id = int(meta.get("chat_id", original_chat_id))
            reply_to_id_raw = meta.get("message_id")
            try:
                reply_to_id = int(reply_to_id_raw) if reply_to_id_raw is not None else None
            except (TypeError, ValueError):
                reply_to_id = None

            kie_event(
                "FINAL_URL",
                request_id=_get_kie_request_id(task_id),
                task_id=task_id,
                final_url=video_url,
            )

            temp_file, file_size = await _download_video(session, video_url)

            await _clear_wait(target_chat_id)
            await _send_message_with_retry(target_chat_id, "🎞️ Рендер завершён — отправляю файл…", reply_to=reply_to_id)
            sent_message = await _send_media_with_retry(target_chat_id, temp_file, file_size, reply_to=reply_to_id)
            media_kind = "video" if file_size <= 48 * 1024 * 1024 else "document"
            log.info(
                "TG_SENT %s: chat_id=%s, message_id=%s",
                media_kind,
                target_chat_id,
                getattr(sent_message, "message_id", None),
            )

            await _send_message_with_retry(
                target_chat_id,
                "✅ Готово!",
                reply_markup=InlineKeyboardMarkup(
                    [[InlineKeyboardButton("🚀 Сгенерировать ещё видео", callback_data="start_new_cycle")]]
                ),
            )
    except TimeoutError as exc:
        log_evt("KIE_TIMEOUT", task_id=task_id, reason="timeout", message=str(exc))
        await _refund("timeout_final", str(exc))
        await _clear_wait()
        await _send_message_with_retry(original_chat_id, RENDER_FAIL_MESSAGE)
    except Exception as exc:
        log.exception("VEO render failed: %s", exc)
        await _refund("exception", str(exc))
        await _clear_wait()
        await _send_message_with_retry(original_chat_id, RENDER_FAIL_MESSAGE)
    finally:
        if temp_file and temp_file.exists():
            with suppress(Exception):
                temp_file.unlink()
        _cleanup()

# ==========================
#   MJ poll (1 авторетрай)
# ==========================
async def poll_mj_and_send_photos(
    chat_id: int,
    task_id: str,
    ctx: ContextTypes.DEFAULT_TYPE,
    prompt: str,
    aspect: str,
    user_id: int,
    price: int,
) -> None:
    start_ts = time.time()
    delay = 12
    max_wait = 12 * 60
    retried = False
    success = False
    aspect_ratio = "9:16" if aspect == "9:16" else "16:9"
    prompt_for_retry = (prompt or "").strip()
    s = state(ctx)
    s["last_mj_task_id"] = task_id

    async def _clear_wait() -> None:
        with suppress(Exception):
            await delete_wait_sticker(ctx, chat_id=chat_id)

    async def _refund(reason_tag: str, message: Optional[str] = None) -> Optional[int]:
        meta: Dict[str, Any] = {"service": "MJ", "reason": reason_tag, "task_id": task_id}
        if message:
            meta["message"] = message
        try:
            new_balance = credit_balance(
                user_id,
                price,
                reason="service:refund",
                meta=meta,
            )
        except Exception as exc:
            log.exception("MJ refund %s failed for %s: %s", reason_tag, user_id, exc)
            return None
        await show_balance_notification(
            chat_id,
            ctx,
            user_id,
            f"⚠️ Ошибка. Возврат {price}💎. Баланс: {new_balance}💎",
        )
        return new_balance

    try:
        while True:
            ok, flag, data = await asyncio.to_thread(mj_status, task_id)
            if not ok:
                await _refund("status_error")
                await _clear_wait()
                await ctx.bot.send_message(chat_id, "❌ MJ: сервис недоступен. 💎 Токены возвращены.")
                return
            if flag == 0:
                if time.time() - start_ts > max_wait:
                    await _refund("timeout")
                    await _clear_wait()
                    await ctx.bot.send_message(chat_id, "⌛ MJ долго отвечает. 💎 Токены возвращены.")
                    return
                await asyncio.sleep(delay)
                delay = min(delay + 6, 30)
                continue
            if flag in (2, 3) or flag is None:
                err_info = None
                if isinstance(data, dict):
                    err_info = (
                        data.get("errorMessage")
                        or data.get("error_message")
                        or data.get("message")
                        or data.get("reason")
                    )
                if isinstance(err_info, str):
                    err = err_info.strip() or "No response from MidJourney Official Website after multiple attempts."
                else:
                    err = "No response from MidJourney Official Website after multiple attempts."
                if (not retried) and prompt_for_retry and _mj_should_retry(err):
                    retried = True
                    await ctx.bot.send_message(chat_id, "🔁 MJ подвис. Перезапускаю задачу бесплатно…")
                    ok2, new_tid, msg2 = await asyncio.to_thread(mj_generate, prompt_for_retry, aspect_ratio)
                    event("MJ_RETRY_SUBMIT", ok=ok2, task_id=new_tid, msg=msg2)
                    if ok2 and new_tid:
                        task_id = new_tid
                        s["last_mj_task_id"] = new_tid
                        start_ts = time.time()
                        delay = 12
                        continue
                await _refund("error", err)
                await _clear_wait()
                await ctx.bot.send_message(chat_id, f"❌ MJ: {err}\n💎 Токены возвращены.")
                return
            if flag == 1:
                payload = data or {}

                urls = _extract_mj_image_urls(payload)
                if not urls:
                    one_url = _extract_result_url(payload)
                    urls = [one_url] if one_url else []

                if not urls:
                    await _refund("empty")
                    await _clear_wait()
                    await ctx.bot.send_message(chat_id, "⚠️ MJ вернул пустой результат. 💎 Токены возвращены.")
                    return

                _store_last_mj_grid(s, user_id, task_id, urls, prompt=prompt_for_retry)

                sent_successfully = await _deliver_mj_grid_documents(
                    ctx,
                    chat_id=chat_id,
                    user_id=user_id,
                    grid_id=task_id,
                    urls=urls,
                    prompt=prompt_for_retry,
                )

                await _clear_wait()

                if not sent_successfully:
                    await _refund("send_failed")
                    await ctx.bot.send_message(
                        chat_id,
                        "❌ Не удалось отправить изображения MJ. 💎 Токены возвращены.",
                    )
                    return

                success = True
                return
    except Exception as e:
        log.exception("MJ poll crash: %s", e)
        await _refund("exception", str(e))
        with suppress(Exception):
            await delete_wait_sticker(ctx, chat_id=chat_id)
        try:
            await ctx.bot.send_message(chat_id, "💥 Внутренняя ошибка MJ. 💎 Токены возвращены.")
        except Exception:
            pass
    finally:
        s = state(ctx)
        s["mj_generating"] = False
        s["last_mj_task_id"] = None
        s["mj_last_wait_ts"] = 0.0
        s["last_prompt"] = None
        s["mj_locale"] = None
        mid = s.get("last_ui_msg_id_mj")
        if mid:
            final_text = "✅ Midjourney: изображение обработано." if success else "ℹ️ Midjourney: поток завершён."
            try:
                await _safe_edit_message_text(
                    ctx.bot.edit_message_text,
                    chat_id=chat_id,
                    message_id=mid,
                    text=final_text,
                    reply_markup=None,
                )
            except Exception:
                pass
            s["last_ui_msg_id_mj"] = None
            s["_last_text_mj"] = None
# ==========================
#   Handlers
# ==========================
def stars_topup_kb() -> InlineKeyboardMarkup:
    rows: List[List[InlineKeyboardButton]] = []
    for stars in STARS_PACK_ORDER:
        diamonds = STARS_TO_DIAMONDS.get(stars)
        if not diamonds:
            continue
        bonus = max(diamonds - stars, 0)
        cap = f"⭐ {stars} → 💎 {diamonds}" + (f" +{bonus}💎 бонус" if bonus else "")
        rows.append(
            [InlineKeyboardButton(cap, callback_data=f"buy:stars:{stars}:{diamonds}")]
        )
    rows.append([InlineKeyboardButton("🛒 Где купить Stars", url=STARS_BUY_URL)])
    rows.append([InlineKeyboardButton(common_text("topup.menu.back"), callback_data="topup:open")])
    return InlineKeyboardMarkup(rows)


async def handle_topup_callback(
    update: Update,
    ctx: ContextTypes.DEFAULT_TYPE,
    data: str,
) -> bool:
    query = update.callback_query
    chat = update.effective_chat
    message = query.message if query else None
    chat_id = None
    if message is not None:
        chat_id = message.chat_id
    elif chat is not None:
        chat_id = chat.id

    if chat_id is None or query is None:
        return False

    user = update.effective_user
    user_id = user.id if user else None

    normalized = data
    if data == CB_PROFILE_TOPUP:
        normalized = "topup:open"
    elif data == CB_PAY_STARS:
        normalized = "topup:stars"
    elif data == CB_PAY_CARD:
        normalized = "topup:yookassa"

    if data == "back_main":
        await query.answer()
        await show_balance_card(chat_id, ctx)
        return True

    if normalized == "topup:open":
        await query.answer()
        markup = menu_pay_unified()
        text = TXT_TOPUP_CHOOSE
        if message is not None:
            try:
                await safe_edit_message(
                    ctx,
                    chat_id,
                    message.message_id,
                    text,
                    markup,
                    parse_mode=ParseMode.HTML,
                    disable_web_page_preview=True,
                )
            except Exception:
                await ctx.bot.send_message(
                    chat_id=chat_id,
                    text=text,
                    reply_markup=markup,
                )
        else:
            await ctx.bot.send_message(
                chat_id=chat_id,
                text=text,
                reply_markup=markup,
            )
        return True

    if data == CB_PAY_CRYPTO:
        await query.answer()
        base_markup = menu_pay_unified()
        rows = [list(row) for row in base_markup.inline_keyboard]
        rows.insert(
            3,
            [InlineKeyboardButton(TXT_PAY_CRYPTO_OPEN_LINK, url=CRYPTO_PAYMENT_URL)],
        )
        markup = InlineKeyboardMarkup(rows)
        if message is not None:
            try:
                await safe_edit_message(
                    ctx,
                    chat_id,
                    message.message_id,
                    TXT_CRYPTO_COMING_SOON,
                    markup,
                    parse_mode=ParseMode.HTML,
                    disable_web_page_preview=True,
                )
            except Exception:
                await ctx.bot.send_message(
                    chat_id=chat_id,
                    text=TXT_CRYPTO_COMING_SOON,
                    reply_markup=markup,
                )
        else:
            await ctx.bot.send_message(
                chat_id=chat_id,
                text=TXT_CRYPTO_COMING_SOON,
                reply_markup=markup,
            )
        return True

    if normalized == "topup:stars":
        await query.answer()
        text = "\n".join(
            filter(
                None,
                [
                    common_text("topup.stars.title"),
                    common_text("topup.stars.info"),
                ],
            )
        )
        edit_callable = getattr(query, "edit_message_text", None)
        try:
            if callable(edit_callable):
                await _safe_edit_message_text(
                    edit_callable,
                    text,
                    reply_markup=stars_topup_kb(),
                )
            elif message is not None and getattr(ctx.bot, "edit_message_text", None):
                await _safe_edit_message_text(
                    ctx.bot.edit_message_text,
                    chat_id=chat_id,
                    message_id=message.message_id,
                    text=text,
                    reply_markup=stars_topup_kb(),
                )
            else:
                raise AttributeError("edit_message_text not available")
        except Exception:
            await ctx.bot.send_message(
                chat_id=chat_id,
                text=text,
                reply_markup=stars_topup_kb(),
            )
        return True

    if normalized == "topup:yookassa":
        await query.answer()
        text = common_text("topup.yookassa.title")
        edit_callable = getattr(query, "edit_message_text", None)
        try:
            if callable(edit_callable):
                await _safe_edit_message_text(
                    edit_callable,
                    text,
                    reply_markup=yookassa_pack_keyboard(),
                )
            elif message is not None and getattr(ctx.bot, "edit_message_text", None):
                await _safe_edit_message_text(
                    ctx.bot.edit_message_text,
                    chat_id=chat_id,
                    message_id=message.message_id,
                    text=text,
                    reply_markup=yookassa_pack_keyboard(),
                )
            else:
                raise AttributeError("edit_message_text not available")
        except Exception:
            await ctx.bot.send_message(
                chat_id=chat_id,
                text=text,
                reply_markup=yookassa_pack_keyboard(),
            )
        return True

    if normalized.startswith("yk:"):
        pack_id = normalized.split(":", 1)[1]
        await query.answer()
        error_keyboard = InlineKeyboardMarkup(
            [
                [InlineKeyboardButton(common_text("topup.yookassa.retry"), callback_data="topup:yookassa")],
                [InlineKeyboardButton(common_text("topup.menu.back"), callback_data="topup:open")],
            ]
        )
        if user_id is None:
            error_text = common_text("topup.yookassa.error")
            try:
                await _safe_edit_message_text(query.edit_message_text, error_text, reply_markup=error_keyboard)
            except Exception:
                await ctx.bot.send_message(chat_id, error_text, reply_markup=error_keyboard)
            return True
        try:
            payment = yookassa_create_payment(user_id, pack_id)
        except YookassaError as exc:
            log.warning(
                "topup.yookassa.error",
                extra={"meta": {"user_id": user_id, "pack_id": pack_id, "err": str(exc)}},
            )
            error_text = common_text("topup.yookassa.error")
            try:
                await _safe_edit_message_text(query.edit_message_text, error_text, reply_markup=error_keyboard)
            except Exception:
                await ctx.bot.send_message(chat_id, error_text, reply_markup=error_keyboard)
            return True
        except Exception as exc:  # pragma: no cover - unexpected failure
            log.exception(
                "topup.yookassa.exception",
                extra={"meta": {"user_id": user_id, "pack_id": pack_id, "err": str(exc)}},
            )
            error_text = common_text("topup.yookassa.error")
            try:
                await _safe_edit_message_text(query.edit_message_text, error_text, reply_markup=error_keyboard)
            except Exception:
                await ctx.bot.send_message(chat_id, error_text, reply_markup=error_keyboard)
            return True

        text = common_text("topup.yookassa.created")
        keyboard = yookassa_payment_keyboard(payment.confirmation_url)
        try:
            await _safe_edit_message_text(query.edit_message_text, text, reply_markup=keyboard)
        except Exception:
            await ctx.bot.send_message(chat_id, text, reply_markup=keyboard)
        return True

    return False

async def handle_menu(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await ensure_user_record(update)

    query = update.callback_query
    if query is not None:
        with suppress(BadRequest):
            await query.answer()

    s = state(ctx)
    s.update({**DEFAULT_STATE})
    _apply_state_defaults(s)

    chat = update.effective_chat
    user = update.effective_user
    chat_id = chat.id if chat else (user.id if user else None)
    if chat_id is None:
        return

    user_id = user.id if user else None
    await _clear_video_menu_state(chat_id, user_id=user_id, ctx=ctx)
    _clear_pm_menu_state(chat_id, user_id=user_id)
    if user_id:
        set_mode(user_id, False)
        clear_wait(user_id)
    await show_emoji_hub_for_chat(chat_id, ctx, user_id=user_id, replace=True)

async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await ensure_user_record(update)
    uid = update.effective_user.id if update.effective_user else None

    await _handle_referral_deeplink(update, ctx)

    if uid is not None:
        try:
            bonus_result = ledger_storage.grant_signup_bonus(uid, 10)
            _set_cached_balance(ctx, bonus_result.balance)
            if bonus_result.applied and update.message is not None:
                await update.message.reply_text("🎁 Добро пожаловать! Начислил +10💎 на баланс.")
        except Exception as exc:
            log.exception("Signup bonus failed for %s: %s", uid, exc)

    await handle_menu(update, ctx)


async def menu_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await handle_menu(update, ctx)


configure_faq(
    show_main_menu=handle_menu,
    on_root_view=_faq_track_root,
    on_section_view=_faq_track_section,
)


async def cancel_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await ensure_user_record(update)
    user = update.effective_user
    if user:
        clear_wait(user.id)
    await handle_menu(update, ctx)


async def suno_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await ensure_user_record(update)
    message = update.effective_message
    chat = update.effective_chat
    if chat is None:
        return
    user = update.effective_user
    if user:
        set_mode(user.id, False)
        clear_wait(user.id)

    if not _suno_configured():
        await _suno_notify(
            ctx,
            chat.id,
            "⚠️ Suno API не настроен. Обратитесь к поддержке.",
            reply_to=message,
        )
        return

    await suno_entry(chat.id, ctx, force_new=True)

    if user:
        s = state(ctx)
        card_id = s.get("last_ui_msg_id_suno") if isinstance(s.get("last_ui_msg_id_suno"), int) else None
        try:
            set_wait(
                user.id,
                WaitKind.SUNO_TITLE.value,
                card_id,
                chat_id=chat.id,
                meta={"source": "command"},
            )
        except ValueError:
            pass


async def _ensure_admin(update: Update) -> bool:
    user = update.effective_user
    if user is None:
        return False
    if not _is_admin(user.id):
        return False
    return True


async def suno_last_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _ensure_admin(update):
        return
    tasks = await asyncio.to_thread(SUNO_SERVICE.list_last_tasks, 5)
    message = update.effective_message
    if message is None:
        return
    if not tasks:
        await message.reply_text("⚠️ Нет сохранённых задач Suno.")
        return
    lines = ["🗂️ Последние задачи Suno:"]
    for item in tasks:
        task_id = item.get("task_id") or "?"
        status = item.get("status") or "unknown"
        prompt = (item.get("prompt") or "").strip() or "—"
        user_id = item.get("user_id") or "?"
        created = item.get("created_at") or item.get("updated_at") or "?"
        lines.append(f"• {task_id} | {status} | user={user_id} | {created}\n  prompt: {prompt}")
    await message.reply_text("\n".join(lines))


async def suno_task_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _ensure_admin(update):
        return
    args = getattr(ctx, "args", None) or []
    if not args:
        await update.effective_message.reply_text("⚠️ Укажите ID задачи: /suno_task <task_id>.")
        return
    task_id = args[0]
    record = await asyncio.to_thread(SUNO_SERVICE.get_task_record, task_id)
    message = update.effective_message
    if message is None:
        return
    if not record:
        await message.reply_text(f"❓ Задача {task_id} не найдена.")
        return
    lines = [
        f"🧾 Suno задача {task_id}",
        f"Статус: {record.get('status') or 'unknown'}",
        f"Пользователь: {record.get('user_id')}",
        f"Чат: {record.get('chat_id')}",
        f"Создана: {record.get('created_at')}",
        f"Обновлена: {record.get('updated_at')}",
        f"Код: {record.get('code')} | Сообщение: {record.get('msg')}",
        f"Промпт: {(record.get('prompt') or '').strip() or '—'}",
        f"Треков: {len(record.get('tracks') or [])}",
    ]
    tracks = record.get("tracks") or []
    if isinstance(tracks, list) and tracks:
        for idx, track in enumerate(tracks, start=1):
            if not isinstance(track, Mapping):
                continue
            lines.append(
                f"  {idx}. {track.get('title') or track.get('id') or 'track'}\n"
                f"     audio: {track.get('audio_url') or '—'}\n"
                f"     image: {track.get('image_url') or '—'}"
            )
    await message.reply_text("\n".join(lines))


async def suno_retry_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _ensure_admin(update):
        return
    args = getattr(ctx, "args", None) or []
    if not args:
        await update.effective_message.reply_text("⚠️ Укажите ID задачи: /suno_retry <task_id>.")
        return
    task_id = args[0]
    success = await asyncio.to_thread(SUNO_SERVICE.resend_links, task_id)
    message = update.effective_message
    if message is None:
        return
    if success:
        await message.reply_text(f"🔁 Повторная отправка для задачи {task_id} запущена.")
    else:
        await message.reply_text(f"⚠️ Не удалось повторно отправить задачу {task_id}.")


async def video_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await ensure_user_record(update)
    chat = update.effective_chat
    if chat is None:
        return
    user = update.effective_user
    user_id = user.id if user else None
    log.debug(
        "video.command",
        extra={"chat_id": chat.id, "user_id": user_id},
    )
    try:
        await start_video_menu(update, ctx)
    except MenuLocked:
        log.info(
            "ui.video_menu.command_locked",
            extra={"chat_id": chat.id, "user_id": user_id},
        )


async def image_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await ensure_user_record(update)
    chat = update.effective_chat
    if chat is None:
        return
    user = update.effective_user
    user_id = user.id if user else None
    if user_id:
        input_state.clear(user_id, reason="card_opened")
        set_mode(user_id, False)
        clear_wait(user_id)

    s = state(ctx)
    engine = s.get("image_engine")
    if engine not in {"mj", "banana"}:
        await show_image_engine_selector(chat.id, ctx, force_new=True)
        return
    try:
        await _open_image_engine(
            chat.id,
            ctx,
            engine,
            user_id=user_id,
            source="image_command",
            force_new=True,
        )
    except Exception:
        log.exception("IMAGE_ENGINE_OPEN_FAIL | engine=%s chat=%s", engine, chat.id)
        await show_image_engine_selector(chat.id, ctx, force_new=True)


async def buy_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await ensure_user_record(update)
    await topup(update, ctx)


async def lang_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await ensure_user_record(update)
    message = update.effective_message
    if message is None:
        return
    await message.reply_text("🌍 Переключение языка будет доступно в ближайшее время.")


async def help_command_entry(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await ensure_user_record(update)
    await help_command(update, ctx)


async def faq_command_entry(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await ensure_user_record(update)
    await faq_command(update, ctx)


async def faq_callback_entry(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await ensure_user_record(update)
    await faq_callback(update, ctx)


async def prompt_master_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await ensure_user_record(update)
    await prompt_master_open(update, ctx)


async def prompt_master_reset_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await ensure_user_record(update)
    await prompt_master_reset(update, ctx)


async def prompt_master_callback_entry(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await ensure_user_record(update)
    await prompt_master_callback(update, ctx)


async def prompt_master_insert_callback_entry(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await ensure_user_record(update)
    await prompt_master_insert_callback(update, ctx)


async def topup(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await ensure_user_record(update)
    chat = update.effective_chat
    if chat is None:
        return
    user = update.effective_user
    await show_topup_menu(ctx, chat.id, user_id=user.id if user else None)


async def promo_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await ensure_user_record(update)
    message = update.effective_message
    if message is None:
        return

    if not PROMO_ENABLED:
        await message.reply_text("🎟️ Промокоды временно отключены.")
        return

    args = getattr(ctx, "args", None) or []
    if args:
        await process_promo_submission(update, ctx, " ".join(args))
        return

    state(ctx)["mode"] = "promo"
    await message.reply_text("Введите промокод одним сообщением…")


# codex/fix-balance-reset-after-deploy
async def balance_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await ensure_user_record(update)
    chat = update.effective_chat
    if chat is None:
        return
    mid = await show_balance_card(chat.id, ctx, force_new=True)
    if mid is None:
        user = update.effective_user
        if user is None or update.message is None:
            return
        balance = _safe_get_balance(user.id)
        _set_cached_balance(ctx, balance)
        await update.message.reply_text(f"💎 Ваш баланс: {balance} 💎")


async def handle_video_entry(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await ensure_user_record(update)
    message = update.effective_message
    if message is None:
        return
    chat = update.effective_chat
    if chat is None:
        return

    user = update.effective_user
    user_id = user.id if user else None

    log.debug(
        "video.menu.reply",
        extra={"chat_id": chat.id, "user_id": user_id},
    )
    try:
        await start_video_menu(update, ctx)
    except MenuLocked:
        log.info(
            "ui.video_menu.reply_locked",
            extra={"chat_id": chat.id, "user_id": user_id},
        )


async def video_menu_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await ensure_user_record(update)
    query = update.callback_query
    if not query or not query.data:
        return

    state_dict = state(ctx)
    raw_data = str(query.data)
    data = VIDEO_CALLBACK_ALIASES.get(raw_data, raw_data)

    message = getattr(query, "message", None)
    chat_obj = getattr(message, "chat", None) or getattr(update, "effective_chat", None)
    chat_id = getattr(chat_obj, "id", None) if chat_obj else None
    if chat_id is None and message is not None:
        chat_id = getattr(message, "chat_id", None)
    user = getattr(query, "from_user", None) or update.effective_user
    user_id = user.id if user else None

    answer_payload: Dict[str, Any] = {"text": "", "show_alert": False}

    async def _answer() -> None:
        text = str(answer_payload.get("text", ""))
        show_alert = bool(answer_payload.get("show_alert", False))
        answer_method = getattr(query, "answer", None)
        try:
            if callable(answer_method):
                if text or show_alert:
                    await answer_method(text=text, show_alert=show_alert)
                else:
                    await answer_method()
                return
            bot_obj = getattr(ctx, "bot", None)
            if bot_obj is not None:
                await bot_obj.answer_callback_query(
                    callback_query_id=getattr(query, "id", ""),
                    text=text,
                    show_alert=show_alert,
                )
        except Exception as exc:
            log.debug(
                "ui.video_menu.answer_fail",
                extra={"chat_id": chat_id, "error": str(exc)},
            )

    log.info(
        "ui.video_menu.click",
        extra={"chat_id": chat_id, "user_id": user_id, "data": data},
    )

    try:
        if data == CB.VIDEO_MENU:
            if chat_id is None:
                return
            try:
                await start_video_menu(update, ctx)
            except MenuLocked:
                answer_payload["text"] = "Обрабатываю…"
            return

        if data == CB.VIDEO_PICK_VEO:
            if chat_id is None:
                return
            try:
                async with with_menu_lock(
                    _VIDEO_MENU_LOCK_NAME,
                    chat_id,
                    ttl=VIDEO_MENU_LOCK_TTL,
                ):
                    fallback_id = getattr(message, "message_id", None)
                    message_id = await safe_edit_or_send_menu(
                        ctx,
                        chat_id=chat_id,
                        text=VIDEO_VEO_MENU_TEXT,
                        reply_markup=veo_modes_kb(),
                        state_key=VIDEO_MENU_STATE_KEY,
                        msg_ids_key=VIDEO_MENU_MSG_IDS_KEY,
                        state_dict=state_dict,
                        fallback_message_id=fallback_id,
                        parse_mode=ParseMode.HTML,
                        disable_web_page_preview=True,
                        log_label="ui.video_menu.veo",
                    )
                    if isinstance(message_id, int):
                        save_menu_message(
                            _VIDEO_MENU_MESSAGE_NAME,
                            chat_id,
                            message_id,
                            _VIDEO_MENU_MESSAGE_TTL,
                        )
            except MenuLocked:
                answer_payload["text"] = "Обрабатываю…"
            return

        if data == CB.VIDEO_PICK_SORA2_DISABLED:
            answer_payload["text"] = "Sora2 скоро будет доступна"
            answer_payload["show_alert"] = True
            return

        if data == CB.VIDEO_PICK_SORA2:
            if not _sora2_is_enabled():
                answer_payload["text"] = "Sora2 временно недоступна. Попробуйте позже или выберите VEO."
                answer_payload["show_alert"] = True
                if chat_id is not None:
                    try:
                        await start_video_menu(update, ctx)
                    except MenuLocked:
                        pass
                return
            if chat_id is not None:
                await _clear_video_menu_state(chat_id, user_id=user_id, ctx=ctx)
                await sora2_entry(chat_id, ctx)
            return

        if data in VIDEO_MODE_CALLBACK_MAP:
            selected_mode = VIDEO_MODE_CALLBACK_MAP[data]
            handled = await _start_video_mode(
                selected_mode,
                chat_id=chat_id,
                ctx=ctx,
                user_id=user_id,
                message=message,
            )
            if handled and chat_id is not None:
                await _clear_video_menu_state(chat_id, user_id=user_id, ctx=ctx)
            return

        if data == CB.VIDEO_MENU_BACK:
            target_chat = chat_id if chat_id is not None else (user_id if user_id else None)
            if target_chat is not None:
                await _clear_video_menu_state(target_chat, user_id=user_id, ctx=ctx)
                await show_emoji_hub_for_chat(
                    target_chat,
                    ctx,
                    user_id=user_id,
                    replace=True,
                )
            return
    finally:
        await _answer()

async def handle_image_entry(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await image_command(update, ctx)


async def handle_music_entry(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await suno_command(update, ctx)


async def handle_chat_entry(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await chat_command(update, ctx)


async def handle_balance_entry(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await balance_command(update, ctx)




async def transactions_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await ensure_user_record(update)
    message = update.effective_message
    user = update.effective_user
    if message is None or user is None:
        return

    text, keyboard, _ = _build_transactions_view(user.id, 0)
    sent = await message.reply_text(text, reply_markup=keyboard)

    if sent is not None:
        s = state(ctx)
        s["last_panel"] = "balance_history"
        msg_ids = s.get("msg_ids")
        if not isinstance(msg_ids, dict):
            msg_ids = {}
            s["msg_ids"] = msg_ids
        msg_ids["balance"] = sent.message_id


async def my_balance_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await ensure_user_record(update)
    message = update.effective_message
    user = update.effective_user
    if message is None or user is None:
        return
    balance = _safe_get_balance(user.id)
    await message.reply_text(f"💎 Ваш баланс: {balance}")


async def add_balance_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await ensure_user_record(update)
    message = update.effective_message
    user = update.effective_user
    if message is None or user is None:
        return
    if not _is_admin(user.id):
        await message.reply_text("⛔️ Только для админа.")
        return
    text = (message.text or "").partition(" ")[2].strip()
    try:
        amount = int(text)
        if amount <= 0:
            raise ValueError
    except Exception:
        await message.reply_text("Использование: /add_balance 10")
        return
    new_balance = credit(user.id, amount, "admin topup", {"admin_id": user.id})
    await message.reply_text(f"✅ Начислено {amount}. Новый баланс: {new_balance}")


async def sub_balance_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await ensure_user_record(update)
    message = update.effective_message
    user = update.effective_user
    if message is None or user is None:
        return
    if not _is_admin(user.id):
        await message.reply_text("⛔️ Только для админа.")
        return
    text = (message.text or "").partition(" ")[2].strip()
    try:
        amount = int(text)
        if amount <= 0:
            raise ValueError
    except Exception:
        await message.reply_text("Использование: /sub_balance 10")
        return
    ok, balance = debit_try(user.id, amount, "admin debit", {"admin_id": user.id})
    if ok:
        await message.reply_text(f"✅ Списано {amount}. Баланс: {balance}")
    else:
        await message.reply_text(f"⚠️ Недостаточно средств. На счету: {balance}")


async def balance_recalc(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await ensure_user_record(update)
    uid = update.effective_user.id
    try:
        result = ledger_storage.recalc_user_balance(uid)
        _set_cached_balance(ctx, result.calculated)
    except Exception as exc:
        log.exception("Balance recalc failed for %s: %s", uid, exc)
        await update.message.reply_text("⚠️ Не удалось пересчитать баланс. Попробуйте позже.")
        return
    if result.updated:
        await update.message.reply_text(
            f"♻️ Баланс обновлён: было {result.previous} 💎 → стало {result.calculated} 💎"
        )
    else:
        await update.message.reply_text(f"✅ Баланс актуален: {result.calculated} 💎")

async def health(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await ensure_user_record(update)
    parts = [
        f"PTB: `{getattr(_tg, '__version__', 'unknown')}`" if _tg else "PTB: `unknown`",
        f"KIEBASEURL: `{KIE_BASE_URL}`",
        f"OPENAI: `{'set' if OPENAI_API_KEY else 'missing'}`",
        f"KIE: `{'set' if KIE_API_KEY else 'missing'}`",
        f"REDIS: `{'on' if REDIS_URL else 'off'}`",
        f"FFMPEG: `{FFMPEG_BIN}`",
    ]
    parts.append(f"DB: `{'ok' if ledger_storage.ping() else 'error'}`")
    lock_status = "disabled"
    if runner_lock_state.get("enabled"):
        lock_status = "owned" if runner_lock_state.get("owned") else "free"
    lock_payload: Dict[str, Any] = {"ok": True, "lock": lock_status}
    if runner_lock_state.get("heartbeat_at"):
        lock_payload["hb"] = runner_lock_state.get("heartbeat_at")
    parts.append(f"LOCK: `{json.dumps(lock_payload, ensure_ascii=False)}`")
    await update.message.reply_text("🩺 *Health*\n" + "\n".join(parts), parse_mode=ParseMode.MARKDOWN)


async def users_count_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await ensure_user_record(update)
    message = update.effective_message
    if message is None:
        return
    count = await get_users_count(redis_client)
    if count is None:
        await message.reply_text("⚠️ Redis недоступен.")
        return
    await message.reply_text(f"👥 Пользователей: {count}")


async def whoami_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await ensure_user_record(update)
    message = update.effective_message
    user = update.effective_user
    if message is None or user is None:
        return
    if redis_client is None:
        status = "⚠️ Redis недоступен"
    else:
        exists = await user_exists(redis_client, user.id)
        status = "✅ Запись найдена" if exists else "⚠️ Запись не найдена"
    await message.reply_text(
        f"🆔 Ваш Telegram ID: {user.id}\n💾 В Redis: {status}"
    )


async def suno_debug_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await ensure_user_record(update)
    message = update.effective_message
    user = update.effective_user
    if message is None or user is None:
        return
    if user.id not in ADMIN_IDS:
        await message.reply_text("⛔ У вас нет прав для этой команды.")
        return
    if not rds:
        await message.reply_text("⚠️ Redis недоступен.")
        return
    try:
        entries = rds.lrange(SUNO_LOG_KEY, 0, 4)
    except Exception as exc:
        log.warning("Suno debug fetch failed: %s", exc)
        await message.reply_text("⚠️ Не удалось получить логи.")
        return
    if not entries:
        await message.reply_text("ℹ️ Логи генерации музыки пусты.")
        return
    lines: List[str] = []
    for raw in entries:
        if isinstance(raw, bytes):
            try:
                raw = raw.decode("utf-8")
            except Exception:
                raw = raw.decode("utf-8", "ignore")
        if not isinstance(raw, str):
            raw = str(raw)
        raw = raw.strip()
        if not raw:
            continue
        try:
            doc = json.loads(raw)
        except Exception:
            doc = None
        if isinstance(doc, dict):
            ts = doc.get("timestamp") or "?"
            uid = doc.get("user_id")
            phase = doc.get("phase")
            status = doc.get("http_status")
            snippet = doc.get("response_snippet") or ""
            line = f"{ts} | user={uid} | {phase} | status={status} | {snippet}"
        else:
            line = raw
        lines.append(f"• <code>{html.escape(line)}</code>")
    text = "🛠️ Логи генерации музыки (последние 5):\n" + "\n".join(lines)
    await message.reply_text(text, parse_mode=ParseMode.HTML)


async def broadcast_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await ensure_user_record(update)
    message = update.effective_message
    user = update.effective_user
    if message is None or user is None:
        return
    if user.id not in ADMIN_IDS:
        await message.reply_text("⛔ У вас нет прав для этой команды.")
        return

    if redis_client is None:
        await message.reply_text("⚠️ Redis недоступен, рассылка невозможна.")
        return

    is_reply_broadcast = message.reply_to_message is not None
    payload = ""
    fallback_reply_payload = ""
    if not is_reply_broadcast:
        text = message.text or message.caption or ""
        payload = text.partition(" ")[2].strip()
        if not payload:
            await message.reply_text("⚠️ Использование: /broadcast текст или ответьте на сообщение.")
            return
    elif message.reply_to_message:
        fallback_reply_payload = (message.reply_to_message.text or message.reply_to_message.caption or "").strip()

    user_ids = await get_all_user_ids(redis_client)
    if not user_ids:
        await message.reply_text("⚠️ Нет пользователей для рассылки.")
        return

    total = len(user_ids)
    sent = 0
    errors = 0
    status_msg = await message.reply_text(f"🚀 Рассылка началась. Всего получателей: {total}")

    async def _update_progress(current: int) -> None:
        nonlocal status_msg
        text_progress = f"{current}/{total} (ok {sent}, err {errors})"
        if status_msg:
            try:
                await status_msg.edit_text(text_progress)
                return
            except BadRequest:
                pass
        status_msg = await message.reply_text(text_progress)

    source_chat_id = message.reply_to_message.chat_id if is_reply_broadcast and message.reply_to_message else None
    source_message_id = message.reply_to_message.message_id if is_reply_broadcast and message.reply_to_message else None

    async def _send_payload(target_id: int) -> None:
        nonlocal sent, errors

        async def _deliver() -> None:
            if is_reply_broadcast and source_chat_id is not None and source_message_id is not None:
                await ctx.bot.copy_message(
                    chat_id=target_id,
                    from_chat_id=source_chat_id,
                    message_id=source_message_id,
                )
            elif is_reply_broadcast and fallback_reply_payload:
                await ctx.bot.send_message(target_id, fallback_reply_payload)
            else:
                await ctx.bot.send_message(target_id, payload)

        attempts = 0
        last_retry_error: Optional[Exception] = None
        while attempts < 3:
            attempts += 1
            try:
                await _deliver()
                sent += 1
                return
            except RetryAfter as exc:
                last_retry_error = exc
                await asyncio.sleep(exc.retry_after + 0.1)
                continue
            except Forbidden:
                errors += 1
                await mark_user_dead(redis_client, target_id)
                return
            except BadRequest as exc:
                errors += 1
                message_text = getattr(exc, "message", "") or ""
                lowered = message_text.lower()
                dead_markers = (
                    "chat not found",
                    "user is deactivated",
                    "not found",
                    "peer_id_invalid",
                    "bot was blocked",
                )
                if any(marker in lowered for marker in dead_markers):
                    await mark_user_dead(redis_client, target_id)
                    return
                log.warning("Broadcast send failed for %s: %s", target_id, exc)
                return
            except Exception as exc:
                errors += 1
                action = "copy" if is_reply_broadcast else "send"
                log.warning("Broadcast %s failed for %s: %s", action, target_id, exc)
                return

        errors += 1
        action = "copy" if is_reply_broadcast else "send"
        log.warning("Broadcast %s failed for %s after retries: %s", action, target_id, last_retry_error)

    batch_size = 30
    delay_between = 0.12
    progress_step = 500

    for batch_start in range(0, total, batch_size):
        batch = user_ids[batch_start : batch_start + batch_size]
        for idx_offset, target_id in enumerate(batch, start=1):
            current_index = batch_start + idx_offset
            await _send_payload(target_id)

            if current_index % progress_step == 0:
                await _update_progress(current_index)

            await asyncio.sleep(delay_between)

        if batch_start + batch_size < total:
            await asyncio.sleep(0.15)

    await _update_progress(total)

    final_text = f"Готово: OK={sent}, Ошибок={errors}"
    if status_msg:
        try:
            await status_msg.edit_text(final_text)
        except BadRequest:
            await message.reply_text(final_text)
    else:
        await message.reply_text(final_text)


async def sora2_health_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _ensure_admin(update):
        return
    message = update.effective_message
    if message is None:
        return
    if not SORA2_ENABLED or not (SORA2.get("API_KEY") or "").strip():
        await message.reply_text("⚠️ Sora2 отключена или не настроена в конфигурации.")
        return
    keyboard = InlineKeyboardMarkup(
        [[InlineKeyboardButton("🔍 Проверить Sora2", callback_data=CB_ADMIN_SORA2_HEALTH)]]
    )
    await message.reply_text("Проверка доступности Sora2", reply_markup=keyboard)


async def sora2_health_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _ensure_admin(update):
        return
    query = update.callback_query
    if query is None or query.data != CB_ADMIN_SORA2_HEALTH:
        return
    message = query.message
    chat = message.chat if message is not None else update.effective_chat
    chat_id = chat.id if chat is not None else None
    try:
        await query.answer("⏳ Проверяю Sora2…", show_alert=False)
    except Exception as exc:
        log.debug("sora2.health.answer_fail", extra={"error": str(exc)})
    payload = {
        "model": "sora2-text-to-video",
        "input": {
            "prompt": "healthcheck: ignore",
            "aspect_ratio": "16:9",
            "duration": 1,
            "quality": "standard",
            "dry_run": True,
        },
        "metadata": {"source": "healthcheck"},
    }
    try:
        response: CreateTaskResponse = await asyncio.to_thread(sora2_create_task, payload)
    except Sora2UnavailableError as exc:
        mark_sora2_unavailable()
        result_text = "⚠️ Sora2 отключена для текущего ключа (422)."
        log.warning("sora2.health.unavailable", extra={"error": str(exc)})
    except Sora2AuthError as exc:
        mark_sora2_unavailable()
        result_text = "⚠️ Ошибка авторизации Sora2 (401/403)."
        log.error("sora2.health.auth_error", extra={"error": str(exc)})
    except Sora2Error as exc:
        result_text = f"⚠️ Ошибка healthcheck: {exc}"[:200]
        log.warning("sora2.health.error", extra={"error": str(exc)})
    else:
        clear_sora2_unavailable()
        result_text = f"✅ Sora2 доступна. task_id={response.task_id}"
        log.info("sora2.health.ok", extra={"task_id": response.task_id})
    if message is not None:
        with suppress(Exception):
            await message.edit_text(result_text)
    elif chat_id is not None:
        await ctx.bot.send_message(chat_id, result_text)
    if chat_id is not None:
        await _refresh_video_menu_ui(ctx, chat_id=chat_id, message=None)

async def error_handler(update: Optional[Update], context: ContextTypes.DEFAULT_TYPE):
    log.exception("Unhandled error: %s", context.error)
    try:
        if update and update.effective_chat:
            await context.bot.send_message(update.effective_chat.id, "⚠️ Системная ошибка. Попробуйте ещё раз.")
    except Exception:
        pass

async def show_banana_card(
    chat_id: int,
    ctx: ContextTypes.DEFAULT_TYPE,
    *,
    force_new: bool = False,
) -> None:
    s = state(ctx)
    text = banana_card_text(s)
    if not force_new and text == s.get("_last_text_banana"):
        return
    kb = banana_kb()
    mid = await upsert_card(
        ctx,
        chat_id,
        s,
        "last_ui_msg_id_banana",
        text,
        kb,
        force_new=force_new,
    )
    if mid:
        s["_last_text_banana"] = text
    else:
        s["_last_text_banana"] = None

async def banana_entry(chat_id: int, ctx: ContextTypes.DEFAULT_TYPE, *, force_new: bool = True) -> None:
    s = state(ctx)
    mid = s.get("last_ui_msg_id_banana")
    if mid:
        with suppress(Exception):
            await ctx.bot.delete_message(chat_id, mid)
    s["last_ui_msg_id_banana"] = None
    s["_last_text_banana"] = None
    await show_banana_card(chat_id, ctx, force_new=force_new)

def _coerce_optional_str(value: Any) -> Optional[str]:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped
    return None


def _normalize_banana_image(value: Any) -> Optional[Dict[str, Any]]:
    if isinstance(value, Mapping):
        file_path = _coerce_optional_str(value.get("file_path"))
        url = _coerce_optional_str(value.get("url"))
        if not url and file_path:
            url = tg_direct_file_url(TELEGRAM_TOKEN, file_path)
        if not url:
            return None
        return {
            "url": url,
            "file_path": file_path,
            "filename": _coerce_optional_str(value.get("filename")),
            "mime": _coerce_optional_str(value.get("mime")),
            "source": _coerce_optional_str(value.get("source")) or "photo",
            "width": value.get("width"),
            "height": value.get("height"),
        }
    if isinstance(value, str):
        url = value.strip()
        if not url:
            return None
        return {
            "url": url,
            "file_path": None,
            "filename": None,
            "mime": None,
            "source": "external",
            "width": None,
            "height": None,
        }
    return None


def _get_banana_images(state_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    images = state_dict.get("banana_images")
    if not isinstance(images, list):
        images = []
        state_dict["banana_images"] = images
        return images
    normalized: List[Dict[str, Any]] = []
    changed = False
    for item in images:
        entry = _normalize_banana_image(item)
        if entry is None:
            changed = True
            continue
        normalized.append(entry)
        if entry is not item:
            changed = True
    if changed:
        state_dict["banana_images"] = normalized
        return normalized
    return images


async def on_banana_photo_received(chat_id: int, ctx: ContextTypes.DEFAULT_TYPE, file_id: Any) -> None:
    s = state(ctx)
    images = _get_banana_images(s)
    entry = _normalize_banana_image(file_id)
    if entry is None:
        return
    images.append(entry)
    s["_last_text_banana"] = None
    await show_banana_card(chat_id, ctx)

async def on_banana_prompt_saved(chat_id: int, ctx: ContextTypes.DEFAULT_TYPE, text_prompt: str) -> None:
    s = state(ctx)
    s["last_prompt"] = text_prompt
    s["_last_text_banana"] = None
    await show_banana_card(chat_id, ctx)

async def show_veo_card(
    chat_id: int,
    ctx: ContextTypes.DEFAULT_TYPE,
    *,
    force_new: bool = False,
) -> None:
    s = state(ctx)
    text = veo_card_text(s)
    if not force_new and text == s.get("_last_text_veo"):
        return
    kb = veo_kb(s)
    mid = await upsert_card(
        ctx,
        chat_id,
        s,
        "last_ui_msg_id_veo",
        text,
        kb,
        force_new=force_new,
    )
    if mid:
        s["_last_text_veo"] = text
    else:
        s["_last_text_veo"] = None

async def veo_entry(chat_id: int, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    s = state(ctx)
    mid = s.get("last_ui_msg_id_veo")
    if mid:
        with suppress(Exception):
            await ctx.bot.delete_message(chat_id, mid)
    s["last_ui_msg_id_veo"] = None
    s["_last_text_veo"] = None
    await show_veo_card(chat_id, ctx, force_new=True)

async def set_veo_card_prompt(chat_id: int, prompt_text: str, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    s = state(ctx)
    s["last_prompt"] = prompt_text
    s["_last_text_veo"] = None
    await show_veo_card(chat_id, ctx)


async def prompt_master_insert_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query or not query.data:
        return

    parts = query.data.split(":", 2)
    engine = parts[2] if len(parts) > 2 else ""
    message = query.message
    chat = message.chat if message is not None else update.effective_chat
    chat_id = chat.id if chat is not None else None
    if chat_id is None:
        await query.answer("Чат недоступен", show_alert=True)
        return

    prompt_obj = get_pm_prompt(chat_id, engine)
    if prompt_obj is None:
        await query.answer("Промпт не найден", show_alert=True)
        return

    s = state(ctx)
    body_text = str(prompt_obj.get("card_text") or prompt_obj.get("copy_text") or "")
    if engine in {"veo", "animate"}:
        await set_veo_card_prompt(chat_id, body_text, ctx)
        await query.answer("Промпт вставлен в карточку VEO")
        return
    if engine == "mj":
        s["last_prompt"] = body_text
        s["_last_text_mj"] = None
        await show_mj_prompt_card(chat_id, ctx)
        await query.answer("Промпт вставлен в карточку Midjourney")
        return
    if engine == "banana":
        s["last_prompt"] = body_text
        s["_last_text_banana"] = None
        await show_banana_card(chat_id, ctx)
        await query.answer("Промпт вставлен в карточку Banana")
        return
    if engine == "suno":
        suno_state_obj = load_suno_state(ctx)
        set_suno_lyrics(suno_state_obj, body_text)
        suno_state_obj.mode = "lyrics"
        save_suno_state(ctx, suno_state_obj)
        s["suno_state"] = suno_state_obj.to_dict()
        s["suno_waiting_state"] = IDLE_SUNO
        _reset_suno_card_cache(s)
        await refresh_suno_card(ctx, chat_id, s, price=PRICE_SUNO)
        await query.answer("Промпт вставлен в карточку Suno")
        return

    await query.answer("Режим не поддерживается", show_alert=True)


async def handle_pm_insert_to_veo(update: Update, ctx: ContextTypes.DEFAULT_TYPE, data: str) -> None:
    q = update.callback_query
    if not q or not q.message:
        return
    chat_id = q.message.chat_id
    kino_prompt = get_cached_pm_prompt(chat_id)
    if not kino_prompt:
        await q.answer("Промпт не найден — пришлите идею заново.", show_alert=True)
        return

    await set_veo_card_prompt(chat_id, kino_prompt, ctx)
    await q.answer("Промпт вставлен в карточку VEO")


async def on_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await ensure_user_record(update)
    q = update.callback_query
    if not q:
        return
    data = (q.data or "").strip()
    s = state(ctx)
    message = q.message
    chat = update.effective_chat
    user = update.effective_user
    chat_id = None
    if message is not None:
        chat_id = message.chat_id
    elif chat is not None:
        chat_id = chat.id

    if data == CB_MODE_CHAT:
        if chat_id is not None:
            _mode_set(chat_id, MODE_CHAT)
        if user:
            set_mode(user.id, True)
        s["mode"] = None
        await q.answer("Режим: Обычный чат")
        if message is not None:
            await _safe_edit_message_text(
                q.edit_message_text,
                "Режим переключён: теперь это обычный чат. Пишите вопрос! /reset — очистить контекст.",
            )
        return

    if data.startswith("topup:") or data.startswith("yk:"):
        handled = await handle_topup_callback(update, ctx, data)
        if handled:
            return

    if data == CB_MODE_PM:
        if chat_id is not None:
            _mode_set(chat_id, MODE_PM)
        if user:
            set_mode(user.id, False)
        s["mode"] = None
        await q.answer("Режим: Prompt-Master")
        await prompt_master_command(update, ctx)
        return

    if data == CB_GO_HOME:
        if chat_id is not None:
            _mode_set(chat_id, MODE_CHAT)
        if user:
            set_mode(user.id, False)
        s.update({**DEFAULT_STATE})
        _apply_state_defaults(s)
        await q.answer()
        if message is not None:
            with suppress(BadRequest):
                await q.edit_message_reply_markup(reply_markup=None)
        target_chat = chat_id if chat_id is not None else (user.id if user else None)
        if target_chat is not None:
            await show_emoji_hub_for_chat(
                target_chat,
                ctx,
                user_id=user.id if user else None,
                replace=True,
            )
        return

    if data.startswith(CB_PM_INSERT_VEO):
        await handle_pm_insert_to_veo(update, ctx, data)
        return

    await q.answer()

    if data.startswith("tx:"):
        user = update.effective_user
        uid = get_user_id(ctx) or (user.id if user else None)
        if uid is None:
            return

        if data == "tx:open":
            await _edit_transactions_message(q, ctx, uid, 0)
            return

        if data.startswith("tx:page:"):
            try:
                offset_text = data.split(":", 2)[2]
                offset = int(offset_text)
            except (IndexError, ValueError):
                offset = 0
            if offset < 0:
                offset = 0
            await _edit_transactions_message(q, ctx, uid, offset)
            return

        if data == "tx:back":
            await _edit_balance_from_history(q, ctx, uid)
            return
    if data == "ref:open":
        user = update.effective_user
        if user is None:
            return
        uid = user.id
        chat_for_card = chat_id if chat_id is not None else uid
        try:
            link = await _build_referral_link(uid, ctx)
        except Exception as exc:
            log.warning("referral_link_failed | user=%s err=%s", uid, exc)
            link = None
        if not link:
            if q.message is not None:
                await q.message.reply_text("⚠️ Не удалось получить ссылку. Попробуйте позже.")
            return
        try:
            referrals, earned = get_ref_stats(uid)
        except Exception as exc:
            log.warning("referral_stats_failed | user=%s err=%s", uid, exc)
            referrals, earned = 0, 0
        await show_referral_card(
            ctx,
            chat_for_card,
            s,
            link=link,
            referrals=referrals,
            earned=earned,
            share_text=_REF_SHARE_TEXT,
        )
        return
    if data == "ref:back":
        target_chat = chat_id if chat_id is not None else (update.effective_user.id if update.effective_user else None)
        if target_chat is None:
            return
        await show_balance_card(target_chat, ctx, force_new=True)
        return

    if data == "promo_open":
        if not PROMO_ENABLED:
            await q.message.reply_text("🎟️ Промокоды временно отключены.")
            return
        s["mode"] = "promo"
        await q.message.reply_text("Введите промокод одним сообщением…")
        return

    if data == "faq":
        await faq_command_entry(update, ctx)
        return

    if data == "back":
        s.update({**DEFAULT_STATE})
        _apply_state_defaults(s)
        target_chat = chat_id if chat_id is not None else (user.id if user else None)
        if target_chat is not None:
            await show_emoji_hub_for_chat(
                target_chat,
                ctx,
                user_id=user.id if user else None,
                replace=True,
            )
        return

    if data == "start_new_cycle":
        s.update({**DEFAULT_STATE})
        _apply_state_defaults(s)
        target_chat = chat_id if chat_id is not None else (user.id if user else None)
        if target_chat is not None:
            await show_emoji_hub_for_chat(
                target_chat,
                ctx,
                user_id=user.id if user else None,
                replace=True,
            )
        return

    # Покупка
    if data.startswith("buy:stars:"):
        try:
            _, _, stars_str, diamonds_str = data.split(":")
            stars = int(stars_str)
            diamonds = int(diamonds_str)
        except (ValueError, TypeError):
            log.warning("stars_purchase_invalid_callback | data=%s", data)
            await q.message.reply_text(
                "⚠️ Не удалось определить пакет. Попробуйте выбрать его заново.",
                reply_markup=stars_topup_kb(),
            )
            return

        title = f"{stars}⭐ → {diamonds}💎"
        payload = json.dumps(
            {
                "type": "stars_pack",
                "stars": stars,
                "diamonds": diamonds,
                "bonus": max(diamonds - stars, 0),
            }
        )
        try:
            await ctx.bot.send_invoice(
                chat_id=update.effective_chat.id,
                title=title,
                description="Пакет пополнения токенов",
                payload=payload,
                provider_token="",
                currency="XTR",
                prices=[LabeledPrice(label=title, amount=stars)],
            )
        except Exception as e:
            event("STARS_INVOICE_ERR", err=str(e))
            await q.message.reply_text(
                f"Если счёт не открылся — активируйте Stars и попробуйте снова, или купите в {STARS_BUY_URL}.",
                reply_markup=stars_topup_kb()
            )
        return

    # Режимы
    if data.startswith("mode:"):
        selected_mode = data.split(":", 1)[1]
        s["mode"] = selected_mode
        chat = update.effective_chat
        chat_id_val = chat.id if chat else None
        user_obj = update.effective_user
        uid_val = user_obj.id if user_obj else None
        if uid_val is not None:
            clear_wait_state(uid_val, reason="mode_switch")
        handled = await _start_video_mode(
            selected_mode,
            chat_id=chat_id_val,
            ctx=ctx,
            user_id=uid_val,
            message=q.message,
        )
        if handled:
            return
        if selected_mode == "chat":
            await q.message.reply_text("💬 Чат активен. Напишите сообщение."); return
        if selected_mode == "mj_txt":
            await _open_image_engine(
                update.effective_chat.id,
                ctx,
                "mj",
                user_id=uid_val,
                source="mode_switch",
            )
            return
        if selected_mode == "mj_upscale":
            locale = _determine_user_locale(update.effective_user)
            s["mode"] = "mj_upscale"
            s["mj_locale"] = locale
            s["mj_upscale_active"] = None
            grid = _load_last_mj_grid(s, uid_val)
            if grid:
                await tg_safe_send(
                    ctx.bot.send_message,
                    method_name="sendMessage",
                    kind="message",
                    chat_id=chat_id_val,
                    text=_mj_ui_text("upscale_choose", locale),
                    reply_markup=_mj_upscale_keyboard(len(grid.get("result_urls", [])), locale),
                )
            else:
                await tg_safe_send(
                    ctx.bot.send_message,
                    method_name="sendMessage",
                    kind="message",
                    chat_id=chat_id_val,
                    text=_mj_ui_text("upscale_need_photo", locale),
                )
            return
        if selected_mode == "banana":
            s["banana_images"] = []
            s["last_prompt"] = None
            await q.message.reply_text(BANANA_MODE_HINT_MD, parse_mode=ParseMode.MARKDOWN)
            await _open_image_engine(
                update.effective_chat.id,
                ctx,
                "banana",
                user_id=uid_val,
                source="mode_switch",
            )
            return

    if data.startswith("img_engine:"):
        choice = data.split(":", 1)[1]
        chat = update.effective_chat
        if not chat:
            return
        user_obj = update.effective_user
        uid_val = user_obj.id if user_obj else None
        if choice not in {"mj", "banana"}:
            await q.answer()
            return
        try:
            await q.answer("Midjourney" if choice == "mj" else "Banana")
        except Exception:
            pass
        await _open_image_engine(
            chat.id,
            ctx,
            choice,
            user_id=uid_val,
            source="engine_select",
        )
        if choice == "banana" and q.message is not None:
            await q.message.reply_text(BANANA_MODE_HINT_MD, parse_mode=ParseMode.MARKDOWN)
        return

    if data.startswith("mj_upscale:"):
        chat = update.effective_chat
        if not chat:
            return
        parts = data.split(":", 2)
        action = parts[1] if len(parts) > 1 else ""
        payload = parts[2] if len(parts) > 2 else ""
        chat_id = chat.id
        user_obj = update.effective_user
        uid_val = user_obj.id if user_obj else None
        locale = s.get("mj_locale") or _determine_user_locale(user_obj)
        s["mj_locale"] = locale
        grid = _load_last_mj_grid(s, uid_val)

        if action == "select":
            try:
                index = int(payload)
            except (TypeError, ValueError):
                await ctx.bot.send_message(chat_id, _mj_ui_text("upscale_choose", locale))
                return
            if uid_val is None:
                await ctx.bot.send_message(chat_id, "⚠️ Не удалось определить пользователя.")
                return
            if not grid or index < 0 or index >= len(grid.get("result_urls", [])):
                await ctx.bot.send_message(chat_id, _mj_ui_text("upscale_need_photo", locale))
                return
            await _launch_mj_upscale(
                chat_id,
                ctx,
                user_id=uid_val,
                grid=grid,
                image_index=index,
                locale=locale,
                source="from_grid",
            )
            return

        if action == "repeat":
            if grid:
                await tg_safe_send(
                    ctx.bot.send_message,
                    method_name="sendMessage",
                    kind="message",
                    chat_id=chat_id,
                    text=_mj_ui_text("upscale_choose", locale),
                    reply_markup=_mj_upscale_keyboard(len(grid.get("result_urls", [])), locale),
                )
            else:
                await tg_safe_send(
                    ctx.bot.send_message,
                    method_name="sendMessage",
                    kind="message",
                    chat_id=chat_id,
                    text=_mj_ui_text("upscale_need_photo", locale),
                )
            return

        await tg_safe_send(
            ctx.bot.send_message,
            method_name="sendMessage",
            kind="message",
            chat_id=chat_id,
            text=_mj_ui_text("upscale_choose", locale),
            reply_markup=_mj_upscale_keyboard(len(grid.get("result_urls", [])) if grid else 4, locale),
        )
        return

    if data.startswith("mj:"):
        chat = update.effective_chat
        if not chat:
            return
        parts = data.split(":", 2)
        action = parts[1] if len(parts) > 1 else ""
        payload = parts[2] if len(parts) > 2 else ""
        chat_id = chat.id
        current_aspect = "9:16" if s.get("aspect") == "9:16" else "16:9"
        user_obj = update.effective_user
        uid_val = user_obj.id if user_obj else None

        if action == "aspect":
            if s.get("mj_generating"):
                await q.message.reply_text("⏳ Дождитесь завершения текущей генерации."); return
            new_aspect = "9:16" if payload == "9:16" else "16:9"
            s["aspect"] = new_aspect
            s["last_prompt"] = None
            await show_mj_prompt_card(chat_id, ctx)
            card_id = s.get("last_ui_msg_id_mj") if isinstance(s.get("last_ui_msg_id_mj"), int) else None
            _activate_wait_state(
                user_id=uid_val,
                chat_id=chat_id,
                card_msg_id=card_id,
                kind=WaitKind.MJ_PROMPT,
                meta={"aspect": new_aspect},
            )
            return

        if action == "change_format":
            if s.get("mj_generating"):
                await q.message.reply_text("⏳ Дождитесь завершения текущей генерации."); return
            if uid_val is not None:
                clear_wait_state(uid_val, reason="mj_change_format")
            await show_mj_format_card(chat_id, ctx)
            return

        if action == "switch_engine":
            if s.get("mj_generating"):
                await q.message.reply_text("⏳ Дождитесь завершения текущей генерации.")
                return
            if uid_val is not None:
                clear_wait_state(uid_val, reason="mj_switch_engine")
            s["image_engine"] = None
            await q.answer()
            await show_image_engine_selector(chat_id, ctx, force_new=True)
            return

        if action == "cancel":
            s["mode"] = None
            s["last_prompt"] = None
            s["mj_generating"] = False
            s["last_mj_task_id"] = None
            s["mj_last_wait_ts"] = 0.0
            if uid_val is not None:
                clear_wait_state(uid_val, reason="mj_cancel")
            mid = s.get("last_ui_msg_id_mj")
            if mid:
                try:
                    await _safe_edit_message_text(
                        ctx.bot.edit_message_text,
                        chat_id=chat_id,
                        message_id=mid,
                        text="❌ Midjourney отменён.",
                        reply_markup=None,
                    )
                except Exception:
                    pass
            s["last_ui_msg_id_mj"] = None
            s["_last_text_mj"] = None
            target_chat = chat_id if chat_id is not None else (update.effective_user.id if update.effective_user else None)
            if target_chat is not None:
                await show_emoji_hub_for_chat(
                    target_chat,
                    ctx,
                    user_id=update.effective_user.id if update.effective_user else None,
                    replace=True,
                )
            return

        if action == "confirm":
            if s.get("mj_generating"):
                await q.message.reply_text("⏳ Уже идёт генерация. Дождитесь результата."); return
            prompt = (s.get("last_prompt") or "").strip()
            if not prompt:
                await q.message.reply_text("❌ Промпт не найден, отправьте текст и повторите."); return
            price = PRICE_MJ
            aspect_value = "9:16" if s.get("aspect") == "9:16" else "16:9"
            user = update.effective_user
            uid = user.id if user else None
            if not uid:
                await q.message.reply_text("⚠️ Не удалось определить пользователя. Попробуйте позже.")
                return
            mj_locale = "ru"
            if user and isinstance(user.language_code, str):
                lowered_lang = user.language_code.lower()
                if lowered_lang.startswith("en"):
                    mj_locale = "en"
            s["mj_locale"] = mj_locale
            try:
                ensure_user(uid)
            except Exception as exc:
                log.exception("MJ ensure_user failed for %s: %s", uid, exc)
                await q.message.reply_text("⚠️ Не удалось проверить баланс. Попробуйте позже.")
                return
            if not await ensure_tokens(ctx, chat_id, uid, price):
                return
            ok, balance_after = debit_try(
                uid,
                price,
                reason="service:start",
                meta={"service": "MJ", "aspect": aspect_value, "prompt": _short_prompt(prompt, 160)},
            )
            if not ok:
                await ensure_tokens(ctx, chat_id, uid, price)
                return
            clear_wait_state(uid, reason="mj_confirm")
            await q.message.reply_text("✅ Промпт принят.")
            await show_balance_notification(
                chat_id,
                ctx,
                uid,
                f"✅ Списано {price}💎. Текущий баланс: {balance_after}💎 — запускаю…",
            )
            s["mj_generating"] = True
            s["mj_last_wait_ts"] = time.time()
            wait_msg_id = await send_wait_sticker(ctx, "mj", chat_id=chat_id)
            if wait_msg_id:
                s["video_wait_message_id"] = wait_msg_id
            await show_mj_generating_card(chat_id, ctx, prompt, aspect_value)
            ok, task_id, msg = await asyncio.to_thread(mj_generate, prompt, aspect_value)
            event("MJ_SUBMIT_RESP", ok=ok, task_id=task_id, msg=msg)
            if not ok or not task_id:
                try:
                    new_balance = credit_balance(
                        uid,
                        price,
                        reason="service:refund",
                        meta={"service": "MJ", "reason": "submit_failed", "message": msg},
                    )
                except Exception as exc:
                    log.exception("MJ submit refund failed for %s: %s", uid, exc)
                    new_balance = None
                s["mj_generating"] = False
                s["last_mj_task_id"] = None
                s["mj_last_wait_ts"] = 0.0
                if new_balance is not None:
                    await show_balance_notification(
                        chat_id,
                        ctx,
                        uid,
                        f"⚠️ Ошибка. Возврат {price}💎. Баланс: {new_balance}💎",
                    )
                await q.message.reply_text(f"❌ Не удалось создать MJ-задачу: {msg}\n💎 Токены возвращены.")
                await show_mj_prompt_card(chat_id, ctx)
                return
            s["last_mj_task_id"] = task_id
            asyncio.create_task(
                poll_mj_and_send_photos(chat_id, task_id, ctx, prompt, aspect_value, uid, price)
            )
            return
        if action == "repeat":
            if s.get("mj_generating"):
                await q.message.reply_text("⏳ Уже идёт генерация. Дождитесь результата."); return
            s["mode"] = "mj_txt"
            s["last_prompt"] = None
            s["mj_generating"] = False
            s["mj_last_wait_ts"] = 0.0
            s["last_mj_task_id"] = None
            await show_mj_prompt_card(chat_id, ctx)
            card_id = s.get("last_ui_msg_id_mj") if isinstance(s.get("last_ui_msg_id_mj"), int) else None
            _activate_wait_state(
                user_id=uid_val,
                chat_id=chat_id,
                card_msg_id=card_id,
                kind=WaitKind.MJ_PROMPT,
                meta={"aspect": s.get("aspect")},
            )
            await q.message.reply_text("✍️ Пришлите новый промпт для Midjourney.")
            return

        return

    # Banana callbacks
    if data.startswith("banana:"):
        act = data.split(":",1)[1]
        if act == "add_more":
            await q.message.reply_text("➕ Пришлите ещё фото (всего до 4)."); return
        if act == "reset_imgs":
            s["banana_images"] = []
            s["_last_text_banana"] = None
            await q.message.reply_text("🧹 Фото очищены."); await show_banana_card(update.effective_chat.id, ctx); return
        if act == "edit_prompt":
            user_obj = update.effective_user
            uid_val = user_obj.id if user_obj else None
            chat_ctx = update.effective_chat
            chat_id_val = chat_ctx.id if chat_ctx else (q.message.chat_id if q.message else None)
            card_id = s.get("last_ui_msg_id_banana") if isinstance(s.get("last_ui_msg_id_banana"), int) else None
            _activate_wait_state(
                user_id=uid_val,
                chat_id=chat_id_val,
                card_msg_id=card_id,
                kind=WaitKind.BANANA_PROMPT,
                meta={"action": "edit"},
            )
            await q.message.reply_text("✍️ Пришлите новый промпт для Banana.")
            return
        if act == "switch_engine":
            user_obj = update.effective_user
            uid_val = user_obj.id if user_obj else None
            chat_ctx = update.effective_chat
            chat_id_val = chat_ctx.id if chat_ctx else (q.message.chat_id if q.message else None)
            if chat_id_val is None:
                await q.answer()
                return
            if uid_val is not None:
                clear_wait_state(uid_val, reason="banana_switch_engine")
            s["image_engine"] = None
            await q.answer()
            await show_image_engine_selector(chat_id_val, ctx, force_new=True)
            return
        if act == "start":
            imgs = list(_get_banana_images(s))
            prompt = (s.get("last_prompt") or "").strip()
            if not imgs:
                await q.message.reply_text("⚠️ Сначала добавьте хотя бы одно фото.")
                return
            if not prompt:
                await q.message.reply_text("⚠️ Добавьте текст-промпт (что изменить).")
                return
            user = update.effective_user
            uid = user.id if user else None
            if not uid:
                await q.message.reply_text("⚠️ Не удалось определить пользователя. Попробуйте позже.")
                return
            try:
                ensure_user(uid)
            except Exception as exc:
                log.exception("Banana ensure_user failed for %s: %s", uid, exc)
                await q.message.reply_text("⚠️ Не удалось проверить баланс. Попробуйте позже.")
                return
            chat_id = update.effective_chat.id
            if not await ensure_tokens(ctx, chat_id, uid, PRICE_BANANA):
                return
            ok, balance_after = debit_try(
                uid,
                PRICE_BANANA,
                reason="service:start",
                meta={"service": "BANANA", "images": len(imgs)},
            )
            if not ok:
                await ensure_tokens(ctx, chat_id, uid, PRICE_BANANA)
                return
            new_balance = balance_after
            s["banana_balance"] = new_balance
            s["_last_text_banana"] = None
            clear_wait_state(uid, reason="banana_confirm")
            await show_banana_card(chat_id, ctx)
            await show_balance_notification(
                chat_id,
                ctx,
                uid,
                f"✅ Списано {PRICE_BANANA}💎. Текущий баланс: {new_balance}💎 — запускаю…",
            )
            asyncio.create_task(
                _banana_run_and_send(
                    update.effective_chat.id,
                    ctx,
                    imgs,
                    prompt,
                    PRICE_BANANA,
                    uid,
                )
            );
            return

    if data.startswith("suno:"):
        parts = data.split(":", 2)
        action = parts[1] if len(parts) > 1 else ""
        argument = parts[2] if len(parts) > 2 else ""
        chat = update.effective_chat
        chat_id = chat.id if chat else None
        user = update.effective_user
        uid = user.id if user else None
        suppress_wait_clear = False
        if action == "card" and argument.startswith("lyrics_source"):
            suppress_wait_clear = True
        if uid is not None and not suppress_wait_clear:
            clear_wait_state(uid, reason="suno_callback")
        s["mode"] = "suno"
        suno_state_obj = load_suno_state(ctx)
        s["suno_state"] = suno_state_obj.to_dict()

        if action == "menu":
            await q.answer()
            await _music_show_main_menu(chat_id, ctx, s)
            return

        if action == "mode":
            if argument not in {"instrumental", "lyrics", "cover"}:
                await q.answer("Неизвестный режим", show_alert=True)
                return
            await q.answer()
            await _music_begin_flow(
                chat_id,
                ctx,
                s,
                flow=argument,
                user_id=uid,
            )
            return

        if action == "cancel":
            await q.answer()
            clear_suno_title(suno_state_obj)
            clear_suno_style(suno_state_obj)
            clear_suno_lyrics(suno_state_obj)
            clear_suno_cover_source(suno_state_obj)
            save_suno_state(ctx, suno_state_obj)
            s["suno_state"] = suno_state_obj.to_dict()
            s["suno_flow"] = None
            s["suno_step"] = None
            s["suno_step_order"] = None
            _reset_suno_card_cache(s)
            if chat_id is not None:
                await refresh_suno_card(ctx, chat_id, s, price=PRICE_SUNO)
                await _music_show_main_menu(chat_id, ctx, s)
                await _suno_notify(ctx, chat_id, "❌ Cancelled. Card reset.", reply_to=q.message)
            return

        if action == "card":
            sub_action, _, sub_argument = argument.partition(":")
            if sub_action == "lyrics_source" and (sub_argument in {"", "toggle"}):
                await q.answer()
                current_source = suno_state_obj.lyrics_source
                current_value = (
                    current_source.value
                    if isinstance(current_source, LyricsSource)
                    else str(current_source).strip().lower()
                )
                new_source = (
                    LyricsSource.AI
                    if current_value == LyricsSource.USER.value
                    else LyricsSource.USER
                )
                if new_source == LyricsSource.AI:
                    clear_suno_lyrics(suno_state_obj)
                set_suno_lyrics_source(suno_state_obj, new_source)
                save_suno_state(ctx, suno_state_obj)
                suno_state_payload = suno_state_obj.to_dict()
                s["suno_state"] = suno_state_payload
                user_data = getattr(ctx, "user_data", None)
                if user_data is not None:
                    try:
                        user_data["suno_state"] = dict(suno_state_payload)
                    except Exception:
                        pass
                s["suno_lyrics_confirmed"] = False
                message_text: Optional[str] = None
                if new_source == LyricsSource.USER:
                    s["suno_waiting_state"] = WAIT_SUNO_LYRICS
                else:
                    s["suno_waiting_state"] = IDLE_SUNO
                _reset_suno_card_cache(s)
                target_chat = chat_id
                if target_chat is None and q.message is not None:
                    target_chat = q.message.chat_id
                if target_chat is not None:
                    await refresh_suno_card(ctx, target_chat, s, price=PRICE_SUNO)
                    ready_flag = suno_is_ready_to_start(suno_state_obj)
                    generating = bool(s.get("suno_generating"))
                    waiting_enqueue = bool(s.get("suno_waiting_enqueue"))
                    await sync_suno_start_message(
                        ctx,
                        target_chat,
                        s,
                        suno_state=suno_state_obj,
                        ready=ready_flag,
                        generating=generating,
                        waiting_enqueue=waiting_enqueue,
                    )
                if new_source == LyricsSource.USER:
                    if uid is not None and target_chat is not None:
                        card_msg = _music_card_message_id(s)
                        _activate_wait_state(
                            user_id=uid,
                            chat_id=target_chat,
                            card_msg_id=card_msg,
                            kind=WaitKind.SUNO_LYRICS,
                            meta={"flow": suno_state_obj.mode, "step": "lyrics", "source": "toggle"},
                        )
                    message_text = (
                        f"🔁 {t('suno.field.lyrics_source')}: {t('suno.lyrics_source.user')}\n"
                        f"📝 Пришлите текст песни (1–{LYRICS_MAX_LENGTH} символов)."
                    )
                else:
                    if uid is not None:
                        clear_wait_state(uid, reason="suno_lyrics_source_toggle")
                    message_text = f"✨ {t('suno.field.lyrics_source')}: {t('suno.lyrics_source.ai')}"
                if target_chat is not None and message_text:
                    await _suno_notify(
                        ctx,
                        target_chat,
                        message_text,
                        reply_to=q.message,
                    )
                return
            await q.answer()
            return

        if action == "edit":
            field = argument
            if field not in {"title", "style", "lyrics", "cover"}:
                await q.answer("Недоступное поле", show_alert=True)
                return
            if field == "cover":
                s["suno_step"] = "source"
                await q.answer()
                await _music_prompt_step(
                    chat_id,
                    ctx,
                    s,
                    flow="cover",
                    step="source",
                    user_id=uid,
                )
                return
            if field == "title":
                waiting_state = WAIT_SUNO_TITLE
                log.info("suno wait input", extra={"field": "title", "user_id": uid})
            elif field == "style":
                waiting_state = WAIT_SUNO_STYLE
                log.info("suno wait input", extra={"field": "style", "user_id": uid})
            else:
                waiting_state = WAIT_SUNO_LYRICS
                log.info("suno wait input", extra={"field": "lyrics", "user_id": uid})
            s["suno_waiting_state"] = waiting_state
            await q.answer()
            target_chat = chat_id
            if target_chat is None and q.message is not None:
                target_chat = q.message.chat_id
            if target_chat is None and uid is not None:
                target_chat = uid
            if target_chat is None:
                await q.answer("Нет чата", show_alert=True)
                return
            prompt_text = _suno_prompt_text(field, suno_state_obj)
            prompt_message = await _suno_notify(ctx, target_chat, prompt_text, reply_to=q.message)
            prompt_msg_id = getattr(prompt_message, "message_id", None) if prompt_message else None
            card_state_meta = s.get("suno_card")
            card_msg_id = None
            if isinstance(card_state_meta, dict):
                raw_card_id = card_state_meta.get("msg_id")
                if isinstance(raw_card_id, int):
                    card_msg_id = raw_card_id
            if card_msg_id is None and isinstance(suno_state_obj.card_message_id, int):
                card_msg_id = suno_state_obj.card_message_id
            log_evt(
                "SUNO_INPUT_START",
                kind=field,
                msg_id_prompt=prompt_msg_id,
                card_msg_id=card_msg_id,
                user_id=uid,
            )
            wait_kind_map = {
                WAIT_SUNO_TITLE: WaitKind.SUNO_TITLE,
                WAIT_SUNO_STYLE: WaitKind.SUNO_STYLE,
                WAIT_SUNO_LYRICS: WaitKind.SUNO_LYRICS,
            }
            wait_kind = wait_kind_map.get(waiting_state)
            if wait_kind is not None:
                _activate_wait_state(
                    user_id=uid,
                    chat_id=target_chat,
                    card_msg_id=card_msg_id,
                    kind=wait_kind,
                    meta={"field": field},
            )
            return

        if action == "confirm" and argument == "auto_lyrics":
            s["suno_lyrics_confirmed"] = True
            await q.answer("Текст подтверждён")
            target_chat = chat_id
            if target_chat is None and q.message is not None:
                target_chat = q.message.chat_id
            if target_chat is not None:
                await _suno_notify(
                    ctx,
                    target_chat,
                    "✅ Текст принят. Можно запускать генерацию.",
                    reply_to=q.message,
                )
            return

        if action == "preset":
            if argument != "ambient":
                await q.answer("Неизвестный пресет", show_alert=True)
                return
            cfg = get_preset_config(AMBIENT_NATURE_PRESET_ID) or AMBIENT_NATURE_PRESET
            suggestions = list(cfg.get("title_suggestions") or [])
            if suggestions:
                suggestion = random.choice(suggestions)
            else:
                suggestion = "Oceanic Dreams"
            set_suno_title(suno_state_obj, suggestion)
            clear_suno_style(suno_state_obj)
            clear_suno_lyrics(suno_state_obj)
            suno_state_obj.mode = "instrumental"
            suno_state_obj.preset = AMBIENT_NATURE_PRESET_ID
            save_suno_state(ctx, suno_state_obj)
            s["suno_state"] = suno_state_obj.to_dict()
            s["suno_waiting_state"] = IDLE_SUNO
            s["suno_flow"] = "instrumental"
            s["suno_step_order"] = _music_flow_steps("instrumental")
            s["suno_step"] = None
            s["suno_auto_lyrics_pending"] = False
            s["suno_auto_lyrics_generated"] = False
            s["suno_lyrics_confirmed"] = False
            _reset_suno_card_cache(s)
            await q.answer()
            target_chat = chat_id
            if target_chat is None and q.message is not None:
                target_chat = q.message.chat_id
            if target_chat is not None:
                await refresh_suno_card(ctx, target_chat, s, price=PRICE_SUNO)
                description = "ocean waves, birds, wind, and experimental instruments"
                message_text = (
                    f"✅ Ambient preset selected ({suggestion})\n"
                    f"🎶 Generating track with {description}..."
                )
                await _suno_notify(
                    ctx,
                    target_chat,
                    message_text,
                    reply_to=q.message,
                )
            return

        if action == "toggle" and argument == "instrumental":
            suno_state_obj.mode = "instrumental" if suno_state_obj.has_lyrics else "lyrics"
            save_suno_state(ctx, suno_state_obj)
            s["suno_state"] = suno_state_obj.to_dict()
            s["suno_waiting_state"] = IDLE_SUNO
            _reset_suno_card_cache(s)
            if chat_id is not None:
                await refresh_suno_card(ctx, chat_id, s, price=PRICE_SUNO)
            mode_label = "Инструментал" if suno_state_obj.mode == "instrumental" else "Со словами"
            await q.answer(f"Режим: {mode_label}")
            return

        if action == "busy":
            await q.answer("Уже идёт генерация музыки — дождитесь завершения.")
            return

        if action == "start":
            if chat_id is None:
                await q.answer("Нет чата", show_alert=True)
                return

            if suno_state_obj.start_clicked or bool(s.get("suno_start_clicked")):
                await q.answer("Уже идёт генерация музыки — дождитесь завершения.")
                return

            if uid is not None:
                start_lock_ok = _acquire_suno_start_lock(int(uid))
                if not start_lock_ok:
                    await q.answer()
                    return

            await q.answer()

            missing_fields = _suno_missing_fields(suno_state_obj)
            if missing_fields:
                fields_text = ", ".join(missing_fields)
                await _suno_notify(
                    ctx,
                    chat_id,
                    f"⚠️ {t('suno.prompt.fill', fields=fields_text)}.",
                    reply_to=q.message,
                )
                return

            raw_start_msg = s.get("suno_start_msg_id")
            start_msg_id = raw_start_msg if isinstance(raw_start_msg, int) else None
            if start_msg_id is None and isinstance(suno_state_obj.start_msg_id, int):
                start_msg_id = suno_state_obj.start_msg_id

            msg_ids_map = s.get("msg_ids")
            if isinstance(msg_ids_map, dict) and isinstance(start_msg_id, int):
                msg_ids_map["suno_start"] = start_msg_id
            if isinstance(start_msg_id, int):
                s["suno_start_msg_id"] = start_msg_id

            suno_state_obj.start_clicked = True
            suno_state_obj.start_msg_id = start_msg_id if isinstance(start_msg_id, int) else None
            suno_state_obj.start_emoji_msg_id = None
            save_suno_state(ctx, suno_state_obj)
            s["suno_state"] = suno_state_obj.to_dict()

            if isinstance(start_msg_id, int):
                try:
                    await safe_edit_message(
                        ctx,
                        chat_id,
                        start_msg_id,
                        SUNO_START_READY_MESSAGE,
                        suno_start_disabled_keyboard(),
                        parse_mode=ParseMode.HTML,
                        disable_web_page_preview=True,
                    )
                except BadRequest as exc:
                    log.debug(
                        "suno start message edit failed | chat=%s msg=%s err=%s",
                        chat_id,
                        start_msg_id,
                        exc,
                    )
                except Exception as exc:
                    log.warning(
                        "suno start message edit error | chat=%s msg=%s err=%s",
                        chat_id,
                        start_msg_id,
                        exc,
                    )

            emoji_msg_id: Optional[int] = None
            sticker_id = START_EMOJI_STICKER_ID.strip()
            if sticker_id:
                try:
                    sticker_message = await safe_send_sticker(ctx.bot, chat_id, sticker_id)
                    if sticker_message is not None:
                        emoji_msg_id = getattr(sticker_message, "message_id", None)
                except Exception as exc:
                    log.warning(
                        "suno start sticker failed | user=%s chat=%s err=%s",
                        uid,
                        chat_id,
                        exc,
                    )

            if emoji_msg_id is None:
                fallback_emoji = START_EMOJI_FALLBACK or "🎬"
                try:
                    fallback_message = await safe_send_placeholder(
                        ctx.bot,
                        chat_id,
                        fallback_emoji,
                    )
                    if fallback_message is not None:
                        emoji_msg_id = getattr(fallback_message, "message_id", None)
                except Exception as exc:
                    log.warning(
                        "suno start fallback emoji failed | user=%s chat=%s err=%s",
                        uid,
                        chat_id,
                        exc,
                    )

            suno_state_obj.start_emoji_msg_id = emoji_msg_id
            save_suno_state(ctx, suno_state_obj)
            s["suno_state"] = suno_state_obj.to_dict()
            s["suno_start_clicked"] = True
            s["suno_can_start"] = False
            if uid is not None:
                s["suno_current_req_id"] = _generate_suno_request_id(int(uid))
            else:
                s["suno_current_req_id"] = f"suno:anon:{uuid.uuid4()}"
            s["suno_current_lyrics_hash"] = suno_state_obj.lyrics_hash

            await _suno_notify(
                ctx,
                chat_id,
                SUNO_STARTING_MESSAGE,
            )
            params = _suno_collect_params(s, suno_state_obj)
            lock_acquired = False
            if uid is not None:
                lock_acquired = _acquire_suno_lock(int(uid))
                if not lock_acquired:
                    await _suno_notify(
                        ctx,
                        chat_id,
                        "⏳ Уже есть активная задача Suno. Дождитесь завершения предыдущей.",
                        reply_to=q.message,
                    )
                    return
            try:
                await _launch_suno_generation(
                    chat_id,
                    ctx,
                    params=params,
                    user_id=uid,
                    reply_to=q.message,
                    trigger="start",
                )
            finally:
                if lock_acquired and uid is not None:
                    _release_suno_lock(int(uid))
            return

        if action == "repeat":
            if chat_id is None:
                await q.answer("Нет чата", show_alert=True)
                return
            params = s.get("suno_last_params")
            if not isinstance(params, dict) or not params:
                await q.answer("Нет предыдущих параметров", show_alert=True)
                return
            await q.answer()
            lock_acquired = False
            if uid is not None:
                lock_acquired = _acquire_suno_lock(int(uid))
                if not lock_acquired:
                    await _suno_notify(
                        ctx,
                        chat_id,
                        "⏳ Уже есть активная задача Suno. Дождитесь завершения предыдущей.",
                        reply_to=q.message,
                    )
                    return
            try:
                await _launch_suno_generation(
                    chat_id,
                    ctx,
                    params=params,
                    user_id=uid,
                    reply_to=q.message,
                    trigger="repeat",
                )
            finally:
                if lock_acquired and uid is not None:
                    _release_suno_lock(int(uid))
            return

        await q.answer()
        return

    # -------- VEO card actions --------
    if data.startswith("veo:set_ar:"):
        s["aspect"] = "9:16" if data.endswith("9:16") else "16:9"
        await show_veo_card(update.effective_chat.id, ctx); return
    if data.startswith("veo:set_model:"):
        s["model"] = "veo3_fast" if data.endswith("fast") else "veo3"
        await show_veo_card(update.effective_chat.id, ctx); return
    if data == "veo:clear_img":
        s["last_image_url"] = None
        await show_veo_card(update.effective_chat.id, ctx); return
    if data == "veo:start":
        prompt = (s.get("last_prompt") or "").strip()
        if not prompt:
            await q.message.reply_text("⚠️ Сначала пришлите текстовый промпт."); return
        user = update.effective_user
        user_id_for_lock = user.id if user else None
        if not _acquire_click_lock(user_id_for_lock, "veo:start"):
            return
        chat_id = update.effective_chat.id
        if ACTIVE_TASKS.get(chat_id):
            await q.message.reply_text("⏳ Уже рендерю вашу предыдущую задачу. Подождите, пожалуйста.")
            return
        uid = user.id if user else None
        if not uid:
            await q.message.reply_text("⚠️ Не удалось определить пользователя. Попробуйте позже.")
            return
        mode = s.get("mode") or ""
        if mode == "veo_photo":
            price = PRICE_VEO_ANIMATE
            service_name = "VEO_ANIMATE"
        elif s.get("model") == "veo3":
            price = PRICE_VEO_QUALITY
            service_name = "VEO_QUALITY"
        else:
            price = PRICE_VEO_FAST
            service_name = "VEO_FAST"
        try:
            ensure_user(uid)
        except Exception as exc:
            log.exception("VEO ensure_user failed for %s: %s", uid, exc)
            await q.message.reply_text("⚠️ Не удалось проверить баланс. Попробуйте позже.")
            return
        meta = {
            "service": service_name,
            "prompt": _short_prompt(prompt, 160),
            "aspect": s.get("aspect") or "16:9",
            "model": s.get("model") or "veo3_fast",
            "has_image": bool(s.get("last_image_url")),
        }
        if not await ensure_tokens(ctx, chat_id, uid, price):
            return
        ok, balance_after = debit_try(
            uid,
            price,
            reason="service:start",
            meta=meta,
        )
        if not ok:
            await ensure_tokens(ctx, chat_id, uid, price)
            return
        clear_wait_state(uid, reason="veo_start")
        await show_balance_notification(
            chat_id,
            ctx,
            uid,
            f"✅ Списано {price}💎. Текущий баланс: {balance_after}💎 — запускаю…",
        )
        ACTIVE_TASKS[chat_id] = "__pending__"
        await q.message.reply_text("🎬 Отправляю задачу в VEO…")
        try:
            ok, task_id, msg = await asyncio.to_thread(
                submit_kie_veo,
                prompt,
                (s.get("aspect") or "16:9"),
                s.get("last_image_url"),
                s.get("model") or "veo3_fast",
            )
        except Exception as exc:
            log.exception("VEO submit crashed: %s", exc)
            ACTIVE_TASKS.pop(chat_id, None)
            try:
                new_balance = credit_balance(
                    uid,
                    price,
                    reason="service:refund",
                    meta={"service": service_name, "reason": "submit_exception", "message": str(exc)},
                )
            except Exception as refund_exc:
                log.exception("VEO submit crash refund failed for %s: %s", uid, refund_exc)
                new_balance = None
            if new_balance is not None:
                await show_balance_notification(
                    chat_id,
                    ctx,
                    uid,
                    f"⚠️ Ошибка. Возврат {price}💎. Баланс: {new_balance}💎",
                )
            await q.message.reply_text("⚠️ Не удалось создать VEO-задачу. 💎 Токены возвращены.")
            return
        if not ok or not task_id:
            ACTIVE_TASKS.pop(chat_id, None)
            try:
                new_balance = credit_balance(
                    uid,
                    price,
                    reason="service:refund",
                    meta={"service": service_name, "reason": "submit_failed", "message": msg},
                )
            except Exception as exc:
                log.exception("VEO submit refund failed for %s: %s", uid, exc)
                new_balance = None
            if new_balance is not None:
                await show_balance_notification(
                    chat_id,
                    ctx,
                    uid,
                    f"⚠️ Ошибка. Возврат {price}💎. Баланс: {new_balance}💎",
                )
            await q.message.reply_text(
                f"❌ Не удалось создать VEO-задачу: {msg}\n💎 Токены возвращены.",
            )
            return
        gen_id = uuid.uuid4().hex
        s["generating"] = True
        s["generation_id"] = gen_id
        s["last_task_id"] = task_id
        ACTIVE_TASKS[chat_id] = task_id
        try:
            effective_message = update.effective_message
            message_id = effective_message.message_id if effective_message else 0
            mode = s.get("mode") or ""
            aspect_for_meta = s.get("aspect") or "16:9"
            save_task_meta(task_id, chat_id, int(message_id), mode, aspect_for_meta)
            log.info("task-meta saved | task=%s chat=%s", task_id, chat_id)
        except Exception:
            log.exception("Failed to save task meta for %s", task_id)
        await q.message.reply_text("🎬 Рендер начат — вернусь с готовым видео.")
        wait_msg_id = await send_wait_sticker(ctx, "veo", chat_id=chat_id)
        if wait_msg_id:
            s["video_wait_message_id"] = wait_msg_id
        asyncio.create_task(
            poll_veo_and_send(update.effective_chat.id, task_id, gen_id, ctx, uid, price, service_name)
        ); return

async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await ensure_user_record(update)
    msg = update.message
    if msg is None:
        return

    s = state(ctx)
    raw_text = msg.text or ""
    text = raw_text.strip()
    chat_id = msg.chat_id
    user = update.effective_user
    user_id = user.id if user else None
    state_mode = s.get("mode")
    user_mode = _mode_get(chat_id) or MODE_CHAT

    waiting_for_input = _chat_state_waiting_input(s)
    if (
        waiting_for_input
        and user_id
        and user_mode != MODE_PM
        and not chat_mode_is_on(user_id)
    ):
        chat_autoswitch_total.labels(outcome="skip_active").inc()

    mapped_command = label_to_command(text)
    if mapped_command:
        handler = LABEL_COMMAND_ROUTES.get(mapped_command)
        if handler is not None:
            await handler(update, ctx)
            return

    lowered = text.lower()
    if lowered in {"меню", "menu", "главное меню"}:
        await handle_menu(update, ctx)
        return

    if state_mode == "mj_upscale":
        locale = s.get("mj_locale") or _determine_user_locale(user)
        s["mj_locale"] = locale
        await msg.reply_text(_mj_ui_text("upscale_need_photo", locale))
        return

    if state_mode == "promo":
        if not PROMO_ENABLED:
            await msg.reply_text("🎟️ Промокоды временно отключены.")
            s["mode"] = None
            return
        await process_promo_submission(update, ctx, text)
        return

    if state_mode == "suno" and lowered == "reset":
        suno_state_obj = load_suno_state(ctx)
        clear_suno_title(suno_state_obj)
        clear_suno_style(suno_state_obj)
        save_suno_state(ctx, suno_state_obj)
        s["suno_state"] = suno_state_obj.to_dict()
        _reset_suno_card_cache(s)
        log.info("suno input cleared", extra={"field": "reset", "user_id": user_id})
        await refresh_suno_card(ctx, chat_id, s, price=PRICE_SUNO)
        await msg.reply_text("Название и стиль очищены.")
        return

    waiting_cover = s.get("suno_waiting_state") == WAIT_SUNO_REFERENCE
    if (
        state_mode == "suno"
        and s.get("suno_flow") == "cover"
        and (waiting_cover or s.get("suno_step") == "source")
    ):
        if not text:
            await msg.reply_text(_COVER_INVALID_INPUT_MESSAGE)
            return
        await _cover_process_url_input(
            ctx,
            chat_id,
            msg,
            s,
            text.strip(),
            user_id=user_id,
        )
        return

    if user_mode == MODE_PM:
        await prompt_master_process(update, ctx)
        return

    low = text.lower()
    if low.startswith(("http://", "https://")) and any(
        low.split("?")[0].endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp", ".heic")
    ):
        if state_mode == "banana":
            if len(_get_banana_images(s)) >= 4:
                await msg.reply_text("⚠️ Достигнут лимит 4 фото.", reply_markup=banana_kb())
                return
            await on_banana_photo_received(chat_id, ctx, text.strip())
            await msg.reply_text(f"📸 Фото принято ({len(s['banana_images'])}/4).")
            return
        s["last_image_url"] = text.strip()
        await msg.reply_text("🧷 Ссылка на изображение принята.")
        if state_mode in ("veo_text_fast", "veo_text_quality", "veo_photo"):
            await show_veo_card(chat_id, ctx)
        return

    if state_mode == "mj_txt":
        if not text:
            await msg.reply_text("⚠️ Отправьте текстовый промпт.")
            return
        s["last_prompt"] = text
        await show_mj_prompt_card(chat_id, ctx)
        await msg.reply_text("📝 Промпт сохранён. Нажмите «Подтвердить».")
        return

    if state_mode == "banana":
        await msg.reply_text("✍️ Промпт сохранён.")
        await on_banana_prompt_saved(chat_id, ctx, text)
        return

    if state_mode in ("veo_text_fast", "veo_text_quality", "veo_photo"):
        s["last_prompt"] = text
        await show_veo_card(chat_id, ctx)
        return

    waiting_for_input = _chat_state_waiting_input(s)
    if (
        user_id
        and user_mode != MODE_PM
        and not chat_mode_is_on(user_id)
        and not waiting_for_input
    ):
        chat_autoswitch_total.labels(outcome="on").inc()
        chat_mode_turn_on(user_id)
        _mode_set(chat_id, MODE_CHAT)
        s["mode"] = None
        await _handle_chat_message(
            ctx=ctx,
            chat_id=chat_id,
            user_id=user_id,
            state_dict=s,
            raw_text=raw_text,
            text=text,
            send_typing_action=True,
            send_hint=True,
            inline_keyboard=main_suggest_kb(),
        )
        return

    if user_mode == MODE_CHAT:
        if not user_id:
            return
        if not chat_mode_is_on(user_id):
            return
        await _handle_chat_message(
            ctx=ctx,
            chat_id=chat_id,
            user_id=user_id,
            state_dict=s,
            raw_text=raw_text,
            text=text,
            send_typing_action=True,
            send_hint=False,
        )
        return

    if user_mode != MODE_CHAT:
        await msg.reply_text("ℹ️ Выберите раздел в /menu")
        return

    s["last_prompt"] = text
    await show_veo_card(chat_id, ctx)

async def _banana_run_and_send(
    chat_id: int,
    ctx: ContextTypes.DEFAULT_TYPE,
    src_images: Optional[Sequence[Any]] = None,
    prompt: str = "",
    price: int = 0,
    user_id: int = 0,
    *,
    src_urls: Optional[Sequence[str]] = None,
) -> None:
    s = state(ctx)
    normalized: list[Dict[str, str]]
    if src_urls is not None:
        normalized = [
            {"url": str(url)}
            for url in src_urls
            if isinstance(url, str) and url.strip()
        ]
    else:
        image_items = src_images or []
        url_sources = [_normalize_banana_image(item) for item in image_items]
        normalized = [entry for entry in url_sources if entry]
    src_urls = [entry["url"] for entry in normalized if entry.get("url")]
    task_info: Dict[str, Optional[str]] = {"id": None}

    async def _refund(reason_tag: str, message: Optional[str] = None) -> Optional[int]:
        meta: Dict[str, Any] = {"service": "BANANA", "reason": reason_tag}
        if message:
            meta["message"] = message
        task_id_val = task_info.get("id")
        if task_id_val:
            meta["task_id"] = task_id_val
        try:
            new_balance = credit_balance(
                user_id,
                price,
                reason="service:refund",
                meta=meta,
            )
        except Exception as exc:
            log.exception("Banana refund %s failed: %s", reason_tag, exc)
            return None
        s["banana_balance"] = new_balance
        s["_last_text_banana"] = None
        await show_banana_card(chat_id, ctx)
        await show_balance_notification(
            chat_id,
            ctx,
            user_id,
            f"⚠️ Ошибка. Возврат {price}💎. Баланс: {new_balance}💎",
        )
        return new_balance

    try:
        task_id = await asyncio.to_thread(
            create_banana_task, prompt, src_urls, "png", "auto", None, None, 60
        )
        task_info["id"] = str(task_id)
        await ctx.bot.send_message(
            chat_id,
            f"🍌 Задача Banana создана.\n🆔 taskId={task_id}\nЖдём результат…",
        )
        urls = await asyncio.to_thread(wait_for_banana_result, task_id, 8 * 60, 3)
        if not urls:
            new_balance = await _refund("empty")
            msg = "⚠️ Banana вернула пустой результат. Произошла ошибка, 5💎 возвращены."
            if new_balance is not None:
                msg += f" Текущий баланс: {new_balance}."
            await ctx.bot.send_message(chat_id, msg)
            return
        u0 = urls[0]
        data, content_type = await asyncio.to_thread(_download_binary, u0)
        if not data:
            raise RuntimeError("banana returned empty file")
        suffix = _banana_guess_suffix(u0, content_type)
        temp_path = save_bytes_to_temp(data, suffix=suffix)
        caption = _banana_caption(prompt)
        delivered = await _deliver_banana_media(
            ctx.bot,
            chat_id=chat_id,
            user_id=user_id,
            file_path=temp_path,
            caption=caption,
            reply_markup=None,
            send_document=BANANA_SEND_AS_DOCUMENT,
        )
        if delivered:
            try:
                await ctx.bot.send_message(
                    chat_id,
                    "Галерея сгенерирована.",
                    reply_markup=banana_result_keyboard(),
                )
            except Exception as exc:
                log.warning(
                    "banana.result.keyboard_fail",
                    extra={"meta": {"chat_id": chat_id, "error": str(exc)}},
                )
        else:
            await ctx.bot.send_message(
                chat_id,
                "❌ Не удалось отправить изображение Banana. Попробуйте позже.",
            )
    except KieBananaError as e:
        new_balance = await _refund("error", str(e))
        msg = f"❌ Banana ошибка: {e}\nПроизошла ошибка, 5💎 возвращены."
        if new_balance is not None:
            msg += f" Текущий баланс: {new_balance}."
        await ctx.bot.send_message(chat_id, msg)
    except Exception as e:
        new_balance = await _refund("exception", str(e))
        log.exception("BANANA unexpected: %s", e)
        msg = "💥 Внутренняя ошибка Banana. Произошла ошибка, 5💎 возвращены."
        if new_balance is not None:
            msg += f" Текущий баланс: {new_balance}."
        await ctx.bot.send_message(chat_id, msg)

async def on_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await ensure_user_record(update)
    message = update.effective_message
    if message is None:
        return

    chat = update.effective_chat
    chat_id = chat.id if chat else None
    s = state(ctx)

    try:
        image = await download_image_from_update(update, ctx.bot)
    except TelegramImageError as exc:
        if exc.reason == "too_large":
            await message.reply_text("Файл слишком большой (лимит 20 MB).")
            return
        await message.reply_text("⚠️ Не удалось обработать фото. Пришлите публичный URL картинки текстом.")
        return
    except Exception as exc:
        log.exception("Get photo failed: %s", exc)
        await message.reply_text("⚠️ Не удалось обработать фото. Пришлите публичный URL картинки текстом.")
        return

    if not image.file_path:
        await message.reply_text("⚠️ Не удалось получить путь к файлу Telegram.")
        return

    url = tg_direct_file_url(TELEGRAM_TOKEN, image.file_path)

    if s.get("mode") == "mj_upscale":
        await _handle_mj_upscale_input(
            update,
            ctx,
            url,
            width=image.width,
            height=image.height,
            source=image.source,
        )
        return

    if s.get("mode") == "banana":
        images = _get_banana_images(s)
        if len(images) >= 4:
            await message.reply_text("⚠️ Достигнут лимит 4 фото.", reply_markup=banana_kb())
            return
        caption = (message.caption or "").strip()
        if caption:
            s["last_prompt"] = caption
            s["_last_text_banana"] = None
        payload = {
            "url": url,
            "file_path": image.file_path,
            "filename": image.filename,
            "mime": image.mime_type,
            "source": image.source,
            "width": image.width,
            "height": image.height,
        }
        entry = _normalize_banana_image(payload)
        if entry is None:
            await message.reply_text("⚠️ Не удалось обработать фото. Пришлите публичный URL картинки текстом.")
            return
        if chat_id is not None:
            await on_banana_photo_received(chat_id, ctx, entry)
        else:
            images.append(entry)
            s["_last_text_banana"] = None
        await message.reply_text(f"📸 Фото принято ({len(s['banana_images'])}/4).")
        return

    s["last_image_url"] = url
    await message.reply_text("🖼️ Фото принято как референс.")
    if chat_id is not None and s.get("mode") in ("veo_text_fast", "veo_text_quality", "veo_photo"):
        await show_veo_card(chat_id, ctx)


async def on_document(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await ensure_user_record(update)
    message = update.effective_message
    if message is None:
        return
    doc = message.document
    if doc is None:
        return

    s = state(ctx)
    try:
        image = await download_image_from_update(update, ctx.bot)
    except TelegramImageError as exc:
        if exc.reason == "invalid_type":
            await message.reply_text("Нужно прислать изображение (PNG/JPG/WEBP) как файл-документ.")
            return
        if exc.reason == "too_large":
            await message.reply_text("Файл слишком большой (лимит 20 MB).")
            return
        await message.reply_text("⚠️ Не удалось получить файл. Попробуйте снова.")
        return
    except Exception as exc:
        log.exception("Get document failed: %s", exc)
        await message.reply_text("⚠️ Не удалось получить файл. Попробуйте снова.")
        return

    if not image.file_path:
        await message.reply_text("⚠️ Не удалось получить путь к файлу Telegram.")
        return

    chat = update.effective_chat
    chat_id = chat.id if chat else None
    url = tg_direct_file_url(TELEGRAM_TOKEN, image.file_path)

    if s.get("mode") == "banana":
        images = _get_banana_images(s)
        if len(images) >= 4:
            await message.reply_text("⚠️ Достигнут лимит 4 фото.", reply_markup=banana_kb())
            return
        caption = (message.caption or "").strip()
        if caption:
            s["last_prompt"] = caption
            s["_last_text_banana"] = None
        payload = {
            "url": url,
            "file_path": image.file_path,
            "filename": image.filename,
            "mime": image.mime_type,
            "source": image.source,
            "width": image.width,
            "height": image.height,
        }
        entry = _normalize_banana_image(payload)
        if entry is None:
            await message.reply_text("⚠️ Не удалось получить файл. Попробуйте снова.")
            return
        if chat_id is not None:
            await on_banana_photo_received(chat_id, ctx, entry)
        else:
            images.append(entry)
            s["_last_text_banana"] = None
        await message.reply_text(f"📸 Фото принято ({len(s['banana_images'])}/4).")
        return

    if s.get("mode") != "mj_upscale":
        return

    await _handle_mj_upscale_input(
        update,
        ctx,
        url,
        width=image.width,
        height=image.height,
        source=image.source,
    )

# --- Voice handling -----------------------------------------------------


async def handle_voice(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await ensure_user_record(update)
    message = update.message
    if message is None:
        return

    document_audio = None
    document_obj = getattr(message, "document", None)
    if document_obj and str(getattr(document_obj, "mime_type", "")).startswith("audio/"):
        document_audio = document_obj

    voice_or_audio = message.voice or message.audio or document_audio
    if voice_or_audio is None:
        return

    chat_id = message.chat_id
    user = update.effective_user
    user_id = user.id if user else None
    start_time = time.time()

    s = state(ctx)
    waiting_cover = s.get("suno_waiting_state") == WAIT_SUNO_REFERENCE
    if (
        s.get("mode") == "suno"
        and s.get("suno_flow") == "cover"
        and (waiting_cover or s.get("suno_step") == "source")
    ):
        await _cover_process_audio_input(
            ctx,
            chat_id,
            message,
            s,
            voice_or_audio,
            user_id=user_id,
        )
        return

    file_size = getattr(voice_or_audio, "file_size", None) or 0
    duration = getattr(voice_or_audio, "duration", None) or 0

    if file_size and file_size > VOICE_MAX_SIZE_BYTES:
        payload = md2_escape(VOICE_TOO_LARGE_TEXT)
        try:
            await safe_send_text(ctx.bot, chat_id, payload)
        except Exception as exc:
            log.warning("chat.voice.too_large_notify_failed | chat=%s err=%s", chat_id, exc)
        chat_voice_total.labels(outcome="too_large", **_VOICE_METRIC_LABELS).inc()
        chat_voice_latency_ms.labels(**_VOICE_METRIC_LABELS).observe(
            (time.time() - start_time) * 1000.0
        )
        return

    if duration and duration > VOICE_MAX_DURATION_SEC:
        payload = md2_escape(VOICE_TOO_LARGE_TEXT)
        try:
            await safe_send_text(ctx.bot, chat_id, payload)
        except Exception as exc:
            log.warning("chat.voice.too_long_notify_failed | chat=%s err=%s", chat_id, exc)
        chat_voice_total.labels(outcome="too_long", **_VOICE_METRIC_LABELS).inc()
        chat_voice_latency_ms.labels(**_VOICE_METRIC_LABELS).observe(
            (time.time() - start_time) * 1000.0
        )
        return

    with suppress(Exception):
        await ctx.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    placeholder = None
    try:
        placeholder = await safe_send_placeholder(
            ctx.bot, chat_id, md2_escape(VOICE_PLACEHOLDER_TEXT)
        )
    except Exception as exc:
        log.warning("chat.voice.placeholder_failed | chat=%s err=%s", chat_id, exc)
        placeholder = None

    transcribe_started: Optional[float] = None
    transcription: Optional[str] = None
    mime_type = getattr(voice_or_audio, "mime_type", None)
    try:
        file = await ctx.bot.get_file(voice_or_audio.file_id)
        file_path = getattr(file, "file_path", None)
        if not file_path:
            raise RuntimeError("telegram file path missing")
        url = tg_direct_file_url(TELEGRAM_TOKEN, file_path)
        audio_bytes = await _download_telegram_file(url)
        actual_mime = mime_type or getattr(file, "mime_type", None)
        if _should_convert_to_wav(actual_mime, file_path):
            audio_bytes = await run_ffmpeg(
                audio_bytes,
                ["-i", "pipe:0", "-ac", "1", "-ar", "16000", "-f", "wav", "pipe:1"],
            )
            actual_mime = "audio/wav"
        lang_hint = _voice_lang_hint(message, user)
        transcribe_started = time.time()
        transcription = await asyncio.to_thread(
            voice_transcribe, audio_bytes, actual_mime, lang_hint
        )
    except VoiceTranscribeError as exc:
        if transcribe_started is not None:
            chat_transcribe_latency_ms.labels(**_VOICE_METRIC_LABELS).observe(
                (time.time() - transcribe_started) * 1000.0
            )
        err_text = md2_escape(VOICE_TRANSCRIBE_ERROR_TEXT)
        handled = False
        if placeholder and getattr(placeholder, "message_id", None) is not None:
            try:
                await safe_edit_markdown_v2(
                    ctx.bot, chat_id, placeholder.message_id, err_text
                )
                handled = True
            except Exception as edit_exc:
                log.warning(
                    "chat.voice.transcribe_edit_failed | chat=%s err=%s",
                    chat_id,
                    edit_exc,
                )
        if not handled:
            with suppress(Exception):
                await safe_send_text(ctx.bot, chat_id, err_text)
        chat_voice_total.labels(outcome="error", **_VOICE_METRIC_LABELS).inc()
        chat_voice_latency_ms.labels(**_VOICE_METRIC_LABELS).observe(
            (time.time() - start_time) * 1000.0
        )
        log.warning("chat.voice.transcribe_failed | chat=%s err=%s", chat_id, exc)
        return
    except Exception as exc:
        if transcribe_started is not None:
            chat_transcribe_latency_ms.labels(**_VOICE_METRIC_LABELS).observe(
                (time.time() - transcribe_started) * 1000.0
            )
        err_text = md2_escape(VOICE_TRANSCRIBE_ERROR_TEXT)
        handled = False
        if placeholder and getattr(placeholder, "message_id", None) is not None:
            try:
                await safe_edit_markdown_v2(
                    ctx.bot, chat_id, placeholder.message_id, err_text
                )
                handled = True
            except Exception as edit_exc:
                log.warning(
                    "chat.voice.general_edit_failed | chat=%s err=%s",
                    chat_id,
                    edit_exc,
                )
        if not handled:
            with suppress(Exception):
                await safe_send_text(ctx.bot, chat_id, err_text)
        chat_voice_total.labels(outcome="error", **_VOICE_METRIC_LABELS).inc()
        chat_voice_latency_ms.labels(**_VOICE_METRIC_LABELS).observe(
            (time.time() - start_time) * 1000.0
        )
        log.exception("chat.voice.error | chat=%s", chat_id)
        return

    if transcribe_started is not None:
        chat_transcribe_latency_ms.labels(**_VOICE_METRIC_LABELS).observe(
            (time.time() - transcribe_started) * 1000.0
        )

    assert transcription is not None
    preview = _voice_preview(transcription)
    thinking_text = md2_escape(
        f"📝 Расшифровка:\n{preview}\n\nДумаю над ответом…"
    )
    placeholder_msg_id = getattr(placeholder, "message_id", None)
    if placeholder_msg_id is not None:
        try:
            await safe_edit_markdown_v2(ctx.bot, chat_id, placeholder_msg_id, thinking_text)
        except Exception as exc:
            log.warning("chat.voice.thinking_edit_failed | chat=%s err=%s", chat_id, exc)
            placeholder_msg_id = None
    if placeholder_msg_id is None:
        try:
            new_placeholder = await safe_send_placeholder(ctx.bot, chat_id, thinking_text)
        except Exception as exc:
            log.warning("chat.voice.thinking_send_failed | chat=%s err=%s", chat_id, exc)
            new_placeholder = None
        placeholder_msg_id = getattr(new_placeholder, "message_id", None)

    user_mode = _mode_get(chat_id) or MODE_CHAT
    chat_enabled = False
    if user_id:
        try:
            chat_enabled = chat_mode_is_on(user_id) or is_mode_on(user_id)
        except Exception:
            chat_enabled = is_mode_on(user_id)
    if user_mode != MODE_CHAT or not user_id or not chat_enabled:
        final_text = md2_escape(
            f"📝 Расшифровка:\n{preview}\n\n💬 Включить чат: /chat"
        )
        delivered = False
        if placeholder_msg_id is not None:
            try:
                await safe_edit_markdown_v2(ctx.bot, chat_id, placeholder_msg_id, final_text)
                delivered = True
            except Exception as exc:
                log.warning("chat.voice.final_edit_failed | chat=%s err=%s", chat_id, exc)
        if not delivered:
            with suppress(Exception):
                await safe_send_text(ctx.bot, chat_id, final_text)
        chat_voice_total.labels(outcome="ok", **_VOICE_METRIC_LABELS).inc()
        chat_voice_latency_ms.labels(**_VOICE_METRIC_LABELS).observe(
            (time.time() - start_time) * 1000.0
        )
        return

    if rate_limit_hit(user_id):
        chat_messages_total.labels(outcome="rate_limited").inc()
        rate_text = md2_escape(
            f"📝 Расшифровка:\n{preview}\n\n⏳ Слишком быстро. Подождите секунду и повторите."
        )
        handled = False
        if placeholder_msg_id is not None:
            try:
                await safe_edit_markdown_v2(ctx.bot, chat_id, placeholder_msg_id, rate_text)
                handled = True
            except Exception as exc:
                log.warning("chat.voice.rate_limit_edit_failed | chat=%s err=%s", chat_id, exc)
        if not handled:
            with suppress(Exception):
                await safe_send_text(ctx.bot, chat_id, rate_text)
        chat_voice_total.labels(outcome="ok", **_VOICE_METRIC_LABELS).inc()
        chat_voice_latency_ms.labels(**_VOICE_METRIC_LABELS).observe(
            (time.time() - start_time) * 1000.0
        )
        return

    history = load_ctx(user_id)
    ctx_tokens = sum(estimate_tokens(str(item.get("content", ""))) for item in history)
    ctx_tokens += estimate_tokens(transcription)
    chat_context_tokens.set(float(min(ctx_tokens, CTX_MAX_TOKENS)))

    append_ctx(user_id, "user", transcription)

    lang = detect_lang(transcription)
    messages = build_messages(CHAT_SYSTEM_PROMPT, history, transcription, lang)

    chat_start = time.time()
    try:
        answer = await asyncio.to_thread(call_llm, messages)
        append_ctx(user_id, "assistant", answer)
        final_payload = md2_escape(answer)
        delivered = False
        if placeholder_msg_id is not None:
            try:
                await safe_edit_markdown_v2(
                    ctx.bot, chat_id, placeholder_msg_id, final_payload
                )
                delivered = True
            except Exception as exc:
                log.warning("chat.voice.answer_edit_failed | chat=%s err=%s", chat_id, exc)
        if not delivered:
            with suppress(Exception):
                await safe_send_text(ctx.bot, chat_id, final_payload)

        chat_messages_total.labels(outcome="ok").inc()
        chat_latency_ms.observe((time.time() - chat_start) * 1000.0)
        chat_voice_total.labels(outcome="ok", **_VOICE_METRIC_LABELS).inc()
        chat_voice_latency_ms.labels(**_VOICE_METRIC_LABELS).observe(
            (time.time() - start_time) * 1000.0
        )
    except Exception as exc:
        chat_messages_total.labels(outcome="error").inc()
        chat_latency_ms.observe((time.time() - chat_start) * 1000.0)
        error_payload = md2_escape("⚠️ Не получилось ответить сейчас. Попробуйте ещё раз.")
        handled = False
        if placeholder_msg_id is not None:
            try:
                await safe_edit_markdown_v2(
                    ctx.bot, chat_id, placeholder_msg_id, error_payload
                )
                handled = True
            except Exception as edit_exc:
                log.warning(
                    "chat.voice.answer_error_edit_failed | chat=%s err=%s",
                    chat_id,
                    edit_exc,
                )
        if not handled:
            with suppress(Exception):
                await safe_send_text(ctx.bot, chat_id, error_payload)

        chat_voice_total.labels(outcome="error", **_VOICE_METRIC_LABELS).inc()
        chat_voice_latency_ms.labels(**_VOICE_METRIC_LABELS).observe(
            (time.time() - start_time) * 1000.0
        )
        app_logger = getattr(getattr(ctx, "application", None), "logger", log)
        app_logger.exception("chat voice error", extra={"user_id": user_id})

# ---------- Payments: Stars (XTR) ----------
async def precheckout_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await ensure_user_record(update)
    try: await update.pre_checkout_query.answer(ok=True)
    except Exception:
        await update.pre_checkout_query.answer(ok=False, error_message=f"Платёж отклонён. Пополните Stars в {STARS_BUY_URL}")

STARS_TO_DIAMONDS = {
    50: 50,
    100: 110,
    200: 220,
    300: 330,
    400: 440,
    500: 550,
}

STARS_PACK_ORDER = [50, 100, 200, 300, 400, 500]


async def successful_payment_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await ensure_user_record(update)
    s = state(ctx)

    message = update.message
    if message is None or message.successful_payment is None:
        return

    user = message.from_user
    if user is None:
        return

    user_id = user.id
    try:
        ensure_user(user_id)
    except Exception as exc:
        log.warning("ensure_user failed during payment for %s: %s", user_id, exc)

    sp = message.successful_payment
    log.info("stars_payment_received | user=%s payment=%s", user_id, sp.to_dict())

    payload_raw = sp.invoice_payload or ""
    parsed_payload: Optional[Dict[str, Any]] = None

    def _to_positive_int(value: Any) -> Optional[int]:
        try:
            ivalue = int(value)
        except (TypeError, ValueError):
            return None
        return ivalue if ivalue > 0 else None

    diamonds_to_credit: Optional[int] = None
    stars_amount: Optional[int] = None

    if payload_raw:
        try:
            payload_candidate = json.loads(payload_raw)
        except json.JSONDecodeError as exc:
            log.warning(
                "stars_payment_payload_invalid | user=%s payload=%s err=%s",
                user_id,
                payload_raw,
                exc,
            )
        else:
            if isinstance(payload_candidate, dict):
                parsed_payload = payload_candidate
                stars_from_payload = _to_positive_int(parsed_payload.get("stars"))
                diamonds_from_payload = _to_positive_int(parsed_payload.get("diamonds"))
                if diamonds_from_payload is None:
                    diamonds_from_payload = _to_positive_int(parsed_payload.get("tokens"))
                if stars_from_payload:
                    stars_amount = stars_from_payload
                if diamonds_from_payload:
                    diamonds_to_credit = diamonds_from_payload
            else:
                log.warning(
                    "stars_payment_payload_unexpected | user=%s payload=%s",
                    user_id,
                    payload_candidate,
                )

    currency = (sp.currency or "").upper()
    total_amount = int(sp.total_amount or 0)
    fallback_stars: Optional[int] = None
    if currency == "XTR":
        if total_amount in STARS_TO_DIAMONDS:
            fallback_stars = total_amount
        elif total_amount % 100 == 0:
            candidate = total_amount // 100
            if candidate in STARS_TO_DIAMONDS:
                fallback_stars = candidate

    if fallback_stars:
        if stars_amount is None:
            stars_amount = fallback_stars
        if diamonds_to_credit is None:
            diamonds_to_credit = STARS_TO_DIAMONDS.get(fallback_stars)

    if not diamonds_to_credit:
        log.error(
            "stars_payment_unrecognized_pack | user=%s currency=%s total_amount=%s payload=%s",
            user_id,
            currency,
            total_amount,
            payload_raw,
        )
        await message.reply_text(
            "⚠️ Платёж получен, но пакет не распознан. Команда уже уведомлена."
        )
        return

    if stars_amount is None and fallback_stars is not None:
        stars_amount = fallback_stars

    charge_id = (
        sp.telegram_payment_charge_id
        or sp.provider_payment_charge_id
        or uuid.uuid4().hex
    )

    credit_meta = {
        "stars": stars_amount,
        "diamonds": diamonds_to_credit,
        "payload": parsed_payload if parsed_payload is not None else payload_raw or None,
        "provider": "telegram_stars",
        "charge_id": charge_id,
        "currency": currency,
        "total_amount": total_amount,
    }
    credit_meta = {k: v for k, v in credit_meta.items() if v is not None}

    try:
        new_balance = credit_balance(
            user_id,
            diamonds_to_credit,
            reason="stars_payment",
            meta=credit_meta,
        )
    except KeyError:
        log.exception(
            "stars_payment_mapping_missing | user=%s stars=%s",
            user_id,
            stars_amount,
        )
        await message.reply_text(
            "⚠️ Платёж получен, но пакет не распознан. Команда уже уведомлена."
        )
        return
    except Exception as exc:
        log.exception("Stars payment processing failed for %s: %s", charge_id, exc)
        await message.reply_text(
            "⚠️ Платёж получен, но обновить баланс не удалось. Поддержка уведомлена."
        )
        return

    _set_cached_balance(ctx, new_balance)

    log.info(
        "stars_payment_success | user=%s charge_id=%s stars=%s diamonds=%s balance=%s",
        user_id,
        charge_id,
        stars_amount,
        diamonds_to_credit,
        new_balance,
    )

    inviter_id: Optional[int] = None
    try:
        inviter_id = get_inviter(user_id)
    except Exception as exc:
        log.warning("referral_lookup_failed | user=%s err=%s", user_id, exc)
        inviter_id = None
    bonus_awarded = False
    total_ref_earned: Optional[int] = None
    inviter_new_balance: Optional[int] = None
    if inviter_id and inviter_id != user_id:
        bonus = max(int(diamonds_to_credit) // 10, 0)
        if bonus > 0:
            try:
                inviter_new_balance = credit_balance(
                    inviter_id,
                    bonus,
                    reason="ref_bonus",
                    meta={"from_user": user_id, "charge_id": charge_id},
                )
                bonus_awarded = True
            except Exception as exc:
                log.warning(
                    "referral_bonus_credit_failed | inviter=%s user=%s err=%s",
                    inviter_id,
                    user_id,
                    exc,
                )
            if bonus_awarded:
                try:
                    total_ref_earned = incr_ref_earned(inviter_id, bonus)
                except Exception as exc:
                    log.warning(
                        "referral_earned_increment_failed | inviter=%s err=%s",
                        inviter_id,
                        exc,
                    )
                    total_ref_earned = None
                payer_display = _format_user_for_notification(update.effective_user, user_id)
                total_text = (
                    f"{total_ref_earned}💎" if isinstance(total_ref_earned, int) else "—"
                )
                notify_text = (
                    f"💎 Реферальный бонус: +{bonus}. За пополнение {payer_display}. "
                    f"Итого с рефералов: {total_text}."
                )
                try:
                    await ctx.bot.send_message(inviter_id, notify_text)
                except Forbidden:
                    pass
                except Exception as exc:
                    log.warning(
                        "referral_bonus_notify_failed | inviter=%s err=%s",
                        inviter_id,
                        exc,
                    )
                inviter_state = None
                if ctx.application and isinstance(ctx.application.user_data, dict):
                    inviter_state = ctx.application.user_data.get(inviter_id)
                if isinstance(inviter_state, dict):
                    if inviter_new_balance is not None:
                        inviter_state["balance"] = inviter_new_balance
                    try:
                        await refresh_balance_card_if_open(
                            inviter_id,
                            inviter_id,
                            ctx=ctx,
                            state_dict=inviter_state,
                            reply_markup=balance_menu_kb(),
                        )
                    except Exception as exc:
                        log.warning(
                            "referral_balance_refresh_failed | inviter=%s err=%s",
                            inviter_id,
                            exc,
                        )
                    if inviter_state.get("last_panel") == "referral":
                        try:
                            link = await _build_referral_link(inviter_id, ctx)
                        except Exception as exc:
                            log.warning(
                                "referral_card_link_failed | inviter=%s err=%s",
                                inviter_id,
                                exc,
                            )
                            link = None
                        try:
                            stats = get_ref_stats(inviter_id)
                        except Exception as exc:
                            log.warning(
                                "referral_stats_refresh_failed | inviter=%s err=%s",
                                inviter_id,
                                exc,
                            )
                            stats = None
                        if link and stats:
                            try:
                                await show_referral_card(
                                    ctx,
                                    inviter_id,
                                    inviter_state,
                                    link=link,
                                    referrals=stats[0],
                                    earned=stats[1],
                                    share_text=_REF_SHARE_TEXT,
                                )
                            except Exception as exc:
                                log.warning(
                                    "referral_card_refresh_failed | inviter=%s err=%s",
                                    inviter_id,
                                    exc,
                                )

    await send_ok_sticker(ctx, "purchase", new_balance, chat_id=message.chat_id)

    chat_id = update.effective_chat.id if update.effective_chat else user_id
    await show_main_menu(chat_id, ctx)
    await refresh_balance_card_if_open(
        user_id,
        chat_id,
        ctx=ctx,
        state_dict=s,
        reply_markup=balance_menu_kb(),
    )

# ==========================
#   Redis runner lock
# ==========================

class RedisLockBusy(RuntimeError):
    """Raised when Redis runner lock is already held by another instance."""


runner_lock_state: Dict[str, Any] = {
    "enabled": bool(REDIS_URL) and REDIS_LOCK_ENABLED and redis_asyncio is not None,
    "owned": False,
    "heartbeat_at": None,
    "started_at": None,
    "host": None,
    "pid": None,
}


class RedisRunnerLock:
    LOCK_TTL_SECONDS = 60
    HEARTBEAT_INTERVAL = 25
    STALE_THRESHOLD_SECONDS = 90

    def __init__(self, redis_url: str, key: str, enabled: bool, version: str):
        self.redis_url = redis_url
        self.key = key
        self.version = version
        self.enabled = enabled and bool(redis_url) and redis_asyncio is not None
        runner_lock_state["enabled"] = self.enabled

        self._redis: Optional["redis_asyncio.Redis"] = None
        self._value: Dict[str, Any] = {}
        self._heartbeat_task: Optional[asyncio.Task[None]] = None
        self._released = False
        self._acquired = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._signal_handlers: List[signal.Signals] = []
        self._stop_callbacks: List[Callable[[Optional[signal.Signals]], None]] = []

    async def __aenter__(self) -> "RedisRunnerLock":
        if not self.enabled:
            log.info("Redis runner lock disabled (enabled=%s, redis_asyncio=%s)", REDIS_LOCK_ENABLED, bool(redis_asyncio))
            return self

        assert redis_asyncio is not None  # for type checkers
        self._redis = redis_asyncio.from_url(self.redis_url, encoding="utf-8", decode_responses=True)
        try:
            await self._acquire()
        except Exception:
            await self._close_redis()
            raise
        self._loop = asyncio.get_running_loop()
        self._install_signal_handlers()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.release()

    def add_stop_callback(self, callback: Callable[[Optional[signal.Signals]], None]) -> None:
        if not callable(callback):
            raise TypeError("callback must be callable")
        self._stop_callbacks.append(callback)

    async def _acquire(self) -> None:
        if not self._redis:
            return

        backoff = 1.0
        while True:
            now_iso = _utcnow_iso()
            host = socket.gethostname()
            pid = os.getpid()
            self._value = {
                "host": host,
                "pid": pid,
                "started_at": now_iso,
                "heartbeat_at": now_iso,
                "version": self.version,
            }
            payload = json.dumps(self._value, ensure_ascii=False)

            try:
                acquired = await self._redis.set(self.key, payload, nx=True, px=self.LOCK_TTL_SECONDS * 1000)
            except Exception as exc:
                log.exception("Redis lock SET failed: %s", exc)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 5.0)
                continue

            if acquired:
                self._on_acquired(takeover=False)
                return

            existing_raw = await self._redis.get(self.key)
            existing = self._decode_existing(existing_raw)
            if self._is_stale(existing):
                event("LOCK_STALE_TAKEOVER", key=self.key, previous_host=existing.get("host"),
                      previous_pid=existing.get("pid"), previous_heartbeat_at=existing.get("heartbeat_at"))
                try:
                    await self._redis.getdel(self.key)
                except Exception as exc:
                    log.warning("Redis lock GETDEL failed: %s", exc)
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 5.0)
                    continue

                takeover = await self._redis.set(self.key, payload, nx=True, px=self.LOCK_TTL_SECONDS * 1000)
                if takeover:
                    self._on_acquired(takeover=True)
                    return

                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 5.0)
                continue

            runner_lock_state.update({
                "owned": False,
                "heartbeat_at": existing.get("heartbeat_at"),
                "started_at": existing.get("started_at"),
                "host": existing.get("host"),
                "pid": existing.get("pid"),
            })
            event("LOCK_BUSY", key=self.key, owner_host=existing.get("host"), owner_pid=existing.get("pid"),
                  owner_heartbeat_at=existing.get("heartbeat_at"))
            raise RedisLockBusy("redis runner lock busy")

    def _on_acquired(self, takeover: bool) -> None:
        self._acquired = True
        runner_lock_state.update({
            "owned": True,
            "heartbeat_at": self._value.get("heartbeat_at"),
            "started_at": self._value.get("started_at"),
            "host": self._value.get("host"),
            "pid": self._value.get("pid"),
        })
        event("LOCK_ACQUIRED", key=self.key, host=self._value.get("host"), pid=self._value.get("pid"), takeover=takeover)
        log.info("Redis runner lock acquired (takeover=%s)", takeover)
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def release(self) -> None:
        if self._released:
            return
        self._released = True

        self._remove_signal_handlers()

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._heartbeat_task

        if not self.enabled or not self._redis:
            runner_lock_state.update({
                "owned": False,
                "heartbeat_at": None,
                "started_at": None,
                "host": None,
                "pid": None,
            })
            return

        try:
            await self._redis.delete(self.key)
            event("LOCK_RELEASED", key=self.key, host=self._value.get("host"), pid=self._value.get("pid"))
            log.info("Redis runner lock released")
        except Exception as exc:
            log.warning("Redis lock delete failed: %s", exc)
        finally:
            runner_lock_state.update({
                "owned": False,
                "heartbeat_at": None,
                "started_at": None,
                "host": None,
                "pid": None,
            })
            await self._close_redis()

    async def _heartbeat_loop(self) -> None:
        if not self._redis:
            return
        try:
            while not self._released:
                await asyncio.sleep(self.HEARTBEAT_INTERVAL)
                if self._released:
                    break
                await self._heartbeat_once()
        except asyncio.CancelledError:
            pass

    async def _heartbeat_once(self) -> None:
        if not self._redis or not self._acquired:
            return
        hb_iso = _utcnow_iso()
        self._value["heartbeat_at"] = hb_iso
        payload = json.dumps(self._value, ensure_ascii=False)
        try:
            updated = await self._redis.set(self.key, payload, xx=True, px=self.LOCK_TTL_SECONDS * 1000)
        except Exception as exc:
            log.warning("Redis heartbeat failed: %s", exc)
            return

        if updated:
            runner_lock_state["heartbeat_at"] = hb_iso
            event("LOCK_HEARTBEAT", key=self.key, heartbeat_at=hb_iso)
        else:
            log.warning("Redis heartbeat lost lock (key missing)")

    def _decode_existing(self, raw: Optional[str]) -> Dict[str, Any]:
        if not raw:
            return {}
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        return {}

    def _is_stale(self, existing: Dict[str, Any]) -> bool:
        if not existing:
            return True
        hb = _parse_iso8601(existing.get("heartbeat_at")) or _parse_iso8601(existing.get("started_at"))
        if not hb:
            return True
        return (datetime.now(timezone.utc) - hb).total_seconds() > self.STALE_THRESHOLD_SECONDS

    def _install_signal_handlers(self) -> None:
        if not self.enabled:
            return
        if not self._loop:
            return
        for sig_name in ("SIGTERM", "SIGINT"):
            if not hasattr(signal, sig_name):
                continue
            sig = getattr(signal, sig_name)
            try:
                def _handler(s: signal.Signals = sig) -> None:
                    for cb in list(self._stop_callbacks):
                        try:
                            cb(s)
                        except Exception as exc:
                            log.warning("Runner lock stop callback failed: %s", exc)
                    asyncio.create_task(self._on_signal(s))

                self._loop.add_signal_handler(sig, _handler)
                self._signal_handlers.append(sig)
            except (NotImplementedError, RuntimeError):
                continue

    def _remove_signal_handlers(self) -> None:
        if not self._loop:
            return
        for sig in self._signal_handlers:
            try:
                self._loop.remove_signal_handler(sig)
            except (NotImplementedError, RuntimeError):
                pass
        self._signal_handlers.clear()

    async def _on_signal(self, sig: signal.Signals) -> None:
        log.info("Signal received: %s. Releasing Redis lock.", sig.name if hasattr(sig, "name") else str(sig))
        await self.release()

    async def _close_redis(self) -> None:
        if not self._redis:
            return
        try:
            close = getattr(self._redis, "aclose", None)
            if callable(close):
                await close()
            else:
                await self._redis.close()
        except Exception:
            pass
        self._redis = None

PRIORITY_COMMAND_SPECS: List[tuple[tuple[str, ...], Any]] = [
    (("start",), start),
    (("menu",), menu_command),
    (("cancel",), cancel_command),
    (("faq",), faq_command_entry),
    (("prompt_master",), prompt_master_command),
    (("pm_reset",), prompt_master_reset_command),
    (("chat",), chat_command),
    (("reset",), chat_reset_command),
    (("history",), chat_history_command),
    (("image", "mj"), image_command),
    (("video", "veo"), video_command),
    (("music", "suno"), suno_command),
    (("balance",), balance_command),
    (("help", "support"), help_command_entry),
]

ADDITIONAL_COMMAND_SPECS: List[tuple[tuple[str, ...], Any]] = [
    (("buy",), buy_command),
    (("suno_last",), suno_last_command),
    (("suno_task",), suno_task_command),
    (("suno_retry",), suno_retry_command),
    (("lang",), lang_command),
    (("health",), health),
    (("topup",), topup),
    (("promo",), promo_command),
    (("users_count",), users_count_command),
    (("whoami",), whoami_command),
    (("suno_debug",), suno_debug_command),
    (("broadcast",), broadcast_command),
    (("my_balance",), my_balance_command),
    (("add_balance",), add_balance_command),
    (("sub_balance",), sub_balance_command),
    (("transactions",), transactions_command),
    (("balance_recalc",), balance_recalc),
    (("sora2_health",), sora2_health_command),
]

CALLBACK_HANDLER_SPECS: List[tuple[Optional[str], Any]] = [
    (rf"^{CB_PM_INSERT_PREFIX}(veo|mj|banana|animate|suno)$", prompt_master_insert_callback_entry),
    (rf"^{CB_PM_PREFIX}", prompt_master_callback_entry),
    (rf"^{CB_FAQ_PREFIX}", faq_callback_entry),
    (r"^(?:cb:|video_menu$|engine:|mode:(?:veo|sora2)_|video:back$)", video_menu_callback),
    (r"^mj\.gallery\.again:", handle_mj_gallery_repeat),
    (r"^mj\.gallery\.back$", handle_mj_gallery_back),
    (r"^mj\.upscale\.menu:", handle_mj_upscale_menu),
    (r"^mj\.upscale:", handle_mj_upscale_choice),
    (r"^(?:hub:|main_|profile_|pay_|nav_|back_main$|ai_modes$|chat_(?:normal|promptmaster)$)", hub_router),
    (r"^go:", main_suggest_router),
    (r"^s2_go_t2v$", sora2_start_t2v),
    (r"^s2_go_i2v$", sora2_start_i2v),
    (rf"^{CB_ADMIN_SORA2_HEALTH}$", sora2_health_callback),
    (None, on_callback),
]

REPLY_BUTTON_ROUTES: List[tuple[str, Callable[[Update, ContextTypes.DEFAULT_TYPE], Awaitable[Any]]]] = [
    (MENU_BTN_VIDEO, handle_video_entry),
    (MENU_BTN_IMAGE, handle_image_entry),
    (MENU_BTN_SUNO, handle_music_entry),
    (MENU_BTN_PM, prompt_master_command),
    (MENU_BTN_CHAT, handle_chat_entry),
    (MENU_BTN_BALANCE, handle_balance_entry),
    (MENU_BTN_SUPPORT, help_command_entry),
]


LABEL_COMMAND_ROUTES: Dict[str, Callable[[Update, ContextTypes.DEFAULT_TYPE], Awaitable[Any]]] = {
    "veo.card": handle_video_entry,
    "mj.card": handle_image_entry,
    "balance.show": handle_balance_entry,
    "help.open": help_command_entry,
    "pm.open": prompt_master_command,
}


_REGISTERED_APPS: set[int] = set()


HANDLERS_FLAG = "bestveo.handlers_registered"


MJ_UPSCALE_PROMPT_PLACEHOLDER = "upscale image"


def register_handlers(application: Any) -> None:
    app_id = id(application)
    if application.bot_data.get(HANDLERS_FLAG) or app_id in _REGISTERED_APPS:
        log.warning("handlers.duplicate_registration", extra={"application": app_id})
        return

    application.bot_data[HANDLERS_FLAG] = True
    _REGISTERED_APPS.add(app_id)

    card_input_handler = MessageHandler(
        filters.TEXT,
        handle_card_input,
    )
    card_input_handler.block = False
    application.add_handler(card_input_handler, group=1)

    for names, callback in PRIORITY_COMMAND_SPECS:
        application.add_handler(CommandHandler(list(names), callback))

    for names, callback in ADDITIONAL_COMMAND_SPECS:
        application.add_handler(CommandHandler(list(names), callback))

    for pattern, callback in CALLBACK_HANDLER_SPECS:
        if pattern is None:
            application.add_handler(CallbackQueryHandler(callback))
        else:
            application.add_handler(CallbackQueryHandler(callback, pattern=pattern))

    application.add_handler(PreCheckoutQueryHandler(precheckout_callback))
    application.add_handler(MessageHandler(filters.SUCCESSFUL_PAYMENT, successful_payment_handler))
    application.add_handler(MessageHandler(filters.PHOTO, on_photo))
    application.add_handler(MessageHandler(filters.Document.ALL, on_document))
    application.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))

    for text, handler in REPLY_BUTTON_ROUTES:
        pattern = rf"^{re.escape(text)}$"
        application.add_handler(
            MessageHandler(
                filters.TEXT & ~filters.COMMAND & filters.Regex(pattern),
                handler,
            )
        )

    pm_handler = MessageHandler(filters.TEXT & ~filters.COMMAND, prompt_master_handle_text)
    pm_handler.block = False
    application.add_handler(pm_handler, group=2)

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text), group=10)


# ==========================
#   Entry (fixed for PTB 21.x)
# ==========================
async def run_bot_async() -> None:
    if not TELEGRAM_TOKEN: raise RuntimeError("TELEGRAM_TOKEN is not set")
    if not KIE_BASE_URL:   raise RuntimeError("KIE_BASE_URL is not set")
    if not KIE_API_KEY:    raise RuntimeError("KIE_API_KEY is not set")

    application = (ApplicationBuilder()
                   .token(TELEGRAM_TOKEN)
                   .rate_limiter(AIORateLimiter())
                   .build())

    try:
        if not application.bot_data.get(HANDLERS_FLAG):
            register_handlers(application)
            application.bot_data[HANDLERS_FLAG] = True
        else:
            log.info(
                "handlers.already_registered",
                extra={"application": id(application)},
            )
    except Exception:
        log.exception("handler registration failed")
        raise
    application.add_error_handler(error_handler)

    lock = RedisRunnerLock(REDIS_URL, _rk("lock", "runner"), REDIS_LOCK_ENABLED, APP_VERSION)

    try:
        async with lock:
            log.info(
                "Bot starting… (Redis=%s, lock=%s)",
                "on" if redis_client else "off",
                "enabled" if lock.enabled else "disabled",
            )

            try:
                await _run_suno_probe()
            except Exception as exc:
                log.warning("SUNO probe execution failed: %s", exc)

            loop = asyncio.get_running_loop()
            stop_event = asyncio.Event()
            manual_signal_handlers: List[signal.Signals] = []

            graceful_task: Optional[asyncio.Task[None]] = None

            async def _await_active_tasks() -> None:
                deadline = time.time() + 10.0
                while time.time() < deadline and ACTIVE_TASKS:
                    await asyncio.sleep(0.1)
                stop_event.set()

            def _trigger_stop(sig: Optional[signal.Signals] = None, *, reason: str = "external") -> None:
                nonlocal graceful_task
                if stop_event.is_set():
                    return
                if sig is not None:
                    sig_name = sig.name if hasattr(sig, "name") else str(sig)
                    log.info("Stop signal received: %s. Triggering shutdown.", sig_name)
                else:
                    log.info("Stop requested (%s). Triggering shutdown.", reason)
                SHUTDOWN_EVENT.set()
                if application.updater:
                    loop.create_task(application.updater.stop())
                if graceful_task is None or graceful_task.done():
                    graceful_task = loop.create_task(_await_active_tasks())

            lock.add_stop_callback(lambda sig: _trigger_stop(sig))

            if not lock.enabled:
                for sig_name in ("SIGINT", "SIGTERM"):
                    if not hasattr(signal, sig_name):
                        continue
                    sig_obj = getattr(signal, sig_name)
                    try:
                        loop.add_signal_handler(sig_obj, lambda s=sig_obj: _trigger_stop(s))
                        manual_signal_handlers.append(sig_obj)
                    except (NotImplementedError, RuntimeError):
                        continue

            previous_post_stop = application.post_stop

            async def _post_stop(app) -> None:
                _trigger_stop(reason="post_stop")
                if previous_post_stop:
                    await previous_post_stop(app)

            application.post_stop = _post_stop

            # ВАЖНО: полный async-жизненный цикл PTB — без run_polling()
            await application.initialize()

            try:
                await application.bot.set_my_commands([
                    BotCommand("menu", "⭐ Главное меню"),
                    BotCommand("buy", "💎 Купить генерации"),
                    BotCommand("video", "🎬 Генерация видео"),
                    BotCommand("image", "🎨 Генерация изображений"),
                    BotCommand("lang", "🌍 Изменить язык"),
                    BotCommand("help", "🆘 Поддержка"),
                    BotCommand("faq", "❓ FAQ"),
                ])
            except Exception as exc:
                log.warning("Failed to set bot commands: %s", exc)

            try:
                try:
                    await application.bot.delete_webhook(drop_pending_updates=True)
                    event("WEBHOOK_DELETE_OK", drop_pending_updates=True)
                    log.info("Webhook deleted")
                except Exception as exc:
                    event("WEBHOOK_DELETE_ERROR", error=str(exc))
                    log.warning("Delete webhook failed: %s", exc)

                await application.start()
                await application.updater.start_polling(
                    allowed_updates=Update.ALL_TYPES,
                    drop_pending_updates=True,
                )

                log.info("Application started")

                try:
                    await stop_event.wait()
                except asyncio.CancelledError:
                    _trigger_stop(reason="cancelled")
                    raise
            finally:
                for sig_obj in manual_signal_handlers:
                    try:
                        loop.remove_signal_handler(sig_obj)
                    except (NotImplementedError, RuntimeError):
                        pass

                if application.updater:
                    try:
                        await application.updater.stop()
                    except RuntimeError as exc:
                        log.warning("Updater stop failed: %s", exc)
                    except Exception as exc:
                        log.warning("Updater stop failed with unexpected error: %s", exc)

                try:
                    await application.stop()
                except Exception as exc:
                    log.warning("Application stop failed: %s", exc)

                try:
                    await application.shutdown()
                except Exception as exc:
                    log.warning("Application shutdown failed: %s", exc)
                application.post_stop = previous_post_stop
    except RedisLockBusy:
        log.error("Another instance is running (redis lock present). Exiting to avoid 409 conflict.")


def main() -> None:
    # Единая точка входа: создаём и закрываем цикл здесь
    acquire_singleton_lock(3600)
    asyncio.run(run_bot_async())


if __name__ == "__main__":
    main()
