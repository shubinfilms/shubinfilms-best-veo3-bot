# -*- coding: utf-8 -*-
# Best VEO3 Bot — PTB 21.x
# Версия: 2025-09-14r4
# Единственное изменение против прежней версии: надежная доставка VEO-видео в Telegram
# (освежение ссылки + повторная попытка + download&reupload с увеличенным таймаутом).
# Остальное (карточки, кнопки, тексты, цены, FAQ, промокоды, бонусы и т.д.) — без изменений.

# odex/fix-balance-reset-after-deploy
import os, json, time, uuid, asyncio, logging, tempfile, subprocess, re, signal, socket, hashlib, io, html
from pathlib import Path
# main
from typing import Dict, Any, Optional, List, Tuple, Callable, Awaitable
from datetime import datetime, timezone
from contextlib import suppress
from urllib.parse import urlparse
from dataclasses import dataclass

import aiohttp
from aiohttp import ClientError, ClientResponseError, ClientTimeout
import requests
from telegram import (
    Update, InlineKeyboardButton, InlineKeyboardMarkup,
    InputFile, LabeledPrice, InputMediaPhoto, ReplyKeyboardMarkup,
    KeyboardButton, BotCommand, User
)
from telegram.constants import ParseMode, ChatAction
from telegram.ext import (
    ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler,
    CallbackQueryHandler, filters, AIORateLimiter, PreCheckoutQueryHandler
)
from telegram.error import BadRequest, Forbidden, RetryAfter

from handlers import (
    PROMPT_MASTER_CANCEL,
    PROMPT_MASTER_OPEN,
    configure_prompt_master,
    prompt_master_conv,
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
    refresh_suno_card,
    show_referral_card,
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
)

from ledger import (
    LedgerStorage,
    LedgerOpResult,
    BalanceRecalcResult,
    InsufficientBalance,
)
from settings import REDIS_PREFIX, SUNO_LOG_KEY
try:
    import redis.asyncio as redis_asyncio  # type: ignore
except Exception:  # pragma: no cover - fallback if asyncio interface unavailable
    redis_asyncio = None

# ==========================
#   ENV / INIT
# ==========================
APP_VERSION = "2025-09-14r4"


ACTIVE_TASKS: Dict[int, str] = {}


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

def _env_int(k: str, default: int) -> int:
    raw = _env(k, str(default))
    if not raw:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default

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
    gen_path: str
    status_path: str
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


def _compose_suno_url(base: str, path: str) -> str:
    if not path:
        return base
    if "://" in path:
        return path
    return f"{(base or '').rstrip('/')}{path}"


def _load_suno_config() -> SunoConfig:
    base = (_env("SUNO_API_URL", "https://api.kie.ai/suno-api") or "https://api.kie.ai/suno-api").strip().rstrip("/")
    gen_path = _normalize_suno_path(_env("SUNO_GEN_PATH", "/generate-music"), "/generate-music")
    status_path = _normalize_suno_path(_env("SUNO_STATUS_PATH", "/record-info"), "/record-info")
    model = (_env("SUNO_MODEL", "V5") or "V5").strip() or "V5"
    price = _env_int("SUNO_PRICE", 30)
    timeout_sec = _env_int("SUNO_TIMEOUT_SEC", 180)
    enabled = _env("SUNO_ENABLED", "false").lower() == "true"
    has_key = bool(_env("KIE_API_KEY"))
    return SunoConfig(
        base=base,
        gen_path=gen_path,
        status_path=status_path,
        model=model,
        price=price,
        timeout_sec=timeout_sec,
        enabled=enabled,
        has_key=has_key,
    )


SUNO_CONFIG = _load_suno_config()
SUNO_MODEL = (SUNO_CONFIG.model or "V5").upper()
SUNO_MODEL_LABEL = SUNO_MODEL
SUNO_BASE_URL = SUNO_CONFIG.base or "https://api.kie.ai/suno-api"
SUNO_GEN_PATH = SUNO_CONFIG.gen_path
SUNO_STATUS_PATH = SUNO_CONFIG.status_path
SUNO_GEN_URL = _compose_suno_url(SUNO_BASE_URL, SUNO_GEN_PATH)
SUNO_STATUS_URL = _compose_suno_url(SUNO_BASE_URL, SUNO_STATUS_PATH)
SUNO_PRICE = SUNO_CONFIG.price
SUNO_POLL_INTERVAL = 3.0
SUNO_POLL_TIMEOUT = float(SUNO_CONFIG.timeout_sec)
TELEGRAM_TOKEN      = _env("TELEGRAM_TOKEN")
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

# Видео
FFMPEG_BIN                = _env("FFMPEG_BIN", "ffmpeg")
ENABLE_VERTICAL_NORMALIZE = _env("ENABLE_VERTICAL_NORMALIZE", "true").lower() == "true"
ALWAYS_FORCE_FHD          = _env("ALWAYS_FORCE_FHD", "true").lower() == "true"
MAX_TG_VIDEO_MB           = int(_env("MAX_TG_VIDEO_MB", "48"))
POLL_INTERVAL_SECS = int(_env("POLL_INTERVAL_SECS", "6"))
POLL_TIMEOUT_SECS  = int(_env("POLL_TIMEOUT_SECS", str(20 * 60)))
KIE_STRICT_POLLING = _env("KIE_STRICT_POLLING", "false").lower() == "true"

LOG_LEVEL = _env("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logging.getLogger("kie").setLevel(logging.INFO)
log = logging.getLogger("veo3-bot")

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


CB_MODE_CHAT = "mode:chat"
CB_MODE_PM = "mode:pm"
CB_PM_INSERT_VEO = "pm_insert_veo"
CB_GO_HOME = "go_home"

# ==========================
#   Tokens / Pricing
# ==========================
PRICE_MJ = 10
PRICE_BANANA = 5
PRICE_VEO_FAST = 50
PRICE_VEO_QUALITY = 150
PRICE_VEO_ANIMATE = 50
PRICE_SUNO = SUNO_PRICE

TOKEN_COSTS = {
    "veo_fast": PRICE_VEO_FAST,
    "veo_quality": PRICE_VEO_QUALITY,
    "veo_photo": PRICE_VEO_ANIMATE,
    "mj": PRICE_MJ,          # 16:9 или 9:16
    "banana": PRICE_BANANA,
    "suno": PRICE_SUNO,
    "chat": 0,
}
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

    await message.reply_text(
        "✅ Промокод активирован!\nНачислено: +100 💎\nНовый баланс: "
        f"{balance_after} 💎"
    )

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


def _mj_guess_filename(url: str, index: int, content_type: Optional[str]) -> str:
    try:
        path = urlparse(url).path
    except Exception:
        path = ""
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    if ext not in _MJ_ALLOWED_EXTENSIONS:
        ext = _mj_content_type_extension(content_type) or ".jpg"
    return f"midjourney_{index + 1:02d}{ext}"


def _download_mj_image_bytes(url: str, index: int) -> Optional[Tuple[bytes, str]]:
    try:
        resp = requests.get(url, timeout=60)
    except requests.RequestException as exc:
        log.warning("MJ image download failed (%s): %s", url, exc)
        return None
    if resp.status_code != 200:
        log.warning("MJ image download status %s for %s", resp.status_code, url)
        return None
    data = resp.content
    if not data:
        log.warning("MJ image download empty response for %s", url)
        return None
    filename = _mj_guess_filename(url, index, resp.headers.get("Content-Type"))
    return data, filename


def _make_input_photo(data: bytes, filename: str) -> InputFile:
    buffer = io.BytesIO(data)
    buffer.name = filename
    buffer.seek(0)
    return InputFile(buffer, filename=filename)


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
DEFAULT_STATE = {
    "mode": None, "aspect": "16:9", "model": None,
    "last_prompt": None, "last_image_url": None,
    "generating": False, "generation_id": None, "last_task_id": None,
    "last_ui_msg_id_menu": None,
    "last_ui_msg_id_balance": None,
    "last_ui_msg_id_veo": None, "last_ui_msg_id_banana": None, "last_ui_msg_id_mj": None,
    "last_ui_msg_id_suno": None,
    "banana_images": [],
    "mj_last_wait_ts": 0.0,
    "mj_generating": False, "last_mj_task_id": None,
    "active_generation_op": None,
    "mj_active_op_key": None,
    "banana_active_op_key": None,
    "_last_text_veo": None,
    "_last_text_banana": None,
    "_last_text_mj": None,
    "_last_text_suno": None,
    "msg_ids": {},
    "last_panel": None,
    "suno_title": None,
    "suno_style": None,
    "suno_lyrics": None,
    "suno_instrumental": True,
    "suno_waiting_field": None,
    "suno_generating": False,
    "suno_last_task_id": None,
    "suno_last_params": None,
    "suno_balance": None,
}


def _apply_state_defaults(target: Dict[str, Any]) -> Dict[str, Any]:
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
            target[key] = {}
    if not isinstance(target.get("banana_images"), list):
        target["banana_images"] = []
    if not isinstance(target.get("msg_ids"), dict):
        target["msg_ids"] = {}
    return target


def state(ctx: ContextTypes.DEFAULT_TYPE) -> Dict[str, Any]:
    ud = ctx.user_data
    return _apply_state_defaults(ud)

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

MENU_BTN_VIDEO = "🎬 ГЕНЕРАЦИЯ ВИДЕО"
MENU_BTN_IMAGE = "🎨 ГЕНЕРАЦИЯ ИЗОБРАЖЕНИЙ"
MENU_BTN_SUNO = "🎵 Генерация музыки"
MENU_BTN_PM = "🧠 Prompt-Master"
MENU_BTN_CHAT = "💬 Обычный чат"
MENU_BTN_BALANCE = "💎 Баланс"
BALANCE_CARD_STATE_KEY = "last_ui_msg_id_balance"
LEDGER_PAGE_SIZE = 10

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


async def show_main_menu(chat_id: int, ctx: ContextTypes.DEFAULT_TYPE) -> Optional[int]:
    s = state(ctx)
    uid = get_user_id(ctx) or chat_id
    balance = _safe_get_balance(uid)
    text = render_welcome_for(uid, ctx, balance=balance)
    return await upsert_card(
        ctx,
        chat_id,
        s,
        "last_ui_msg_id_menu",
        text,
        reply_markup=main_menu_kb(),
        parse_mode=ParseMode.MARKDOWN,
        disable_web_page_preview=True,
    )


def video_menu_kb() -> InlineKeyboardMarkup:
    keyboard = [
        [InlineKeyboardButton(
            f"🎬 Генерация видео (Veo Fast) — 💎 {TOKEN_COSTS['veo_fast']}",
            callback_data="mode:veo_text_fast",
        )],
        [InlineKeyboardButton(
            f"🎬 Генерация видео (Veo Quality) — 💎 {TOKEN_COSTS['veo_quality']}",
            callback_data="mode:veo_text_quality",
        )],
        [InlineKeyboardButton(
            f"🖼️ Оживить изображение (Veo) — 💎 {TOKEN_COSTS['veo_photo']}",
            callback_data="mode:veo_photo",
        )],
        [InlineKeyboardButton("⬅️ Назад", callback_data="back")],
    ]
    return InlineKeyboardMarkup(keyboard)


def image_menu_kb() -> InlineKeyboardMarkup:
    keyboard = [
        [InlineKeyboardButton(
            f"🖼️ Генерация изображений (MJ) — 💎 {TOKEN_COSTS['mj']}",
            callback_data="mode:mj_txt",
        )],
        [InlineKeyboardButton(
            f"🍌 Редактор изображений (Banana) — 💎 {TOKEN_COSTS['banana']}",
            callback_data="mode:banana",
        )],
        [InlineKeyboardButton("⬅️ Назад", callback_data="back")],
    ]
    return InlineKeyboardMarkup(keyboard)


def inline_topup_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("💳 Пополнить баланс", callback_data="topup_open")],
            [InlineKeyboardButton("🎁 Активировать промокод", callback_data="promo_open")],
            [InlineKeyboardButton("⬅️ Назад", callback_data="back")],
        ]
    )


def balance_menu_kb() -> InlineKeyboardMarkup:
    keyboard = [
        [
            InlineKeyboardButton("💳 Пополнить баланс", callback_data="topup_open"),
            InlineKeyboardButton("🧾 История операций", callback_data="tx:open"),
        ],
        [InlineKeyboardButton("👥 Пригласить друга", callback_data="ref:open")],
        [InlineKeyboardButton("🎁 Активировать промокод", callback_data="promo_open")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="back")],
    ]
    return InlineKeyboardMarkup(keyboard)


async def show_balance_card(chat_id: int, ctx: ContextTypes.DEFAULT_TYPE) -> Optional[int]:
    s = state(ctx)
    uid = get_user_id(ctx) or chat_id
    balance = _safe_get_balance(uid)
    _set_cached_balance(ctx, balance)
    text = f"💎 Ваш баланс: {balance}"
    mid = await upsert_card(
        ctx,
        chat_id,
        s,
        BALANCE_CARD_STATE_KEY,
        text,
        reply_markup=balance_menu_kb(),
        parse_mode=ParseMode.MARKDOWN,
        disable_web_page_preview=True,
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


def render_faq_text() -> str:
    return (
        "📘 *FAQ*\n"
        "— *Как начать с VEO?*\n"
        "1) Выберите «Veo Fast» или «Veo Quality». 2) Пришлите идею текстом и/или фото. "
        "3) Карточка откроется автоматически — проверьте параметры и жмите «🚀 Сгенерировать».\n\n"
        "— *Fast vs Quality?* Fast — быстрее и дешевле. Quality — дольше, но лучше детализация.\n\n"
        "— *Форматы VEO?* Готовые клипы загружаем в чат как видеофайлы.\n\n"
        "— *MJ:* Цена 10💎. Один бесплатный перезапуск при сетевой ошибке. На выходе одно изображение.\n\n"
        "— *Banana:* до 4 фото, затем текст — что поменять (фон, одежда, макияж, удаление объектов, объединение людей).\n\n"
        "— *Время ожидания:* VEO 2–10 мин, MJ 1–3 мин, Banana 1–5 мин (может быть дольше при нагрузке).\n\n"
        "— *Токены/возвраты:* списываются при старте; при ошибке/таймауте бот автоматически возвращает 💎.\n\n"
        f"— *Пополнение:* через Stars в меню. Где купить: {STARS_BUY_URL}\n"
        f"— *Примеры и идеи:* {PROMPTS_CHANNEL_URL}."
    )

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
    if snippet:
        lines.extend(["", f"Последний промпт: <i>{html.escape(snippet)}</i>"])
    return "\n".join(lines)

def _mj_prompt_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Подтвердить", callback_data="mj:confirm")],
        [
            InlineKeyboardButton("Отменить", callback_data="mj:cancel"),
            InlineKeyboardButton("Сменить формат", callback_data="mj:change_format"),
        ],
    ])

async def _update_mj_card(chat_id: int, ctx: ContextTypes.DEFAULT_TYPE, text: str,
                                reply_markup: Optional[InlineKeyboardMarkup], *, force: bool = False) -> None:
    s = state(ctx)
    if not force and text == s.get("_last_text_mj"):
        return
    mid = await upsert_card(ctx, chat_id, s, "last_ui_msg_id_mj", text, reply_markup)
    if mid:
        s["_last_text_mj"] = text
    elif force:
        s["_last_text_mj"] = None

async def show_mj_format_card(chat_id: int, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    s = state(ctx)
    aspect = "9:16" if s.get("aspect") == "9:16" else "16:9"
    s["aspect"] = aspect
    s["last_prompt"] = None
    await _update_mj_card(chat_id, ctx, _mj_format_card_text(aspect), _mj_format_keyboard(aspect))

async def show_mj_prompt_card(chat_id: int, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    s = state(ctx)
    aspect = "9:16" if s.get("aspect") == "9:16" else "16:9"
    s["aspect"] = aspect
    await _update_mj_card(chat_id, ctx, _mj_prompt_card_text(aspect, s.get("last_prompt")), _mj_prompt_keyboard())

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
    await show_mj_format_card(chat_id, ctx)

def banana_examples_block() -> str:
    return (
        "💡 <b>Примеры запросов:</b>\n"
        "• поменяй фон на городской вечер\n"
        "• смени одежду на чёрный пиджак\n"
        "• добавь лёгкий макияж, подчеркни глаза\n"
        "• убери лишние предметы со стола\n"
        "• поставь нас на одну фотографию"
    )

def banana_card_text(s: Dict[str, Any]) -> str:
    n = len(s.get("banana_images") or [])
    prompt = (s.get("last_prompt") or "—").strip() or "—"
    prompt_html = html.escape(prompt)
    has_prompt = "есть" if s.get("last_prompt") else "нет"
    lines = [
        "🍌 <b>Карточка Banana</b>",
        f"🧩 Фото: <b>{n}/4</b>  •  Промпт: <b>{has_prompt}</b>",
        "",
        "🖊️ <b>Промпт:</b>",
        f"<code>{prompt_html}</code>",
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
        [InlineKeyboardButton("⬅️ Назад", callback_data="back")],
    ]
    return InlineKeyboardMarkup(rows)


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
    if phase == "error":
        log.warning("Suno %s error | user=%s status=%s url=%s snippet=%s", phase, user_id, http_status, request_url, snippet)
    else:
        log.info("Suno %s | user=%s status=%s url=%s snippet=%s", phase, user_id, http_status, request_url, snippet)
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
    headers: Dict[str, str] = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    headers.update(_kie_auth_header())
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

    url = join_url(SUNO_CONFIG.base, path)
    headers = _suno_headers()
    method_upper = method.upper()

    if method_upper == "POST":
        try:
            log_payload = json.dumps(_suno_sanitize_log_payload(json_payload or {}), ensure_ascii=False)
        except Exception:
            log_payload = repr(json_payload)
        if log_payload is None:
            log_payload = "{}"
        if len(log_payload) > 200:
            log_payload = log_payload[:197] + "…"
        log.info("[SUNO] POST %s body=%s", url, log_payload)

    timeout_cfg = ClientTimeout(total=timeout)
    attempt = 0
    dynamic_attempts = int(timeout // 10) if timeout else 0
    max_attempts = max(1, min(10, dynamic_attempts if dynamic_attempts > 0 else 3))

    async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
        while attempt < max_attempts:
            attempt += 1
            try:
                async with session.request(
                    method_upper,
                    url,
                    headers=headers,
                    json=json_payload,
                    params=params,
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
    params = {"task_id": task_id, "id": task_id, "taskId": task_id}
    return await _suno_request(
        "GET",
        SUNO_CONFIG.status_path,
        json_payload=None,
        params=params,
        timeout=timeout,
        log_user_id=user_id,
        log_phase="poll",
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
            "prompt": "",
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


def _suno_collect_params(state_obj: Dict[str, Any]) -> Dict[str, Any]:
    title = (state_obj.get("suno_title") or "").strip()
    style = (state_obj.get("suno_style") or "").strip()
    lyrics = state_obj.get("suno_lyrics") or ""
    instrumental = bool(state_obj.get("suno_instrumental", True))
    return {
        "title": title,
        "style": style,
        "lyrics": lyrics,
        "instrumental": instrumental,
    }


def _suno_make_preview(text: str, limit: int = 160) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""
    if len(raw) <= limit:
        return raw
    clipped = raw[: limit - 1].rstrip()
    return clipped + "…"


def _suno_result_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("🔁 Повторить", callback_data="suno:repeat")],
            [InlineKeyboardButton("⬅️ В меню", callback_data="back")],
        ]
    )


async def suno_entry(chat_id: int, ctx: ContextTypes.DEFAULT_TYPE, *, refresh_balance: bool = True) -> None:
    s = state(ctx)
    s["suno_waiting_field"] = None
    if not _suno_configured():
        await _suno_notify(
            ctx,
            chat_id,
            "⚠️ Suno API не настроен. Обратитесь к поддержке.",
        )
        return
    if refresh_balance:
        uid = get_user_id(ctx) or chat_id
        if uid:
            s["suno_balance"] = _safe_get_balance(int(uid))
    s["mode"] = "suno"
    s.setdefault("suno_instrumental", True)
    s["_last_text_suno"] = None
    await refresh_suno_card(ctx, chat_id, s, price=PRICE_SUNO)


async def _suno_notify(
    ctx: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    text: str,
    *,
    reply_to: Optional["telegram.Message"] = None,
    parse_mode: Optional[ParseMode] = None,
    reply_markup: Optional[InlineKeyboardMarkup] = None,
) -> None:
    if reply_to is not None:
        await _send_with_retry(
            lambda: reply_to.reply_text(text, parse_mode=parse_mode, reply_markup=reply_markup)
        )
    else:
        await _send_with_retry(
            lambda: ctx.bot.send_message(
                chat_id,
                text,
                parse_mode=parse_mode,
                reply_markup=reply_markup,
            )
        )


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
) -> None:
    meta = dict(base_meta or {})
    if task_id:
        meta["task_id"] = task_id
    if error_text:
        meta["error"] = error_text

    try:
        new_balance = credit_balance(user_id, PRICE_SUNO, reason, meta=meta)
    except Exception as exc:
        log.exception("Suno refund failed | task=%s err=%s", task_id, exc)
        new_balance = None

    s = state(ctx)
    s["suno_generating"] = False
    if task_id and s.get("suno_last_task_id") == task_id:
        s["suno_last_task_id"] = None
    if isinstance(new_balance, int):
        s["suno_balance"] = new_balance
    s["_last_text_suno"] = None
    s["suno_waiting_field"] = None
    await refresh_suno_card(ctx, chat_id, s, price=PRICE_SUNO)
    await refresh_balance_card_if_open(user_id, chat_id, ctx=ctx, state_dict=s)

    await _suno_notify(
        ctx,
        chat_id,
        f"❌ Не удалось сгенерировать. Средства возвращены (+{PRICE_SUNO}💎).",
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
    if s.get("suno_generating"):
        await _suno_notify(ctx, chat_id, "⏳ Уже идёт генерация музыки — дождитесь завершения.", reply_to=reply_to)
        return

    if not _suno_configured():
        await _suno_notify(ctx, chat_id, "⚠️ Suno API не настроен. Обратитесь к поддержке.", reply_to=reply_to)
        return

    if not user_id:
        await _suno_notify(ctx, chat_id, "⚠️ Не удалось определить пользователя.", reply_to=reply_to)
        return

    try:
        ensure_user(user_id)
    except Exception as exc:
        log.exception("Suno ensure_user failed | user=%s err=%s", user_id, exc)
        await _suno_notify(ctx, chat_id, "⚠️ Ошибка доступа к балансу. Попробуйте позже.", reply_to=reply_to)
        return

    instrumental = bool(params.get("instrumental", True))
    title = (params.get("title") or "").strip()
    style = (params.get("style") or "").strip()
    lyrics = params.get("lyrics") or ""

    payload: Dict[str, Any] = {
        "model": SUNO_MODEL,
        "title": title,
        "style": style,
        "prompt": lyrics or "",
        "instrumental": instrumental,
    }

    meta: Dict[str, Any] = {
        "task_id": None,
        "model": SUNO_MODEL,
        "instrumental": instrumental,
        "has_lyrics": bool(lyrics),
        "prompt_len": len(lyrics or ""),
        "title": title,
        "style": style,
        "trigger": trigger,
    }

    ok, new_balance = debit_try(user_id, PRICE_SUNO, "suno:start", meta=meta)
    if not ok:
        balance_after = new_balance if isinstance(new_balance, int) else _safe_get_balance(user_id)
        await show_balance_notification(
            chat_id,
            ctx,
            user_id,
            f"🙇 Недостаточно токенов. Нужно: {PRICE_SUNO}💎, у вас: {balance_after}💎.",
            reply_markup=inline_topup_keyboard(),
        )
        return

    s["suno_generating"] = True
    s["suno_last_task_id"] = None
    s["suno_last_params"] = {
        "title": title,
        "style": style,
        "lyrics": lyrics,
        "instrumental": instrumental,
    }
    s["suno_balance"] = new_balance
    s["_last_text_suno"] = None
    s["suno_waiting_field"] = None

    await refresh_suno_card(ctx, chat_id, s, price=PRICE_SUNO)
    await refresh_balance_card_if_open(user_id, chat_id, ctx=ctx, state_dict=s)

    await _suno_notify(
        ctx,
        chat_id,
        f"✅ Списано {PRICE_SUNO}💎. Запускаю генерацию музыки…",
        reply_to=reply_to,
    )

    try:
        response = await suno_create_task(payload, user_id=user_id)
    except SunoConfigError as exc:
        log.warning("Suno config error: %s", exc)
        _suno_log(
            user_id=user_id,
            phase="error",
            request_url=join_url(SUNO_CONFIG.base, SUNO_CONFIG.gen_path),
            http_status=None,
            response_snippet="config_error",
        )
        await _suno_issue_refund(
            ctx,
            chat_id,
            user_id,
            base_meta=meta,
            task_id=None,
            error_text="config_error",
            reason="suno:refund:create_err",
            reply_to=reply_to,
        )
        return
    except SunoApiError as exc:
        log.warning("Suno create failed | status=%s payload=%s", exc.status, exc.payload)
        await _suno_issue_refund(
            ctx,
            chat_id,
            user_id,
            base_meta=meta,
            task_id=None,
            error_text=str(exc.payload or exc),
            reason="suno:refund:create_err",
            reply_to=reply_to,
        )
        return
    except Exception as exc:
        log.exception("Suno create crashed: %s", exc)
        _suno_log(
            user_id=user_id,
            phase="error",
            request_url=join_url(SUNO_CONFIG.base, SUNO_CONFIG.gen_path),
            http_status=None,
            response_snippet=f"exception:{exc}",
        )
        await _suno_issue_refund(
            ctx,
            chat_id,
            user_id,
            base_meta=meta,
            task_id=None,
            error_text=str(exc),
            reason="suno:refund:create_err",
            reply_to=reply_to,
        )
        return

    task_id = (
        str(
            response.get("task_id")
            or response.get("id")
            or response.get("taskId")
            or response.get("job_id")
            or ""
        ).strip()
    )
    if not task_id:
        log.warning("Suno create response missing task_id | resp=%s", response)
        _suno_log(
            user_id=user_id,
            phase="error",
            request_url=join_url(SUNO_CONFIG.base, SUNO_CONFIG.gen_path),
            http_status=None,
            response_snippet="missing_task_id",
        )
        await _suno_issue_refund(
            ctx,
            chat_id,
            user_id,
            base_meta=meta,
            task_id=None,
            error_text="missing_task_id",
            reason="suno:refund:create_err",
            reply_to=reply_to,
        )
        return

    meta["task_id"] = task_id
    _suno_update_last_debit_meta(user_id, {"task_id": task_id})

    s["suno_last_task_id"] = task_id
    s["suno_last_params"] = {
        "title": title,
        "style": style,
        "lyrics": lyrics,
        "instrumental": instrumental,
    }
    await refresh_suno_card(ctx, chat_id, s, price=PRICE_SUNO)

    asyncio.create_task(
        _poll_suno_and_send(
            chat_id,
            ctx,
            user_id,
            task_id,
            s["suno_last_params"],
            dict(meta),
        )
    )


async def _poll_suno_and_send(
    chat_id: int,
    ctx: ContextTypes.DEFAULT_TYPE,
    user_id: int,
    task_id: str,
    params: Dict[str, Any],
    meta: Dict[str, Any],
) -> None:
    start_time = time.monotonic()
    refunded = False

    async def _refund(error_text: str) -> None:
        nonlocal refunded
        if refunded:
            return
        refunded = True
        _suno_log(
            user_id=user_id,
            phase="error",
            request_url=f"task:{task_id}",
            http_status=None,
            response_snippet=error_text,
        )
        await _suno_issue_refund(
            ctx,
            chat_id,
            user_id,
            base_meta=meta,
            task_id=task_id,
            error_text=error_text,
            reason="suno:refund:timeout",
            reply_markup=_suno_result_keyboard(),
        )

    try:
        while True:
            elapsed = time.monotonic() - start_time
            if elapsed > SUNO_POLL_TIMEOUT:
                await _refund("timeout")
                return

            try:
                details = await suno_poll_task(task_id, user_id=user_id)
            except SunoApiError as exc:
                log.warning("Suno status error | task=%s status=%s payload=%s", task_id, exc.status, exc.payload)
                await asyncio.sleep(SUNO_POLL_INTERVAL)
                continue
            except Exception as exc:
                log.exception("Suno status crash | task=%s err=%s", task_id, exc)
                _suno_log(
                    user_id=user_id,
                    phase="error",
                    request_url=f"task:{task_id}",
                    http_status=None,
                    response_snippet=f"poll_exception:{exc}",
                )
                await asyncio.sleep(SUNO_POLL_INTERVAL)
                continue

            status = (details.get("status") or "").lower()
            if status in {"succeeded", "ready", "ok", "completed"}:
                audio_url = (
                    details.get("audio_url")
                    or details.get("audioUrl")
                    or details.get("url")
                    or ""
                )
                if not audio_url:
                    await _refund("missing_audio_url")
                    return

                title = details.get("title") or params.get("title") or "Без названия"
                style = details.get("style") or params.get("style") or "—"
                mode_label = "Инструментал" if params.get("instrumental", True) else "Со словами"
                lyrics_text = (
                    (details.get("lyrics") or details.get("lyric") or params.get("lyrics") or "")
                    .strip()
                )
                lyrics_preview = _suno_make_preview(lyrics_text)
                safe_preview = html.escape(lyrics_preview) if lyrics_preview else "—"
                current_balance = _safe_get_balance(user_id)

                caption_lines = [
                    f"🎵 {html.escape(title)}",
                    f"Режим: {html.escape(mode_label)}",
                    f"Стиль: {html.escape(style)}",
                ]
                caption_lines.append(f"Текст: {safe_preview}")
                caption_lines.append(f"Модель: {html.escape(SUNO_MODEL_LABEL)}")
                caption_lines.append(f"Готово. Списано {PRICE_SUNO}💎.")
                if isinstance(current_balance, int):
                    caption_lines.append(f"Баланс: {current_balance}💎")
                caption = "\n".join(caption_lines)

                keyboard = _suno_result_keyboard()

                await _send_with_retry(
                    lambda: ctx.bot.send_audio(
                        chat_id,
                        audio=audio_url,
                        caption=caption,
                        parse_mode=ParseMode.HTML,
                        reply_markup=keyboard,
                    ),
                    attempts=4,
                )

                _suno_log(
                    user_id=user_id,
                    phase="done",
                    request_url=audio_url,
                    http_status=200,
                    response_snippet="delivered",
                )

                s = state(ctx)
                s["suno_generating"] = False
                s["suno_last_task_id"] = None
                s["suno_balance"] = current_balance
                s["_last_text_suno"] = None
                await refresh_suno_card(ctx, chat_id, s, price=PRICE_SUNO)
                await refresh_balance_card_if_open(user_id, chat_id, ctx=ctx, state_dict=s)
                return

            if status in {"failed", "timeout", "error"}:
                await _refund(details.get("error") or status)
                return

            await asyncio.sleep(SUNO_POLL_INTERVAL)

    except Exception as exc:
        log.exception("Suno poll unexpected failure | task=%s err=%s", task_id, exc)
        await _refund(str(exc))
    finally:
        if not refunded:
            s = state(ctx)
            if s.get("suno_last_task_id") == task_id:
                s["suno_last_task_id"] = None
            s["suno_generating"] = False
            s["_last_text_suno"] = None
            await refresh_suno_card(ctx, chat_id, s, price=PRICE_SUNO)


# --------- VEO Card ----------
def veo_card_text(s: Dict[str, Any]) -> str:
    prompt = (s.get("last_prompt") or "—").strip() or "—"
    prompt_html = html.escape(prompt)
    aspect = html.escape(s.get("aspect") or "16:9")
    model = "Veo Quality" if s.get("model") == "veo3" else "Veo Fast"
    img = "есть" if s.get("last_image_url") else "нет"
    lines = [
        "🟦 <b>Карточка VEO</b>",
        f"• Формат: <b>{aspect}</b>",
        f"• Модель: <b>{model}</b>",
        f"• Фото-референс: <b>{img}</b>",
        "",
        "🖊️ <b>Промпт:</b>",
        f"<code>{prompt_html}</code>",
    ]
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
        async with aiohttp.ClientSession(timeout=ClientTimeout(total=600)) as session:
            try:
                video_url = await _poll_record_info()
            except TimeoutError as exc:
                log_evt("KIE_TIMEOUT", task_id=task_id, reason="poll_exception", message=str(exc))
                await _refund("timeout", str(exc))
                await _send_message_with_retry(original_chat_id, RENDER_FAIL_MESSAGE)
                return
            except Exception as exc:
                log.exception("VEO status polling failed: %s", exc)
                await _refund("poll_exception", str(exc))
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
        await _send_message_with_retry(original_chat_id, RENDER_FAIL_MESSAGE)
    except Exception as exc:
        log.exception("VEO render failed: %s", exc)
        await _refund("exception", str(exc))
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
                await ctx.bot.send_message(chat_id, "❌ MJ: сервис недоступен. 💎 Токены возвращены.")
                return
            if flag == 0:
                if time.time() - start_ts > max_wait:
                    await _refund("timeout")
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
                    await ctx.bot.send_message(chat_id, "⚠️ MJ вернул пустой результат. 💎 Токены возвращены.")
                    return

                base_prompt = re.sub(r"\s+", " ", prompt_for_retry).strip()
                snippet = base_prompt[:100] if base_prompt else "—"
                caption = "🖼 Midjourney\n• Формат: {ar}\n• Промпт: \"{snip}\"".format(ar=aspect_ratio, snip=snippet)

                downloaded: List[Tuple[bytes, str]] = []
                for idx, u in enumerate(urls[:10]):
                    result = await asyncio.to_thread(_download_mj_image_bytes, u, idx)
                    if result:
                        downloaded.append(result)
                    else:
                        log.warning("MJ skip image due to download failure: %s", u)

                if not downloaded:
                    await _refund("download_failed")
                    await ctx.bot.send_message(
                        chat_id,
                        "⚠️ MJ вернул результат, но изображения не удалось загрузить. 💎 Токены возвращены.",
                    )
                    return

                async def _send_photos_one_by_one() -> bool:
                    sent_any = False
                    for idx, (data, filename) in enumerate(downloaded):
                        try:
                            await ctx.bot.send_photo(
                                chat_id=chat_id,
                                photo=_make_input_photo(data, filename),
                                caption=caption if idx == 0 else None,
                            )
                            sent_any = True
                        except Exception as send_exc:
                            log.warning("MJ send_photo #%s failed: %s", idx, send_exc)
                    return sent_any

                sent_successfully = False
                if len(downloaded) >= 2:
                    media: List[InputMediaPhoto] = []
                    for idx, (data, filename) in enumerate(downloaded):
                        media.append(
                            InputMediaPhoto(
                                media=_make_input_photo(data, filename),
                                caption=caption if idx == 0 else None,
                            )
                        )
                    try:
                        await ctx.bot.send_media_group(chat_id=chat_id, media=media)
                        sent_successfully = True
                    except Exception as e:
                        log.warning("MJ send_media_group failed: %s", e)
                        sent_successfully = await _send_photos_one_by_one()
                else:
                    sent_successfully = await _send_photos_one_by_one()

                if not sent_successfully:
                    await _refund("send_failed")
                    await ctx.bot.send_message(
                        chat_id,
                        "❌ Не удалось отправить изображения MJ. 💎 Токены возвращены.",
                    )
                    return

                keyboard = InlineKeyboardMarkup([
                    [InlineKeyboardButton("Повторить", callback_data="mj:repeat")],
                    [InlineKeyboardButton("Назад в меню", callback_data="back")],
                ])
                await ctx.bot.send_message(chat_id, "Галерея сгенерирована.", reply_markup=keyboard)

                success = True
                return
    except Exception as e:
        log.exception("MJ poll crash: %s", e)
        await _refund("exception", str(e))
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
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="back")])
    return InlineKeyboardMarkup(rows)

async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await ensure_user_record(update)
    s = state(ctx); s.update({**DEFAULT_STATE}); _apply_state_defaults(s)
    uid = update.effective_user.id

    await _handle_referral_deeplink(update, ctx)

    try:
        bonus_result = ledger_storage.grant_signup_bonus(uid, 10)
        _set_cached_balance(ctx, bonus_result.balance)
        if bonus_result.applied:
            await update.message.reply_text("🎁 Добро пожаловать! Начислил +10💎 на баланс.")
    except Exception as exc:
        log.exception("Signup bonus failed for %s: %s", uid, exc)

    chat_id = update.effective_chat.id if update.effective_chat else uid
    await show_main_menu(chat_id, ctx)

async def menu_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await ensure_user_record(update)
    message = update.effective_message
    if message is None:
        return
    s = state(ctx)
    s.update({**DEFAULT_STATE})
    _apply_state_defaults(s)
    uid = update.effective_user.id if update.effective_user else 0
    chat_id = update.effective_chat.id if update.effective_chat else uid
    await show_main_menu(chat_id, ctx)


async def suno_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await ensure_user_record(update)
    message = update.effective_message
    chat = update.effective_chat
    if chat is None:
        return

    if not _suno_configured():
        await _suno_notify(
            ctx,
            chat.id,
            "⚠️ Suno API не настроен. Обратитесь к поддержке.",
            reply_to=message,
        )
        return

    await suno_entry(chat.id, ctx)


async def video_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await ensure_user_record(update)
    message = update.effective_message
    if message is None:
        return
    state(ctx)["mode"] = None
    await message.reply_text("🎬 Выберите тип генерации видео:", reply_markup=video_menu_kb())


async def image_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await ensure_user_record(update)
    message = update.effective_message
    if message is None:
        return
    state(ctx)["mode"] = None
    await message.reply_text("🖼️ Выберите режим работы с изображениями:", reply_markup=image_menu_kb())


async def buy_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await ensure_user_record(update)
    await topup(update, ctx)


async def lang_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await ensure_user_record(update)
    message = update.effective_message
    if message is None:
        return
    await message.reply_text("🌍 Переключение языка будет доступно в ближайшее время.")


async def help_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await ensure_user_record(update)
    message = update.effective_message
    if message is None:
        return
    await message.reply_text("🆘 Поддержка появится скоро. Напишите сюда свой вопрос, и мы ответим позже.")


async def faq_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await ensure_user_record(update)
    message = update.effective_message
    if message is None:
        return
    await message.reply_text(
        render_faq_text(),
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=main_menu_kb(),
    )

async def topup(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await ensure_user_record(update)
    await update.message.reply_text(
        "💳 Пополнение через *Telegram Stars*.\nЕсли звёзд не хватает — купите в официальном боте:",
        parse_mode=ParseMode.MARKDOWN, reply_markup=stars_topup_kb()
    )


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
    mid = await show_balance_card(chat.id, ctx)
    if mid is None:
        user = update.effective_user
        if user is None or update.message is None:
            return
        balance = _safe_get_balance(user.id)
        _set_cached_balance(ctx, balance)
        await update.message.reply_text(f"💎 Ваш баланс: {balance} 💎")


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
async def error_handler(update: Optional[Update], context: ContextTypes.DEFAULT_TYPE):
    log.exception("Unhandled error: %s", context.error)
    try:
        if update and update.effective_chat:
            await context.bot.send_message(update.effective_chat.id, "⚠️ Системная ошибка. Попробуйте ещё раз.")
    except Exception:
        pass

async def show_banana_card(chat_id: int, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    s = state(ctx)
    text = banana_card_text(s)
    if text == s.get("_last_text_banana"):
        return
    kb = banana_kb()
    mid = await upsert_card(ctx, chat_id, s, "last_ui_msg_id_banana", text, kb)
    if mid:
        s["_last_text_banana"] = text
    else:
        s["_last_text_banana"] = None

async def banana_entry(chat_id: int, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    s = state(ctx)
    mid = s.get("last_ui_msg_id_banana")
    if mid:
        with suppress(Exception):
            await ctx.bot.delete_message(chat_id, mid)
    s["last_ui_msg_id_banana"] = None
    s["_last_text_banana"] = None
    await show_banana_card(chat_id, ctx)

async def on_banana_photo_received(chat_id: int, ctx: ContextTypes.DEFAULT_TYPE, file_id: str) -> None:
    s = state(ctx)
    s.setdefault("banana_images", []).append(file_id)
    s["_last_text_banana"] = None
    await show_banana_card(chat_id, ctx)

async def on_banana_prompt_saved(chat_id: int, ctx: ContextTypes.DEFAULT_TYPE, text_prompt: str) -> None:
    s = state(ctx)
    s["last_prompt"] = text_prompt
    s["_last_text_banana"] = None
    await show_banana_card(chat_id, ctx)

async def show_veo_card(chat_id: int, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    s = state(ctx)
    text = veo_card_text(s)
    if text == s.get("_last_text_veo"):
        return
    kb = veo_kb(s)
    mid = await upsert_card(ctx, chat_id, s, "last_ui_msg_id_veo", text, kb)
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
    await show_veo_card(chat_id, ctx)

async def set_veo_card_prompt(chat_id: int, prompt_text: str, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    s = state(ctx)
    s["last_prompt"] = prompt_text
    s["_last_text_veo"] = None
    await show_veo_card(chat_id, ctx)


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


configure_prompt_master(update_veo_card=lambda chat_id, context: show_veo_card(chat_id, context))

async def on_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await ensure_user_record(update)
    q = update.callback_query
    if not q:
        return
    data = (q.data or "").strip()
    s = state(ctx)
    message = q.message
    chat = update.effective_chat
    chat_id = None
    if message is not None:
        chat_id = message.chat_id
    elif chat is not None:
        chat_id = chat.id

    if data == CB_MODE_CHAT:
        if chat_id is not None:
            _mode_set(chat_id, MODE_CHAT)
        s["mode"] = None
        await q.answer("Режим: Обычный чат")
        if message is not None:
            await _safe_edit_message_text(
                q.edit_message_text,
                "Режим переключён: теперь это обычный чат. Напишите сообщение.",
            )
        return

    if data == CB_MODE_PM:
        if chat_id is not None:
            _mode_set(chat_id, MODE_PM)
        s["mode"] = None
        await q.answer("Режим: Prompt-Master")
        if message is not None:
            await _safe_edit_message_text(
                q.edit_message_text,
                "Режим переключён: Prompt-Master. Пришлите идею/сцену — верну кинопромпт.",
            )
        return

    if data == CB_GO_HOME:
        if chat_id is not None:
            _mode_set(chat_id, MODE_CHAT)
        s.update({**DEFAULT_STATE})
        _apply_state_defaults(s)
        await q.answer()
        if message is not None:
            with suppress(BadRequest):
                await q.edit_message_reply_markup(reply_markup=None)
            await message.reply_text("🏠 Главное меню:", reply_markup=main_menu_kb())
        return

    if data.startswith(CB_PM_INSERT_VEO):
        await handle_pm_insert_to_veo(update, ctx, data)
        return

    if data in (PROMPT_MASTER_OPEN, PROMPT_MASTER_CANCEL):
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
        await show_balance_card(target_chat, ctx)
        return

    if data == "promo_open":
        if not PROMO_ENABLED:
            await q.message.reply_text("🎟️ Промокоды временно отключены.")
            return
        s["mode"] = "promo"
        await q.message.reply_text("Введите промокод одним сообщением…")
        return

    if data == "faq":
        await q.message.reply_text(
            render_faq_text(),
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=main_menu_kb(),
        )
        return

    if data == "back":
        s.update({**DEFAULT_STATE})
        _apply_state_defaults(s)
        await q.message.reply_text("🏠 Главное меню:", reply_markup=main_menu_kb()); return

    if data == "start_new_cycle":
        s.update({**DEFAULT_STATE})
        _apply_state_defaults(s)
        await q.message.reply_text("Выберите режим:", reply_markup=main_menu_kb()); return

    if data == "topup_open":
        await q.message.reply_text("💳 Выберите пакет Stars ниже:", reply_markup=stars_topup_kb()); return

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
        if selected_mode in ("veo_text_fast", "veo_text_quality"):
            s["aspect"] = "16:9"; s["model"] = "veo3_fast" if selected_mode.endswith("fast") else "veo3"
            await veo_entry(update.effective_chat.id, ctx)
            await q.message.reply_text("✍️ Пришлите текст идеи и/или фото-референс — карточка обновится автоматически.")
            return
        if selected_mode == "veo_photo":
            s["aspect"] = "9:16"; s["model"] = "veo3_fast"
            await veo_entry(update.effective_chat.id, ctx)
            await q.message.reply_text("📸 Пришлите фото (подпись-промпт — по желанию). Карточка обновится автоматически.")
            return
        if selected_mode == "chat":
            await q.message.reply_text("💬 Чат активен. Напишите сообщение."); return
        if selected_mode == "mj_txt":
            s["aspect"] = "9:16" if s.get("aspect") == "9:16" else "16:9"
            s["last_prompt"] = None
            s["mj_generating"] = False
            s["mj_last_wait_ts"] = 0.0
            s["last_mj_task_id"] = None
            await mj_entry(update.effective_chat.id, ctx)
            return
        if selected_mode == "banana":
            s["banana_images"] = []; s["last_prompt"] = None
            await q.message.reply_text("🍌 Banana включён\nСначала пришлите до *4 фото* (можно по одному). Когда будут готовы — пришлите *текст-промпт*, что изменить.", parse_mode=ParseMode.MARKDOWN)
            await banana_entry(update.effective_chat.id, ctx); return

    if data.startswith("mj:"):
        chat = update.effective_chat
        if not chat:
            return
        parts = data.split(":", 2)
        action = parts[1] if len(parts) > 1 else ""
        payload = parts[2] if len(parts) > 2 else ""
        chat_id = chat.id
        current_aspect = "9:16" if s.get("aspect") == "9:16" else "16:9"

        if action == "aspect":
            if s.get("mj_generating"):
                await q.message.reply_text("⏳ Дождитесь завершения текущей генерации."); return
            new_aspect = "9:16" if payload == "9:16" else "16:9"
            s["aspect"] = new_aspect
            s["last_prompt"] = None
            await show_mj_prompt_card(chat_id, ctx)
            return

        if action == "change_format":
            if s.get("mj_generating"):
                await q.message.reply_text("⏳ Дождитесь завершения текущей генерации."); return
            await show_mj_format_card(chat_id, ctx)
            return

        if action == "cancel":
            s["mode"] = None
            s["last_prompt"] = None
            s["mj_generating"] = False
            s["last_mj_task_id"] = None
            s["mj_last_wait_ts"] = 0.0
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
            await q.message.reply_text("🏠 Главное меню:", reply_markup=main_menu_kb()); return

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
            try:
                ensure_user(uid)
            except Exception as exc:
                log.exception("MJ ensure_user failed for %s: %s", uid, exc)
                await q.message.reply_text("⚠️ Не удалось проверить баланс. Попробуйте позже.")
                return
            ok, balance_after = debit_try(
                uid,
                price,
                reason="service:start",
                meta={"service": "MJ", "aspect": aspect_value, "prompt": _short_prompt(prompt, 160)},
            )
            if not ok:
                current_balance = balance_after if isinstance(balance_after, int) else get_balance(uid)
                await show_balance_notification(
                    chat_id,
                    ctx,
                    uid,
                    f"🙇 Недостаточно токенов. Нужно: {price}💎, у вас: {current_balance}💎.",
                    reply_markup=inline_topup_keyboard(),
                )
                return
            await q.message.reply_text("✅ Промпт принят.")
            await show_balance_notification(
                chat_id,
                ctx,
                uid,
                f"✅ Списано {price}💎. Текущий баланс: {balance_after}💎 — запускаю…",
            )
            s["mj_generating"] = True
            s["mj_last_wait_ts"] = time.time()
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
            await q.message.reply_text("✍️ Пришлите новый промпт для Banana."); return
        if act == "start":
            imgs = s.get("banana_images") or []
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
            ok, balance_after = debit_try(
                uid,
                PRICE_BANANA,
                reason="service:start",
                meta={"service": "BANANA", "images": len(imgs)},
            )
            if not ok:
                current_balance = balance_after if isinstance(balance_after, int) else get_balance(uid)
                await show_balance_notification(
                    update.effective_chat.id,
                    ctx,
                    uid,
                    f"🙇 Недостаточно токенов. Нужно: {PRICE_BANANA}💎, у вас: {current_balance}💎.",
                    reply_markup=inline_topup_keyboard(),
                )
                return
            new_balance = balance_after
            s["banana_balance"] = new_balance
            s["_last_text_banana"] = None
            chat_id = update.effective_chat.id
            await show_banana_card(chat_id, ctx)
            await show_main_menu(chat_id, ctx)
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
        s["mode"] = "suno"

        if action == "edit":
            field = argument
            if field not in {"title", "style", "lyrics"}:
                await q.answer("Недоступное поле", show_alert=True)
                return
            s["suno_waiting_field"] = field
            prompts = {
                "title": "Введите название трека. Для очистки отправьте «-». Отмена — /cancel.",
                "style": "Опишите стиль музыки. Для очистки отправьте «-». Отмена — /cancel.",
                "lyrics": "Пришлите текст песни (8–16 строк). Для очистки отправьте «-». Отмена — /cancel.",
            }
            msg_text = prompts.get(field, "Введите значение.")
            await q.answer()
            target_chat = chat_id
            if target_chat is None and q.message is not None:
                target_chat = q.message.chat_id
            if target_chat is None and uid is not None:
                target_chat = uid
            if target_chat is None:
                await q.answer("Нет чата", show_alert=True)
                return
            await _suno_notify(ctx, target_chat, msg_text, reply_to=q.message)
            return

        if action == "toggle" and argument == "instrumental":
            new_value = not bool(s.get("suno_instrumental", True))
            s["suno_instrumental"] = new_value
            s["suno_waiting_field"] = None
            s["_last_text_suno"] = None
            if chat_id is not None:
                await refresh_suno_card(ctx, chat_id, s, price=PRICE_SUNO)
            await q.answer("Режим: " + ("Инструментал" if new_value else "Со словами"))
            return

        if action == "start":
            if chat_id is None:
                await q.answer("Нет чата", show_alert=True)
                return
            await q.answer()
            params = _suno_collect_params(s)
            await _launch_suno_generation(
                chat_id,
                ctx,
                params=params,
                user_id=uid,
                reply_to=q.message,
                trigger="start",
            )
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
            await _launch_suno_generation(
                chat_id,
                ctx,
                params=params,
                user_id=uid,
                reply_to=q.message,
                trigger="repeat",
            )
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
        ok, balance_after = debit_try(
            uid,
            price,
            reason="service:start",
            meta=meta,
        )
        if not ok:
            current_balance = balance_after if isinstance(balance_after, int) else get_balance(uid)
            await show_balance_notification(
                chat_id,
                ctx,
                uid,
                f"🙇 Недостаточно токенов. Нужно: {price}💎, у вас: {current_balance}💎.",
                reply_markup=inline_topup_keyboard(),
            )
            return
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
    state_mode = s.get("mode")
    user_mode = _mode_get(chat_id) or MODE_CHAT

    waiting_field = s.get("suno_waiting_field")
    if waiting_field:
        lowered = text.lower()
        if lowered == "/cancel":
            s["suno_waiting_field"] = None
            await msg.reply_text("✏️ Изменение отменено.")
            return

        clear_requested = text in {"-", "—"}
        updated_message = "Обновлено."
        if waiting_field == "title":
            new_value = "" if clear_requested else text
            s["suno_title"] = new_value or None
            updated_message = "Название обновлено." if new_value else "Название очищено."
        elif waiting_field == "style":
            new_value = "" if clear_requested else text
            s["suno_style"] = new_value or None
            updated_message = "Стиль обновлён." if new_value else "Стиль очищен."
        elif waiting_field == "lyrics":
            new_value = "" if clear_requested else raw_text.strip()
            s["suno_lyrics"] = new_value or None
            updated_message = "Текст песни обновлён." if new_value else "Текст песни очищен."
        else:
            new_value = text

        s["suno_waiting_field"] = None
        s["_last_text_suno"] = None
        await refresh_suno_card(ctx, chat_id, s, price=PRICE_SUNO)
        await _send_with_retry(lambda: msg.reply_text(updated_message))
        return

    if text == MENU_BTN_VIDEO:
        s["mode"] = None
        await msg.reply_text("🎬 Выберите тип генерации видео:", reply_markup=video_menu_kb())
        return

    if text == MENU_BTN_IMAGE:
        s["mode"] = None
        await msg.reply_text("🖼️ Выберите режим работы с изображениями:", reply_markup=image_menu_kb())
        return

    if text == MENU_BTN_SUNO:
        await suno_entry(chat_id, ctx)
        return

    if text == MENU_BTN_BALANCE:
        chat_id = msg.chat_id
        mid = await show_balance_card(chat_id, ctx)
        if mid is None:
            balance = get_user_balance_value(ctx, force_refresh=True)
            await msg.reply_text(
                f"💎 Ваш баланс: {balance} 💎",
                reply_markup=balance_menu_kb(),
            )
        return

    if text == MENU_BTN_PM:
        _mode_set(chat_id, MODE_PM)
        s["mode"] = None
        await msg.reply_text("Режим переключён: Prompt-Master. Пришлите идею/сцену — верну кинопромпт.")
        return

    if text == MENU_BTN_CHAT:
        _mode_set(chat_id, MODE_CHAT)
        s["mode"] = None
        await msg.reply_text("Режим переключён: Обычный чат. Напишите сообщение.")
        return

    if state_mode == "promo":
        if not PROMO_ENABLED:
            await msg.reply_text("🎟️ Промокоды временно отключены.")
            s["mode"] = None
            return
        await process_promo_submission(update, ctx, text)
        return

    if user_mode == MODE_PM:
        if not text:
            await msg.reply_text("⚠️ Пришлите идею или сцену для Prompt-Master.")
            return
        from prompt_master import build_cinema_prompt

        status_msg = await msg.reply_text("🧠 Пишу промпт…")
        with suppress(Exception):
            await ctx.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

        try:
            kino_prompt, _ = await build_cinema_prompt(text, user_lang=detect_lang(text))
        except Exception as exc:
            log.exception("Prompt-Master generation failed: %s", exc)
            err_text = "⚠️ Не удалось сгенерировать промпт, попробуйте ещё раз."
            if status_msg:
                try:
                    result = await _safe_edit_message_text(
                        ctx.bot.edit_message_text,
                        chat_id=chat_id,
                        message_id=status_msg.message_id,
                        text=err_text,
                    )
                    if isinstance(result, BadRequest):
                        await msg.reply_text(err_text)
                except Exception:
                    await msg.reply_text(err_text)
            else:
                await msg.reply_text(err_text)
            return

        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("🎬 Вставить в VEO", callback_data=CB_PM_INSERT_VEO)],
            [InlineKeyboardButton("⬅️ Назад", callback_data=CB_GO_HOME)],
        ])
        block = f"```\n{kino_prompt.strip()}\n```"
        final_text = f"🧠 Готово! Вот ваш кинопромпт:\n\n{block}"

        try:
            result = await _safe_edit_message_text(
                ctx.bot.edit_message_text,
                chat_id=chat_id,
                message_id=status_msg.message_id,
                text=final_text,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=kb,
                disable_web_page_preview=True,
            )
            if isinstance(result, BadRequest):
                await msg.reply_text(
                    final_text,
                    parse_mode=ParseMode.MARKDOWN,
                    reply_markup=kb,
                    disable_web_page_preview=True,
                )
        except Exception:
            await msg.reply_text(
                final_text,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=kb,
                disable_web_page_preview=True,
            )
        cache_pm_prompt(chat_id, kino_prompt.strip())
        return

    low = text.lower()
    if low.startswith(("http://", "https://")) and any(
        low.split("?")[0].endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp", ".heic")
    ):
        if state_mode == "banana":
            if len(s["banana_images"]) >= 4:
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

    if user_mode == MODE_CHAT:
        if not text:
            await msg.reply_text("⚠️ Отправьте сообщение.")
            return
        if openai is None or not OPENAI_API_KEY:
            await msg.reply_text("⚠️ ChatGPT недоступен (нет OPENAI_API_KEY).")
            return
        await msg.reply_text("💬 Думаю над ответом…")
        try:
            answer = await chatgpt_smalltalk(text, chat_id)
        except Exception as exc:
            log.exception("Chat error: %s", exc)
            await msg.reply_text("⚠️ Ошибка запроса к ChatGPT.")
            return
        await msg.reply_text(
            answer,
            reply_markup=None,
            disable_web_page_preview=True,
        )
        return

    # Если режим не распознан — по умолчанию обновляем карточку VEO
    s["last_prompt"] = text
    await show_veo_card(chat_id, ctx)

async def _banana_run_and_send(
    chat_id: int,
    ctx: ContextTypes.DEFAULT_TYPE,
    src_urls: List[str],
    prompt: str,
    price: int,
    user_id: int,
) -> None:
    s = state(ctx)
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
        await show_main_menu(chat_id, ctx)
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
        try:
            await ctx.bot.send_photo(chat_id=chat_id, photo=u0, caption="✅ Banana готово")
        except Exception:
            r = requests.get(u0, timeout=180)
            r.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                f.write(r.content)
                path = f.name
            with open(path, "rb") as f:
                await ctx.bot.send_document(
                    chat_id=chat_id,
                    document=InputFile(f, filename="banana.png"),
                    caption="✅ Banana готово",
                )
            try:
                os.unlink(path)
            except Exception:
                pass
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
    s = state(ctx)
    photos = update.message.photo
    if not photos: return
    ph = photos[-1]
    try:
        file = await ctx.bot.get_file(ph.file_id)
        if not file.file_path:
            await update.message.reply_text("⚠️ Не удалось получить путь к файлу Telegram."); return
        url = tg_direct_file_url(TELEGRAM_TOKEN, file.file_path)
        if s.get("mode") == "banana":
            if len(s["banana_images"]) >= 4:
                await update.message.reply_text("⚠️ Достигнут лимит 4 фото.", reply_markup=banana_kb()); return
            cap = (update.message.caption or "").strip()
            if cap:
                s["last_prompt"] = cap
                s["_last_text_banana"] = None
            await on_banana_photo_received(update.effective_chat.id, ctx, url)
            await update.message.reply_text(f"📸 Фото принято ({len(s['banana_images'])}/4).")
            return
        s["last_image_url"] = url
        await update.message.reply_text("🖼️ Фото принято как референс.")
        if s.get("mode") in ("veo_text_fast","veo_text_quality","veo_photo"):
            await show_veo_card(update.effective_chat.id, ctx)
    except Exception as e:
        log.exception("Get photo failed: %s", e)
        await update.message.reply_text("⚠️ Не удалось обработать фото. Пришлите публичный URL картинки текстом.")

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

    await message.reply_text(
        f"✅ Начислено +{diamonds_to_credit} 💎. Новый баланс: {new_balance} 💎."
    )

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
            await self._redis.close()
        except Exception:
            pass
        self._redis = None

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

    # Handlers (оставляем как есть)
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("menu", menu_command))
    application.add_handler(CommandHandler("buy", buy_command))
    application.add_handler(CommandHandler("video", video_command))
    application.add_handler(CommandHandler("image", image_command))
    application.add_handler(CommandHandler("suno", suno_command))
    application.add_handler(CommandHandler("lang", lang_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("faq", faq_command))
    application.add_handler(CommandHandler("health", health))
    application.add_handler(CommandHandler("topup", topup))
    application.add_handler(CommandHandler("promo", promo_command))
    application.add_handler(CommandHandler("users_count", users_count_command))
    application.add_handler(CommandHandler("whoami", whoami_command))
    application.add_handler(CommandHandler("suno_debug", suno_debug_command))
    application.add_handler(CommandHandler("broadcast", broadcast_command))
    application.add_handler(CommandHandler("my_balance", my_balance_command))
    application.add_handler(CommandHandler("add_balance", add_balance_command))
    application.add_handler(CommandHandler("sub_balance", sub_balance_command))
    application.add_handler(CommandHandler("transactions", transactions_command))
# codex/fix-balance-reset-after-deploy
    application.add_handler(CommandHandler("balance", balance_command))
    application.add_handler(CommandHandler("balance_recalc", balance_recalc))
    application.add_handler(prompt_master_conv)
# main
    application.add_handler(PreCheckoutQueryHandler(precheckout_callback))
    application.add_handler(MessageHandler(filters.SUCCESSFUL_PAYMENT, successful_payment_handler))
    application.add_handler(CallbackQueryHandler(on_callback))
    application.add_handler(MessageHandler(filters.PHOTO, on_photo))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
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

            def _trigger_stop(sig: Optional[signal.Signals] = None, *, reason: str = "external") -> None:
                if stop_event.is_set():
                    return
                if sig is not None:
                    sig_name = sig.name if hasattr(sig, "name") else str(sig)
                    log.info("Stop signal received: %s. Triggering shutdown.", sig_name)
                else:
                    log.info("Stop requested (%s). Triggering shutdown.", reason)
                stop_event.set()

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
    asyncio.run(run_bot_async())


if __name__ == "__main__":
    main()
