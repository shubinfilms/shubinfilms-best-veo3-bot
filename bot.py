# -*- coding: utf-8 -*-
# Best VEO3 Bot — PTB 21.x
# Версия: 2025-09-14r4
# Единственное изменение против прежней версии: надежная доставка VEO-видео в Telegram
# (освежение ссылки + повторная попытка + download&reupload с увеличенным таймаутом).
# Остальное (карточки, кнопки, тексты, цены, FAQ, промокоды, бонусы и т.д.) — без изменений.

# odex/fix-balance-reset-after-deploy
import os, json, time, uuid, asyncio, logging, tempfile, subprocess, re, signal, socket, hashlib, io
from pathlib import Path
# main
from typing import Dict, Any, Optional, List, Tuple, Callable
from datetime import datetime, timezone
from contextlib import suppress
from urllib.parse import urlparse

import aiohttp
from aiohttp import ClientError, ClientResponseError, ClientTimeout
import requests
from dotenv import load_dotenv

from telegram import (
    Update, InlineKeyboardButton, InlineKeyboardMarkup,
    InputFile, LabeledPrice, InputMediaPhoto
)
from telegram.constants import ParseMode, ChatAction
from telegram.ext import (
    ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler,
    CallbackQueryHandler, filters, AIORateLimiter, PreCheckoutQueryHandler
)
from telegram.error import BadRequest, RetryAfter

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

from redis_utils import clear_task_meta, load_task_meta, save_task_meta

from ledger import (
    LedgerStorage,
    LedgerOpResult,
    BalanceRecalcResult,
    InsufficientBalance,
)
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


load_dotenv()
def _env(k: str, d: str = "") -> str:
    v = os.getenv(k)
    return (v if v is not None else d).strip()

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

# Redis
REDIS_URL           = _env("REDIS_URL")
REDIS_PREFIX        = _env("REDIS_PREFIX", "veo3:prod")
REDIS_LOCK_ENABLED  = _env("REDIS_LOCK_ENABLED", "true").lower() == "true"
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True) if REDIS_URL else None

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
CB_PM_INSERT_VEO = "pm:insert_to_veo"

# ==========================
#   Tokens / Pricing
# ==========================
TOKEN_COSTS = {
    "veo_fast": 50,
    "veo_quality": 150,
    "veo_photo": 50,
    "mj": 10,          # 16:9 или 9:16
    "banana": 5,
    "chat": 0,
}
CHAT_UNLOCK_PRICE = 0

# ==========================
#   Promo codes (one-time / global)
# ==========================
PROMO_CODES = {
    "WELCOME50": 50,
    "FREE10": 10,
    "LABACCENT100": 100,
    "BONUS50": 50,
    "FRIENDS150": 150,
}

def promo_amount(code: str) -> Optional[int]:
    code = (code or "").strip().upper()
    if not code: return None
    if redis_client:
        v = redis_client.get(_rk("promo", "amount", code))
        if v:
            try: return int(v)
            except: pass
    return PROMO_CODES.get(code)

def promo_used_global(code: str) -> Optional[int]:
    code = (code or "").strip().upper()
    if not code: return None
    if redis_client:
        u = redis_client.get(_rk("promo", "used_by", code))
        try: return int(u) if u is not None else None
        except: return None
    try:
        owner = ledger_storage.get_promo_owner(code)
        return owner
    except Exception as exc:
        log.warning("Failed to fetch promo owner for %s: %s", code, exc)
        return None

def promo_mark_used(code: str, uid: int):
    code = (code or "").strip().upper()
    if not code: return
    if redis_client:
        redis_client.setnx(_rk("promo", "used_by", code), str(uid))

# локальный кэш процесса (если Redis выключен)
app_cache: Dict[Any, Any] = {}


_CYRILLIC_RE = re.compile(r"[а-яА-ЯёЁ]")


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
    if not force_refresh:
        cached = ctx.user_data.get("balance")
        if cached is not None:
            try:
                return int(cached)
            except (TypeError, ValueError):
                pass

    uid = get_user_id(ctx)
    if not uid:
        return 0

    balance = ledger_storage.get_balance(uid)
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
    "last_ui_msg_id": None, "last_ui_msg_id_banana": None,
    "banana_images": [],
    "mj_last_wait_ts": 0.0,
    "mj_generating": False, "last_mj_task_id": None, "last_mj_msg_id": None,
    "active_generation_op": None,
    "mj_active_op_key": None,
    "banana_active_op_key": None,
}
def state(ctx: ContextTypes.DEFAULT_TYPE) -> Dict[str, Any]:
    ud = ctx.user_data
    for k, v in DEFAULT_STATE.items():
        if k not in ud: ud[k] = [] if isinstance(v, list) else v
    if not isinstance(ud.get("banana_images"), list): ud["banana_images"] = []
    return ud

# odex/fix-balance-reset-after-deploy
# main
# ==========================
#   UI / Texts
# ==========================
WELCOME = (
    "🎬 *Veo 3 — съёмочная команда*: опиши идею и получи *готовый клип*.\n"
    "🖌️ *MJ — художник*: рисует изображение по тексту (16:9 или 9:16).\n"
    "🍌 *Banana — редактор из будущего*: меняет фон, одежду, макияж, убирает лишнее, объединяет людей.\n"
    "🧠 *Prompt-Master* — вернёт профессиональный *кинопромпт*.\n"
    "💬 *Обычный чат* — ответы на любые вопросы.\n\n"
    "💎 *Ваш баланс:* {balance}\n"
    "📈 Больше идей и примеров: {prompts_url}\n\n"
    "Выберите режим 👇"
)

def render_welcome_for(uid: int, ctx: ContextTypes.DEFAULT_TYPE) -> str:
    return WELCOME.format(balance=get_user_balance_value(ctx), prompts_url=PROMPTS_CHANNEL_URL)

def main_menu_kb() -> InlineKeyboardMarkup:
    keyboard = [
        [InlineKeyboardButton("🧠 Prompt-Master", callback_data=CB_MODE_PM)],
        [InlineKeyboardButton("💬 Обычный чат (ChatGPT)", callback_data=CB_MODE_CHAT)],
        [InlineKeyboardButton(f"🎬 Генерация видео (Veo Fast) 💎 {TOKEN_COSTS['veo_fast']}", callback_data="mode:veo_text_fast")],
        [InlineKeyboardButton(f"🎬 Генерация видео (Veo Quality) 💎 {TOKEN_COSTS['veo_quality']}", callback_data="mode:veo_text_quality")],
        [InlineKeyboardButton(f"🖼️ Генерация изображений (MJ) 💎 {TOKEN_COSTS['mj']}", callback_data="mode:mj_txt")],
        [InlineKeyboardButton(f"🍌 Редактор изображений (Banana) 💎 {TOKEN_COSTS['banana']}", callback_data="mode:banana")],
        [InlineKeyboardButton(f"📸 Оживить изображение (Veo) 💎 {TOKEN_COSTS['veo_photo']}", callback_data="mode:veo_photo")],
        [
            InlineKeyboardButton("❓ FAQ", callback_data="faq"),
            InlineKeyboardButton("📈 Канал с промптами", url=PROMPTS_CHANNEL_URL),
        ],
        [InlineKeyboardButton("💳 Пополнить баланс", callback_data="topup_open")],
    ]

    if PROMO_ENABLED:
        keyboard.append([
            InlineKeyboardButton("🎁 Активировать промокод", callback_data="promo_open")
        ])

    return InlineKeyboardMarkup(keyboard)

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
        "🖼 Midjourney\n"
        "Выберите формат изображения.\n\n"
        "• Горизонтальный — 16:9\n"
        "• Вертикальный — 9:16\n\n"
        f"Текущий выбор: {choice}"
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
        "🖼 Midjourney",
        "",
        'Введите промпт сообщением. После этого нажмите "Подтвердить".',
        f"Текущий формат: {aspect}",
    ]
    snippet = _short_prompt(prompt)
    if snippet:
        lines.extend(["", f'Последний промпт: "{snippet}"'])
    return "\n".join(lines)

def _mj_prompt_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Подтвердить", callback_data="mj:confirm")],
        [
            InlineKeyboardButton("Отменить", callback_data="mj:cancel"),
            InlineKeyboardButton("Сменить формат", callback_data="mj:change_format"),
        ],
    ])

async def _send_or_edit_mj_card(chat_id: int, ctx: ContextTypes.DEFAULT_TYPE, text: str,
                                reply_markup: Optional[InlineKeyboardMarkup]) -> None:
    s = state(ctx)
    mid = s.get("last_mj_msg_id")
    try:
        if mid:
            await ctx.bot.edit_message_text(
                chat_id=chat_id,
                message_id=mid,
                text=text,
                reply_markup=reply_markup,
                disable_web_page_preview=True,
            )
        else:
            msg = await ctx.bot.send_message(
                chat_id,
                text,
                reply_markup=reply_markup,
                disable_web_page_preview=True,
            )
            s["last_mj_msg_id"] = msg.message_id
    except Exception as e:
        if "message is not modified" in str(e).lower():
            return
        log.warning("MJ card send/edit failed: %s", e)
        try:
            msg = await ctx.bot.send_message(
                chat_id,
                text,
                reply_markup=reply_markup,
                disable_web_page_preview=True,
            )
            s["last_mj_msg_id"] = msg.message_id
        except Exception as e2:
            log.warning("MJ card send fallback failed: %s", e2)

async def show_mj_format_card(chat_id: int, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    s = state(ctx)
    aspect = "9:16" if s.get("aspect") == "9:16" else "16:9"
    s["aspect"] = aspect
    s["last_prompt"] = None
    await _send_or_edit_mj_card(chat_id, ctx, _mj_format_card_text(aspect), _mj_format_keyboard(aspect))

async def show_mj_prompt_card(chat_id: int, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    s = state(ctx)
    aspect = "9:16" if s.get("aspect") == "9:16" else "16:9"
    s["aspect"] = aspect
    await _send_or_edit_mj_card(chat_id, ctx, _mj_prompt_card_text(aspect, s.get("last_prompt")), _mj_prompt_keyboard())

async def show_mj_generating_card(chat_id: int, ctx: ContextTypes.DEFAULT_TYPE, prompt: str, aspect: str) -> None:
    aspect = "9:16" if aspect == "9:16" else "16:9"
    snippet = _short_prompt(prompt, 160)
    text = (
        "⏳ Midjourney генерирует изображение…\n"
        f"Формат: {aspect}\n"
        f'Промпт: "{snippet}"'
    )
    await _send_or_edit_mj_card(chat_id, ctx, text, None)

def banana_examples_block() -> str:
    return (
        "💡 *Примеры запросов:*\n"
        "• поменяй фон на городской вечер\n"
        "• смени одежду на чёрный пиджак\n"
        "• добавь лёгкий макияж, подчеркни глаза\n"
        "• убери лишние предметы со стола\n"
        "• поставь нас на одну фотографию\n"
    )

def banana_card_text(s: Dict[str, Any]) -> str:
    n = len(s.get("banana_images") or [])
    prompt = (s.get("last_prompt") or "—").strip()
    lines = [
        "🍌 *Карточка Banana*",
        f"🧩 Фото: *{n}/4*  •  Промпт: *{'есть' if s.get('last_prompt') else 'нет'}*",
        "",
        "🖊️ *Промпт:*",
        f"`{prompt}`",
        "",
        banana_examples_block()
    ]
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

# --------- VEO Card ----------
def veo_card_text(s: Dict[str, Any]) -> str:
    prompt = (s.get("last_prompt") or "—").strip()
    img = "есть" if s.get("last_image_url") else "нет"
    return (
        "🟦 *Карточка VEO*\n"
        f"• Формат: *{s.get('aspect') or '16:9'}*\n"
        f"• Модель: *{'Veo Quality' if s.get('model')=='veo3' else 'Veo Fast'}*\n"
        f"• Фото-референс: *{img}*\n\n"
        "🖊️ *Промпт:*\n"
        f"`{prompt}`"
    )

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
async def poll_veo_and_send(chat_id: int, task_id: str, gen_id: str, ctx: ContextTypes.DEFAULT_TYPE):
    original_chat_id = chat_id
    s = state(ctx)
    price = TOKEN_COSTS['veo_quality'] if s.get('model') == 'veo3' else TOKEN_COSTS['veo_fast']
    op_key = s.get("active_generation_op")

    def _cleanup() -> None:
        ACTIVE_TASKS.pop(original_chat_id, None)
        if s.get("generation_id") == gen_id:
            s["generating"] = False
            s["generation_id"] = None
        if op_key:
            _clear_operation(ctx, op_key)
        s.pop("active_generation_op", None)
        _clear_kie_request_id(task_id)
        with suppress(Exception):
            clear_task_meta(task_id)

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
                refund(
                    ctx,
                    price,
                    f"refund:{task_id}:timeout",
                    {"reason": "timeout"},
                )
                await _send_message_with_retry(original_chat_id, RENDER_FAIL_MESSAGE)
                return
            except Exception as exc:
                log.exception("VEO status polling failed: %s", exc)
                refund(
                    ctx,
                    price,
                    f"refund:{task_id}:poll_exception",
                    {"reason": "poll_exception", "message": str(exc)},
                )
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
        refund(
            ctx,
            price,
            f"refund:{task_id}:timeout_final",
            {"reason": "timeout"},
        )
        await _send_message_with_retry(original_chat_id, RENDER_FAIL_MESSAGE)
    except Exception as exc:
        log.exception("VEO render failed: %s", exc)
        refund(
            ctx,
            price,
            f"refund:{task_id}:exception",
            {"reason": "exception", "message": str(exc)},
        )
        await _send_message_with_retry(original_chat_id, RENDER_FAIL_MESSAGE)
    finally:
        if temp_file and temp_file.exists():
            with suppress(Exception):
                temp_file.unlink()
        _cleanup()

# ==========================
#   MJ poll (1 авторетрай)
# ==========================
async def poll_mj_and_send_photos(chat_id: int, task_id: str, ctx: ContextTypes.DEFAULT_TYPE,
                                  prompt: str, aspect: str) -> None:
    price = TOKEN_COSTS["mj"]
    start_ts = time.time()
    delay = 12
    max_wait = 12 * 60
    retried = False
    success = False
    aspect_ratio = "9:16" if aspect == "9:16" else "16:9"
    prompt_for_retry = (prompt or "").strip()
    s = state(ctx)
    s["last_mj_task_id"] = task_id

    op_key = s.get("mj_active_op_key")

    def _refund(reason_tag: str, message: Optional[str] = None) -> None:
        meta: Dict[str, Any] = {"reason": reason_tag}
        if message:
            meta["message"] = message
        refund_op_id = f"refund:{task_id}:{reason_tag}"
        try:
            credit_tokens(ctx, price, "mj_refund", refund_op_id, meta)
        except Exception as exc:
            log.exception("MJ refund %s failed: %s", reason_tag, exc)

    try:
        while True:
            ok, flag, data = await asyncio.to_thread(mj_status, task_id)
            if not ok:
                _refund("status_error")
                await ctx.bot.send_message(chat_id, "❌ MJ: сервис недоступен. 💎 Токены возвращены.")
                return
            if flag == 0:
                if time.time() - start_ts > max_wait:
                    _refund("timeout")
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
                _refund("error", err)
                await ctx.bot.send_message(chat_id, f"❌ MJ: {err}\n💎 Токены возвращены.")
                return
            if flag == 1:
                payload = data or {}

                urls = _extract_mj_image_urls(payload)
                if not urls:
                    one_url = _extract_result_url(payload)
                    urls = [one_url] if one_url else []

                if not urls:
                    _refund("empty")
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
                    _refund("download_failed")
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
                    _refund("send_failed")
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
        _refund("exception", str(e))
        try: await ctx.bot.send_message(chat_id, "💥 Внутренняя ошибка MJ. 💎 Токены возвращены.")
        except Exception: pass
    finally:
        s = state(ctx)
        s["mj_generating"] = False
        s["last_mj_task_id"] = None
        s["mj_last_wait_ts"] = 0.0
        s["last_prompt"] = None
        mid = s.get("last_mj_msg_id")
        if mid:
            final_text = "✅ Midjourney: изображение обработано." if success else "ℹ️ Midjourney: поток завершён."
            try: await ctx.bot.edit_message_text(chat_id=chat_id, message_id=mid, text=final_text, reply_markup=None)
            except Exception: pass
            s["last_mj_msg_id"] = None
        if op_key:
            _clear_operation(ctx, op_key)
        s.pop("mj_active_op_key", None)

# ==========================
#   Handlers
# ==========================
def stars_topup_kb() -> InlineKeyboardMarkup:
    rows: List[List[InlineKeyboardButton]] = []
    packs = [
        (50, 50, 0),
        (100, 110, 10),
        (200, 220, 20),
        (300, 330, 30),
        (400, 440, 40),
        (500, 550, 50),
    ]
    for stars, tokens, bonus in packs:
        cap = f"⭐ {stars} → 💎 {tokens}" + (f" +{bonus}💎 бонус" if bonus else "")
        rows.append([InlineKeyboardButton(cap, callback_data=f"buy:stars:{stars}:{tokens}")])
    rows.append([InlineKeyboardButton("🛒 Где купить Stars", url=STARS_BUY_URL)])
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="back")])
    return InlineKeyboardMarkup(rows)

async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = state(ctx); s.update({**DEFAULT_STATE})
    uid = update.effective_user.id

    try:
        bonus_result = ledger_storage.grant_signup_bonus(uid, 10)
        _set_cached_balance(ctx, bonus_result.balance)
        if bonus_result.applied:
            await update.message.reply_text("🎁 Добро пожаловать! Начислил +10💎 на баланс.")
    except Exception as exc:
        log.exception("Signup bonus failed for %s: %s", uid, exc)

    await update.message.reply_text(
        render_welcome_for(uid, ctx),
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=main_menu_kb(),
    )

async def topup(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "💳 Пополнение через *Telegram Stars*.\nЕсли звёзд не хватает — купите в официальном боте:",
        parse_mode=ParseMode.MARKDOWN, reply_markup=stars_topup_kb()
    )


# codex/fix-balance-reset-after-deploy
async def balance_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    balance = get_user_balance_value(ctx, force_refresh=True)
    await update.message.reply_text(f"💎 Ваш баланс: {balance} 💎")


async def balance_recalc(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
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

async def error_handler(update: Optional[Update], context: ContextTypes.DEFAULT_TYPE):
    log.exception("Unhandled error: %s", context.error)
    try:
        if update and update.effective_chat:
            await context.bot.send_message(update.effective_chat.id, "⚠️ Системная ошибка. Попробуйте ещё раз.")
    except Exception:
        pass

async def show_or_update_banana_card(chat_id: int, ctx: ContextTypes.DEFAULT_TYPE):
    s = state(ctx)
    text = banana_card_text(s)
    kb = banana_kb()
    mid = s.get("last_ui_msg_id_banana")
    try:
        if mid:
            await ctx.bot.edit_message_text(chat_id=chat_id, message_id=mid, text=text,
                                            parse_mode=ParseMode.MARKDOWN, reply_markup=kb, disable_web_page_preview=True)
        else:
            m = await ctx.bot.send_message(chat_id, text, parse_mode=ParseMode.MARKDOWN, reply_markup=kb, disable_web_page_preview=True)
            s["last_ui_msg_id_banana"] = m.message_id
    except Exception as e:
        log.warning("banana card edit/send failed: %s", e)

async def show_or_update_veo_card(chat_id: int, ctx: ContextTypes.DEFAULT_TYPE):
    s = state(ctx)
    text = veo_card_text(s)
    kb = veo_kb(s)
    mid = s.get("last_ui_msg_id")
    try:
        if mid:
            await ctx.bot.edit_message_text(chat_id=chat_id, message_id=mid, text=text,
                                            parse_mode=ParseMode.MARKDOWN, reply_markup=kb, disable_web_page_preview=True)
        else:
            m = await ctx.bot.send_message(chat_id, text, parse_mode=ParseMode.MARKDOWN, reply_markup=kb, disable_web_page_preview=True)
            s["last_ui_msg_id"] = m.message_id
    except Exception as e:
        log.warning("veo card edit/send failed: %s", e)


async def set_veo_card_prompt(chat_id: int, prompt_text: str, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    s = state(ctx)
    s["last_prompt"] = prompt_text
    await show_or_update_veo_card(chat_id, ctx)


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
    await q.edit_message_text(
        "Промпт добавлен в карточку VEO. Можно запускать генерацию. ✨",
        reply_markup=main_menu_kb(),
    )


configure_prompt_master(update_veo_card=show_or_update_veo_card)

async def on_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
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
            await q.edit_message_text(
                "Режим переключён: теперь это обычный чат. Напишите сообщение.",
                reply_markup=main_menu_kb(),
            )
        return

    if data == CB_MODE_PM:
        if chat_id is not None:
            _mode_set(chat_id, MODE_PM)
        s["mode"] = None
        await q.answer("Режим: Prompt-Master")
        if message is not None:
            await q.edit_message_text(
                "Режим переключён: Prompt-Master. Пришлите идею/сцену — верну кинопромпт.",
                reply_markup=main_menu_kb(),
            )
        return

    if data.startswith(CB_PM_INSERT_VEO):
        await handle_pm_insert_to_veo(update, ctx, data)
        return

    if data in (PROMPT_MASTER_OPEN, PROMPT_MASTER_CANCEL):
        return

    await q.answer()

    if data == "promo_open":
        if not PROMO_ENABLED:
            await q.message.reply_text("🎟️ Промокоды временно отключены.")
            return
        s["mode"] = "promo"
        await q.message.reply_text("🎟️ Введите промокод одним сообщением:"); return

    if data == "faq":
        await q.message.reply_text(
            "📘 *FAQ*\n"
            "— *Как начать с VEO?*\n"
            "1) Выберите «Veo Fast» или «Veo Quality». 2) Пришлите идею текстом и/или фото. "
            "3) Карточка откроется автоматически — проверьте параметры и жмите «🚀 Сгенерировать».\n\n"
            "— *Fast vs Quality?* Fast — быстрее и дешевле. Quality — дольше, но лучше детализация. Оба: 16:9 и 9:16.\n\n"
            "— *Форматы VEO?* 16:9 и 9:16. Готовые клипы загружаем в чат как видеофайлы.\n\n"
            "— *MJ:* 16:9 или 9:16, цена 10💎. Один бесплатный перезапуск при сетевой ошибке. На выходе одно изображение.\n\n"
            "— *Banana:* до 4 фото, затем текст — что поменять (фон, одежда, макияж, удаление объектов, объединение людей).\n\n"
            "— *Время ожидания:* VEO 2–10 мин, MJ 1–3 мин, Banana 1–5 мин (может быть дольше при нагрузке).\n\n"
            "— *Токены/возвраты:* списываются при старте; при ошибке/таймауте бот автоматически возвращает 💎.\n\n"
            f"— *Пополнение:* через Stars в меню. Где купить: {STARS_BUY_URL}\n"
            "— *Примеры и идеи:* кнопка «Канал с промптами».",
            parse_mode=ParseMode.MARKDOWN, reply_markup=main_menu_kb()
        ); return

    if data == "back":
        s.update({**DEFAULT_STATE})
        await q.message.reply_text("🏠 Главное меню:", reply_markup=main_menu_kb()); return

    if data == "start_new_cycle":
        s.update({**DEFAULT_STATE})
        await q.message.reply_text("Выберите режим:", reply_markup=main_menu_kb()); return

    if data == "topup_open":
        await q.message.reply_text("💳 Выберите пакет Stars ниже:", reply_markup=stars_topup_kb()); return

    # Покупка
    if data.startswith("buy:stars:"):
        _, _, stars_str, tokens_str = data.split(":")
        stars = int(stars_str); tokens = int(tokens_str)
        title = f"{stars}⭐ → {tokens}💎"
        payload = json.dumps({"kind": "stars_pack", "stars": stars, "tokens": tokens})
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
            await show_or_update_veo_card(update.effective_chat.id, ctx)
            await q.message.reply_text("✍️ Пришлите текст идеи и/или фото-референс — карточка обновится автоматически.")
            return
        if selected_mode == "veo_photo":
            s["aspect"] = "9:16"; s["model"] = "veo3_fast"
            await show_or_update_veo_card(update.effective_chat.id, ctx)
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
            mid = s.get("last_mj_msg_id")
            if mid:
                try: await ctx.bot.delete_message(update.effective_chat.id, mid)
                except Exception: pass
            s["last_mj_msg_id"] = None
            await show_mj_format_card(update.effective_chat.id, ctx)
            return
        if selected_mode == "banana":
            s["banana_images"] = []; s["last_prompt"] = None
            await q.message.reply_text("🍌 Banana включён\nСначала пришлите до *4 фото* (можно по одному). Когда будут готовы — пришлите *текст-промпт*, что изменить.", parse_mode=ParseMode.MARKDOWN)
            await show_or_update_banana_card(update.effective_chat.id, ctx); return

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
            mid = s.get("last_mj_msg_id")
            if mid:
                try: await ctx.bot.edit_message_text(chat_id=chat_id, message_id=mid, text="❌ Midjourney отменён.", reply_markup=None)
                except Exception: pass
            s["last_mj_msg_id"] = None
            await q.message.reply_text("🏠 Главное меню:", reply_markup=main_menu_kb()); return

        if action == "confirm":
            if s.get("mj_generating"):
                await q.message.reply_text("⏳ Уже идёт генерация. Дождитесь результата."); return
            prompt = (s.get("last_prompt") or "").strip()
            if not prompt:
                await q.message.reply_text("❌ Промпт не найден, отправьте текст и повторите."); return
            price = TOKEN_COSTS['mj']
            aspect_value = "9:16" if s.get("aspect") == "9:16" else "16:9"
            fingerprint = _make_fingerprint({"prompt": prompt, "aspect": aspect_value})
            op_key = f"mj:{fingerprint}"
            op_id, _ = _ensure_operation(ctx, op_key)
            status, rest, _ = try_charge(
                ctx,
                price,
                "mj_charge",
                op_id,
                {"prompt": _short_prompt(prompt, 160), "aspect": aspect_value},
            )
            if status == "insufficient":
                _clear_operation(ctx, op_key)
                await q.message.reply_text(
                    f"💎 Недостаточно токенов: нужно {price}, на балансе {rest}.",
                    reply_markup=stars_topup_kb(),
                );
                return
            if status == "duplicate":
                await q.message.reply_text("⏳ Уже выполняю этот промпт. Дождитесь результата.")
                return
            await q.message.reply_text("✅ Промпт принят.")
            s["mj_generating"] = True
            s["mj_last_wait_ts"] = time.time()
            await show_mj_generating_card(chat_id, ctx, prompt, aspect_value)
            ok, task_id, msg = await asyncio.to_thread(mj_generate, prompt, aspect_value)
            event("MJ_SUBMIT_RESP", ok=ok, task_id=task_id, msg=msg)
            if not ok or not task_id:
                refund_id = f"refund:{op_id}:submit"
                try:
                    credit_tokens(
                        ctx,
                        price,
                        "mj_refund",
                        refund_id,
                        {"reason": "submit_failed", "message": msg},
                    )
                except Exception as exc:
                    log.exception("MJ submit refund failed for %s: %s", update.effective_user.id, exc)
                _clear_operation(ctx, op_key)
                s["mj_generating"] = False
                s["last_mj_task_id"] = None
                s["mj_last_wait_ts"] = 0.0
                await q.message.reply_text(f"❌ Не удалось создать MJ-задачу: {msg}\n💎 Токены возвращены.")
                await show_mj_prompt_card(chat_id, ctx)
                return
            final_op_id = f"gen:{task_id}"
            if not rename_operation(op_id, final_op_id, {"task_id": task_id}):
                log.warning("Failed to rename MJ op %s -> %s", op_id, final_op_id)
            _update_operation(ctx, op_key, op_id=final_op_id, task_id=task_id, price=price)
            s["mj_active_op_key"] = op_key
            s["last_mj_task_id"] = task_id
            asyncio.create_task(poll_mj_and_send_photos(chat_id, task_id, ctx, prompt, aspect_value))
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
            await q.message.reply_text("🧹 Фото очищены."); await show_or_update_banana_card(update.effective_chat.id, ctx); return
        if act == "edit_prompt":
            await q.message.reply_text("✍️ Пришлите новый промпт для Banana."); return
        if act == "start":
            imgs = s.get("banana_images") or []
            prompt = (s.get("last_prompt") or "").strip()
            if not imgs:   await q.message.reply_text("⚠️ Сначала добавьте хотя бы одно фото."); return
            if not prompt: await q.message.reply_text("⚠️ Добавьте текст-промпт (что изменить)."); return
            price = TOKEN_COSTS['banana']
            fingerprint = _make_fingerprint({"prompt": prompt, "images": imgs})
            op_key = f"banana:{fingerprint}"
            op_id, _ = _ensure_operation(ctx, op_key)
            status, rest, _ = try_charge(
                ctx,
                price,
                "banana_charge",
                op_id,
                {"prompt": _short_prompt(prompt, 160), "images": len(imgs)},
            )
            if status == "insufficient":
                _clear_operation(ctx, op_key)
                await q.message.reply_text(
                    f"💎 Недостаточно токенов: нужно {price}, на балансе {rest}.",
                    reply_markup=stars_topup_kb(),
                );
                return
            if status == "duplicate":
                await q.message.reply_text("⏳ Уже обрабатываю предыдущий запрос Banana. Дождитесь результата.")
                return
            await q.message.reply_text("🍌 Запускаю Banana…")
            s["banana_active_op_key"] = op_key
            asyncio.create_task(_banana_run_and_send(update.effective_chat.id, ctx, imgs, prompt, op_key, op_id, price)); return

    # -------- VEO card actions --------
    if data.startswith("veo:set_ar:"):
        s["aspect"] = "9:16" if data.endswith("9:16") else "16:9"
        await show_or_update_veo_card(update.effective_chat.id, ctx); return
    if data.startswith("veo:set_model:"):
        s["model"] = "veo3_fast" if data.endswith("fast") else "veo3"
        await show_or_update_veo_card(update.effective_chat.id, ctx); return
    if data == "veo:clear_img":
        s["last_image_url"] = None
        await show_or_update_veo_card(update.effective_chat.id, ctx); return
    if data == "veo:start":
        prompt = (s.get("last_prompt") or "").strip()
        if not prompt:
            await q.message.reply_text("⚠️ Сначала пришлите текстовый промпт."); return
        user = update.effective_user
        user_id_for_lock = user.id if user else None
        if not _acquire_click_lock(user_id_for_lock, "veo:start"):
            return
        price = TOKEN_COSTS['veo_quality'] if s.get('model') == 'veo3' else TOKEN_COSTS['veo_fast']
        chat_id = update.effective_chat.id
        if ACTIVE_TASKS.get(chat_id):
            await q.message.reply_text("⏳ Уже рендерю вашу предыдущую задачу. Подождите, пожалуйста.")
            return
        uid = user.id if user else 0
        fingerprint = _make_fingerprint({
            "prompt": prompt,
            "image": s.get("last_image_url"),
            "aspect": s.get("aspect"),
            "model": s.get("model"),
        })
        op_key = f"veo:{fingerprint}"
        op_id, _ = _ensure_operation(ctx, op_key)
        meta = {
            "prompt": _short_prompt(prompt, 160),
            "aspect": s.get("aspect") or "16:9",
            "model": s.get("model") or "veo3_fast",
        }
        status, rest, _ = try_charge(ctx, price, "veo_charge", op_id, meta)
        if status == "insufficient":
            _clear_operation(ctx, op_key)
            await q.message.reply_text(
                f"💎 Недостаточно токенов: нужно {price}, на балансе {rest}.",
                reply_markup=stars_topup_kb(),
            );
            return
        if status == "duplicate":
            await q.message.reply_text("⏳ Уже выполняю предыдущий запрос. Дождитесь результата.")
            return
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
            refund_id = f"refund:{op_id}:submit_exc"
            try:
                credit_tokens(
                    ctx,
                    price,
                    "veo_refund",
                    refund_id,
                    {"reason": "submit_exception", "message": str(exc)},
                )
            except Exception as refund_exc:
                log.exception("VEO submit crash refund failed for %s: %s", uid, refund_exc)
            _clear_operation(ctx, op_key)
            ACTIVE_TASKS.pop(chat_id, None)
            await q.message.reply_text("⚠️ Не удалось создать VEO-задачу. 💎 Токены возвращены.")
            return
        if not ok or not task_id:
            refund_id = f"refund:{op_id}:submit"
            try:
                credit_tokens(
                    ctx,
                    price,
                    "veo_refund",
                    refund_id,
                    {"reason": "submit_failed", "message": msg},
                )
            except Exception as exc:
                log.exception("VEO submit refund failed for %s: %s", uid, exc)
            _clear_operation(ctx, op_key)
            ACTIVE_TASKS.pop(chat_id, None)
            await q.message.reply_text(
                f"❌ Не удалось создать VEO-задачу: {msg}\n💎 Токены возвращены.",
            )
            return
        final_op_id = f"gen:{task_id}"
        if not rename_operation(op_id, final_op_id, {"task_id": task_id}):
            log.warning("Failed to rename ledger op %s -> %s", op_id, final_op_id)
        _update_operation(ctx, op_key, op_id=final_op_id, task_id=task_id, price=price)
        s["active_generation_op"] = op_key
        gen_id = uuid.uuid4().hex
        s["generating"] = True; s["generation_id"] = gen_id; s["last_task_id"] = task_id
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
        asyncio.create_task(poll_veo_and_send(update.effective_chat.id, task_id, gen_id, ctx)); return

async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if msg is None:
        return

    s = state(ctx)
    text = (msg.text or "").strip()
    chat_id = msg.chat_id
    state_mode = s.get("mode")
    user_mode = _mode_get(chat_id) or MODE_CHAT

    if state_mode == "promo":
        if not PROMO_ENABLED:
            await msg.reply_text("🎟️ Промокоды временно отключены.")
            s["mode"] = None
            return
        code = text.upper()
        uid = update.effective_user.id
        bonus = promo_amount(code)
        if not bonus:
            await msg.reply_text("❌ Неверный промокод.")
            s["mode"] = None
            return
        used_by = promo_used_global(code)
        if used_by and used_by != uid:
            await msg.reply_text("❌ Этот промокод уже использован.")
            s["mode"] = None
            return
        try:
            result = ledger_storage.apply_promo(
                uid,
                code,
                bonus,
                {"source": "promo_command"},
            )
            _set_cached_balance(ctx, result.balance)
            if not result.applied:
                if used_by == uid or result.duplicate:
                    await msg.reply_text("❌ Этот промокод уже использован.")
                else:
                    await msg.reply_text("⚠️ Промокод сейчас недоступен. Попробуйте другой.")
                s["mode"] = None
                return
        except Exception as exc:
            log.exception("Promo apply failed for %s (%s): %s", uid, code, exc)
            await msg.reply_text("⚠️ Не удалось применить промокод. Попробуйте позже.")
            s["mode"] = None
            return

        promo_mark_used(code, uid)
        get_user_balance_value(ctx, force_refresh=True)
        await msg.reply_text(
            f"✅ Промокод активирован! Вам начислено {bonus} токенов."
        )
        s["mode"] = None
        return

    if user_mode == MODE_PM:
        if not text:
            await msg.reply_text("⚠️ Пришлите идею или сцену для Prompt-Master.")
            return
        from prompt_master import build_cinema_prompt

        kino_prompt, _ = await build_cinema_prompt(text, user_lang=detect_lang(text))
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("🎬 Вставить в VEO", callback_data=f"{CB_PM_INSERT_VEO}")],
            [InlineKeyboardButton("⬅️ Назад", callback_data=CB_MODE_CHAT)],
        ])
        await msg.reply_text(
            f"🧠 Готово! Вот ваш кинопромпт:\n\n{kino_prompt}",
            reply_markup=kb,
        )
        cache_pm_prompt(chat_id, kino_prompt)
        return

    low = text.lower()
    if low.startswith(("http://", "https://")) and any(
        low.split("?")[0].endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp", ".heic")
    ):
        if state_mode == "banana":
            if len(s["banana_images"]) >= 4:
                await msg.reply_text("⚠️ Достигнут лимит 4 фото.", reply_markup=banana_kb())
                return
            s["banana_images"].append(text.strip())
            await msg.reply_text(f"📸 Фото принято ({len(s['banana_images'])}/4).")
            await show_or_update_banana_card(chat_id, ctx)
            return
        s["last_image_url"] = text.strip()
        await msg.reply_text("🧷 Ссылка на изображение принята.")
        if state_mode in ("veo_text_fast", "veo_text_quality", "veo_photo"):
            await show_or_update_veo_card(chat_id, ctx)
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
        s["last_prompt"] = text
        await msg.reply_text("✍️ Промпт сохранён.")
        await show_or_update_banana_card(chat_id, ctx)
        return

    if state_mode in ("veo_text_fast", "veo_text_quality", "veo_photo"):
        s["last_prompt"] = text
        await show_or_update_veo_card(chat_id, ctx)
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
        await msg.reply_text(answer, reply_markup=main_menu_kb())
        return

    # Если режим не распознан — по умолчанию обновляем карточку VEO
    s["last_prompt"] = text
    await show_or_update_veo_card(chat_id, ctx)

async def _banana_run_and_send(
    chat_id: int,
    ctx: ContextTypes.DEFAULT_TYPE,
    src_urls: List[str],
    prompt: str,
    op_key: str,
    op_id: str,
    price: int,
) -> None:
    s = state(ctx)

    def _refund(reason_tag: str, message: Optional[str] = None) -> None:
        meta: Dict[str, Any] = {"reason": reason_tag}
        if message:
            meta["message"] = message
        refund_op_id = f"refund:{op_id}:{reason_tag}"
        try:
            credit_tokens(ctx, price, "banana_refund", refund_op_id, meta)
        except Exception as exc:
            log.exception("Banana refund %s failed: %s", reason_tag, exc)

    try:
        task_id = await asyncio.to_thread(create_banana_task, prompt, src_urls, "png", "auto", None, None, 60)
        final_op_id = f"gen:{task_id}"
        if not rename_operation(op_id, final_op_id, {"task_id": task_id}):
            log.warning("Failed to rename Banana op %s -> %s", op_id, final_op_id)
        _update_operation(ctx, op_key, op_id=final_op_id, task_id=task_id, price=price)
        s["banana_active_op_key"] = op_key
        await ctx.bot.send_message(chat_id, f"🍌 Задача Banana создана.\n🆔 taskId={task_id}\nЖдём результат…")
        urls = await asyncio.to_thread(wait_for_banana_result, task_id, 8 * 60, 3)
        if not urls:
            _refund("empty")
            await ctx.bot.send_message(chat_id, "⚠️ Banana вернула пустой результат. 💎 Токены возвращены."); return
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
        _refund("error", str(e))
        await ctx.bot.send_message(chat_id, f"❌ Banana ошибка: {e}\n💎 Токены возвращены.")
    except Exception as e:
        _refund("exception", str(e))
        log.exception("BANANA unexpected: %s", e)
        await ctx.bot.send_message(chat_id, "💥 Внутренняя ошибка Banana. 💎 Токены возвращены.")
    finally:
        _clear_operation(ctx, op_key)
        s.pop("banana_active_op_key", None)

async def on_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
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
            s["banana_images"].append(url)
            cap = (update.message.caption or "").strip()
            if cap: s["last_prompt"] = cap
            await update.message.reply_text(f"📸 Фото принято ({len(s['banana_images'])}/4).")
            await show_or_update_banana_card(update.effective_chat.id, ctx); return
        s["last_image_url"] = url
        await update.message.reply_text("🖼️ Фото принято как референс.")
        if s.get("mode") in ("veo_text_fast","veo_text_quality","veo_photo"):
            await show_or_update_veo_card(update.effective_chat.id, ctx)
    except Exception as e:
        log.exception("Get photo failed: %s", e)
        await update.message.reply_text("⚠️ Не удалось обработать фото. Пришлите публичный URL картинки текстом.")

# ---------- Payments: Stars (XTR) ----------
async def precheckout_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try: await update.pre_checkout_query.answer(ok=True)
    except Exception:
        await update.pre_checkout_query.answer(ok=False, error_message=f"Платёж отклонён. Пополните Stars в {STARS_BUY_URL}")

async def successful_payment_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    sp = update.message.successful_payment
    try: meta = json.loads(sp.invoice_payload)
    except Exception: meta = {}
    stars = int(sp.total_amount)
    if meta.get("kind") == "stars_pack":
        tokens = int(meta.get("tokens", 0))
        if tokens <= 0:
            mapping = {50:50,100:110,200:220,300:330,400:440,500:550}
            tokens = mapping.get(stars, 0)
        if tokens > 0:
            charge_id = sp.telegram_payment_charge_id or sp.provider_payment_charge_id or uuid.uuid4().hex
            payment_meta = {
                "source": "telegram_stars",
                "stars": stars,
                "payload": meta,
            }
            try:
                result = credit_tokens(ctx, tokens, "topup_stars", f"payment:{charge_id}", payment_meta)
            except Exception as exc:
                log.exception("Top-up credit failed for %s: %s", charge_id, exc)
                await update.message.reply_text("⚠️ Платёж получен, но обновить баланс не удалось. Свяжитесь с поддержкой.")
                return
            balance = result.balance
            msg = "✅ Оплата учтена повторно." if not result.applied else f"✅ Оплата получена: +{tokens} токенов."
            await update.message.reply_text(f"{msg}\nБаланс: {balance} 💎")
            return
        return
    await update.message.reply_text("✅ Оплата получена.")

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
    application.add_handler(CommandHandler("health", health))
    application.add_handler(CommandHandler("topup", topup))
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
