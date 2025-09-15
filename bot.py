# -*- coding: utf-8 -*-
# Best VEO3 Bot — PTB 21.x
# Версия: 2025-09-14r4
# Единственное изменение против прежней версии: надежная доставка VEO-видео в Telegram
# (освежение ссылки + повторная попытка + download&reupload с увеличенным таймаутом).
# Остальное (карточки, кнопки, тексты, цены, FAQ, промокоды, бонусы и т.д.) — без изменений.

import os, json, time, uuid, asyncio, logging, tempfile, subprocess, re, signal, socket, hashlib
from typing import Dict, Any, Optional, List, Tuple, Callable
from datetime import datetime, timezone
from contextlib import suppress

import requests
from dotenv import load_dotenv

from telegram import (
    Update, InlineKeyboardButton, InlineKeyboardMarkup,
    InputFile, LabeledPrice
)
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler,
    CallbackQueryHandler, filters, AIORateLimiter, PreCheckoutQueryHandler
)

from handlers.prompt_master_handler import (
    PROMPT_MASTER_BODY,
    PROMPT_MASTER_HEADER,
    prompt_master_conv,
)

# === KIE Banana wrapper ===
from kie_banana import create_banana_task, wait_for_banana_result, KieBananaError

import redis

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

LOG_LEVEL = _env("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
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

DATABASE_URL = _env("DATABASE_URL") or _env("POSTGRES_DSN")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL (or POSTGRES_DSN) must be set for persistent ledger storage")

def _rk(*parts: str) -> str: return ":".join([REDIS_PREFIX, *parts])

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

# Ledger storage (Postgres)
ledger_storage = LedgerStorage(DATABASE_URL)


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
    code = payload.get("code", resp.status_code)
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
            for key in ("resultUrls", "resultUrl", "originUrls", "originUrl", "videoUrls", "videoUrl",
                        "videos", "urls", "url", "downloadUrl", "fileUrl", "cdnUrl", "outputUrl"):
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

def event(tag: str, **kw):
    try: log.info("EVT %s | %s", tag, json.dumps(kw, ensure_ascii=False))
    except Exception: log.info("EVT %s | %s", tag, kw)

def kie_event(stage: str, **kw):
    event(f"KIE_{stage}", **kw)

def tg_direct_file_url(bot_token: str, file_path: str) -> str:
    p = (file_path or "").strip()
    if p.startswith("http://") or p.startswith("https://"): return p
    return f"https://api.telegram.org/file/bot{bot_token}/{p.lstrip('/')}"

# ---------- User state ----------
DEFAULT_STATE = {
    "mode": None, "aspect": None, "model": None,
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

# ==========================
#   UI / Texts
# ==========================
WELCOME = (
    "🎬 *Veo 3 — съёмочная команда*: опиши идею и получи *готовый клип*.\n"
    "🖌️ *MJ — художник*: рисует изображение по тексту (16:9 или 9:16).\n"
    "🍌 *Banana — редактор из будущего*: меняет фон, одежду, макияж, убирает лишнее, объединяет людей.\n"
    "🧠 *Prompt-Master (/promptmaster)* — вернёт профессиональный *кинопромпт*.\n"
    "💬 *Обычный чат* — ответы на любые вопросы.\n\n"
    "💎 *Ваш баланс:* {balance}\n"
    "📈 Больше идей и примеров: {prompts_url}\n\n"
    "Выберите режим 👇"
)

def render_welcome_for(uid: int, ctx: ContextTypes.DEFAULT_TYPE) -> str:
    return WELCOME.format(balance=get_user_balance_value(ctx), prompts_url=PROMPTS_CHANNEL_URL)

def main_menu_kb() -> InlineKeyboardMarkup:
    keyboard = [
        [InlineKeyboardButton(f"🎬 Генерация видео (Veo Fast) 💎 {TOKEN_COSTS['veo_fast']}", callback_data="mode:veo_text_fast")],
        [InlineKeyboardButton(f"🎬 Генерация видео (Veo Quality) 💎 {TOKEN_COSTS['veo_quality']}", callback_data="mode:veo_text_quality")],
        [InlineKeyboardButton(f"🖼️ Генерация изображений (MJ) 💎 {TOKEN_COSTS['mj']}", callback_data="mode:mj_txt")],
        [InlineKeyboardButton(f"🍌 Редактор изображений (Banana) 💎 {TOKEN_COSTS['banana']}", callback_data="mode:banana")],
        [InlineKeyboardButton(f"📸 Оживить изображение (Veo) 💎 {TOKEN_COSTS['veo_photo']}", callback_data="mode:veo_photo")],
        [InlineKeyboardButton("🧠 Prompt-Master (/promptmaster)", callback_data="mode:prompt_master")],
        [InlineKeyboardButton("💬 Обычный чат (ChatGPT)", callback_data="mode:chat")],
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
        [InlineKeyboardButton("⬅️ Назад", callback_data="back")],
    ]
    return InlineKeyboardMarkup(keyboard)

def _mj_prompt_card_text(aspect: str, prompt: Optional[str]) -> str:
    aspect = "9:16" if aspect == "9:16" else "16:9"
    lines = [
        "Введите промпт сообщением. После этого нажмите «Подтвердить».",
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
    status, resp, req_id = _kie_request("POST", KIE_VEO_GEN_PATH, json_payload=payload)
    code = resp.get("code", status)
    tid = _extract_task_id(resp)
    message = resp.get("msg") or resp.get("message")
    kie_event("SUBMIT", request_id=req_id, status=status, code=code, task_id=tid, message=message)
    if status == 200 and code == 200:
        if tid:
            _remember_kie_request_id(tid, req_id)
            return True, tid, "Задача создана."
        return False, None, "Ответ KIE без taskId."
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
    code = resp.get("code", status)
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

def try_get_1080_url(task_id: str, attempts: int = 3, per_try_timeout: int = 15) -> Optional[str]:
    last_err: Optional[str] = None
    req_id = _get_kie_request_id(task_id)
    for attempt in range(1, attempts + 1):
        status, resp, req_id_used, path_used = _kie_request_with_endpoint(
            "veo",
            "1080",
            "GET",
            KIE_VEO_1080_PATHS,
            params={"taskId": task_id},
            timeout=per_try_timeout,
            request_id=req_id,
        )
        if not req_id:
            req_id = req_id_used
            _remember_kie_request_id(task_id, req_id)
        code = resp.get("code", status)
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
        url = data.get("url") if isinstance(data.get("url"), str) else None
        if status == 200 and code == 200:
            if not (isinstance(url, str) and url.startswith("http")):
                url = _extract_result_url(data)
            if isinstance(url, str) and url.startswith("http"):
                kie_event(
                    "FETCH_1080_SUCCESS",
                    request_id=req_id,
                    task_id=task_id,
                    attempt=attempt,
                    path=path_used,
                )
                return url
            last_err = "empty_url"
            kie_event(
                "FETCH_1080_EMPTY",
                request_id=req_id,
                task_id=task_id,
                attempt=attempt,
                path=path_used,
            )
        else:
            last_err = f"{status}/{code}"
            message = resp.get("msg") or resp.get("message") or resp.get("error")
            kie_event(
                "FETCH_1080_FAIL",
                request_id=req_id,
                task_id=task_id,
                attempt=attempt,
                status=status,
                code=code,
                message=message,
                path=path_used,
            )
        kie_event(
            "1080_RETRY",
            task_id=task_id,
            attempt=attempt,
            status=status,
            code=code,
            path=path_used,
        )
        time.sleep(attempt)
    log.warning("1080p retries failed: %s", last_err)
    kie_event("FETCH_1080_GIVEUP", request_id=req_id, task_id=task_id, error=last_err)
    return None

# ==========================
#   MJ
# ==========================
def _kie_error_message(status_code: int, j: Dict[str, Any]) -> str:
    code = j.get("code", status_code)
    msg = j.get("msg") or j.get("message") or j.get("error") or ""
    mapping = {401: "Доступ запрещён.", 402: "Недостаточно кредитов.",
               429: "Превышен лимит.", 500: "Внутренняя ошибка KIE.",
               422: "Запрос отклонён модерацией.", 400: "Неверный запрос."}
    base = mapping.get(code, f"KIE code {code}.")
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
    code = resp.get("code", status)
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
    code = resp.get("code", status)
    raw_data = resp.get("data")
    if isinstance(raw_data, str):
        try:
            parsed = json.loads(raw_data)
            data = parsed if isinstance(parsed, dict) else {"value": parsed}
        except Exception:
            data = {"raw": raw_data}
    elif isinstance(raw_data, dict):
        data = raw_data
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
    rj = status_data.get("resultInfoJson") or {}
    if isinstance(rj, str):
        try:
            rj = json.loads(rj)
        except Exception:
            rj = {}
    urls = _coerce_url_list((rj or {}).get("resultUrls"))
    for u in urls:
        if isinstance(u, str) and u.startswith("http"): res.append(u)
    return res

def _mj_should_retry(msg: Optional[str]) -> bool:
    if not msg: return False
    m = msg.lower()
    return ("no response from midjourney official website" in m) or ("timeout" in m) or ("server error" in m)

# ==========================
#   ffmpeg helpers (видео)
# ==========================
def _ffmpeg_available() -> bool:
    from shutil import which
    return bool(which(FFMPEG_BIN))

def _ffmpeg_normalize_vertical(inp: str, outp: str) -> bool:
    cmd = [
        FFMPEG_BIN, "-y", "-i", inp,
        "-vf", "scale=1080:1920:force_original_aspect_ratio=decrease,"
               "pad=1080:1920:(ow-iw)/2:(oh-ih)/2,format=yuv420p",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-c:a", "aac", "-b:a", "128k",
        "-metadata:s:v:0", "rotate=0",
        outp
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE); return True
    except Exception as e:
        log.warning("ffmpeg vertical failed: %s", e); return False

def _ffmpeg_force_16x9_fhd(inp: str, outp: str, target_mb: int) -> bool:
    target_bytes = max(8, int(target_mb)) * 1024 * 1024
    cmd = [
        FFMPEG_BIN, "-y", "-i", inp,
        "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,"
               "pad=1920:1080:(ow-iw)/2:(oh-ih)/2,format=yuv420p",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-c:a", "aac", "-b:a", "128k",
        "-fs", str(target_bytes),
        outp
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE); return True
    except Exception as e:
        log.warning("ffmpeg 16x9 FHD failed: %s", e); return False

# ==========================
#   Sending video (FIXED)
# ==========================
async def send_video_with_fallback(ctx: ContextTypes.DEFAULT_TYPE, chat_id: int, url: str,
                                   expect_vertical: bool = False, task_id: Optional[str] = None) -> bool:
    """
    Надёжная отправка VEO-видео:
    1) пробуем прямой URL (если не вертикаль);
    2) если не вышло — освежаем ссылку у KIE (1080p/record-info) и пробуем ещё раз;
    3) скачиваем и перезаливаем (таймаут 300с). Остальной UX не меняется.
    """
    event("SEND_TRY_URL", url=url, expect_vertical=expect_vertical)

    # 1) прямая отправка
    if not expect_vertical:
        try:
            await ctx.bot.send_video(chat_id=chat_id, video=url, supports_streaming=True)
            return True
        except Exception as e:
            event("SEND_FAIL_DIRECT", err=str(e))

    # 2) освежим ссылку и попробуем снова
    refreshed = None
    try:
        if task_id:
            u1080 = await asyncio.to_thread(try_get_1080_url, task_id)
            if isinstance(u1080, str) and u1080.startswith("http"):
                refreshed = u1080
            else:
                ok2, _, _, u2 = await asyncio.to_thread(get_kie_veo_status, task_id)
                if ok2 and isinstance(u2, str) and u2.startswith("http"):
                    refreshed = u2
    except Exception as e:
        event("SEND_REFRESH_ERR", err=str(e))

    if refreshed:
        event("SEND_TRY_REFRESHED", url=refreshed)
        if not expect_vertical:
            try:
                await ctx.bot.send_video(chat_id=chat_id, video=refreshed, supports_streaming=True)
                return True
            except Exception as e:
                event("SEND_FAIL_REFRESHED_DIRECT", err=str(e))
        url = refreshed  # перейдём к скачиванию с обновлённой ссылки

    # 3) download & reupload
    tmp_path = None
    try:
        r = requests.get(url, stream=True, timeout=300)  # увеличили таймаут
        r.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
            for c in r.iter_content(256 * 1024):
                if c: f.write(c)
            tmp_path = f.name

        if expect_vertical and ENABLE_VERTICAL_NORMALIZE and _ffmpeg_available():
            out = tmp_path + "_v.mp4"
            if _ffmpeg_normalize_vertical(tmp_path, out):
                with open(out, "rb") as f:
                    await ctx.bot.send_video(chat_id, InputFile(f, filename="result_vertical.mp4"), supports_streaming=True)
                return True

        if (not expect_vertical) and ALWAYS_FORCE_FHD and _ffmpeg_available():
            out = tmp_path + "_1080.mp4"
            if _ffmpeg_force_16x9_fhd(tmp_path, out, MAX_TG_VIDEO_MB):
                with open(out, "rb") as f:
                    await ctx.bot.send_video(chat_id, InputFile(f, filename="result_1080p.mp4"), supports_streaming=True)
                return True

        with open(tmp_path, "rb") as f:
            await ctx.bot.send_video(chat_id, InputFile(f, filename="result.mp4"), supports_streaming=True)
        return True
    except Exception as e:
        log.exception("send_video failed: %s", e)
        try:
            await ctx.bot.send_message(chat_id, f"🔗 Результат готов, но загрузка в Telegram не удалась. Ссылка:\n{url}")
            return True
        except Exception:
            return False
    finally:
        if tmp_path:
            try: os.unlink(tmp_path)
            except Exception: pass

# ==========================
#   VEO polling
# ==========================
async def poll_veo_and_send(chat_id: int, task_id: str, gen_id: str, ctx: ContextTypes.DEFAULT_TYPE):
    s = state(ctx)
    start_ts = time.time()
    price = TOKEN_COSTS['veo_quality'] if s.get('model') == 'veo3' else TOKEN_COSTS['veo_fast']
    op_key = s.get("active_generation_op")

    def _refund(reason_tag: str, message: Optional[str] = None) -> None:
        meta: Dict[str, Any] = {"reason": reason_tag}
        if message:
            meta["message"] = message
        refund_op_id = f"refund:{task_id}:{reason_tag}"
        try:
            credit_tokens(ctx, price, "veo_refund", refund_op_id, meta)
        except Exception as exc:
            log.exception("VEO refund %s failed: %s", reason_tag, exc)

    try:
        while True:
            if s.get("generation_id") != gen_id: return
            ok, flag, msg, res_url = await asyncio.to_thread(get_kie_veo_status, task_id)
            if not ok:
                _refund("status_error", msg)
                await ctx.bot.send_message(chat_id, f"❌ Ошибка статуса VEO. 💎 Токены возвращены.\n{msg or ''}")
                break
            if isinstance(res_url, str) and res_url.startswith("http"):
                # 🔄 освежаем ссылку непосредственно перед отправкой
                final_url = res_url
                if (s.get("aspect") or "16:9") == "16:9":
                    u1080 = await asyncio.to_thread(try_get_1080_url, task_id)
                    if isinstance(u1080, str) and u1080.startswith("http"):
                        final_url = u1080
                else:
                    ok_r2, _, _, u2 = await asyncio.to_thread(get_kie_veo_status, task_id)
                    if ok_r2 and isinstance(u2, str) and u2.startswith("http"):
                        final_url = u2

                kie_event(
                    "FINAL_URL",
                    request_id=_get_kie_request_id(task_id),
                    task_id=task_id,
                    final_url=final_url,
                )
                await ctx.bot.send_message(chat_id, "🎞️ Рендер завершён — отправляю файл…")
                await send_video_with_fallback(ctx, chat_id, final_url,
                                               expect_vertical=(s.get("aspect") == "9:16"),
                                               task_id=task_id)
                await ctx.bot.send_message(chat_id, "✅ *Готово!*", parse_mode=ParseMode.MARKDOWN,
                    reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🚀 Сгенерировать ещё видео", callback_data="start_new_cycle")]]))
                break
            if flag in (2, 3):
                _refund("no_url", msg)
                await ctx.bot.send_message(chat_id, f"❌ KIE не вернул ссылку на видео. 💎 Токены возвращены.\n{msg or ''}")
                break
            if (time.time() - start_ts) > POLL_TIMEOUT_SECS:
                _refund("timeout")
                await ctx.bot.send_message(chat_id, "⌛ Превышено время ожидания VEO. 💎 Токены возвращены.")
                break
            await asyncio.sleep(POLL_INTERVAL_SECS)
    except Exception as e:
        log.exception("VEO poll crash: %s", e)
        _refund("exception", str(e))
        try: await ctx.bot.send_message(chat_id, "💥 Внутренняя ошибка при опросе VEO. 💎 Токены возвращены.")
        except Exception: pass
    finally:
        if s.get("generation_id") == gen_id:
            s["generating"] = False
            s["generation_id"] = None
        if op_key:
            _clear_operation(ctx, op_key)
        s.pop("active_generation_op", None)
        _clear_kie_request_id(task_id)

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
                err = (data or {}).get("errorMessage") or "No response from MidJourney Official Website after multiple attempts."
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
                url = _extract_result_url(payload)
                if not url:
                    urls = _extract_mj_image_urls(payload)
                    url = urls[0] if urls else None
                if not url:
                    _refund("empty")
                    await ctx.bot.send_message(chat_id, "⚠️ MJ вернул пустой результат. 💎 Токены возвращены.")
                    return
                base_prompt = re.sub(r"\s+", " ", prompt_for_retry).strip()
                snippet = base_prompt[:100] if base_prompt else ""
                if not snippet:
                    snippet = "—"
                caption = "\n".join([
                    "🖼 Midjourney",
                    f"• Формат: {aspect_ratio}",
                    f'• Промпт: "{snippet}"',
                ])
                keyboard = InlineKeyboardMarkup([
                    [InlineKeyboardButton("Открыть", url=url)],
                    [InlineKeyboardButton("Повторить", callback_data="mj:repeat")],
                    [InlineKeyboardButton("Назад в меню", callback_data="back")],
                ])
                try:
                    await ctx.bot.send_photo(chat_id=chat_id, photo=url, caption=caption, reply_markup=keyboard)
                except Exception as e:
                    log.warning("MJ send_photo failed: %s", e)
                    try:
                        await ctx.bot.send_message(chat_id, caption + f"\n{url}", reply_markup=keyboard)
                    except Exception as e2:
                        log.warning("MJ send_message fallback failed: %s", e2)
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

async def on_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; data = (q.data or "").strip()
    await q.answer()
    s = state(ctx)

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
            "— *Форматы VEO?* 16:9 и 9:16. Для 16:9 стараемся получить 1080p; вертикаль нормализуется локально для Telegram.\n\n"
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
        mode = data.split(":",1)[1]
        if mode == "prompt_master":
            s["mode"] = None
            await q.message.reply_text(
                f"{PROMPT_MASTER_HEADER} 2.0 запускается командой /promptmaster.\n\n{PROMPT_MASTER_BODY}"
            )
            return
        s["mode"] = mode
        if mode in ("veo_text_fast","veo_text_quality"):
            s["aspect"] = "16:9"; s["model"] = "veo3_fast" if mode.endswith("fast") else "veo3"
            await show_or_update_veo_card(update.effective_chat.id, ctx)
            await q.message.reply_text("✍️ Пришлите текст идеи и/или фото-референс — карточка обновится автоматически.")
            return
        if mode == "veo_photo":
            s["aspect"] = "9:16"; s["model"] = "veo3_fast"
            await show_or_update_veo_card(update.effective_chat.id, ctx)
            await q.message.reply_text("📸 Пришлите фото (подпись-промпт — по желанию). Карточка обновится автоматически.")
            return
        if mode == "chat":
            await q.message.reply_text("💬 Чат активен. Напишите сообщение."); return
        if mode == "mj_txt":
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
        if mode == "banana":
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
        price = TOKEN_COSTS['veo_quality'] if s.get('model') == 'veo3' else TOKEN_COSTS['veo_fast']
        uid = update.effective_user.id
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
        await q.message.reply_text("🎬 Отправляю задачу в VEO…")
        ok, task_id, msg = await asyncio.to_thread(
            submit_kie_veo,
            prompt,
            (s.get("aspect") or "16:9"),
            s.get("last_image_url"),
            s.get("model") or "veo3_fast",
        )
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
        await q.message.reply_text(f"🆔 VEO taskId: `{task_id}`\n🎞 Рендер начат — вернусь с готовым видео.", parse_mode=ParseMode.MARKDOWN)
        asyncio.create_task(poll_veo_and_send(update.effective_chat.id, task_id, gen_id, ctx)); return

async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = state(ctx)
    text = (update.message.text or "").strip()
    mode = s.get("mode")

    # PROMO
    if mode == "promo":
        if not PROMO_ENABLED:
            await update.message.reply_text("🎟️ Промокоды временно отключены.")
            s["mode"] = None
            return
        code = text.upper()
        uid = update.effective_user.id
        bonus = promo_amount(code)
        if not bonus:
            await update.message.reply_text("❌ Неверный промокод.")
            s["mode"] = None
            return
        used_by = promo_used_global(code)
        if used_by and used_by != uid:
            await update.message.reply_text("⛔ Этот промокод уже был активирован другим пользователем.")
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
                    await update.message.reply_text("ℹ️ Вы уже активировали этот промокод ранее.")
                else:
                    await update.message.reply_text("⚠️ Промокод сейчас недоступен. Попробуйте другой.")
                s["mode"] = None
                return
        except Exception as exc:
            log.exception("Promo apply failed for %s (%s): %s", uid, code, exc)
            await update.message.reply_text("⚠️ Не удалось применить промокод. Попробуйте позже.")
            s["mode"] = None
            return

        promo_mark_used(code, uid)
        balance = get_user_balance_value(ctx, force_refresh=True)
        await update.message.reply_text(
            f"✅ Промокод принят! +{bonus}💎\nБаланс: {balance} 💎"
        )
        s["mode"] = None
        return

    # Ссылка на картинку как текст
    low = text.lower()
    if low.startswith(("http://", "https://")) and any(low.split("?")[0].endswith(ext) for ext in (".jpg",".jpeg",".png",".webp",".heic")):
        if mode == "banana":
            if len(s["banana_images"]) >= 4:
                await update.message.reply_text("⚠️ Достигнут лимит 4 фото.", reply_markup=banana_kb()); return
            s["banana_images"].append(text.strip())
            await update.message.reply_text(f"📸 Фото принято ({len(s['banana_images'])}/4).")
            await show_or_update_banana_card(update.effective_chat.id, ctx); return
        s["last_image_url"] = text.strip()
        await update.message.reply_text("🧷 Ссылка на изображение принята.")
        if mode in ("veo_text_fast","veo_text_quality","veo_photo"):
            await show_or_update_veo_card(update.effective_chat.id, ctx)
        return

    if mode == "chat":
        if openai is None or not OPENAI_API_KEY:
            await update.message.reply_text("⚠️ ChatGPT недоступен (нет OPENAI_API_KEY)."); return
        try:
            await update.message.reply_text("💬 Думаю над ответом…")
            resp = await asyncio.to_thread(
                openai.ChatCompletion.create,
                model="gpt-4o-mini",
                messages=[{"role":"system","content":"You are a helpful, concise assistant."},
                          {"role":"user","content":text}],
                temperature=0.5, max_tokens=700,
            )
            answer = resp["choices"][0]["message"]["content"].strip()
            await update.message.reply_text(answer)
        except Exception as e:
            log.exception("Chat error: %s", e)
            await update.message.reply_text("⚠️ Ошибка запроса к ChatGPT.")
        return

    if mode == "mj_txt":
        if not text:
            await update.message.reply_text("⚠️ Отправьте текстовый промпт.")
            return
        s["last_prompt"] = text
        await show_mj_prompt_card(update.effective_chat.id, ctx)
        await update.message.reply_text("📝 Промпт сохранён. Нажмите «Подтвердить».")
        return

    if mode == "banana":
        s["last_prompt"] = text
        await update.message.reply_text("✍️ Промпт сохранён.")
        await show_or_update_banana_card(update.effective_chat.id, ctx)
        return

    # VEO по умолчанию: сохраняем prompt и мгновенно обновляем карточку
    s["last_prompt"] = text
    await show_or_update_veo_card(update.effective_chat.id, ctx)

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
    application.add_handler(CommandHandler("balance", balance_command))
    application.add_handler(CommandHandler("balance_recalc", balance_recalc))
    application.add_handler(prompt_master_conv, group=10)
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
