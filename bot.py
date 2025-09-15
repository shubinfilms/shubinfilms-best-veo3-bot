# -*- coding: utf-8 -*-
# Best VEO3 Bot — PTB 21.x
# Версия: 2025-09-14r4
# Единственное изменение против прежней версии: надежная доставка VEO-видео в Telegram
# (освежение ссылки + повторная попытка + download&reupload с увеличенным таймаутом).
# Остальное (карточки, кнопки, тексты, цены, FAQ, промокоды, бонусы и т.д.) — без изменений.

import os, json, time, uuid, asyncio, logging, tempfile, subprocess, re, threading, signal, socket
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Callable
from datetime import datetime, timezone
from contextlib import suppress

import requests
from dotenv import load_dotenv

from telegram import (
    Update, InlineKeyboardButton, InlineKeyboardMarkup,
    InputFile, LabeledPrice
)
from telegram.constants import ParseMode, ChatAction
from telegram.ext import (
    ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler,
    CallbackQueryHandler, filters, AIORateLimiter, PreCheckoutQueryHandler
)

from handlers.prompt_master_handler import PROMPT_MASTER_HINT
from prompt_master import generate_prompt_master

# === KIE Banana wrapper ===
from kie_banana import create_banana_task, wait_for_banana_result, KieBananaError

import redis
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
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
BALANCE_BACKUP_PATH = _env("BALANCE_BACKUP_PATH", os.path.join(SCRIPT_DIR, "balances.json")).strip()
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
    return None

def promo_mark_used(code: str, uid: int):
    code = (code or "").strip().upper()
    if not code: return
    if redis_client:
        redis_client.setnx(_rk("promo", "used_by", code), str(uid))

# локальный кэш процесса (если Redis выключен)
app_cache: Dict[Any, Any] = {}


class PersistentBalanceStorage:
    def __init__(self, redis_conn: Optional["redis.Redis"], backup_path: str):
        self.redis = redis_conn
        self.backup_path = backup_path
        self.backup_enabled = bool(self.backup_path)
        self._lock = threading.RLock()
        self._balances: Dict[str, int] = {}
        if self.backup_enabled:
            self.backup_path = os.path.abspath(self.backup_path)
        self._load_backup()
        self._sync_backup_to_redis()

    @staticmethod
    def _parse_int(value: Any) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _redis_get(self, key: str) -> Optional[int]:
        if not self.redis:
            return None
        try:
            raw = self.redis.get(key)
        except redis.RedisError as exc:
            log.warning("Redis get failed for %s: %s", key, exc)
            return None
        if raw is None:
            return None
        parsed = self._parse_int(raw)
        if parsed is None:
            log.warning("Invalid balance value in Redis for %s: %r", key, raw)
        return parsed

    def _redis_set(self, key: str, value: int) -> bool:
        if not self.redis:
            return False
        try:
            self.redis.set(key, value)
            return True
        except redis.RedisError as exc:
            log.warning("Redis set failed for %s: %s", key, exc)
            return False

    def _load_backup(self):
        if not self.backup_enabled:
            return
        data: Dict[str, Any] = {}
        try:
            with open(self.backup_path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            if isinstance(raw, dict):
                data = raw
        except FileNotFoundError:
            data = {}
        except Exception as exc:
            log.warning("Failed to load balance backup %s: %s", self.backup_path, exc)
            data = {}

        cleaned: Dict[str, int] = {}
        for key, value in data.items():
            parsed = self._parse_int(value)
            if parsed is not None:
                cleaned[str(key)] = parsed

        with self._lock:
            self._balances = cleaned

    def _write_backup_locked(self):
        if not self.backup_enabled:
            return
        try:
            directory = os.path.dirname(self.backup_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            tmp_path = f"{self.backup_path}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(self._balances, fh, ensure_ascii=False, indent=2, sort_keys=True)
            os.replace(tmp_path, self.backup_path)
        except Exception as exc:
            log.warning("Failed to write balance backup %s: %s", self.backup_path, exc)

    def _update_backup(self, uid: int, value: int):
        if not self.backup_enabled:
            return
        with self._lock:
            key = str(uid)
            int_value = int(value)
            if self._balances.get(key) == int_value:
                return
            self._balances[key] = int_value
            self._write_backup_locked()

    def _get_backup(self, uid: int) -> Optional[int]:
        if not self.backup_enabled:
            return None
        with self._lock:
            value = self._balances.get(str(uid))
        if value is None:
            return None
        return int(value)

    def _sync_backup_to_redis(self):
        if not self.redis or not self.backup_enabled or not self._balances:
            return
        for uid, value in list(self._balances.items()):
            key = _rk("balance", str(uid))
            try:
                if self.redis.get(key) is None:
                    self.redis.set(key, value)
            except redis.RedisError as exc:
                log.warning("Redis sync failed for %s: %s", key, exc)
                break

    def get(self, uid: int) -> int:
        if not uid:
            return 0
        key = _rk("balance", str(uid))
        value = self._redis_get(key)
        if value is not None:
            self._update_backup(uid, value)
            return value
        backup = self._get_backup(uid)
        if backup is not None:
            return backup
        return 0

    def set(self, uid: int, value: int) -> int:
        value = max(0, int(value))
        if not uid:
            return value
        key = _rk("balance", str(uid))
        self._redis_set(key, value)
        self._update_backup(uid, value)
        return value


balance_storage = PersistentBalanceStorage(redis_client, BALANCE_BACKUP_PATH)

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
    "mode": None, "aspect": "16:9", "model": None,
    "last_prompt": None, "last_image_url": None,
    "generating": False, "generation_id": None, "last_task_id": None,
    "last_ui_msg_id": None, "last_ui_msg_id_banana": None,
    "banana_images": [],
    "mj_last_wait_ts": 0.0,
    "mj_generating": False, "last_mj_task_id": None, "last_mj_msg_id": None,
}

MODE_PROMPTMASTER = "MODE_PROMPTMASTER"
PROMPT_MASTER_TIMEOUT = 27.0
PROMPT_MASTER_ERROR_MESSAGE = "❌ Не удалось собрать промпт. Попробуй сформулировать короче."
PROMPT_MASTER_CARD_TEMPLATE = (
    "🟦 Карточка Prompt-Master\n"
    "✍️ Промпт:\n{prompt}"
)
def state(ctx: ContextTypes.DEFAULT_TYPE) -> Dict[str, Any]:
    ud = ctx.user_data
    for k, v in DEFAULT_STATE.items():
        if k not in ud: ud[k] = [] if isinstance(v, list) else v
    if not isinstance(ud.get("banana_images"), list): ud["banana_images"] = []
    return ud


def activate_prompt_master_mode(ctx: ContextTypes.DEFAULT_TYPE) -> Dict[str, Any]:
    s = state(ctx)
    s["mode"] = MODE_PROMPTMASTER
    s["last_prompt"] = None
    s["last_image_url"] = None
    return s

# ---------- Balance ----------
def get_user_id(ctx: ContextTypes.DEFAULT_TYPE) -> Optional[int]:
    try: return ctx._user_id_and_data[0]  # type: ignore[attr-defined]
    except Exception: return None

def get_user_balance_value(ctx: ContextTypes.DEFAULT_TYPE) -> int:
    cached = ctx.user_data.get("balance")
    if cached is not None:
        try:
            return int(cached)
        except (TypeError, ValueError):
            pass

    uid = get_user_id(ctx)
    if not uid:
        return 0

    balance = balance_storage.get(uid)
    ctx.user_data["balance"] = balance
    return balance

def set_user_balance_value(ctx: ContextTypes.DEFAULT_TYPE, v: int):
    v = max(0, int(v))
    ctx.user_data["balance"] = v
    uid = get_user_id(ctx)
    if uid:
        balance_storage.set(uid, v)

def add_tokens(ctx: ContextTypes.DEFAULT_TYPE, add: int):
    set_user_balance_value(ctx, get_user_balance_value(ctx) + int(add))

def try_charge(ctx: ContextTypes.DEFAULT_TYPE, need: int) -> Tuple[bool, int]:
    bal = get_user_balance_value(ctx)
    if bal < need: return False, bal
    set_user_balance_value(ctx, bal - need)
    return True, bal - need

def has_signup_bonus(uid: int) -> bool:
    if not redis_client: return False
    return bool(redis_client.get(_rk("signup_bonus", str(uid))))

def set_signup_bonus(uid: int):
    if redis_client: redis_client.set(_rk("signup_bonus", str(uid)), "1")

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
        [InlineKeyboardButton(f"🎬 Генерация видео (Veo Fast) 💎 {TOKEN_COSTS['veo_fast']}", callback_data="mode:veo_text_fast")],
        [InlineKeyboardButton(f"🎬 Генерация видео (Veo Quality) 💎 {TOKEN_COSTS['veo_quality']}", callback_data="mode:veo_text_quality")],
        [InlineKeyboardButton(f"🖼️ Генерация изображений (MJ) 💎 {TOKEN_COSTS['mj']}", callback_data="mode:mj_txt")],
        [InlineKeyboardButton(f"🍌 Редактор изображений (Banana) 💎 {TOKEN_COSTS['banana']}", callback_data="mode:banana")],
        [InlineKeyboardButton(f"📸 Оживить изображение (Veo) 💎 {TOKEN_COSTS['veo_photo']}", callback_data="mode:veo_photo")],
        [InlineKeyboardButton("🧠 Prompt-Master", callback_data="mode:prompt_master")],
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


def download_file(url: str) -> Path:
    tmp_path = Path(tempfile.gettempdir()) / f"{uuid.uuid4()}.mp4"
    with KIE_1080_SESSION.get(url, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        with tmp_path.open("wb") as fh:
            for chunk in resp.iter_content(1024 * 1024):
                if chunk:
                    fh.write(chunk)
    return tmp_path


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
    code_for_user = meta.get("code") or meta.get("http_status") or "n/a"

    if not hd_url:
        reason = meta.get("error") or meta.get("message")
        if reason:
            kie_event("1080_UNAVAILABLE", taskId=task_id, index=index, reason=reason, code=meta.get("code"), http_status=meta.get("http_status"))
        message = f"KIE не вернул 1080p: {code_for_user}"
        await ctx.bot.send_message(chat_id, f"ℹ️ {message}")
        if fallback_url:
            await ctx.bot.send_message(chat_id, f"🔗 Исходный URL: {fallback_url}")
        return False

    try:
        path = download_file(hd_url)
    except Exception as exc:
        kie_event("1080_DOWNLOAD_FAIL", taskId=task_id, index=index, error=str(exc))
        await ctx.bot.send_message(chat_id, "❌ Не удалось скачать 1080p видео.")
        if fallback_url:
            await ctx.bot.send_message(chat_id, f"🔗 Исходный URL: {fallback_url}")
        return False

    try:
        wh = probe_size(path)
        width, height = (wh if wh else (None, None))
        resolution = f"{width}x{height}" if width and height else None
        main
        kie_event(
            "1080_LOCAL",
            taskId=task_id,
            index=index,
            http_status=meta.get("http_status"),
            code=meta.get("code"),
            resultUrl=hd_url,
            local_path=str(path),
            resolution=resolution,
            width=width,
            height=height,
        )

        caption = (f"{width}×{height}" if width and height else "1080p")
        with path.open("rb") as fh:
            await ctx.bot.send_video(
                chat_id=chat_id,
                video=InputFile(fh, filename="veo_result_1080p.mp4"),
                supports_streaming=True,
                caption=caption,
            )

        if width and height:
            expected = (1080, 1920) if is_vertical else (1920, 1080)
            if (width, height) != expected:
                log.warning("KIE returned non-1080p (%s×%s), expected %s×%s", width, height, expected[0], expected[1])
                await ctx.bot.send_message(chat_id, "ℹ️ KIE вернул не 1080p. Проверим тариф/параметры.")
        return True
    finally:
        with suppress(Exception):
            path.unlink()


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
#   VEO polling
# ==========================
async def poll_veo_and_send(chat_id: int, task_id: str, gen_id: str, ctx: ContextTypes.DEFAULT_TYPE):
    s = state(ctx)
    start_ts = time.time()
    try:
        while True:
            if s.get("generation_id") != gen_id: return
            ok, flag, msg, res_url = await asyncio.to_thread(get_kie_veo_status, task_id)
            if not ok:
                add_tokens(ctx, TOKEN_COSTS['veo_quality'] if s.get('model') == 'veo3' else TOKEN_COSTS['veo_fast'])
                await ctx.bot.send_message(chat_id, f"❌ Ошибка статуса VEO. 💎 Токены возвращены.\n{msg or ''}")
                break
            if isinstance(res_url, str) and res_url.startswith("http"):
                # 🔄 освежаем ссылку непосредственно перед отправкой
                final_url = res_url
                if (s.get("aspect") or "16:9") == "9:16":
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
                await send_kie_1080p_to_tg(
                    ctx,
                    chat_id,
                    task_id,
                    index=None,
                    fallback_url=final_url,
                    is_vertical=(s.get("aspect") == "9:16"),
                )
                await ctx.bot.send_message(chat_id, "✅ *Готово!*", parse_mode=ParseMode.MARKDOWN,
                    reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🚀 Сгенерировать ещё видео", callback_data="start_new_cycle")]]))
                break
            if flag in (2, 3):
                add_tokens(ctx, TOKEN_COSTS['veo_quality'] if s.get('model') == 'veo3' else TOKEN_COSTS['veo_fast'])
                await ctx.bot.send_message(chat_id, f"❌ KIE не вернул ссылку на видео. 💎 Токены возвращены.\n{msg or ''}")
                break
            if (time.time() - start_ts) > POLL_TIMEOUT_SECS:
                add_tokens(ctx, TOKEN_COSTS['veo_quality'] if s.get('model') == 'veo3' else TOKEN_COSTS['veo_fast'])
                await ctx.bot.send_message(chat_id, "⌛ Превышено время ожидания VEO. 💎 Токены возвращены.")
                break
            await asyncio.sleep(POLL_INTERVAL_SECS)
    except Exception as e:
        log.exception("VEO poll crash: %s", e)
        add_tokens(ctx, TOKEN_COSTS['veo_quality'] if s.get('model') == 'veo3' else TOKEN_COSTS['veo_fast'])
        try: await ctx.bot.send_message(chat_id, "💥 Внутренняя ошибка при опросе VEO. 💎 Токены возвращены.")
        except Exception: pass
    finally:
        if s.get("generation_id") == gen_id:
            s["generating"] = False
            s["generation_id"] = None
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
    try:
        while True:
            ok, flag, data = await asyncio.to_thread(mj_status, task_id)
            if not ok:
                add_tokens(ctx, price)
                await ctx.bot.send_message(chat_id, "❌ MJ: сервис недоступен. 💎 Токены возвращены.")
                return
            if flag == 0:
                if time.time() - start_ts > max_wait:
                    add_tokens(ctx, price)
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
                add_tokens(ctx, price)
                await ctx.bot.send_message(chat_id, f"❌ MJ: {err}\n💎 Токены возвращены.")
                return
            if flag == 1:
                payload = data or {}
                url = _extract_result_url(payload)
                if not url:
                    urls = _extract_mj_image_urls(payload)
                    url = urls[0] if urls else None
                if not url:
                    add_tokens(ctx, price)
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
        add_tokens(ctx, price)
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

    got_bonus = False
    if redis_client:
        if not has_signup_bonus(uid):
            set_signup_bonus(uid); got_bonus = True
    else:
        if not ctx.user_data.get("__signup_bonus"):
            ctx.user_data["__signup_bonus"] = True; got_bonus = True
    if got_bonus:
        add_tokens(ctx, 10)
        await update.message.reply_text("🎁 Добро пожаловать! Начислил +10💎 на баланс.")

    await update.message.reply_text(render_welcome_for(uid, ctx), parse_mode=ParseMode.MARKDOWN, reply_markup=main_menu_kb())

async def topup(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "💳 Пополнение через *Telegram Stars*.\nЕсли звёзд не хватает — купите в официальном боте:",
        parse_mode=ParseMode.MARKDOWN, reply_markup=stars_topup_kb()
    )


async def prompt_master_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    activate_prompt_master_mode(ctx)
    msg = update.message
    if msg:
        await msg.reply_text(PROMPT_MASTER_HINT)
        return
    chat = update.effective_chat
    if chat:
        await ctx.bot.send_message(chat_id=chat.id, text=PROMPT_MASTER_HINT)


async def health(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    parts = [
        f"PTB: `{getattr(_tg, '__version__', 'unknown')}`" if _tg else "PTB: `unknown`",
        f"KIEBASEURL: `{KIE_BASE_URL}`",
        f"OPENAI: `{'set' if OPENAI_API_KEY else 'missing'}`",
        f"KIE: `{'set' if KIE_API_KEY else 'missing'}`",
        f"REDIS: `{'on' if REDIS_URL else 'off'}`",
        f"FFMPEG: `{FFMPEG_BIN}`",
    ]
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
        selected_mode = data.split(":", 1)[1]
        if selected_mode == "prompt_master":
            activate_prompt_master_mode(ctx)
            await q.message.reply_text(PROMPT_MASTER_HINT)
            return
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
            ok_balance, rest = try_charge(ctx, price)
            if not ok_balance:
                await q.message.reply_text(f"💎 Недостаточно токенов: нужно {price}, на балансе {rest}.", reply_markup=stars_topup_kb()); return
            await q.message.reply_text("✅ Промпт принят.")
            s["mj_generating"] = True
            s["mj_last_wait_ts"] = time.time()
            aspect_value = "9:16" if s.get("aspect") == "9:16" else "16:9"
            await show_mj_generating_card(chat_id, ctx, prompt, aspect_value)
            ok, task_id, msg = await asyncio.to_thread(mj_generate, prompt, aspect_value)
            event("MJ_SUBMIT_RESP", ok=ok, task_id=task_id, msg=msg)
            if not ok or not task_id:
                add_tokens(ctx, price)
                s["mj_generating"] = False
                s["last_mj_task_id"] = None
                s["mj_last_wait_ts"] = 0.0
                await q.message.reply_text(f"❌ Не удалось создать MJ-задачу: {msg}\n💎 Токены возвращены.")
                await show_mj_prompt_card(chat_id, ctx)
                return
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
            ok, rest = try_charge(ctx, price)
            if not ok:
                await q.message.reply_text(f"💎 Недостаточно токенов: нужно {price}, на балансе {rest}.", reply_markup=stars_topup_kb()); return
            await q.message.reply_text("🍌 Запускаю Banana…")
            asyncio.create_task(_banana_run_and_send(update.effective_chat.id, ctx, imgs, prompt)); return

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
        ok_balance, rest = try_charge(ctx, price)
        if not ok_balance:
            await q.message.reply_text(f"💎 Недостаточно токенов: нужно {price}, на балансе {rest}.", reply_markup=stars_topup_kb()); return
        await q.message.reply_text("🎬 Отправляю задачу в VEO…")
        ok, task_id, msg = await asyncio.to_thread(submit_kie_veo, prompt, (s.get("aspect") or "16:9"), s.get("last_image_url"), s.get("model") or "veo3_fast")
        if not ok or not task_id:
            add_tokens(ctx, price)
            await q.message.reply_text(f"❌ Не удалось создать VEO-задачу: {msg}\n💎 Токены возвращены."); return
        gen_id = uuid.uuid4().hex
        s["generating"] = True; s["generation_id"] = gen_id; s["last_task_id"] = task_id
        await q.message.reply_text(f"🆔 VEO taskId: `{task_id}`\n🎞 Рендер начат — вернусь с готовым видео.", parse_mode=ParseMode.MARKDOWN)
        asyncio.create_task(poll_veo_and_send(update.effective_chat.id, task_id, gen_id, ctx)); return

async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = state(ctx)
    text = (update.message.text or "").strip()
    mode = s.get("mode")

    if mode == MODE_PROMPTMASTER:
        if not text:
            return
        chat = update.effective_chat
        if chat:
            with suppress(Exception):
                await ctx.bot.send_chat_action(chat_id=chat.id, action=ChatAction.TYPING)
        user_id = update.effective_user.id if update.effective_user else None
        try:
            prompt_text = await asyncio.wait_for(
                asyncio.to_thread(generate_prompt_master, text),
                timeout=PROMPT_MASTER_TIMEOUT,
            )
        except asyncio.TimeoutError:
            log.error("PromptMaster timeout: uid=%s len=%s", user_id, len(text))
            await update.message.reply_text(PROMPT_MASTER_ERROR_MESSAGE)
            return
        except Exception:
            log.exception("PromptMaster error: uid=%s", user_id)
            await update.message.reply_text(PROMPT_MASTER_ERROR_MESSAGE)
            return

        prompt_text = (prompt_text or "").strip()
        if not prompt_text:
            log.error("PromptMaster empty response: uid=%s", user_id)
            await update.message.reply_text(PROMPT_MASTER_ERROR_MESSAGE)
            return

        card_text = PROMPT_MASTER_CARD_TEMPLATE.format(prompt=prompt_text)
        await update.message.reply_text(card_text)
        return

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
        promo_mark_used(code, uid)
        add_tokens(ctx, bonus)
        await update.message.reply_text(
            f"✅ Промокод принят! +{bonus}💎\nБаланс: {get_user_balance_value(ctx)} 💎"
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

async def _banana_run_and_send(chat_id: int, ctx: ContextTypes.DEFAULT_TYPE, src_urls: List[str], prompt: str):
    try:
        task_id = await asyncio.to_thread(create_banana_task, prompt, src_urls, "png", "auto", None, None, 60)
        await ctx.bot.send_message(chat_id, f"🍌 Задача Banana создана.\n🆔 taskId={task_id}\nЖдём результат…")
        urls = await asyncio.to_thread(wait_for_banana_result, task_id, 8*60, 3)
        if not urls:
            add_tokens(ctx, TOKEN_COSTS["banana"])
            await ctx.bot.send_message(chat_id, "⚠️ Banana вернула пустой результат. 💎 Токены возвращены."); return
        u0 = urls[0]
        try:
            await ctx.bot.send_photo(chat_id=chat_id, photo=u0, caption="✅ Banana готово")
        except Exception:
            r = requests.get(u0, timeout=180); r.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                f.write(r.content); path = f.name
            with open(path, "rb") as f:
                await ctx.bot.send_document(chat_id=chat_id, document=InputFile(f, filename="banana.png"), caption="✅ Banana готово")
            try: os.unlink(path)
            except Exception: pass
    except KieBananaError as e:
        add_tokens(ctx, TOKEN_COSTS["banana"])
        await ctx.bot.send_message(chat_id, f"❌ Banana ошибка: {e}\n💎 Токены возвращены.")
    except Exception as e:
        add_tokens(ctx, TOKEN_COSTS["banana"])
        log.exception("BANANA unexpected: %s", e)
        await ctx.bot.send_message(chat_id, "💥 Внутренняя ошибка Banana. 💎 Токены возвращены.")

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
        add_tokens(ctx, tokens)
        await update.message.reply_text(f"✅ Оплата получена: +{tokens} токенов.\nБаланс: {get_user_balance_value(ctx)} 💎")
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
    application.add_handler(CommandHandler("promptmaster", prompt_master_command))
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
