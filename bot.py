# -*- coding: utf-8 -*-
# Best VEO3 Bot — PTB 21.x
# Версия: 2025-09-14r4
# Единственное изменение против прежней версии: надежная доставка VEO-видео в Telegram
# (освежение ссылки + повторная попытка + download&reupload с увеличенным таймаутом).
# Остальное (карточки, кнопки, тексты, цены, FAQ, промокоды, бонусы и т.д.) — без изменений.

import os, json, time, uuid, asyncio, logging, tempfile, subprocess, re
from typing import Dict, Any, Optional, List, Tuple

import requests
from dotenv import load_dotenv

from telegram import (
    Update, InlineKeyboardButton, InlineKeyboardMarkup,
    InputFile, InputMediaPhoto, LabeledPrice, Bot
)
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler,
    CallbackQueryHandler, filters, AIORateLimiter, PreCheckoutQueryHandler
)

# === KIE Banana wrapper ===
from kie_banana import create_banana_task, wait_for_banana_result, KieBananaError

import redis

# ==========================
#   ENV / INIT
# ==========================
load_dotenv()
def _env(k: str, d: str = "") -> str:
    v = os.getenv(k)
    return (v if v is not None else d).strip()

TELEGRAM_TOKEN      = _env("TELEGRAM_TOKEN")
PROMPTS_CHANNEL_URL = _env("PROMPTS_CHANNEL_URL", "https://t.me/bestveo3promts")
STARS_BUY_URL       = _env("STARS_BUY_URL", "https://t.me/PremiumBot")
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
KIE_VEO_GEN_PATH    = _env("KIE_VEO_GEN_PATH",    "/api/v1/veo/generate")
KIE_VEO_STATUS_PATH = _env("KIE_VEO_STATUS_PATH", "/api/v1/veo/record-info")
KIE_VEO_1080_PATH   = _env("KIE_VEO_1080_PATH",   "/api/v1/veo/get-1080p-video")

# MJ
KIE_MJ_GENERATE = _env("KIE_MJ_GENERATE", "/api/v1/mj/generate")
KIE_MJ_STATUS   = _env("KIE_MJ_STATUS",   "/api/v1/mj/record-info")

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
REDIS_URL    = _env("REDIS_URL")
REDIS_PREFIX = _env("REDIS_PREFIX", "veo3:prod")
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True) if REDIS_URL else None
def _rk(*parts: str) -> str: return ":".join([REDIS_PREFIX, *parts])

# ==========================
#   Tokens / Pricing
# ==========================
TOKEN_COSTS = {
    "veo_fast": 50,
    "veo_quality": 150,
    "veo_photo": 50,
    "mj": 10,          # только 16:9
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

# ==========================
#   Helpers / storage
# ==========================
def join_url(base: str, path: str) -> str:
    u = f"{base.rstrip('/')}/{path.lstrip('/')}"
    return u.replace("://", "§§").replace("//", "/").replace("§§", "://")

def _kie_headers_json() -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    tok = (KIE_API_KEY or "").strip()
    if tok and not tok.lower().startswith("bearer "): tok = f"Bearer {tok}"
    if tok: h["Authorization"] = tok
    return h

def _post_json(url: str, payload: Dict[str, Any], timeout: int = 50) -> Tuple[int, Dict[str, Any]]:
    r = requests.post(url, json=payload, headers=_kie_headers_json(), timeout=timeout)
    try: return r.status_code, r.json()
    except Exception: return r.status_code, {"error": r.text}

def _get_json(url: str, params: Dict[str, Any], timeout: int = 50) -> Tuple[int, Dict[str, Any]]:
    r = requests.get(url, params=params, headers=_kie_headers_json(), timeout=timeout)
    try: return r.status_code, r.json()
    except Exception: return r.status_code, {"error": r.text}

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
    for key in ("originUrls", "resultUrls", "videoUrls"):
        urls = _coerce_url_list(data.get(key))
        if urls: return urls[0]
    for cont in ("info", "response", "resultInfoJson"):
        v = data.get(cont)
        if isinstance(v, dict):
            for key in ("originUrls", "resultUrls", "videoUrls", "urls"):
                urls = _coerce_url_list(v.get(key))
                if urls: return urls[0]
    return None

def event(tag: str, **kw):
    try: log.info("EVT %s | %s", tag, json.dumps(kw, ensure_ascii=False))
    except Exception: log.info("EVT %s | %s", tag, kw)

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
}
def state(ctx: ContextTypes.DEFAULT_TYPE) -> Dict[str, Any]:
    ud = ctx.user_data
    for k, v in DEFAULT_STATE.items():
        if k not in ud: ud[k] = [] if isinstance(v, list) else v
    if not isinstance(ud.get("banana_images"), list): ud["banana_images"] = []
    return ud

# ---------- Balance ----------
def get_user_id(ctx: ContextTypes.DEFAULT_TYPE) -> Optional[int]:
    try: return ctx._user_id_and_data[0]  # type: ignore[attr-defined]
    except Exception: return None

def get_user_balance_value(ctx: ContextTypes.DEFAULT_TYPE) -> int:
    uid = get_user_id(ctx)
    if redis_client and uid:
        v = redis_client.get(_rk("balance", str(uid)))
        if v is not None:
            try: return int(v)
            except: return 0
    return int(ctx.user_data.get("balance", 0))

def set_user_balance_value(ctx: ContextTypes.DEFAULT_TYPE, v: int):
    v = max(0, int(v))
    ctx.user_data["balance"] = v
    uid = get_user_id(ctx)
    if redis_client and uid: redis_client.set(_rk("balance", str(uid)), v)

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
    "🖌️ *MJ — художник*: рисует изображение по тексту (*только 16:9*).\n"
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
    rows = [
        [InlineKeyboardButton(f"🎬 Генерация видео (Veo Fast) 💎 {TOKEN_COSTS['veo_fast']}", callback_data="mode:veo_text_fast")],
        [InlineKeyboardButton(f"🎬 Генерация видео (Veo Quality) 💎 {TOKEN_COSTS['veo_quality']}", callback_data="mode:veo_text_quality")],
        [InlineKeyboardButton(f"🖼️ Генерация изображений (MJ) 💎 {TOKEN_COSTS['mj']}", callback_data="mode:mj_txt")],
        [InlineKeyboardButton(f"🍌 Редактор изображений (Banana) 💎 {TOKEN_COSTS['banana']}", callback_data="mode:banana")],
        [InlineKeyboardButton(f"📸 Оживить изображение (Veo) 💎 {TOKEN_COSTS['veo_photo']}", callback_data="mode:veo_photo")],
        [InlineKeyboardButton("🧠 Prompt-Master (ChatGPT)", callback_data="mode:prompt_master")],
        [InlineKeyboardButton("💬 Обычный чат (ChatGPT)", callback_data="mode:chat")],
        [
            InlineKeyboardButton("❓ FAQ", callback_data="faq"),
            InlineKeyboardButton("📈 Канал с промптами", url=PROMPTS_CHANNEL_URL),
        ],
        [InlineKeyboardButton("💳 Пополнить баланс", callback_data="topup_open")],
        [InlineKeyboardButton("🎟️ Активировать промокод", callback_data="promo_open")],
    ]
    return InlineKeyboardMarkup(rows)

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
#   Prompt-Master (ChatGPT)
# ==========================
PM_HINT = (
    "🧠 *Prompt-Master готов!* Коротко опишите идею сцены — сделаю проф. кинопромпт.\n"
    "Подсказка: локация, атмосфера/свет, действие, камера, реплики (в кавычках), детали.\n"
    "Диалоги и lip-sync будут на *языке вашего сообщения*; остальное — на английском для качества."
)
async def oai_prompt_master(idea_text: str) -> Optional[str]:
    if openai is None or not OPENAI_API_KEY: return None
    dialogue_lang = "Russian" if re.search(r"[\u0400-\u04FF]", idea_text or "") else "English"
    system = (
        "You are a Prompt-Master for cinematic AI video generation (Veo-style). "
        "Return ONE multi-line prompt with these labeled sections exactly:\n"
        "Scene:\nCamera:\nAction:\nDialogue:\nLip-sync:\nAudio:\nLighting:\nWardrobe/props:\nFraming:\n"
        f"Write ALL sections in English EXCEPT 'Dialogue' and 'Lip-sync', which must be in {dialogue_lang}. "
        "Dialogue must be short ad lines in quotes. "
        "No subtitles/logos/on-screen text in the video. Keep 16:9 framing. Total 600–1100 chars."
    )
    try:
        user = (idea_text or "").strip()[:900]
        resp = await asyncio.to_thread(
            openai.ChatCompletion.create,
            model="gpt-4o-mini",
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            temperature=0.8, max_tokens=800,
        )
        return (resp["choices"][0]["message"]["content"] or "").strip()[:1400]
    except Exception as e:
        log.exception("Prompt-Master error: %s", e)
        return None

# ==========================
#   VEO
# ==========================
def _build_payload_for_veo(prompt: str, aspect: str, image_url: Optional[str], model_key: str) -> Dict[str, Any]:
    payload = {
        "prompt": prompt,
        "aspectRatio": "9:16" if aspect == "9:16" else "16:9",
        "model": "veo3" if model_key == "veo3" else "veo3_fast",
        "enableFallback": aspect == "16:9",
    }
    if image_url: payload["imageUrls"] = [image_url]
    return payload

def submit_kie_veo(prompt: str, aspect: str, image_url: Optional[str], model_key: str) -> Tuple[bool, Optional[str], str]:
    status, j = _post_json(join_url(KIE_BASE_URL, KIE_VEO_GEN_PATH),
                           _build_payload_for_veo(prompt, aspect, image_url, model_key))
    code = j.get("code", status)
    if status == 200 and code == 200:
        tid = _extract_task_id(j)
        if tid: return True, tid, "Задача создана."
        return False, None, "Ответ KIE без taskId."
    return False, None, f"Ошибка VEO: {j}"

def get_kie_veo_status(task_id: str) -> Tuple[bool, Optional[int], Optional[str], Optional[str]]:
    status, j = _get_json(join_url(KIE_BASE_URL, KIE_VEO_STATUS_PATH), {"taskId": task_id})
    code = j.get("code", status)
    if status == 200 and code == 200:
        data = j.get("data") or {}
        try: flag = int(data.get("successFlag"))
        except Exception: flag = None
        return True, flag, (j.get("msg") or j.get("message")), _extract_result_url(data)
    return False, None, f"Ошибка статуса VEO: {j}", None

def try_get_1080_url(task_id: str, attempts: int = 3, per_try_timeout: int = 15) -> Optional[str]:
    last_err = None
    for i in range(attempts):
        try:
            status, j = _get_json(join_url(KIE_BASE_URL, KIE_VEO_1080_PATH), {"taskId": task_id}, timeout=per_try_timeout)
            code = j.get("code", status)
            if status == 200 and code == 200:
                data = j.get("data") or {}
                u = data.get("url") or _extract_result_url(data)
                if isinstance(u, str) and u.startswith("http"): return u
                last_err = "empty_url"
            else:
                last_err = f"{status}/{code}"
        except Exception as e:
            last_err = str(e)
        time.sleep(1+i)
    log.warning("1080p retries failed: %s", last_err)
    return None

# ==========================
#   MJ (только 16:9)
# ==========================
def _kie_error_message(status_code: int, j: Dict[str, Any]) -> str:
    code = j.get("code", status_code)
    msg = j.get("msg") or j.get("message") or j.get("error") or ""
    mapping = {401: "Доступ запрещён.", 402: "Недостаточно кредитов.",
               429: "Превышен лимит.", 500: "Внутренняя ошибка KIE.",
               422: "Запрос отклонён модерацией.", 400: "Неверный запрос."}
    base = mapping.get(code, f"KIE code {code}.")
    return f"{base} {msg}".strip()

def mj_generate(prompt: str) -> Tuple[bool, Optional[str], str]:
    payload = {
        "taskType": "mj_txt2img",
        "prompt": prompt,
        "speed": "fast",
        "aspectRatio": "16:9",
        "version": "7",
        "enableTranslation": True,
    }
    status, j = _post_json(join_url(KIE_BASE_URL, KIE_MJ_GENERATE), payload)
    code = j.get("code", status)
    if status == 200 and code == 200:
        tid = _extract_task_id(j)
        if tid: return True, tid, "MJ задача создана."
        return False, None, "Ответ MJ без taskId."
    return False, None, _kie_error_message(status, j)

def mj_status(task_id: str) -> Tuple[bool, Optional[int], Optional[Dict[str, Any]]]:
    status, j = _get_json(join_url(KIE_BASE_URL, KIE_MJ_STATUS), {"taskId": task_id})
    code = j.get("code", status)
    if status == 200 and code == 200:
        data = j.get("data") or {}
        try: flag = int(data.get("successFlag"))
        except Exception: flag = None
        return True, flag, data
    return False, None, None

def _extract_mj_image_urls(status_data: Dict[str, Any]) -> List[str]:
    res: List[str] = []
    rj = status_data.get("resultInfoJson") or {}
    urls = _coerce_url_list(rj.get("resultUrls"))
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
    Унифицированная надёжная отправка для 16:9 и 9:16:
    1) Освежаем ссылку у KIE (для 16:9 пробуем явный 1080p).
    2) Скачиваем и перезаливаем в Telegram.
       — 9:16: нормализуем scale/pad 1080x1920.
       — 16:9: если ALWAYS_FORCE_FHD=True и есть ffmpeg — приводим к 1080p.
    Никакой прямой отправки по внешней ссылке.
    """
    event("SEND_TRY_URL", url=url, expect_vertical=expect_vertical)

    # 1) Освежаем ссылку непосредственно перед закачкой
    try:
        if task_id:
            if not expect_vertical:
                u1080 = await asyncio.to_thread(try_get_1080_url, task_id)
                if isinstance(u1080, str) and u1080.startswith("http"):
                    url = u1080
                else:
                    ok2, _, _, u2 = await asyncio.to_thread(get_kie_veo_status, task_id)
                    if ok2 and isinstance(u2, str) and u2.startswith("http"):
                        url = u2
            else:
                ok2, _, _, u2 = await asyncio.to_thread(get_kie_veo_status, task_id)
                if ok2 and isinstance(u2, str) and u2.startswith("http"):
                    url = u2
    except Exception as e:
        event("SEND_REFRESH_ERR", err=str(e))

    # 2) Скачиваем и перезаливаем
    tmp_path = None
    try:
        r = requests.get(url, stream=True, timeout=300)
        r.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
            for c in r.iter_content(256 * 1024):
                if c:
                    f.write(c)
            tmp_path = f.name

        # 9:16 — нормализуем и заливаем
        if expect_vertical and _ffmpeg_available():
            out = tmp_path + "_v.mp4"
            if _ffmpeg_normalize_vertical(tmp_path, out):
                with open(out, "rb") as f:
                    await ctx.bot.send_video(chat_id, InputFile(f, filename="result_vertical.mp4"), supports_streaming=True)
                return True

        # 16:9 — гарантируем 1080p при необходимости
        if (not expect_vertical) and ALWAYS_FORCE_FHD and _ffmpeg_available():
            out = tmp_path + "_1080.mp4"
            if _ffmpeg_force_16x9_fhd(tmp_path, out, MAX_TG_VIDEO_MB):
                with open(out, "rb") as f:
                    await ctx.bot.send_video(chat_id, InputFile(f, filename="result_1080p.mp4"), supports_streaming=True)
                return True

        # Если ffmpeg не требуется — шлём как есть
        with open(tmp_path, "rb") as f:
            await ctx.bot.send_video(chat_id, InputFile(f, filename="result.mp4"), supports_streaming=True)
        return True

    except Exception as e:
        log.exception("send_video reupload failed: %s", e)
        try:
            await ctx.bot.send_message(chat_id, f"🔗 Результат готов, но загрузка в Telegram не удалась. Ссылка:\n{url}")
            return True
        except Exception:
            return False
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

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
                if (s.get("aspect") or "16:9") == "16:9":
                    u1080 = await asyncio.to_thread(try_get_1080_url, task_id)
                    if isinstance(u1080, str) and u1080.startswith("http"):
                        final_url = u1080
                else:
                    ok_r2, _, _, u2 = await asyncio.to_thread(get_kie_veo_status, task_id)
                    if ok_r2 and isinstance(u2, str) and u2.startswith("http"):
                        final_url = u2

                await ctx.bot.send_message(chat_id, "🎞️ Рендер завершён — отправляю файл…")
                await send_video_with_fallback(ctx, chat_id, final_url,
                                               expect_vertical=(s.get("aspect") == "9:16"),
                                               task_id=task_id)
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

# ==========================
#   MJ poll (1 авторетрай)
# ==========================
async def poll_mj_and_send_photos(chat_id: int, task_id: str, ctx: ContextTypes.DEFAULT_TYPE,
                                  orig_prompt: Optional[str] = None):
    price = TOKEN_COSTS["mj"]
    start_ts = time.time()
    delay = 12
    max_wait = 12 * 60
    retried = False
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
                if (not retried) and orig_prompt and _mj_should_retry(err):
                    retried = True
                    await ctx.bot.send_message(chat_id, "🔁 MJ подвис. Перезапускаю задачу бесплатно…")
                    ok2, new_tid, msg2 = await asyncio.to_thread(mj_generate, orig_prompt.strip())
                    event("MJ_RETRY_SUBMIT", ok=ok2, task_id=new_tid, msg=msg2)
                    if ok2 and new_tid:
                        task_id = new_tid
                        start_ts = time.time()
                        delay = 12
                        continue
                add_tokens(ctx, price)
                await ctx.bot.send_message(chat_id, f"❌ MJ: {err}\n💎 Токены возвращены.")
                return
            if flag == 1:
                urls = _extract_mj_image_urls(data or {})
                if not urls:
                    add_tokens(ctx, price)
                    await ctx.bot.send_message(chat_id, "⚠️ MJ вернул пустой результат. 💎 Токены возвращены.")
                    return
                if len(urls) == 1:
                    await ctx.bot.send_photo(chat_id=chat_id, photo=urls[0])
                else:
                    await ctx.bot.send_media_group(chat_id=chat_id, media=[InputMediaPhoto(u) for u in urls[:10]])
                await ctx.bot.send_message(chat_id, "✅ *Готово!*", parse_mode=ParseMode.MARKDOWN,
                    reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🚀 Ещё", callback_data="start_new_cycle")]]))
                return
    except Exception as e:
        log.exception("MJ poll crash: %s", e)
        add_tokens(ctx, price)
        try: await ctx.bot.send_message(chat_id, "💥 Внутренняя ошибка MJ. 💎 Токены возвращены.")
        except Exception: pass

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

async def health(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    parts = [
        f"PTB: `{getattr(_tg, '__version__', 'unknown')}`" if _tg else "PTB: `unknown`",
        f"KIEBASEURL: `{KIE_BASE_URL}`",
        f"OPENAI: `{'set' if OPENAI_API_KEY else 'missing'}`",
        f"KIE: `{'set' if KIE_API_KEY else 'missing'}`",
        f"REDIS: `{'on' if REDIS_URL else 'off'}`",
        f"FFMPEG: `{FFMPEG_BIN}`",
    ]
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
            "— *MJ:* только 16:9, цена 10💎. Один бесплатный перезапуск при сетевой ошибке. На выходе до 4 изображений.\n\n"
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
        if mode == "prompt_master":
            await q.message.reply_text(PM_HINT, parse_mode=ParseMode.MARKDOWN); return
        if mode == "chat":
            await q.message.reply_text("💬 Чат активен. Напишите сообщение."); return
        if mode == "mj_txt":
            await q.message.reply_text("🖼️ Пришлите текстовый *prompt* для картинки (формат *16:9*).", parse_mode=ParseMode.MARKDOWN); return
        if mode == "banana":
            s["banana_images"] = []; s["last_prompt"] = None
            await q.message.reply_text("🍌 Banana включён\nСначала пришлите до *4 фото* (можно по одному). Когда будут готовы — пришлите *текст-промпт*, что изменить.", parse_mode=ParseMode.MARKDOWN)
            await show_or_update_banana_card(update.effective_chat.id, ctx); return

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

    # MJ запуск (кнопка "mj:start" сохраняется как раньше)
    if data == "mj:start":
        prompt = (s.get("last_prompt") or "").strip()
        if not prompt:
            await q.message.reply_text("⚠️ Сначала отправьте текстовый prompt."); return
        price = TOKEN_COSTS['mj']
        ok_balance, rest = try_charge(ctx, price)
        if not ok_balance:
            await q.message.reply_text(f"💎 Недостаточно токенов: нужно {price}, на балансе {rest}.", reply_markup=stars_topup_kb()); return
        await q.message.reply_text(f"🎨 Генерация фото запущена…\nФормат: *16:9*\nPrompt: `{prompt}`", parse_mode=ParseMode.MARKDOWN)
        ok, task_id, msg = await asyncio.to_thread(mj_generate, prompt.strip())
        event("MJ_SUBMIT_RESP", ok=ok, task_id=task_id, msg=msg)
        if not ok or not task_id:
            add_tokens(ctx, price)
            await q.message.reply_text(f"❌ Не удалось создать MJ-задачу: {msg}\n💎 Токены возвращены."); return
        await q.message.reply_text(f"🆔 MJ taskId: `{task_id}`\n🖌️ Рисую эскиз и детали…", parse_mode=ParseMode.MARKDOWN)
        asyncio.create_task(poll_mj_and_send_photos(update.effective_chat.id, task_id, ctx, (s.get("last_prompt") or ""))); return

async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = state(ctx)
    text = (update.message.text or "").strip()
    mode = s.get("mode")

    # PROMO
    if mode == "promo":
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
        await update.message.reply_text(f"✅ Промокод принят! +{bonus}💎\nБаланс: {get_user_balance_value(ctx)} 💎")
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

    if mode == "prompt_master":
        if not text:
            await update.message.reply_text("✍️ Напишите идею по подсказке выше."); return
        if len(text) > 700:
            await update.message.reply_text("ℹ️ Урежу ввод до 700 символов для лучшего качества.")
        prompt = await oai_prompt_master(text[:700])
        if not prompt:
            await update.message.reply_text("⚠️ Prompt-Master недоступен или ответ пуст."); return
        s["last_prompt"] = prompt
        await update.message.reply_text(f"🧠 Готово! Вот ваш кинопромпт:\n\n```\n{prompt}\n```", parse_mode=ParseMode.MARKDOWN)
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
        s["last_prompt"] = text
        await update.message.reply_text(
            f"✅ Prompt сохранён:\n\n`{text}`\n\nНажмите, чтобы запустить (16:9):",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🖼️ Сгенерировать (16:9)", callback_data="mj:start")]])
        ); return

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
#   Entry
# ==========================
def main():
    if not TELEGRAM_TOKEN: raise RuntimeError("TELEGRAM_TOKEN is not set")
    if not KIE_BASE_URL:   raise RuntimeError("KIE_BASE_URL is not set")
    if not KIE_API_KEY:    raise RuntimeError("KIE_API_KEY is not set")

    # удалить webhook перед polling
    try:
        Bot(TELEGRAM_TOKEN).delete_webhook(drop_pending_updates=True)
        log.info("Webhook deleted")
    except Exception as e:
        log.warning("Delete webhook failed: %s", e)

    # (опциональный) Redis-замок от дублей — можно оставить как есть или убрать
    lock_key = _rk("poll_lock")
    if redis_client:
        got_lock = redis_client.set(lock_key, str(time.time()), nx=True, ex=30*60)
        if not got_lock:
            log.error("Another instance is running (redis lock present). Exiting to avoid 409 conflict.")
            return

    app = (ApplicationBuilder()
           .token(TELEGRAM_TOKEN)
           .rate_limiter(AIORateLimiter())
           .build())

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("health", health))
    app.add_handler(CommandHandler("topup", topup))
    app.add_handler(PreCheckoutQueryHandler(precheckout_callback))
    app.add_handler(MessageHandler(filters.SUCCESSFUL_PAYMENT, successful_payment_handler))
    app.add_handler(CallbackQueryHandler(on_callback))
    app.add_handler(MessageHandler(filters.PHOTO, on_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_error_handler(error_handler)

    try:
        log.info("Bot starting… (Redis=%s)", "on" if redis_client else "off")
        app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True, stop_signals=None)
    finally:
        try:
            if redis_client: redis_client.delete(lock_key)
        except Exception:
            pass

if __name__ == "__main__":
    main()
