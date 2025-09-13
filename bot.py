# -*- coding: utf-8 -*-
# Best VEO3 Bot — PTB 21.x
# Версия: 2025-09-12 (FULL) — Banana fix (Telegram file → public URL), остальное без изменений

import os
import json
import time
import uuid
import base64
import asyncio
import logging
import tempfile
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

import requests
from dotenv import load_dotenv

from telegram import (
    Update, InlineKeyboardButton, InlineKeyboardMarkup,
    InputMediaPhoto, LabeledPrice, Bot
)
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler,
    CallbackQueryHandler, filters, AIORateLimiter, PreCheckoutQueryHandler
)

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

# ---- KIE core ----
KIE_API_KEY  = _env("KIE_API_KEY")
KIE_BASE_URL = _env("KIE_BASE_URL", "https://api.kie.ai")
KIE_VEO_GEN_PATH    = _env("KIE_VEO_GEN_PATH",    "/api/v1/veo/generate")
KIE_VEO_STATUS_PATH = _env("KIE_VEO_STATUS_PATH", "/api/v1/veo/record-info")
KIE_VEO_1080_PATH   = _env("KIE_VEO_1080_PATH",   "/api/v1/veo/get-1080p-video")

# ---- MJ
KIE_MJ_GENERATE = _env("KIE_MJ_GENERATE", "/api/v1/mj/generate")
KIE_MJ_STATUS   = _env("KIE_MJ_STATUS",   "/api/v1/mj/record-info")

# ---- Banana
KIE_BANANA_GENERATE = _env("KIE_BANANA_GENERATE", "/api/v1/jobs/createTask")
KIE_BANANA_STATUS   = _env("KIE_BANANA_STATUS",   "/api/v1/jobs/recordInfo")

# ---- Upload API (публичная ссылка)
UPLOAD_BASE_URL     = _env("UPLOAD_BASE_URL", "https://kieai.redpandaai.co")
UPLOAD_STREAM_PATH  = _env("UPLOAD_STREAM_PATH", "/api/file-stream-upload")

POLL_INTERVAL_SECS = int(_env("POLL_INTERVAL_SECS", "6"))
POLL_TIMEOUT_SECS  = int(_env("POLL_TIMEOUT_SECS", str(20 * 60)))

LOG_LEVEL = _env("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("veo3-bot")

# Optional Redis (persistent balance)
try:
    import redis  # type: ignore
except Exception:
    redis = None

REDIS_URL    = _env("REDIS_URL")
REDIS_PREFIX = _env("REDIS_PREFIX", "veo3:prod")
redis_client = None
if redis and REDIS_URL:
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()
        log.info("Redis connected")
    except Exception as e:
        log.warning("Redis connect failed: %s", e)
        redis_client = None

def _rk(*parts: str) -> str:
    return ":".join([REDIS_PREFIX, *[p for p in parts if p]])

# ==========================
#   Pricing / Packs / Promo
# ==========================
TOKEN_COSTS = {
    "veo_fast": 50,
    "veo_quality": 150,
    "veo_photo": 50,
    "mj": 15,
    "banana": 5,
    "chat": 0,
}
SIGNUP_BONUS = int(_env("SIGNUP_BONUS", "10"))

STAR_PACKS = [
    (50, 50,  ""),
    (100,110, "+10💎 бонус"),
    (200,220, "+20💎 бонус"),
    (300,330, "+30💎 бонус"),
    (400,440, "+40💎 бонус"),
    (500,550, "+50💎 бонус"),
    (1000,1100, "+100💎 бонус"),
]
if DEV_MODE:
    STAR_PACKS = [(1,1,"DEV"), *STAR_PACKS]

PROMO_CODES_RAW = os.getenv("PROMO_CODES", "")
def _parse_promo_env(s: str) -> Dict[str, int]:
    res: Dict[str, int] = {}
    for part in (s or "").split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        code, val = part.split("=", 1)
        code = code.strip().upper()
        try:
            amount = int(val.strip())
        except:
            continue
        if code and amount > 0:
            res[code] = amount
    return res
PROMO_CODES = _parse_promo_env(PROMO_CODES_RAW)

# ==========================
#   State
# ==========================
DEFAULT_STATE = {
    "mode": None,
    "aspect": None,
    "model": None,
    "last_prompt": None,
    "last_image_url": None,
    "generating": False,
    "generation_id": None,
    "last_ui_msg_id": None,
    "last_task_id": None,
    "last_result_url": None,
    "chat_unlocked": True,
    "mj_wait_sent": False,
    "mj_wait_last_ts": 0.0,
    # Banana
    "banana_images": [],
    "banana_prompt": None,
    "banana_task_id": None,
    "banana_wait_prompt": False,
}

def state(ctx: ContextTypes.DEFAULT_TYPE) -> Dict[str, Any]:
    ud = ctx.user_data
    for k, v in DEFAULT_STATE.items():
        ud.setdefault(k, v)
    return ud

# ==========================
#   Balance helpers (Redis)
# ==========================
def _get_balance(uid: int) -> int:
    if redis_client:
        v = redis_client.get(_rk("balance", str(uid)))
        if v is None: return 0
        try: return int(v)
        except: return 0
    return int(uid and 0 or 0)

def _set_balance(uid: int, val: int):
    if redis_client:
        redis_client.set(_rk("balance", str(uid)), max(0, int(val)))

def get_user_balance_value(ctx: ContextTypes.DEFAULT_TYPE) -> int:
    uid = ctx._user_id_and_data[0] if hasattr(ctx, "_user_id_and_data") else None
    if uid is None: return int(ctx.user_data.get("balance", 0))
    bal = _get_balance(uid)
    ctx.user_data["balance"] = bal
    return bal

def set_user_balance_value(ctx: ContextTypes.DEFAULT_TYPE, v: int):
    uid = ctx._user_id_and_data[0] if hasattr(ctx, "_user_id_and_data") else None
    ctx.user_data["balance"] = max(0, int(v))
    if uid is not None:
        _set_balance(uid, ctx.user_data["balance"])

def add_tokens(ctx: ContextTypes.DEFAULT_TYPE, add: int):
    set_user_balance_value(ctx, get_user_balance_value(ctx) + int(add))

def try_charge(ctx: ContextTypes.DEFAULT_TYPE, need: int) -> Tuple[bool, int]:
    bal = get_user_balance_value(ctx)
    if bal < need:
        return False, bal
    set_user_balance_value(ctx, bal - need)
    return True, bal - need

def ensure_signup_bonus(ctx: ContextTypes.DEFAULT_TYPE, uid: int):
    if redis_client:
        key = _rk("balance", str(uid))
        if redis_client.get(key) is None:
            redis_client.set(key, SIGNUP_BONUS)
            ctx.user_data["balance"] = SIGNUP_BONUS
            return
    if "balance" not in ctx.user_data:
        ctx.user_data["balance"] = SIGNUP_BONUS

# ==========================
#   UI & helpers
# ==========================
WELCOME = (
    "🎬 Veo 3 — съёмочная команда: опиши идею и получи готовый клип!\n"
    "🖌️ MJ — художник: нарисует изображение по твоему тексту.\n"
    "🍌 Banana — Редактор изображений из будущего\n"
    "🧠 Prompt-Master — вернёт профессиональный кинопромпт.\n"
    "💬 Обычный чат — общение с ИИ.\n\n"
    "💎 Ваш баланс: *{balance}*\n"
    "📈 Больше идей и примеров: {prompts_url}\n\n"
    "Выберите режим 👇"
)

def render_welcome_for(uid: int, ctx: ContextTypes.DEFAULT_TYPE) -> str:
    return WELCOME.format(balance=get_user_balance_value(ctx), prompts_url=PROMPTS_CHANNEL_URL)

def main_menu_kb() -> InlineKeyboardMarkup:
    vf = TOKEN_COSTS["veo_fast"]
    vq = TOKEN_COSTS["veo_quality"]
    vp = TOKEN_COSTS["veo_photo"]
    mj = TOKEN_COSTS["mj"]
    bn = TOKEN_COSTS["banana"]
    rows = [
        [InlineKeyboardButton(f"🎬 Генерация видео (Veo Fast) 💎{vf}", callback_data="mode:veo_text_fast")],
        [InlineKeyboardButton(f"🎬 Генерация видео (Veo Quality) 💎{vq}", callback_data="mode:veo_text_quality")],
        [InlineKeyboardButton(f"🖼️ Генерация изображений (MJ) 💎{mj}", callback_data="mode:mj_txt")],
        [InlineKeyboardButton(f"🍌 Редактор изображений (Banana) 💎{bn}", callback_data="mode:banana")],
        [InlineKeyboardButton(f"📸 Оживить изображение (Veo) 💎{vp}", callback_data="mode:veo_photo")],
        [InlineKeyboardButton("🧠 Prompt-Master (ChatGPT)", callback_data="mode:prompt_master")],
        [InlineKeyboardButton("💬 Обычный чат (ChatGPT)", callback_data="mode:chat")],
        [
            InlineKeyboardButton("❓ FAQ", callback_data="faq"),
            InlineKeyboardButton("📈 Канал с промптами", url=PROMPTS_CHANNEL_URL),
        ],
        [InlineKeyboardButton("💳 Пополнить баланс", callback_data="topup_open")],
    ]
    return InlineKeyboardMarkup(rows)

def aspect_row(current: str) -> List[InlineKeyboardButton]:
    if current == "9:16":
        return [InlineKeyboardButton("16:9", callback_data="aspect:16:9"),
                InlineKeyboardButton("9:16 ✅", callback_data="aspect:9:16")]
    return [InlineKeyboardButton("16:9 ✅", callback_data="aspect:16:9"),
            InlineKeyboardButton("9:16",     callback_data="aspect:9:16")]

def model_row(current: str) -> List[InlineKeyboardButton]:
    if current == "veo3":
        return [InlineKeyboardButton("⚡ Fast",       callback_data="model:veo3_fast"),
                InlineKeyboardButton("💎 Quality ✅", callback_data="model:veo3")]
    return [InlineKeyboardButton("⚡ Fast ✅", callback_data="model:veo3_fast"),
            InlineKeyboardButton("💎 Quality", callback_data="model:veo3")]

def build_card_text_veo(s: Dict[str, Any]) -> str:
    prompt_preview = (s.get("last_prompt") or "").strip()
    if len(prompt_preview) > 900: prompt_preview = prompt_preview[:900] + "…"
    has_prompt = "есть" if s.get("last_prompt") else "нет"
    has_ref = "есть" if s.get("last_image_url") else "нет"
    model = "Fast" if s.get("model") == "veo3_fast" else ("Quality" if s.get("model") else "—")
    price = TOKEN_COSTS['veo_quality'] if s.get('model') == 'veo3' else TOKEN_COSTS['veo_fast']
    return "\n".join([
        "🎬 *Карточка VEO*",
        "",
        "✍️ *Промпт:*",
        f"`{prompt_preview or '—'}`",
        "",
        "🧩 *Параметры:*",
        f"• Aspect: *{s.get('aspect') or '—'}*",
        f"• Mode: *{model}*",
        f"• Промпт: *{has_prompt}*",
        f"• Референс: *{has_ref}*",
        "",
        f"💎 *Стоимость запуска:* {price}",
    ])

def build_card_text_banana(s: Dict[str, Any]) -> str:
    imgs = s.get("banana_images") or []
    prompt = (s.get("banana_prompt") or "").strip()
    if len(prompt) > 400: prompt = prompt[:400] + "…"
    have_prompt = "есть" if prompt else "нет"
    return "\n".join([
        "🍌 *Карточка Banana*",
        "",
        f"🧩 Фото: *{len(imgs)}/4*   •   Промпт: *{have_prompt}*",
        "",
        "✍️ *Промпт:*",
        f"`{prompt or '—'}`",
        "",
        "💡 Примеры: *поменяй фон на вечер*, *смени одежду на чёрный пиджак*, *добавь лёгкий макияж*",
    ])

def card_keyboard_veo(s: Dict[str, Any]) -> InlineKeyboardMarkup:
    rows: List[List[InlineKeyboardButton]] = []
    rows.append([InlineKeyboardButton("🖼️ Добавить/Удалить фото", callback_data="card:toggle_photo"),
                 InlineKeyboardButton("✍️ Изменить промпт",       callback_data="card:edit_prompt")])
    rows.append([InlineKeyboardButton("🧠 Prompt-Master (помочь с текстом)", callback_data="mode:prompt_master")])
    rows.append(aspect_row(s.get("aspect") or "16:9"))
    rows.append(model_row(s.get("model") or "veo3_fast"))
    rows.append([InlineKeyboardButton("🚀 Сгенерировать", callback_data="card:generate")])
    rows.append([InlineKeyboardButton("🔁 Начать заново", callback_data="card:reset"),
                 InlineKeyboardButton("⬅️ Назад",         callback_data="back")])
    rows.append([InlineKeyboardButton("💳 Пополнить баланс", callback_data="topup_open")])
    return InlineKeyboardMarkup(rows)

def banana_ready_kb(s: Dict[str, Any]) -> InlineKeyboardMarkup:
    imgs = s.get("banana_images") or []
    has_prompt = bool((s.get("banana_prompt") or "").strip())
    rows: List[List[InlineKeyboardButton]] = []
    if len(imgs) < 4:
        rows.append([InlineKeyboardButton("➕ Добавить ещё фото", callback_data="banana:hint_addphoto")])
    else:
        rows.append([InlineKeyboardButton("🧰 Фото загружены (4/4)", callback_data="noop")])
    rows.append([InlineKeyboardButton("🧹 Очистить фото", callback_data="banana:clear")])
    rows.append([InlineKeyboardButton("✍️ " + ("Добавить промпт" if not has_prompt else "Изменить промпт"),
                                      callback_data="banana:askprompt")])
    rows.append([InlineKeyboardButton("🚀 Начать генерацию Banana", callback_data="banana:start")])
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="back")])
    return InlineKeyboardMarkup(rows)

def mj_start_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton("🌆 Сгенерировать 16:9", callback_data="mj:ar:16:9")]])

# ==========================
#   HTTP helpers
# ==========================
def join_url(base: str, path: str) -> str:
    u = f"{base.rstrip('/')}/{path.lstrip('/')}"
    return u.replace("://", "§§").replace("//", "/").replace("§§", "://")

def _kie_headers_json() -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    token = (KIE_API_KEY or "").strip()
    if token and not token.lower().startswith("bearer "):
        token = f"Bearer {token}"
    if token:
        h["Authorization"] = token
    return h

def _post_json(url: str, payload: Dict[str, Any], timeout: int = 40) -> Tuple[int, Dict[str, Any]]:
    r = requests.post(url, json=payload, headers=_kie_headers_json(), timeout=timeout)
    try: return r.status_code, r.json()
    except Exception: return r.status_code, {"error": r.text}

def _get_json(url: str, params: Dict[str, Any], timeout: int = 40) -> Tuple[int, Dict[str, Any]]:
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
            if s.startswith("http"):
                urls.append(s)
    if not value: return urls
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("["):
            try:
                arr = json.loads(s)
                if isinstance(arr, list):
                    for v in arr:
                        if isinstance(v, str): add(v)
                return urls
            except Exception:
                add(s); return urls
        else:
            add(s); return urls
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
    return urls

def _extract_result_url(data: Dict[str, Any]) -> Optional[str]:
    for key in ("originUrls", "resultUrls", "videoUrls"):
        urls = _coerce_url_list(data.get(key))
        if urls:
            return urls[0]
    for container in ("info", "response", "resultInfoJson"):
        v = data.get(container)
        if isinstance(v, dict):
            for key in ("originUrls", "resultUrls", "videoUrls"):
                urls = _coerce_url_list(v.get(key))
                if urls:
                    return urls[0]
    def walk(x):
        if isinstance(x, dict):
            for vv in x.values():
                r = walk(vv); 
                if r: return r
        elif isinstance(x, list):
            for vv in x:
                r = walk(vv); 
                if r: return r
        elif isinstance(x, str):
            s = x.strip().split("?")[0].lower()
            if s.startswith("http") and s.endswith((".mp4",".mov",".webm",".jpg",".png",".jpeg",".webp")):
                return x.strip()
        return None
    return walk(data)

# ---------- Upload helpers ----------
def _upload_headers() -> Dict[str, str]:
    tok = (KIE_API_KEY or "").strip()
    if tok and not tok.lower().startswith("bearer "):
        tok = f"Bearer {tok}"
    return {"Authorization": tok} if tok else {}

def upload_local_file(local_path: str, upload_path: str = "tg-uploads", timeout: int = 120) -> Optional[str]:
    """
    Загружает локальный файл на UPLOAD_BASE_URL и возвращает публичный URL.
    """
    try:
        url = join_url(UPLOAD_BASE_URL, UPLOAD_STREAM_PATH)
        with open(local_path, "rb") as f:
            files = {"file": (os.path.basename(local_path), f)}
            data  = {"uploadPath": upload_path, "fileName": os.path.basename(local_path)}
            r = requests.post(url, headers=_upload_headers(), files=files, data=data, timeout=timeout)
        try:
            j = r.json()
        except Exception:
            j = {"error": r.text}
        if r.status_code == 200 and (j.get("code", 200) == 200):
            d = j.get("data") or {}
            return d.get("downloadUrl") or d.get("fileUrl")
    except Exception as e:
        log.warning("upload_local_file error: %s", e)
    return None

async def upload_tg_file_to_public(bot: Bot, file_id: str) -> Optional[str]:
    """
    Скачивает файл из Telegram на диск и загружает его на наш аплоадер.
    Это устраняет 404 при прямом обращении KIE к telegram.org/file/... .
    """
    try:
        tgf = await bot.get_file(file_id)
        suffix = ".jpg"
        if tgf.file_path and tgf.file_path.lower().endswith(".png"):
            suffix = ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            local = tmp.name
        await tgf.download_to_drive(local)
        url = await asyncio.to_thread(upload_local_file, local, "banana")
        try:
            os.unlink(local)
        except Exception:
            pass
        return url
    except Exception as e:
        log.warning("upload_tg_file_to_public failed: %s", e)
        return None

# ==========================
#   VEO (как раньше)
# ==========================
def _build_payload_for_veo(prompt: str, aspect: str, image_url: Optional[str], model_key: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "prompt": prompt,
        "aspectRatio": "9:16" if aspect == "9:16" else "16:9",
        "model": "veo3" if model_key == "veo3" else "veo3_fast",
        "enableFallback": aspect == "16:9",
    }
    if image_url: payload["imageUrls"] = [image_url]
    return payload

def submit_kie_veo(prompt: str, aspect: str, image_url: Optional[str], model_key: str) -> Tuple[bool, Optional[str], str]:
    payload = _build_payload_for_veo(prompt, aspect, image_url, model_key)
    status, j = _post_json(join_url(KIE_BASE_URL, KIE_VEO_GEN_PATH), payload)
    code = j.get("code", status)
    if status == 200 and code == 200:
        tid = _extract_task_id(j)
        if tid: return True, tid, "Задача создана."
        return False, None, "Ответ KIE без taskId."
    msg = (j.get("msg") or j.get("message") or j.get("error") or "")
    return False, None, f"KIE error code={code}: {msg}"

def get_kie_veo_status(task_id: str) -> Tuple[bool, Optional[int], Optional[str], Optional[str]]:
    status, j = _get_json(join_url(KIE_BASE_URL, KIE_VEO_STATUS_PATH), {"taskId": task_id})
    code = j.get("code", status)
    if status == 200 and code == 200:
        data = (j.get("data") or {})
        flag = data.get("successFlag")
        try: flag = int(flag)
        except Exception: flag = None
        msg = j.get("msg") or j.get("message")
        return True, flag, msg, _extract_result_url(data)
    return False, None, f"KIE status error code={code}", None

async def send_video_with_fallback(ctx: ContextTypes.DEFAULT_TYPE, chat_id: int, url: str) -> bool:
    try:
        await ctx.bot.send_video(chat_id=chat_id, video=url, supports_streaming=True)
        return True
    except Exception:
        pass
    await ctx.bot.send_message(chat_id, f"🔗 Результат готов: {url}")
    return True

async def poll_veo_and_send(chat_id: int, task_id: str, ctx: ContextTypes.DEFAULT_TYPE, expect_vertical: bool):
    s = state(ctx)
    s["generating"] = True
    s["last_task_id"] = task_id
    start_ts = time.time()
    while True:
        ok, flag, msg, res_url = await asyncio.to_thread(get_kie_veo_status, task_id)
        if not ok:
            await ctx.bot.send_message(chat_id, f"❌ Ошибка статуса: {msg or 'неизвестно'}"); break
        if res_url:
            await ctx.bot.send_message(chat_id, "🎞️ Рендер завершён — отправляю файл…")
            await send_video_with_fallback(ctx, chat_id, res_url)
            await ctx.bot.send_message(chat_id, "✅ Готово!",
                                       reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🚀 Сгенерировать ещё", callback_data="start_new_cycle")]]))
            break
        if flag in (2, 3):
            await ctx.bot.send_message(chat_id, f"❌ Ошибка генерации: {msg or 'нет'}"); break
        if (time.time() - start_ts) > POLL_TIMEOUT_SECS:
            await ctx.bot.send_message(chat_id, "⌛ Превышено время ожидания VEO."); break
        await asyncio.sleep(POLL_INTERVAL_SECS)
    s["generating"] = False

# ==========================
#   MJ (как раньше)
# ==========================
def mj_generate(prompt: str) -> Tuple[bool, Optional[str], str]:
    payload = {
        "taskType": "mj_txt2img",
        "prompt": prompt,
        "speed": "turbo",
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
    return False, None, f"MJ error: code={code}"

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
    res = []
    rj = status_data.get("resultInfoJson") or {}
    arr = rj.get("resultUrls") or []
    urls = _coerce_url_list(arr)
    for u in urls:
        if isinstance(u, str) and u.startswith("http"):
            res.append(u)
    return res

async def poll_mj_and_send_photos(chat_id: int, task_id: str, ctx: ContextTypes.DEFAULT_TYPE):
    s = state(ctx)
    start_ts = time.time()
    while True:
        ok, flag, data = await asyncio.to_thread(mj_status, task_id)
        if not ok:
            await ctx.bot.send_message(chat_id, "❌ MJ сейчас недоступен."); return
        if flag == 0:
            now = time.time()
            if not s.get("mj_wait_sent") or (now - s.get("mj_wait_last_ts", 0)) >= 40:
                s["mj_wait_sent"] = True
                s["mj_wait_last_ts"] = now
                await ctx.bot.send_message(chat_id, f"🌀 Идёт генерация… ✨")
            if (now - start_ts) > 15*60:
                await ctx.bot.send_message(chat_id, "⌛ MJ долго не отвечает. Попробуйте позже.")
                add_tokens(ctx, TOKEN_COSTS["mj"])
                return
            await asyncio.sleep(6)
            continue
        if flag in (2, 3):
            msg = (data or {}).get("errorMessage") or "No response from MidJourney."
            await ctx.bot.send_message(chat_id, f"❌ MJ: {msg}\n💎 Токены возвращены.")
            add_tokens(ctx, TOKEN_COSTS["mj"])
            return
        if flag == 1:
            urls = _extract_mj_image_urls(data or {})
            if not urls:
                await ctx.bot.send_message(chat_id, "⚠️ Готово, но ссылки не найдены.\n💎 Токены возвращены.")
                add_tokens(ctx, TOKEN_COSTS["mj"]); return
            if len(urls) == 1:
                await ctx.bot.send_photo(chat_id=chat_id, photo=urls[0])
            else:
                media = [InputMediaPhoto(u) for u in urls[:10]]
                await ctx.bot.send_media_group(chat_id=chat_id, media=media)
            await ctx.bot.send_message(chat_id, "✅ *Готово!*", parse_mode=ParseMode.MARKDOWN,
                                       reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🚀 Сгенерировать ещё", callback_data="start_new_cycle")]]))
            return

# ==========================
#   Banana (исправлено)
# ==========================
def banana_generate(prompt: str, image_urls: List[str]) -> Tuple[bool, Optional[str], str]:
    """
    Если есть фото → google/nano-banana-edit (image_urls),
    если нет фото → google/nano-banana (без image_urls).
    Тут мы предполагаем, что image_urls уже публичные (мы их сделали через upload_tg_file_to_public).
    """
    prompt = (prompt or "").strip()
    src_urls: List[str] = [u for u in (image_urls or []) if isinstance(u, str) and u.startswith("http")]
    use_model = "google/nano-banana-edit" if src_urls else "google/nano-banana"

    input_obj: Dict[str, Any] = {
        "prompt": prompt,
        "output_format": "png",
        "image_size": "auto",
    }
    if src_urls:
        input_obj["image_urls"] = src_urls[:4]

    payload = {"model": use_model, "input": input_obj}
    status, j = _post_json(join_url(KIE_BASE_URL, KIE_BANANA_GENERATE), payload)
    code = j.get("code", status)
    if status == 200 and code == 200:
        tid = _extract_task_id(j)
        if tid: return True, tid, f"Banana ({use_model}) задача создана."
        return False, None, "Ответ без taskId."
    msg = j.get("msg") or j.get("message") or j.get("error") or ""
    return False, None, f"Banana error code={code}: {msg}"

def banana_status(task_id: str) -> Tuple[bool, Optional[int], Optional[Dict[str, Any]]]:
    status, j = _get_json(join_url(KIE_BASE_URL, KIE_BANANA_STATUS), {"taskId": task_id})
    code = j.get("code", status)
    if status == 200 and code == 200:
        data = j.get("data") or {}
        try: flag = int(data.get("successFlag"))
        except Exception: flag = None
        return True, flag, data
    return False, None, None

def _banana_result_urls(data: Dict[str, Any]) -> List[str]:
    urls = []
    d = data.get("data") or data
    rj = d.get("resultJson") or d.get("resultInfoJson") or {}
    try:
        if isinstance(rj, str):
            rj = json.loads(rj)
    except Exception:
        rj = {}
    arr = (rj.get("resultUrls") or [])
    return _coerce_url_list(arr)

async def poll_banana_and_send(chat_id: int, task_id: str, ctx: ContextTypes.DEFAULT_TYPE):
    start_ts = time.time()
    while True:
        ok, flag, data = await asyncio.to_thread(banana_status, task_id)
        if not ok:
            await ctx.bot.send_message(chat_id, "❌ Banana сейчас недоступен.")
            add_tokens(ctx, TOKEN_COSTS["banana"])
            return
        if flag == 0:
            await asyncio.sleep(6); continue
        if flag in (2, 3):
            await ctx.bot.send_message(chat_id, "❌ Banana: ошибка генерации.\n💎 Токены возвращены.")
            add_tokens(ctx, TOKEN_COSTS["banana"]); return
        if flag == 1:
            urls = _banana_result_urls({"data": data})
            if not urls:
                await ctx.bot.send_message(chat_id, "⚠️ Готово, но ссылки не найдены.\n💎 Токены возвращены.")
                add_tokens(ctx, TOKEN_COSTS["banana"]); return
            if len(urls) == 1:
                await ctx.bot.send_photo(chat_id=chat_id, photo=urls[0])
            else:
                media = [InputMediaPhoto(u) for u in urls[:10]]
                await ctx.bot.send_media_group(chat_id=chat_id, media=media)
            await ctx.bot.send_message(chat_id, "✅ *Готово!*", parse_mode=ParseMode.MARKDOWN,
                                       reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🚀 Сгенерировать ещё", callback_data="start_new_cycle")]]))
            return
        if (time.time() - start_ts) > 15*60:
            await ctx.bot.send_message(chat_id, "⌛ Banana долго не отвечает. Попробуйте позже.\n💎 Токены возвращены.")
            add_tokens(ctx, TOKEN_COSTS["banana"]); return

# ==========================
#   Prompt-Master (как раньше)
# ==========================
async def run_prompt_master(raw: str) -> str:
    base = (raw or "").strip()
    if not base:
        return ""
    sys_msg = (
        "You are a senior cinematic prompt writer for Google Veo 3. "
        "Rewrite the user's idea into ONE compact, production-ready, English prompt. "
        "Include: setting/location & time, mood, lighting, camera movement, composition, lens/focal length, "
        "color grading, key action/beat, and optional one short line of dialogue in quotes if helpful. "
        "Avoid meta-instructions, avoid placeholders, no bullet lists — return a single paragraph (3–6 sentences)."
    )
    if openai and OPENAI_API_KEY:
        try:
            resp = await asyncio.to_thread(
                openai.ChatCompletion.create,
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": base},
                ],
                temperature=0.7,
                max_tokens=500,
            )
            out = (resp["choices"][0]["message"]["content"] or "").strip()
            if out:
                return out
        except Exception as e:
            log.warning("PromptMaster OpenAI error: %s", e)
    return (
        f"Cinematic scene: {base}. Moody, natural light with soft contrast. "
        "Wide establishing push-in, 35mm lens, shallow depth of field, balanced composition, "
        "subtle handheld energy, teal-and-orange grade, 4K, filmic grain."
    )

# ==========================
#   Handlers
# ==========================
FAQ_TEXT = (
    "📖 *FAQ — Вопросы и ответы*\n\n"
    "⭐ *Общая инфа*\n"
    "Что умеет бот?\n"
    "Генерация видео (VEO), картинок (MJ), редактирование фото (Banana), подсказки для идей (Prompt-Master) и чат (ChatGPT).\n\n"
    "Звёзды (💎) — внутренняя валюта бота. Баланс видно в меню. Покупаются через Telegram.\n\n"
    "— Нужна ли регистрация/VPN? *Нет*, всё работает прямо в Telegram.\n"
    "— Если бот молчит: проверьте баланс, нажмите /start и попробуйте снова.\n"
    "— Можно ли использовать результаты в коммерции? Обычно да, вы автор промпта.\n\n"
    "⸻\n\n"
    "🎬 *VEO (Видео)*\n"
    f"• *Fast* — быстрый ролик, 2–5 мин. Стоимость: 💎{TOKEN_COSTS['veo_fast']}.\n"
    f"• *Quality* — высокое качество, 5–10 мин. Стоимость: 💎{TOKEN_COSTS['veo_quality']}.\n"
    f"• *Animate* — оживление фото. Стоимость: 💎{TOKEN_COSTS['veo_photo']}.\n"
    "👉 Опишите идею (локация, стиль, настроение) и ждите готовый клип.\n\n"
    "⸻\n\n"
    "🖼️ *MJ (Картинки)*\n"
    f"• Стоимость: 💎{TOKEN_COSTS['mj']}.\n"
    "• Время: 30–90 сек.\n"
    "• Формат: *только 16:9*.\n"
    "👉 Чем детальнее промпт (цвет, свет, стиль), тем лучше результат.\n\n"
    "⸻\n\n"
    "🍌 *Banana (Редактор фото)*\n"
    f"• Стоимость: 💎{TOKEN_COSTS['banana']}.\n"
    "• Сначала пришлите *до 4 фото*, потом текст-промпт.\n"
    "• После загрузки фото бот пишет «📸 Фото принято (n/4)».\n"
    "• Генерация только после нажатия «🚀 Начать».\n\n"
    "⸻\n\n"
    "🧠 *Prompt-Master*\n"
    "• Бесплатно.\n"
    "• Опишите идею: локация, стиль/настроение, свет, камера, действие, реплики в кавычках.\n"
    "• Получите готовый кинопромпт.\n\n"
    "⸻\n\n"
    "💬 *ChatGPT*\n"
    "• Бесплатно. Общение и ответы на вопросы.\n\n"
    "Если вопроса нет в списке — просто напишите сюда, я помогу."
)

def stars_topup_kb() -> InlineKeyboardMarkup:
    rows: List[List[InlineKeyboardButton]] = []
    for stars, diamonds, note in STAR_PACKS:
        note_txt = f" {note}" if note else ""
        cap = f"⭐ {stars} → 💎 {diamonds}{note_txt}"
        rows.append([InlineKeyboardButton(cap, callback_data=f"buy:stars:{stars}:{diamonds}")])
    rows.append([InlineKeyboardButton("🛒 Где купить Stars", url=STARS_BUY_URL)])
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="back")])
    return InlineKeyboardMarkup(rows)

async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    ensure_signup_bonus(ctx, update.effective_user.id)
    s = state(ctx); s.update({**DEFAULT_STATE})
    await update.message.reply_text(render_welcome_for(update.effective_user.id, ctx),
                                    parse_mode=ParseMode.MARKDOWN,
                                    reply_markup=main_menu_kb())

async def topup(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "💳 Пополнение токенов через *Telegram Stars*.\n"
        f"Если не хватает звёзд — купите их в {STARS_BUY_URL}",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=stars_topup_kb()
    )

async def on_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    data = (query.data or "").strip()
    await query.answer()
    s = state(ctx)

    if data == "faq":
        await query.message.reply_text(FAQ_TEXT, parse_mode=ParseMode.MARKDOWN, reply_markup=main_menu_kb()); return
    if data == "back":
        s.update({**DEFAULT_STATE}); await query.message.reply_text("🏠 Главное меню:", reply_markup=main_menu_kb()); return
    if data == "start_new_cycle":
        s.update({**DEFAULT_STATE}); await query.message.reply_text("Выберите режим:", reply_markup=main_menu_kb()); return
    if data == "topup_open":
        await query.message.reply_text(
            f"💳 Пополнение через Telegram Stars. Если звёзд не хватает — купите в {STARS_BUY_URL}",
            reply_markup=stars_topup_kb()
        ); return
    if data == "noop":
        return

    if data.startswith("buy:stars:"):
        parts = data.split(":")
        stars = int(parts[2]); diamonds = int(parts[3])
        title = f"{stars}⭐ → {diamonds}💎"
        payload = json.dumps({"kind": "stars_pack", "stars": stars, "tokens": diamonds})
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
        except Exception:
            await query.message.reply_text(
                f"Если счёт не открылся — у аккаунта могут быть не активированы Stars.\n"
                f"Купите 1⭐ в {STARS_BUY_URL} и попробуйте снова.",
                reply_markup=stars_topup_kb()
            )
        return

    if data.startswith("mode:"):
        _, mode = data.split(":", 1); s["mode"] = mode
        if mode == "veo_text_fast":
            s["aspect"] = "16:9"; s["model"] = "veo3_fast"
            await query.message.reply_text("📝 Пришлите идею/промпт для видео (VEO Fast)."); return
        if mode == "veo_text_quality":
            s["aspect"] = "16:9"; s["model"] = "veo3"
            await query.message.reply_text("📝 Пришлите идею/промпт для видео (VEO Quality)."); return
        if mode == "veo_photo":
            s["aspect"] = "9:16"; s["model"] = "veo3_fast"
            await query.message.reply_text("🖼️ Пришлите фото (подпись-промпт — по желанию)."); return
        if mode == "mj_txt":
            s["aspect"] = "16:9"; s["last_prompt"] = None
            await query.message.reply_text("🖼️ Пришлите текстовый prompt для картинки (16:9)."); return
        if mode == "banana":
            s["banana_images"] = []; s["banana_prompt"] = None; s["banana_task_id"] = None; s["banana_wait_prompt"] = False
            await show_card_banana(update, ctx); return
        if mode == "prompt_master":
            await query.message.reply_text("🧠 *Prompt-Master готов!* Коротко опишите идею сцены — сделаю проф. кинопромпт.",
                                           parse_mode=ParseMode.MARKDOWN); return
        if mode == "chat":
            await query.message.reply_text("✍️ Напишите вопрос для ChatGPT."); return

    if data == "mj:ar:16:9":
        if not s.get("last_prompt"):
            await query.message.reply_text("✍️ Сначала пришлите текстовый prompt для картинки (16:9)."); return
        price = TOKEN_COSTS["mj"]
        ok_balance, rest = try_charge(ctx, price)
        if not ok_balance:
            await query.message.reply_text(
                f"💎 Недостаточно токенов: нужно {price}, на балансе *{rest}*.\n"
                f"Пополните через Stars: {STARS_BUY_URL}",
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=stars_topup_kb()
            ); return
        ok, task_id, msg = await asyncio.to_thread(mj_generate, s["last_prompt"].strip())
        if not ok or not task_id:
            add_tokens(ctx, price)
            await query.message.reply_text(f"❌ Не удалось создать MJ-задачу: {msg}\n💎 Токены возвращены."); return
        await query.message.reply_text("🖼️ MJ запущен\n🌀 Идёт генерация… ✨")
        asyncio.create_task(poll_mj_and_send_photos(update.effective_chat.id, task_id, ctx)); return

    if data.startswith("aspect:"):
        _, val = data.split(":", 1); s["aspect"] = "9:16" if val.strip() == "9:16" else "16:9"
        await query.message.reply_text("✅ Aspect обновлён."); return
    if data.startswith("model:"):
        _, val = data.split(":", 1); s["model"] = "veo3" if val.strip() == "veo3" else "veo3_fast"
        await query.message.reply_text("✅ Mode обновлён."); return

    if data == "card:toggle_photo":
        if s.get("last_image_url"):
            s["last_image_url"] = None; await query.message.reply_text("🧹 Фото-референс удалён.")
        else:
            await query.message.reply_text("📎 Пришлите фото вложением или публичный URL изображения.")
        return
    if data == "card:edit_prompt":
        await query.message.reply_text("✍️ Пришлите новый текст промпта."); return
    if data == "card:reset":
        keep_aspect = s.get("aspect") or "16:9"; keep_model = s.get("model") or "veo3_fast"
        s.update({**DEFAULT_STATE}); s["aspect"] = keep_aspect; s["model"] = keep_model
        await query.message.reply_text("🗂️ Карточка очищена."); return
    if data == "card:generate":
        if s.get("generating"): await query.message.reply_text("⏳ Уже рендерю это видео — подождите."); return
        if not s.get("last_prompt"): await query.message.reply_text("✍️ Сначала укажите текст промпта."); return
        price = TOKEN_COSTS['veo_quality'] if s.get('model') == 'veo3' else TOKEN_COSTS['veo_fast']
        ok_balance, rest = try_charge(ctx, price)
        if not ok_balance:
            await query.message.reply_text(
                f"💎 Недостаточно токенов: нужно {price}, на балансе *{rest}*.\n"
                f"Пополните через Stars: {STARS_BUY_URL}", parse_mode=ParseMode.MARKDOWN, reply_markup=stars_topup_kb()
            ); return
        ok, task_id, msg = await asyncio.to_thread(
            submit_kie_veo, s["last_prompt"].strip(), s.get("aspect", "16:9"), s.get("last_image_url"), s.get("model", "veo3_fast")
        )
        if not ok or not task_id:
            add_tokens(ctx, price)
            await query.message.reply_text(f"❌ Не удалось создать VEO-задачу: {msg}\n💎 Токены возвращены."); return
        s["generating"] = True; s["last_task_id"] = task_id
        await query.message.reply_text(f"🚀 Задача отправлена. ⏳ Идёт процесс… {datetime.now().strftime('%H:%M:%S')}")
        asyncio.create_task(poll_veo_and_send(update.effective_chat.id, task_id, ctx, expect_vertical=(s.get("aspect") == "9:16"))); return

    if data == "banana:askprompt":
        s["banana_wait_prompt"] = True
        await query.message.reply_text("✍️ Пришлите текст-промпт для Banana (что изменить на фото)."); return
    if data == "banana:hint_addphoto":
        await query.message.reply_text("➕ Пришлите до 4 фото одним сообщением. Я посчитаю: (n/4)."); return
    if data == "banana:start":
        imgs = s.get("banana_images") or []
        prompt = (s.get("banana_prompt") or "").strip()
        if not imgs:
            await query.message.reply_text("⚠️ Сначала добавьте хотя бы одно фото (до 4).", reply_markup=banana_ready_kb(s)); return
        if not prompt:
            await query.message.reply_text("⚠️ Добавьте текст-промпт (что изменить на фото).", reply_markup=banana_ready_kb(s)); return
        price = TOKEN_COSTS["banana"]
        ok_balance, rest = try_charge(ctx, price)
        if not ok_balance:
            await query.message.reply_text(
                f"💎 Недостаточно токенов: нужно {price}, на балансе *{rest}*.\n"
                f"Пополните через Stars: {STARS_BUY_URL}", parse_mode=ParseMode.MARKDOWN, reply_markup=stars_topup_kb()
            ); return
        ok, task_id, msg = await asyncio.to_thread(banana_generate, prompt, imgs)
        if not ok or not task_id:
            add_tokens(ctx, price)
            await query.message.reply_text(f"❌ Не удалось создать Banana-задачу: {msg}\n💎 Токены возвращены."); return
        s["banana_task_id"] = task_id
        await query.message.reply_text("🍌 Banana запущен\n🌀 Идёт генерация… ✨")
        asyncio.create_task(poll_banana_and_send(update.effective_chat.id, task_id, ctx)); return

    if data == "banana:clear":
        s["banana_images"] = []; s["banana_prompt"] = None; s["banana_task_id"] = None
        await query.message.reply_text("🧹 Сессия Banana очищена.", reply_markup=banana_ready_kb(s)); return

async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = state(ctx)
    text = (update.message.text or "").strip()
    mode = s.get("mode")

    if mode == "banana":
        if text:
            s["banana_prompt"] = text[:1000]
            s["banana_wait_prompt"] = False
            await update.message.reply_text("✅ Промпт для Banana сохранён.", reply_markup=banana_ready_kb(s))
            await update.message.reply_text(build_card_text_banana(s), parse_mode=ParseMode.MARKDOWN)
        return

    if mode == "prompt_master":
        await update.message.reply_text("🧠 Генерирую кинопромпт…")
        pm = await run_prompt_master(text)
        if not pm:
            await update.message.reply_text("⚠️ Не получилось сформировать промпт. Напишите идею ещё раз.")
            return
        s["last_prompt"] = pm[:2000]
        await update.message.reply_text(
            "✅ Готово! Вот кинопромпт, я добавил его в карточку:\n\n"
            f"`{s['last_prompt']}`",
            parse_mode=ParseMode.MARKDOWN,
            disable_web_page_preview=True
        )
        return

    if mode == "mj_txt":
        s["last_prompt"] = text[:1000]
        await update.message.reply_text(
            f"✅ Prompt сохранён:\n\n`{s['last_prompt']}`\n\nНажмите ниже для запуска 16:9:",
            parse_mode=ParseMode.MARKDOWN, reply_markup=mj_start_kb()
        ); return

    if mode == "chat":
        if openai is None or not OPENAI_API_KEY:
            await update.message.reply_text("⚠️ ChatGPT недоступен (нет OPENAI_API_KEY)."); return
        try:
            await update.message.reply_text("💬 Думаю над ответом…")
            resp = await asyncio.to_thread(
                openai.ChatCompletion.create,
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": "You are a helpful, concise assistant."},
                          {"role": "user", "content": text}],
                temperature=0.5, max_tokens=700,
            )
            answer = resp["choices"][0]["message"]["content"].strip()
            await update.message.reply_text(answer)
        except Exception as e:
            log.exception("Chat error: %s", e)
            await update.message.reply_text("⚠️ Ошибка запроса к ChatGPT.")
        return

    s["last_prompt"] = text[:2000]
    await update.message.reply_text("🟦 *VEO — подготовка к рендеру*\nПроверь карточку и жми «Сгенерировать».",
                                    parse_mode=ParseMode.MARKDOWN)

async def on_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = state(ctx)
    photos = update.message.photo
    if not photos: return
    ph = photos[-1]
    # 👉 КЛЮЧЕВОЙ ФИКС: сразу делаем публичный URL через наш аплоадер
    public_url = await upload_tg_file_to_public(ctx.bot, ph.file_id)
    if not public_url:
        await update.message.reply_text("⚠️ Не удалось сохранить фото. Отправьте ещё раз.")
        return

    if s.get("mode") == "banana":
        imgs = s.get("banana_images") or []
        if len(imgs) >= 4:
            await update.message.reply_text("⚠️ Принято уже 4 фото. Нажмите «🚀 Начать генерацию».", reply_markup=banana_ready_kb(s)); return
        imgs.append(public_url); s["banana_images"] = imgs
        await update.message.reply_text(f"📸 Фото принято ({len(imgs)}/4).", reply_markup=banana_ready_kb(s))
        if not (s.get("banana_prompt") or "").strip() and not (update.message.caption or "").strip():
            await update.message.reply_text(
                "✍️ Теперь пришлите *текст-промпт*, что сделать с фото "
                "(например: «поменяй фон на вечер», «смени одежду на чёрный пиджак», "
                "«добавь лёгкий макияж»).",
                parse_mode=ParseMode.MARKDOWN
            )
        if (update.message.caption or "").strip() and not (s.get("banana_prompt") or "").strip():
            s["banana_prompt"] = (update.message.caption or "").strip()[:1000]
            await update.message.reply_text("✅ Промпт для Banana сохранён из подписи.", reply_markup=banana_ready_kb(s))
        await update.message.reply_text(build_card_text_banana(s), parse_mode=ParseMode.MARKDOWN)
        return

    # VEO / др.: используем этот же публичный URL как референс (без изменений остальной логики)
    s["last_image_url"] = public_url
    if update.message.caption:
        s["last_prompt"] = (update.message.caption or "").strip()
    await update.message.reply_text("🖼️ Фото принято как референс.")

# ==========================
#   Payments: Stars
# ==========================
async def precheckout_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        await update.pre_checkout_query.answer(ok=True)
    except Exception:
        await update.pre_checkout_query.answer(ok=False, error_message=f"Платёж отклонён. Пополните Stars в {STARS_BUY_URL}")

async def successful_payment_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    sp = update.message.successful_payment
    try:
        meta = json.loads(sp.invoice_payload)
    except Exception:
        meta = {}
    stars = int(sp.total_amount)
    if meta.get("kind") == "stars_pack":
        tokens = int(meta.get("tokens", 0))
        if tokens <= 0:
            mapv = {s: d for (s, d, _) in STAR_PACKS}
            tokens = mapv.get(stars, stars)
        add_tokens(ctx, tokens)
        await update.message.reply_text(
            f"✅ Оплата получена: +*{tokens}* токенов.\nБаланс: *{get_user_balance_value(ctx)}* 💎",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    await update.message.reply_text("✅ Оплата получена.")

# ==========================
#   Promo
# ==========================
async def promo_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    args = (update.message.text or "").split()
    if len(args) < 2:
        await update.message.reply_text("Введите код: `/promo MYCODE`", parse_mode=ParseMode.MARKDOWN)
        return
    code = args[1].strip().upper()
    uid = update.effective_user.id
    amount = PROMO_CODES.get(code, 0)
    if amount <= 0:
        await update.message.reply_text("🚫 Неверный промокод.")
        return
    used_key = _rk("promo_used", str(uid), code)
    if redis_client and redis_client.get(used_key):
        await update.message.reply_text("⚠️ Этот промокод уже был активирован.")
        return
    add_tokens(ctx, amount)
    if redis_client:
        redis_client.set(used_key, "1")
    await update.message.reply_text(
        f"✅ Промокод активирован!\n+*{amount}* 💎\nВаш баланс: *{get_user_balance_value(ctx)}* 💎",
        parse_mode=ParseMode.MARKDOWN
    )

# ==========================
#   Health / Errors / Entry
# ==========================
async def health(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    parts = [
        f"KIE_BASE: `{KIE_BASE_URL}`",
        f"BANANA_GEN: `{KIE_BANANA_GENERATE}`",
        f"Redis: `{'on' if redis_client else 'off'}`",
        f"Balance: *{get_user_balance_value(ctx)}* 💎",
    ]
    await update.message.reply_text("🩺 *Health*\n" + "\n".join(parts), parse_mode=ParseMode.MARKDOWN)

async def error_handler(update: Optional[Update], context: ContextTypes.DEFAULT_TYPE):
    log.exception("Unhandled error: %s", context.error)
    try:
        if update and update.effective_chat:
            await context.bot.send_message(update.effective_chat.id, "⚠️ Системная ошибка. Попробуйте ещё раз.")
    except Exception:
        pass

def main():
    if not TELEGRAM_TOKEN: raise RuntimeError("TELEGRAM_TOKEN is not set")
    try:
        Bot(TELEGRAM_TOKEN).delete_webhook(drop_pending_updates=True)
    except Exception:
        pass

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).rate_limiter(AIORateLimiter()).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("topup", topup))
    app.add_handler(CommandHandler("health", health))
    app.add_handler(CommandHandler("promo", promo_cmd))
    app.add_handler(PreCheckoutQueryHandler(precheckout_callback))
    app.add_handler(MessageHandler(filters.SUCCESSFUL_PAYMENT, successful_payment_handler))
    app.add_handler(CallbackQueryHandler(on_callback))
    app.add_handler(MessageHandler(filters.PHOTO, on_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_error_handler(error_handler)

    log.info("Bot starting…")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
