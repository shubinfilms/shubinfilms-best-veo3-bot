# -*- coding: utf-8 -*-
# Best VEO3 Bot — PTB 21.x
# Версия: 2025-09-13
#
# Включено:
# • Redis-баланс/Stars; бонус новичка; логгирование.
# • VEO (Fast/Quality) с 16:9/9:16, локальные ffmpeg-нормализации, 1080p-фетч.
# • MidJourney txt2img (антиспам «Рисую…» раз в 40с).
# • Prompt-Master (OpenAI) — формирует кинопромпт.
# • Banana (google/nano-banana-edit) до 4 фото:
#   - единая карточка (без дубля),
#   - перезалив TG-файлов на стабильный CDN (устраняет 404/422),
#   - строгий input по докам.
# • Минимум изменений вне Banana относительно прежней логики.
#
import os
import json
import time
import uuid
import math
import asyncio
import logging
import tempfile
import subprocess
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

# --- KIE Banana wrapper ---
from kie_banana import create_banana_task, wait_for_banana_result, KieBananaError

# --- Redis (баланс/флаги) ---
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

# OpenAI (Prompt-Master)
OPENAI_API_KEY = _env("OPENAI_API_KEY")
try:
    import openai  # type: ignore
    if OPENAI_API_KEY:
        openai.api_key = OPENAI_API_KEY
except Exception:
    openai = None

# ---- KIE: VEO / MJ / Upload
KIE_API_KEY  = _env("KIE_API_KEY")
KIE_BASE_URL = _env("KIE_BASE_URL", "https://api.kie.ai")

KIE_VEO_GEN_PATH    = _env("KIE_VEO_GEN_PATH",    "/api/v1/veo/generate")
KIE_VEO_STATUS_PATH = _env("KIE_VEO_STATUS_PATH", "/api/v1/veo/record-info")
KIE_VEO_1080_PATH   = _env("KIE_VEO_1080_PATH",   "/api/v1/veo/get-1080p-video")

KIE_MJ_GENERATE = _env("KIE_MJ_GENERATE", "/api/v1/mj/generate")
KIE_MJ_STATUS   = _env("KIE_MJ_STATUS",   "/api/v1/mj/record-info")

# Upload API (перезалив TG-картинок → стабильные https-ссылки)
UPLOAD_BASE_URL     = _env("UPLOAD_BASE_URL", "https://kieai.redpandaai.co")
UPLOAD_STREAM_PATH  = _env("UPLOAD_STREAM_PATH", "/api/file-stream-upload")

# ---- Видео-доставка
ENABLE_VERTICAL_NORMALIZE = _env("ENABLE_VERTICAL_NORMALIZE", "true").lower() == "true"
ALWAYS_FORCE_FHD          = _env("ALWAYS_FORCE_FHD", "true").lower() == "true"
FFMPEG_BIN                = _env("FFMPEG_BIN", "ffmpeg")
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

# ---- Redis
REDIS_URL    = _env("REDIS_URL")
REDIS_PREFIX = _env("REDIS_PREFIX", "veo3:prod")
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True) if REDIS_URL else None

def _rk(*parts: str) -> str:
    return ":".join([REDIS_PREFIX, *parts])

# ==========================
#   Tokens / Pricing
# ==========================
TOKEN_COSTS = {
    "veo_fast": 50,
    "veo_quality": 200,
    "veo_photo": 50,
    "mj": 15,
    "banana": 10,
    "chat": 0,
}
CHAT_UNLOCK_PRICE = 0

# ---------------- Redis-backed balance helpers ----------------
def get_user_id(ctx: ContextTypes.DEFAULT_TYPE) -> Optional[int]:
    try:
        return ctx._user_id_and_data[0]  # type: ignore[attr-defined]
    except Exception:
        return None

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
    if redis_client and uid:
        redis_client.set(_rk("balance", str(uid)), v)

def add_tokens(ctx: ContextTypes.DEFAULT_TYPE, add: int):
    set_user_balance_value(ctx, get_user_balance_value(ctx) + int(add))

def try_charge(ctx: ContextTypes.DEFAULT_TYPE, need: int) -> Tuple[bool, int]:
    bal = get_user_balance_value(ctx)
    if bal < need:
        return False, bal
    set_user_balance_value(ctx, bal - need)
    return True, bal - need

def has_signup_bonus(uid: int) -> bool:
    if not redis_client:
        return False
    return bool(redis_client.get(_rk("signup_bonus", str(uid))))

def set_signup_bonus(uid: int):
    if redis_client:
        redis_client.set(_rk("signup_bonus", str(uid)), "1")

# ==========================
#   Utils / State
# ==========================
def join_url(base: str, path: str) -> str:
    u = f"{base.rstrip('/')}/{path.lstrip('/')}"
    return u.replace("://", "§§").replace("//", "/").replace("§§", "://")

def _nz(s: Optional[str]) -> Optional[str]:
    if not s: return None
    s2 = s.strip()
    return s2 if s2 else None

def event(tag: str, **kw):
    try:
        log.info("EVT %s | %s", tag, json.dumps(kw, ensure_ascii=False))
    except Exception:
        log.info("EVT %s | %s", tag, kw)

def tg_direct_file_url(bot_token: str, file_path: str) -> str:
    p = (file_path or "").strip()
    if p.startswith("http://") or p.startswith("https://"):
        return p
    return f"https://api.telegram.org/file/bot{bot_token}/{p.lstrip('/')}"

def upload_image_stream(tg_url: str) -> Optional[str]:
    """
    Перезаливаем Telegram file URL на стабильный CDN (устраняет 404/422 на стороне KIE).
    Ожидаемый ответ: { url: "https://..." } или data.url / resultUrl
    """
    try:
        url = join_url(UPLOAD_BASE_URL, UPLOAD_STREAM_PATH)
        r = requests.post(url, json={"url": tg_url}, timeout=60)
        j = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        u = j.get("url") or j.get("resultUrl") or j.get("data", {}).get("url")
        if r.status_code == 200 and isinstance(u, str) and u.startswith("http"):
            return u
    except Exception as e:
        log.warning("upload_image_stream failed: %s", e)
    return None

DEFAULT_STATE = {
    "mode": None,          # 'veo_text' | 'veo_photo' | 'prompt_master' | 'chat' | 'mj_txt' | 'banana'
    "aspect": None,        # '16:9' | '9:16'
    "model": None,         # 'veo3_fast' | 'veo3'
    "last_prompt": None,
    "last_image_url": None,
    "generating": False,
    "generation_id": None,
    "last_ui_msg_id": None,
    "last_task_id": None,
    "last_result_url": None,
    "chat_unlocked": True,
    "mj_wait_sent": False,
    "mj_last_wait_ts": 0.0,
    "banana_images": [],     # список URL до 4 штук (публичные)
}

def state(ctx: ContextTypes.DEFAULT_TYPE) -> Dict[str, Any]:
    ud = ctx.user_data
    for k, v in DEFAULT_STATE.items():
        if k not in ud:
            ud[k] = [] if isinstance(v, list) else v
    if not isinstance(ud.get("banana_images"), list):
        ud["banana_images"] = []
    return ud

# ==========================
#   UI
# ==========================
WELCOME = (
    "🎬 Veo 3 — опиши идею и получи готовый клип!\n"
    "🖌️ MJ — рисует по тексту.\n"
    "🍌 Banana — редактор изображений.\n"
    "🧠 Prompt-Master — собирает кинопромпт.\n"
    "💬 Чат — отвечает на вопросы.\n"
    "💎 Баланс: {balance}\n"
    "📈 Примеры: {prompts_url}\n\n"
    "Выберите режим 👇"
)

def render_welcome_for(uid: int, ctx: ContextTypes.DEFAULT_TYPE) -> str:
    return WELCOME.format(balance=get_user_balance_value(ctx), prompts_url=PROMPTS_CHANNEL_URL)

def main_menu_kb() -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton("🎬 Генерация видео (Veo)", callback_data="mode:veo_text")],
        [InlineKeyboardButton("🖼️ Генерация изображений (MJ)", callback_data="mode:mj_txt")],
        [InlineKeyboardButton("🍌 Редактор изображений (Banana)", callback_data="mode:banana")],
        [InlineKeyboardButton("📸 Оживить изображение (Veo)", callback_data="mode:veo_photo")],
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
    model = "Fast" if s.get("model") == "veo3_fast" else ("Quality" if s.get("model") == "veo3" else "—")
    price = TOKEN_COSTS['veo_quality'] if s.get('model') == 'veo3' else TOKEN_COSTS['veo_fast']
    lines = [
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
    ]
    return "\n".join(lines)

def card_keyboard_veo(s: Dict[str, Any]) -> InlineKeyboardMarkup:
    rows: List[List[InlineKeyboardButton]] = []
    rows.append([InlineKeyboardButton("🖼️ Добавить/Удалить фото", callback_data="card:toggle_photo"),
                 InlineKeyboardButton("✍️ Изменить промпт",       callback_data="card:edit_prompt")])
    rows.append(aspect_row(s.get("aspect") or "16:9"))
    rows.append(model_row(s.get("model") or "veo3_fast"))
    rows.append([InlineKeyboardButton("🚀 Сгенерировать", callback_data="card:generate")])
    rows.append([InlineKeyboardButton("🔁 Начать заново", callback_data="card:reset"),
                 InlineKeyboardButton("⬅️ Назад",         callback_data="back")])
    rows.append([InlineKeyboardButton("💳 Пополнить баланс", callback_data="topup_open")])
    return InlineKeyboardMarkup(rows)

# ===== Banana Card =====
def banana_card_text(s: Dict[str, Any]) -> str:
    n = len(s["banana_images"])
    p = (s.get("last_prompt") or "—").strip()
    if len(p) > 700: p = p[:700] + "…"
    return (
        "🍌 *Карточка Banana*\n"
        f"🧩 Фото: *{n}/4*   •   Промпт: *{'есть' if s.get('last_prompt') else 'нет'}*\n\n"
        "✍️ *Промпт:*\n"
        f"`{p}`\n\n"
        "💡 Примеры запросов:\n"
        "• поменяй фон на городской вечер\n"
        "• смени одежду на чёрный пиджак\n"
        "• добавь лёгкий макияж, подчеркни глаза\n"
        "• убери лишние предметы со стола"
    )

def banana_kb(s: Dict[str, Any]) -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton("➕ Добавить ещё фото", callback_data="banana:add_more")],
        [InlineKeyboardButton("🧹 Очистить фото", callback_data="banana:reset_imgs")],
        [InlineKeyboardButton("✍️ Изменить промпт", callback_data="banana:edit_prompt")],
        [InlineKeyboardButton("🚀 Начать генерацию Banana", callback_data="banana:start")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="back")]
    ]
    return InlineKeyboardMarkup(rows)

async def show_banana_card(update: Update, ctx: ContextTypes.DEFAULT_TYPE, edit_only_markup: bool = False):
    s = state(ctx)
    text = banana_card_text(s)
    kb = banana_kb(s)
    chat_id = update.effective_chat.id
    last_id = s.get("last_ui_msg_id")
    try:
        if last_id:
            if edit_only_markup:
                await ctx.bot.edit_message_reply_markup(chat_id, last_id, reply_markup=kb)
            else:
                await ctx.bot.edit_message_text(chat_id=chat_id, message_id=last_id, text=text,
                                                parse_mode=ParseMode.MARKDOWN, reply_markup=kb,
                                                disable_web_page_preview=True)
        else:
            m = await (update.callback_query.message.reply_text(text, parse_mode=ParseMode.MARKDOWN,
                                                               reply_markup=kb, disable_web_page_preview=True)
                       if update.callback_query else
                       update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN,
                                                 reply_markup=kb, disable_web_page_preview=True))
            s["last_ui_msg_id"] = m.message_id
    except Exception as e:
        log.warning("show_banana_card edit/send failed: %s", e)
        try:
            m = await ctx.bot.send_message(chat_id=chat_id, text=text, parse_mode=ParseMode.MARKDOWN,
                                           reply_markup=kb, disable_web_page_preview=True)
            s["last_ui_msg_id"] = m.message_id
        except Exception as e2:
            log.exception("show_banana_card send failed: %s", e2)

# ==========================
#   Prompt-Master
# ==========================
PM_HINT = (
    "🧠 *Prompt-Master готов!*\n"
    "Напишите:\n"
    "• Локацию, атмосферу, действие, камеру, диалоги (в кавычках), детали.\n"
    "Я соберу один кинопромпт на англ."
)

async def oai_prompt_master(idea_text: str) -> Optional[str]:
    if openai is None or not OPENAI_API_KEY:
        return None
    system = (
        "You are a Prompt-Master for cinematic AI video generation. "
        "Return ONE multi-line prompt in ENGLISH with labels: "
        "High-quality cinematic 4K video (16:9).\n"
        "Scene: ...\nCamera: ...\nAction: ...\nDialogue: ...\nLip-sync: ...\nAudio: ...\n"
        "Lighting: ...\nWardrobe/props: ...\nFraming: ...\n"
        "Constraints: No subtitles. No on-screen text. No logos."
    )
    try:
        user = idea_text.strip()
        if len(user) > 800:
            user = user[:800] + "..."
        resp = await asyncio.to_thread(
            openai.ChatCompletion.create,
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.8, max_tokens=800,
        )
        txt = resp["choices"][0]["message"]["content"].strip()
        return txt[:1400]
    except Exception as e:
        log.exception("Prompt-Master error: %s", e)
        return None

# ==========================
#   HTTP helpers (KIE)
# ==========================
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
    for key in ("originUrls", "resultUrls"):
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
            if s.startswith("http") and s.endswith((".mp4", ".mov", ".webm")):
                return x.strip()
        return None
    return walk(data)

# ---------- VEO ----------
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
    return False, None, f"Ошибка VEO: {j}"

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
    return False, None, f"Ошибка статуса VEO: {j}", None

def try_get_1080_url(task_id: str, attempts: int = 3, per_try_timeout: int = 15) -> Optional[str]:
    last_err = None
    for i in range(attempts):
        try:
            status, j = _get_json(join_url(KIE_BASE_URL, KIE_VEO_1080_PATH), {"taskId": task_id}, timeout=per_try_timeout)
            code = j.get("code", status)
            if status == 200 and code == 200:
                data = j.get("data") or {}
                u = (_nz(data.get("url")) or _extract_result_url(data))
                if _nz(u): return u
                last_err = "empty_1080"
            else:
                last_err = f"status={status}, code={code}"
        except Exception as e:
            last_err = str(e)
        time.sleep(1 + i)
    log.warning("1080p fetch retries failed: %s", last_err)
    return None

# ---------- MJ ----------
def _kie_error_message(status_code: int, j: Dict[str, Any]) -> str:
    code = j.get("code", status_code)
    msg = j.get("msg") or j.get("message") or j.get("error") or ""
    mapping = {401: "Доступ запрещён (Bearer).", 402: "Недостаточно кредитов.",
               429: "Превышен лимит запросов.", 500: "Внутренняя ошибка KIE.",
               422: "Запрос отклонён модерацией.", 400: "Неверный запрос (400)."}
    base = mapping.get(code, f"KIE code {code}.")
    return f"{base} {('Сообщение: ' + msg) if msg else ''}".strip()

def mj_generate(prompt: str, ar: str) -> Tuple[bool, Optional[str], str]:
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
    return False, None, _kie_error_message(status, j)

def mj_status(task_id: str) -> Tuple[bool, Optional[int], Optional[Dict[str, Any]]]:
    status, j = _get_json(join_url(KIE_BASE_URL, KIE_MJ_STATUS), {"taskId": task_id})
    code = j.get("code", status)
    if status == 200 and code == 200:
        data = j.get("data") or {}
        try:
            flag = int(data.get("successFlag"))
        except Exception:
            flag = None
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

# ==========================
#   ffmpeg helpers (для видео)
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
#   Sending video
# ==========================
async def send_video_with_fallback(ctx: ContextTypes.DEFAULT_TYPE, chat_id: int, url: str, expect_vertical: bool = False) -> bool:
    event("SEND_TRY_URL", url=url, expect_vertical=expect_vertical)
    if not expect_vertical:
        try:
            await ctx.bot.send_video(chat_id=chat_id, video=url, supports_streaming=True)
            event("SEND_OK", mode="direct_url"); return True
        except Exception as e:
            event("SEND_FAIL", mode="direct_url", err=str(e))

    tmp_path = None
    try:
        r = requests.get(url, stream=True, timeout=180); r.raise_for_status()
        ct = (r.headers.get("Content-Type") or "").lower()
        ext = ".mp4"
        if ".mov" in url.lower() or "quicktime" in ct: ext = ".mov"
        elif ".webm" in url.lower() or "webm" in ct:   ext = ".webm"

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
            for c in r.iter_content(256*1024):
                if c: f.write(c)
            tmp_path = f.name
        event("DOWNLOAD_OK", path=tmp_path, content_type=ct)

        if expect_vertical and ENABLE_VERTICAL_NORMALIZE and _ffmpeg_available():
            out = tmp_path + "_v.mp4"
            if _ffmpeg_normalize_vertical(tmp_path, out):
                try:
                    with open(out, "rb") as f:
                        await ctx.bot.send_video(chat_id=chat_id, video=InputFile(f, filename="result_vertical.mp4"),
                                                 supports_streaming=True)
                    event("SEND_OK", mode="upload_video_norm"); return True
                except Exception as e:
                    event("SEND_FAIL", mode="upload_video_norm", err=str(e))
                    with open(out, "rb") as f:
                        await ctx.bot.send_document(chat_id=chat_id, document=InputFile(f, filename="result_vertical.mp4"))
                    event("SEND_OK", mode="upload_document_norm"); return True

        if (not expect_vertical) and ALWAYS_FORCE_FHD and _ffmpeg_available():
            out = tmp_path + "_1080.mp4"
            if _ffmpeg_force_16x9_fhd(tmp_path, out, MAX_TG_VIDEO_MB):
                try:
                    with open(out, "rb") as f:
                        await ctx.bot.send_video(chat_id=chat_id, video=InputFile(f, filename="result_1080p.mp4"),
                                                 supports_streaming=True)
                    event("SEND_OK", mode="upload_16x9_forced"); return True
                except Exception as e:
                    event("SEND_FAIL", mode="upload_16x9_forced", err=str(e))
                    with open(out, "rb") as f:
                        await ctx.bot.send_document(chat_id=chat_id, document=InputFile(f, filename="result_1080p.mp4"))
                    event("SEND_OK", mode="upload_document_16x9_forced"); return True

        try:
            with open(tmp_path, "rb") as f:
                await ctx.bot.send_video(chat_id=chat_id, video=InputFile(f, filename=f"result{ext}"),
                                         supports_streaming=True)
            event("SEND_OK", mode="upload_video_raw"); return True
        except Exception as e:
            event("SEND_FAIL", mode="upload_video_raw", err=str(e))
            with open(tmp_path, "rb") as f:
                await ctx.bot.send_document(chat_id=chat_id, document=InputFile(f, filename=f"result{ext}"))
            event("SEND_OK", mode="upload_document_raw"); return True

    except Exception as e:
        log.exception("File send failed: %s", e)
        try:
            await ctx.bot.send_message(chat_id, f"🔗 Результат готов, но вложить файл не удалось. Ссылка:\n{url}")
            event("SEND_OK", mode="link_fallback_on_error"); return True
        except Exception:
            return False
    finally:
        if tmp_path:
            try: os.unlink(tmp_path)
            except Exception: pass

# ==========================
#   Poll VEO
# ==========================
async def poll_veo_and_send(chat_id: int, task_id: str, gen_id: str, ctx: ContextTypes.DEFAULT_TYPE):
    s = state(ctx)
    s["generating"] = True
    s["generation_id"] = gen_id
    s["last_task_id"] = task_id

    start_ts = time.time()
    try:
        while True:
            if s.get("generation_id") != gen_id:
                return

            ok, flag, msg, res_url = await asyncio.to_thread(get_kie_veo_status, task_id)
            event("VEO_STATUS", task_id=task_id, flag=flag, has_url=bool(res_url), msg=msg)

            if not ok:
                add_tokens(ctx, TOKEN_COSTS['veo_quality'] if s.get('model') == 'veo3' else TOKEN_COSTS['veo_fast'])
                await ctx.bot.send_message(chat_id, f"❌ Ошибка статуса: {msg or 'неизвестно'}\n💎 Токены возвращены."); break

            if _nz(res_url):
                final_url = res_url
                if (s.get("aspect") or "16:9") == "16:9":
                    u1080 = await asyncio.to_thread(try_get_1080_url, task_id)
                    if _nz(u1080):
                        final_url = u1080; event("VEO_1080_OK", task_id=task_id)
                    else:
                        event("VEO_1080_MISS", task_id=task_id)

                await ctx.bot.send_message(chat_id, "🎞️ Рендер завершён — отправляю файл…")
                sent = await send_video_with_fallback(
                    ctx, chat_id, final_url, expect_vertical=(s.get("aspect") == "9:16")
                )
                s["last_result_url"] = final_url if sent else None
                await ctx.bot.send_message(
                    chat_id, "✅ *Готово!*", parse_mode=ParseMode.MARKDOWN,
                    reply_markup=InlineKeyboardMarkup(
                        [[InlineKeyboardButton("🚀 Сгенерировать ещё видео", callback_data="start_new_cycle")]]
                    ),
                )
                break

            if flag in (2, 3):
                add_tokens(ctx, TOKEN_COSTS['veo_quality'] if s.get('model') == 'veo3' else TOKEN_COSTS['veo_fast'])
                await ctx.bot.send_message(chat_id, f"❌ KIE не вернул ссылку на видео.\nℹ️ Сообщение: {msg or 'нет'}\n💎 Токены возвращены.")
                break

            if (time.time() - start_ts) > POLL_TIMEOUT_SECS:
                add_tokens(ctx, TOKEN_COSTS['veo_quality'] if s.get('model') == 'veo3' else TOKEN_COSTS['veo_fast'])
                await ctx.bot.send_message(chat_id, "⌛ Превышено время ожидания VEO.\n💎 Токены возвращены."); break

            await asyncio.sleep(POLL_INTERVAL_SECS)

    except Exception as e:
        log.exception("[VEO_POLL] crash: %s", e)
        add_tokens(ctx, TOKEN_COSTS['veo_quality'] if s.get('model') == 'veo3' else TOKEN_COSTS['veo_fast'])
        try: await ctx.bot.send_message(chat_id, "💥 Внутренняя ошибка при опросе VEO.\n💎 Токены возвращены.")
        except Exception: pass
    finally:
        if s.get("generation_id") == gen_id:
            s["generating"] = False
            s["generation_id"] = None

# ==========================
#   MJ poll & send (anti-spam 40s)
# ==========================
async def poll_mj_and_send_photos(chat_id: int, task_id: str, ctx: ContextTypes.DEFAULT_TYPE):
    s = state(ctx)
    s["mj_wait_sent"] = False
    s["mj_last_wait_ts"] = 0.0
    start_ts = time.time()
    delay = 6
    max_wait = 15 * 60
    try:
        while True:
            ok, flag, data = await asyncio.to_thread(mj_status, task_id)
            event("MJ_STATUS", task_id=task_id, flag=flag, has_data=bool(data))

            if not ok:
                add_tokens(ctx, TOKEN_COSTS["mj"])
                await ctx.bot.send_message(chat_id, "❌ MJ сейчас недоступен. 💎 Токены возвращены."); return

            if flag == 0:
                now = time.time()
                if (now - s.get("mj_last_wait_ts", 0.0)) >= 40.0:
                    await ctx.bot.send_message(chat_id, "🖼️✨ Рисую… Подождите немного.", disable_notification=True)
                    s["mj_last_wait_ts"] = now
                if (time.time() - start_ts) > max_wait:
                    add_tokens(ctx, TOKEN_COSTS["mj"])
                    await ctx.bot.send_message(chat_id, "⌛ MJ долго не отвечает. Попробуйте позже. 💎 Токены возвращены."); return
                await asyncio.sleep(delay)
                delay = min(delay * 1.5, 20)
                continue

            if flag in (2, 3):
                msg = (data or {}).get("errorMessage") or "No response from MidJourney Official Website."
                add_tokens(ctx, TOKEN_COSTS["mj"])
                await ctx.bot.send_message(chat_id, f"❌ MJ: {msg}\n💎 Токены возвращены."); return

            if flag == 1:
                urls = _extract_mj_image_urls(data or {})
                if not urls:
                    add_tokens(ctx, TOKEN_COSTS["mj"])
                    await ctx.bot.send_message(chat_id, "⚠️ Готово, но ссылки на изображения не найдены. 💎 Токены возвращены."); return
                if len(urls) == 1:
                    await ctx.bot.send_photo(chat_id=chat_id, photo=urls[0])
                else:
                    media = [InputMediaPhoto(u) for u in urls[:10]]
                    await ctx.bot.send_media_group(chat_id=chat_id, media=media)
                await ctx.bot.send_message(
                    chat_id, "✅ *Готово!*",
                    parse_mode=ParseMode.MARKDOWN,
                    reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🚀 Сгенерировать ещё", callback_data="start_new_cycle")]])
                )
                s["mj_wait_sent"] = False
                s["mj_last_wait_ts"] = 0.0
                return
    except Exception as e:
        log.exception("[MJ_POLL] crash: %s", e)
        add_tokens(ctx, TOKEN_COSTS["mj"])
        try: await ctx.bot.send_message(chat_id, "💥 Внутренняя ошибка при опросе MJ. 💎 Токены возвращены.")
        except Exception: pass

# ==========================
#   Handlers
# ==========================
def stars_topup_kb() -> InlineKeyboardMarkup:
    rows: List[List[InlineKeyboardButton]] = []
    packs = {100:100, 200:200, 300:300, 400:400, 500:500}
    if DEV_MODE:
        packs = {1:1, **packs}
    for stars, tokens in sorted(packs.items()):
        cap = f"⭐ {stars} → 💎 {tokens}" + ("  (DEV)" if DEV_MODE and stars == 1 else "")
        rows.append([InlineKeyboardButton(cap, callback_data=f"buy:stars:{stars}")])
    rows.append([InlineKeyboardButton("🛒 Где купить Stars", url=STARS_BUY_URL)])
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="back")])
    return InlineKeyboardMarkup(rows)

async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = state(ctx); s.update({**DEFAULT_STATE})
    uid = update.effective_user.id

    # Бонус новичка
    if redis_client and not has_signup_bonus(uid):
        if get_user_balance_value(ctx) == 0:
            set_user_balance_value(ctx, 10)
            await update.message.reply_text("🎁 Добро пожаловать! На баланс зачислено 10 бесплатных токенов 💎")
        set_signup_bonus(uid)

    await update.message.reply_text(render_welcome_for(uid, ctx),
                                    parse_mode=ParseMode.MARKDOWN,
                                    reply_markup=main_menu_kb())

async def topup(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "💳 Пополнение токенов через *Telegram Stars*.\n"
        f"Если не хватает звёзд — купите их в официальном боте: {STARS_BUY_URL}",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=stars_topup_kb()
    )

async def health(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    parts = [
        f"PTB: `{getattr(_tg, '__version__', 'unknown')}`" if _tg else "PTB: `unknown`",
        f"KIEBASEURL: `{KIE_BASE_URL}`",
        f"OPENAI key: `{'set' if OPENAI_API_KEY else 'missing'}`",
        f"KIE key: `{'set' if KIE_API_KEY else 'missing'}`",
        f"REDIS: `{'set' if REDIS_URL else 'missing'}`",
        f"DEV_MODE: `{DEV_MODE}`",
        f"FFMPEGBIN: `{FFMPEG_BIN}`",
        f"MAXTGVIDEOMB: `{MAX_TG_VIDEO_MB}`",
    ]
    await update.message.reply_text("🩺 *Health*\n" + "\n".join(parts), parse_mode=ParseMode.MARKDOWN)

async def error_handler(update: Optional[Update], context: ContextTypes.DEFAULT_TYPE):
    log.exception("Unhandled error: %s", context.error)
    try:
        if update and update.effective_chat:
            await context.bot.send_message(update.effective_chat.id, "⚠️ Системная ошибка. Попробуйте ещё раз.")
    except Exception:
        pass

async def show_card_veo(update: Update, ctx: ContextTypes.DEFAULT_TYPE, edit_only_markup: bool = False):
    s = state(ctx)
    text = build_card_text_veo(s)
    kb = card_keyboard_veo(s)
    chat_id = update.effective_chat.id
    last_id = s.get("last_ui_msg_id")
    try:
        if last_id:
            if edit_only_markup:
                await ctx.bot.edit_message_reply_markup(chat_id, last_id, reply_markup=kb)
            else:
                await ctx.bot.edit_message_text(chat_id=chat_id, message_id=last_id, text=text,
                                                parse_mode=ParseMode.MARKDOWN, reply_markup=kb,
                                                disable_web_page_preview=True)
        else:
            m = await (update.callback_query.message.reply_text(text, parse_mode=ParseMode.MARKDOWN,
                                                               reply_markup=kb, disable_web_page_preview=True)
                       if update.callback_query else
                       update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN,
                                                 reply_markup=kb, disable_web_page_preview=True))
            s["last_ui_msg_id"] = m.message_id
    except Exception as e:
        log.warning("show_card_veo edit/send failed: %s", e)
        try:
            m = await ctx.bot.send_message(chat_id=chat_id, text=text, parse_mode=ParseMode.MARKDOWN,
                                           reply_markup=kb, disable_web_page_preview=True)
            s["last_ui_msg_id"] = m.message_id
        except Exception as e2:
            log.exception("show_card_veo send failed: %s", e2)

async def on_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    data = (query.data or "").strip()
    await query.answer()

    s = state(ctx)

    if data == "faq":
        await query.message.reply_text(
            "FAQ:\n"
            "• VEO: поддерживает 16:9 и 9:16 (вертикаль нормализуется локально).\n"
            "• MJ: 16:9; апскейл отключён.\n"
            "• Prompt-Master: возвращает кинопромпт.\n"
            f"• Stars: {STARS_BUY_URL}",
            reply_markup=main_menu_kb(),
        ); return

    if data == "back":
        s.update({**DEFAULT_STATE}); await query.message.reply_text("🏠 Главное меню:", reply_markup=main_menu_kb()); return

    if data == "start_new_cycle":
        s.update({**DEFAULT_STATE}); await query.message.reply_text("Выберите режим:", reply_markup=main_menu_kb()); return

    if data == "topup_open":
        await query.message.reply_text(
            f"💳 Пополнение через Telegram Stars. Если звёзд не хватает — купите в {STARS_BUY_URL}",
            reply_markup=stars_topup_kb()
        ); return

    # Покупка Stars
    if data.startswith("buy:stars:"):
        stars = int(data.split(":")[-1])
        tokens = {1:1,100:100,200:200,300:300,400:400,500:500}.get(stars, 0) if DEV_MODE else {100:100,200:200,300:300,400:400,500:500}.get(stars,0)
        if tokens <= 0:
            await query.message.reply_text("⚠️ Такой пакет недоступен."); return

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
                need_name=False, need_phone_number=False, need_email=False,
                need_shipping_address=False, is_flexible=False
            )
        except Exception as e:
            event("STARS_INVOICE_ERR", err=str(e))
            await query.message.reply_text(
                f"Если счёт не открылся — у аккаунта могут быть не активированы Stars.\n"
                f"Купите 1⭐ в {STARS_BUY_URL} и попробуйте снова.",
                reply_markup=stars_topup_kb()
            )
        return

    # --- Режимы
    if data.startswith("mode:"):
        _, mode = data.split(":", 1); s["mode"] = mode
        if mode == "veo_text":
            s["aspect"] = "16:9"; s["model"] = "veo3_fast"
            await query.message.reply_text("📝 Пришлите идею/промпт для видео."); await show_card_veo(update, ctx); return
        if mode == "veo_photo":
            s["aspect"] = "9:16"; s["model"] = "veo3_fast"
            await query.message.reply_text("🖼️ Пришлите фото (подпись-промпт — по желанию)."); await show_card_veo(update, ctx); return
        if mode == "prompt_master":
            await query.message.reply_text(PM_HINT, parse_mode=ParseMode.MARKDOWN); return
        if mode == "chat":
            await query.message.reply_text("💬 Чат доступен! Напишите вопрос."); return
        if mode == "mj_txt":
            s["aspect"] = "16:9"
            await query.message.reply_text("🖼️ Пришлите текстовый prompt для картинки (формат 16:9)."); return
        if mode == "banana":
            s["banana_images"] = []; s["last_prompt"] = None
            await query.message.reply_text(
                "🍌 Banana включён\n"
                "Сначала пришлите до *4 фото* одним сообщением (можно по одному). Я посчитаю: *n/4*.\n"
                "Когда фото будут готовы — пришлите *текст-промпт*, что изменить.",
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=banana_kb(s)
            )
            await show_banana_card(update, ctx); return

    # --- Banana UI
    if data.startswith("banana:"):
        action = data.split(":", 1)[1]
        if action == "add_more":
            await query.message.reply_text("➕ Пришлите ещё фото (всего до 4)."); return
        if action == "reset_imgs":
            s["banana_images"] = []
            await query.message.reply_text("🧹 Фото очищены. Пришлите новые.", reply_markup=banana_kb(s))
            await show_banana_card(update, ctx); return
        if action == "edit_prompt":
            await query.message.reply_text("✍️ Пришлите новый промпт для Banana."); return
        if action == "start":
            imgs = s.get("banana_images") or []
            prompt = (s.get("last_prompt") or "").strip()
            if not imgs:
                await query.message.reply_text("⚠️ Сначала добавьте хотя бы одно фото."); return
            if not prompt:
                await query.message.reply_text("⚠️ Добавьте текст-промпт (что изменить)."); return
            price = TOKEN_COSTS['banana']
            ok, rest = try_charge(ctx, price)
            if not ok:
                await query.message.reply_text(
                    f"💎 Недостаточно токенов для Banana: нужно {price}, на балансе {rest}.\n"
                    f"Пополните через Stars: {STARS_BUY_URL}", reply_markup=stars_topup_kb()
                ); return
            await query.message.reply_text("🍌 Banana запущен…")
            asyncio.create_task(_banana_run_and_send(update.effective_chat.id, ctx, imgs, prompt))
            return

    # --- VEO параметры
    if data.startswith("aspect:"):
        _, val = data.split(":", 1); s["aspect"] = "9:16" if val.strip() == "9:16" else "16:9"
        await show_card_veo(update, ctx, edit_only_markup=True); return

    if data.startswith("model:"):
        _, val = data.split(":", 1); s["model"] = "veo3" if val.strip() == "veo3" else "veo3_fast"
        await show_card_veo(update, ctx, edit_only_markup=True); return

    if data == "card:toggle_photo":
        if s.get("last_image_url"):
            s["last_image_url"] = None; await query.message.reply_text("🧹 Фото-референс удалён."); await show_card_veo(update, ctx)
        else:
            await query.message.reply_text("📎 Пришлите фото вложением или публичный URL изображения.")
        return

    if data == "card:edit_prompt":
        await query.message.reply_text("✍️ Пришлите новый текст промпта."); return

    if data == "card:reset":
        keep_aspect = s.get("aspect") or "16:9"; keep_model = s.get("model") or "veo3_fast"
        s.update({**DEFAULT_STATE}); s["aspect"] = keep_aspect; s["model"] = keep_model
        await query.message.reply_text("🗂️ Карточка очищена."); await show_card_veo(update, ctx); return

    if data == "card:generate":
        if s.get("generating"): await query.message.reply_text("⏳ Уже рендерю это видео — подождите чуть-чуть."); return
        if not s.get("last_prompt"): await query.message.reply_text("✍️ Сначала укажите текст промпта."); return

        price = TOKEN_COSTS['veo_quality'] if s.get('model') == 'veo3' else TOKEN_COSTS['veo_fast']
        ok_balance, rest = try_charge(ctx, price)
        if not ok_balance:
            await query.message.reply_text(
                f"💎 Недостаточно токенов: нужно {price}, на балансе {rest}.\n"
                f"Пополните через Stars: {STARS_BUY_URL}", reply_markup=stars_topup_kb()
            ); return

        event("VEO_SUBMIT_REQ", aspect=s.get("aspect"), model=s.get("model"),
              with_image=bool(s.get("last_image_url")), prompt_len=len(s.get("last_prompt") or ""))

        ok, task_id, msg = await asyncio.to_thread(
            submit_kie_veo, s["last_prompt"].strip(), s.get("aspect", "16:9"),
            s.get("last_image_url"), s.get("model", "veo3_fast")
        )
        event("VEO_SUBMIT_RESP", ok=ok, task_id=task_id, msg=msg)

        if not ok or not task_id:
            add_tokens(ctx, price)
            await query.message.reply_text(f"❌ Не удалось создать VEO-задачу: {msg}\n💎 Токены возвращены."); return

        gen_id = uuid.uuid4().hex[:12]
        s["generating"] = True; s["generation_id"] = gen_id; s["last_task_id"] = task_id
        mode_txt = "⚡ Fast" if s.get('model')=='veo3_fast' else "💎 Quality"
        await query.message.reply_text(
            f"🚀 Задача отправлена ({mode_txt}).\n🆔 taskId={task_id}\n"
            "🎛️ Подождите — подбираем кадры, свет и ритм…"
        )
        await query.message.reply_text("🎥 Рендер запущен… это может занять несколько минут.")
        asyncio.create_task(poll_veo_and_send(update.effective_chat.id, task_id, gen_id, ctx)); return

    # --- MJ запуск (кнопка из текста)
    if data.startswith("mj:ar:"):
        ar = "16:9"
        prompt = s.get("last_prompt")
        if not prompt:
            await query.message.reply_text("⚠️ Сначала отправьте текстовый prompt."); return

        price = TOKEN_COSTS['mj']
        ok_balance, rest = try_charge(ctx, price)
        if not ok_balance:
            await query.message.reply_text(
                f"💎 Недостаточно токенов: нужно {price}, на балансе {rest}.\nПополните через Stars: {STARS_BUY_URL}",
                reply_markup=stars_topup_kb()
            ); return

        await query.message.reply_text(f"🎨 Генерация фото запущена…\nФормат: *{ar}*\nPrompt: `{prompt}`",
                                       parse_mode=ParseMode.MARKDOWN)
        ok, task_id, msg = await asyncio.to_thread(mj_generate, prompt.strip(), ar)
        event("MJ_SUBMIT_RESP", ok=ok, task_id=task_id, msg=msg)
        if not ok or not task_id:
            add_tokens(ctx, price)
            await query.message.reply_text(f"❌ Не удалось создать MJ-задачу: {msg}\n💎 Токены возвращены."); return
        await query.message.reply_text(f"🆔 MJ taskId: `{task_id}`\n🖌️ Рисую эскиз и детали…", parse_mode=ParseMode.MARKDOWN)
        asyncio.create_task(poll_mj_and_send_photos(update.effective_chat.id, task_id, ctx))
        return

async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = state(ctx)
    text = (update.message.text or "").strip()
    mode = s.get("mode")

    low = text.lower()
    if low.startswith(("http://", "https://")) and any(low.split("?")[0].endswith(ext) for ext in (".jpg",".jpeg",".png",".webp",".heic")):
        if mode == "banana":
            if len(s["banana_images"]) >= 4:
                await update.message.reply_text("⚠️ Лимит 4 фото. Нажмите «🚀 Начать генерацию».", reply_markup=banana_kb(s)); return
            s["banana_images"].append(text.strip())
            await show_banana_card(update, ctx)
            return
        s["last_image_url"] = text.strip(); await update.message.reply_text("🧷 Ссылка на изображение принята."); await show_card_veo(update, ctx); return

    if mode == "prompt_master":
        if not text:
            await update.message.reply_text("✍️ Напишите идею по подсказке выше.")
            return
        if len(text) > 500:
            await update.message.reply_text("ℹ️ Урежу ввод до 500 символов для лучшего качества.")
        prompt = await oai_prompt_master(text[:500])
        if not prompt: await update.message.reply_text("⚠️ Prompt-Master недоступен или ответ пуст."); return
        s["last_prompt"] = prompt; await update.message.reply_text("🧠 Готово! Промпт добавлен в карточку."); await show_card_veo(update, ctx); return

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

    if mode == "mj_txt":
        s["last_prompt"] = text
        await update.message.reply_text(
            f"✅ Prompt сохранён:\n\n`{text}`\n\nНажмите ниже для запуска 16:9:",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🌆 Сгенерировать 16:9", callback_data="mj:ar:16:9")]])
        )
        return

    if mode == "banana":
        s["last_prompt"] = text
        await show_banana_card(update, ctx)
        return

    # VEO по умолчанию
    s["last_prompt"] = text
    await update.message.reply_text("🟦 *VEO — подготовка к рендеру*\nПроверь карточку ниже и жми «Сгенерировать».",
                                    parse_mode=ParseMode.MARKDOWN)
    await show_card_veo(update, ctx)

async def _banana_run_and_send(chat_id: int, ctx: ContextTypes.DEFAULT_TYPE, src_urls: List[str], prompt: str):
    try:
        task_id = await asyncio.to_thread(create_banana_task, prompt, src_urls, "png", "auto", None, None, 60)
        event("BANANA_SUBMIT_OK", task_id=task_id, imgs=len(src_urls))

        await ctx.bot.send_message(chat_id, f"🆔 Banana taskId: `{task_id}`", parse_mode=ParseMode.MARKDOWN)
        urls = await asyncio.to_thread(wait_for_banana_result, task_id, 8*60, 3)

        if not urls:
            await ctx.bot.send_message(chat_id, "⚠️ Banana вернула пустой результат. 💎 Токены возвращены.")
            add_tokens(ctx, TOKEN_COSTS["banana"]); return

        u0 = urls[0]
        try:
            await ctx.bot.send_photo(chat_id=chat_id, photo=u0, caption="✅ Banana готово")
        except Exception:
            path = None
            try:
                r = requests.get(u0, timeout=180); r.raise_for_status()
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                    f.write(r.content)
                    path = f.name
                with open(path, "rb") as f:
                    await ctx.bot.send_document(chat_id=chat_id, document=InputFile(f, filename="banana.png"), caption="✅ Banana готово")
            finally:
                if path:
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
        tgurl = tg_direct_file_url(TELEGRAM_TOKEN, file.file_path)

        # перезалив на стабильный URL (важно для устранения 404/422 у KIE)
        public_url = upload_image_stream(tgurl) or tgurl

        if s.get("mode") == "banana":
            if len(s["banana_images"]) >= 4:
                await update.message.reply_text("⚠️ Лимит 4 фото. Нажмите «🚀 Начать генерацию».", reply_markup=banana_kb(s)); return
            s["banana_images"].append(public_url)
            cap = (update.message.caption or "").strip()
            if cap:
                s["last_prompt"] = cap
            await show_banana_card(update, ctx)
            return

        s["last_image_url"] = public_url
        await update.message.reply_text("🖼️ Фото принято как референс."); await show_card_veo(update, ctx)
    except Exception as e:
        log.exception("Get photo failed: %s", e)
        await update.message.reply_text("⚠️ Не удалось обработать фото. Пришлите публичный URL картинки текстом.")

# ---------- Payments: Stars (XTR) ----------
async def precheckout_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        await update.pre_checkout_query.answer(ok=True)
    except Exception:
        await update.pre_checkout_query.answer(
            ok=False,
            error_message=f"Платёж отклонён. Проверьте баланс Stars или пополните в {STARS_BUY_URL}"
        )

async def successful_payment_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    sp = update.message.successful_payment
    try:
        meta = json.loads(sp.invoice_payload)
    except Exception:
        meta = {}

    stars = int(sp.total_amount)
    charge_id = getattr(sp, "telegram_payment_charge_id", None)
    if charge_id:
        pass

    if meta.get("kind") == "stars_pack":
        tokens = int(meta.get("tokens", 0))
        if tokens <= 0:
            map_dev = {1:1,100:100,200:200,300:300,400:400,500:500}
            map_prod = {100:100,200:200,300:300,400:400,500:500}
            tokens = (map_dev if DEV_MODE else map_prod).get(stars, 0)
        add_tokens(ctx, tokens)
        await update.message.reply_text(
            f"✅ Оплата получена: +{tokens} токенов.\nБаланс: {get_user_balance_value(ctx)} 💎"
        )
        return

    await update.message.reply_text("✅ Оплата получена. Баланс обновлён.")

# ==========================
#   Entry
# ==========================
def main():
    if not TELEGRAM_TOKEN: raise RuntimeError("TELEGRAM_TOKEN is not set")
    if not KIE_BASE_URL:   raise RuntimeError("KIE_BASE_URL is not set")
    if not KIE_API_KEY:    raise RuntimeError("KIE_API_KEY is not set")

    # удаляем вебхук перед polling
    try:
        Bot(TELEGRAM_TOKEN).delete_webhook(drop_pending_updates=True)
        log.info("Webhook deleted (drop_pending_updates=True)")
    except Exception as e:
        log.warning("Delete webhook failed: %s", e)

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

    log.info("Bot starting… (PTB polling, Redis=%s)", "on" if redis_client else "off")
    app.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
        stop_signals=None
    )

if __name__ == "__main__":
    main()
