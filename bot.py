# -*- coding: utf-8 -*-
# Best VEO3 Bot — PTB 21.x
# Версия: 2025-09-12 (Redis balance + signup bonus + Banana 4 imgs + MJ anti-spam 40s)
#
# Ключевые моменты:
# • Redis-хранилище баланса/флагов → токены не теряются при деплоях.
# • Бонус новичка: +10💎 при первом /start (проверка по Redis).
# • VEO: Fast = 50💎, Quality = 200💎. Сообщения Fast/Quality корректные.
# • MJ: антиспам «Рисую…» — не чаще 1 раза в 40 сек. Без упоминания апскейла.
# • Banana: до 4 фото, генерация только по кнопке «🚀 Начать генерацию (Banana)».
# • Удаление вебхука на старте (safety), polling «залипает» корректно.
#
# Версия: 2025-09-12 (FULL) — Banana=5, PromoCodes, Bonus Packs, Quality-50, Bold Balance, FAQ, PM button, progress pings

import os
import json
import time
import uuid
import base64
import asyncio
import logging
import tempfile
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

import requests
@@ -33,12 +27,6 @@
CallbackQueryHandler, filters, AIORateLimiter, PreCheckoutQueryHandler
)

# === Banana wrapper ===
from kie_banana import create_banana_task, wait_for_banana_result, KieBananaError

# === Redis (надёжное хранилище баланса/флагов) ===
import redis

# ==========================
#   ENV / INIT
# ==========================
@@ -53,6 +41,7 @@ def _env(k: str, d: str = "") -> str:
STARS_BUY_URL       = _env("STARS_BUY_URL", "https://t.me/PremiumBot")
DEV_MODE            = _env("DEV_MODE", "false").lower() == "true"

# Optional OpenAI for Prompt-Master / Chat
OPENAI_API_KEY = _env("OPENAI_API_KEY")
try:
import openai  # type: ignore
@@ -64,21 +53,24 @@ def _env(k: str, d: str = "") -> str:
# ---- KIE core ----
KIE_API_KEY  = _env("KIE_API_KEY")
KIE_BASE_URL = _env("KIE_BASE_URL", "https://api.kie.ai")
KIE_VEO_GEN_PATH    = _env("KIE_VEO_GEN_PATH",    _env("KIE_GEN_PATH",    "/api/v1/veo/generate"))
KIE_VEO_STATUS_PATH = _env("KIE_VEO_STATUS_PATH", _env("KIE_STATUS_PATH", "/api/v1/veo/record-info"))
KIE_VEO_1080_PATH   = _env("KIE_VEO_1080_PATH",   _env("KIE_HD_PATH",     "/api/v1/veo/get-1080p-video"))
KIE_VEO_GEN_PATH    = _env("KIE_VEO_GEN_PATH",    "/api/v1/veo/generate")
KIE_VEO_STATUS_PATH = _env("KIE_VEO_STATUS_PATH", "/api/v1/veo/record-info")
KIE_VEO_1080_PATH   = _env("KIE_VEO_1080_PATH",   "/api/v1/veo/get-1080p-video")

# ---- MJ
# ---- MJ (Midjourney)
KIE_MJ_GENERATE = _env("KIE_MJ_GENERATE", "/api/v1/mj/generate")
KIE_MJ_STATUS   = _env("KIE_MJ_STATUS",   "/api/v1/mj/record-info")

# ---- Upload API (для VEO reference, если нужно)
# ---- Banana (Nano-Banana)
KIE_BANANA_GENERATE = _env("KIE_BANANA_GENERATE", "/api/v1/jobs/generate")
KIE_BANANA_STATUS   = _env("KIE_BANANA_STATUS",   "/api/v1/jobs/recordInfo")
KIE_BANANA_MODEL    = _env("KIE_BANANA_MODEL",    "google/nano-banana-edit")

# ---- Upload API (опционально)
UPLOAD_BASE_URL     = _env("UPLOAD_BASE_URL", "https://kieai.redpandaai.co")
UPLOAD_STREAM_PATH  = _env("UPLOAD_STREAM_PATH", "/api/file-stream-upload")
UPLOAD_URL_PATH     = _env("UPLOAD_URL_PATH",    "/api/file-url-upload")
UPLOAD_BASE64_PATH  = _env("UPLOAD_BASE64_PATH", "/api/file-base64-upload")

# ---- Видео-доставка
# ---- Видео-отправка
ENABLE_VERTICAL_NORMALIZE = _env("ENABLE_VERTICAL_NORMALIZE", "true").lower() == "true"
ALWAYS_FORCE_FHD          = _env("ALWAYS_FORCE_FHD", "true").lower() == "true"
FFMPEG_BIN                = _env("FFMPEG_BIN", "ffmpeg")
@@ -91,128 +83,148 @@ def _env(k: str, d: str = "") -> str:
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("veo3-bot")

# Optional Redis (persistent balance)
try:
    import telegram as _tg
    log.info("PTB version: %s", getattr(_tg, "__version__", "unknown"))
    import redis  # type: ignore
except Exception:
    _tg = None
    redis = None

# ---- Redis
REDIS_URL    = _env("REDIS_URL")
REDIS_PREFIX = _env("REDIS_PREFIX", "veo3:prod")
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True) if REDIS_URL else None
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
    return ":".join([REDIS_PREFIX, *parts])
    return ":".join([REDIS_PREFIX, *[p for p in parts if p]])

# ==========================
#   Tokens / Pricing
#   Pricing / Packs / Promo
# ==========================
TOKEN_COSTS = {
    "veo_fast": 50,
    "veo_quality": 200,   # фиксировано по вашему решению
    "veo_photo": 50,
    "veo_fast": 50,     # Fast
    "veo_quality": 150, # Quality (на 50 дешевле прежних 200)
    "veo_photo": 50,    # Animate (оживление фото)
"mj": 15,
    "banana": 10,
    "banana": 5,
"chat": 0,
}

CHAT_UNLOCK_PRICE = 0  # чат бесплатный

# ---------------- Redis-backed balance helpers ----------------
def get_user_id(ctx: ContextTypes.DEFAULT_TYPE) -> Optional[int]:
    # PTB 21: безопасно доставать user_id из контекста
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
    if not redis_client:  # без Redis — не выдаём бонус «навсегда», чтобы не дублировать
        return uid in {}  # всегда False
    return bool(redis_client.get(_rk("signup_bonus", str(uid))))

def set_signup_bonus(uid: int):
    if redis_client:
        # хранить флаг бесконечно
        redis_client.set(_rk("signup_bonus", str(uid)), "1")
SIGNUP_BONUS = int(_env("SIGNUP_BONUS", "10"))

# Stars → Diamonds (final credited diamonds)
STAR_PACKS = [
    (50, 50,  ""),
    (100,110, "+10💎 бонус"),
    (200,220, "+20💎 бонус"),
    (300,330, "+30💎 бонус"),
    (400,440, "+40💎 бонус"),
    (500,550, "+50💎 бонус"),
]
if DEV_MODE:
    STAR_PACKS = [(1,1,"DEV"), *STAR_PACKS]

# Promo codes
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
#   Utils / State
#   State
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

DEFAULT_STATE = {
    "mode": None,          # 'veo_text' | 'veo_photo' | 'prompt_master' | 'chat' | 'mj_txt' | 'banana'
    "aspect": None,        # '16:9' | '9:16'
    "model": None,         # 'veo3_fast' | 'veo3'
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
    "chat_unlocked": True,   # чат бесплатный
    "chat_unlocked": True,     # чат бесплатный
"mj_wait_sent": False,
    "mj_last_wait_ts": 0.0,
    "banana_images": [],     # список URL до 4 штук
    "mj_wait_last_ts": 0.0,    # throttle 40s
    # Banana session
    "banana_images": [],       # list[str]
    "banana_prompt": None,
    "banana_task_id": None,
}

def state(ctx: ContextTypes.DEFAULT_TYPE) -> Dict[str, Any]:
ud = ctx.user_data
for k, v in DEFAULT_STATE.items():
        if k not in ud:
            ud[k] = [] if isinstance(v, list) else v
    if not isinstance(ud.get("banana_images"), list):
        ud["banana_images"] = []
        ud.setdefault(k, v)
return ud

# ==========================
#   Balance helpers
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
#   UI
# ==========================
@@ -221,8 +233,8 @@ def state(ctx: ContextTypes.DEFAULT_TYPE) -> Dict[str, Any]:
"🖌️ MJ — художник: нарисует изображение по твоему тексту.\n"
"🍌 Banana — Редактор изображений из будущего\n"
"🧠 Prompt-Master — вернёт профессиональный кинопромпт.\n"
    "💬 Обычный чат — общение с ИИ.\n"
    "💎 Ваш баланс: {balance}\n"
    "💬 Обычный чат — общение с ИИ.\n\n"
    "💎 Ваш баланс: *{balance}*\n"
"📈 Больше идей и примеров: {prompts_url}\n\n"
"Выберите режим 👇"
)
@@ -231,11 +243,17 @@ def render_welcome_for(uid: int, ctx: ContextTypes.DEFAULT_TYPE) -> str:
return WELCOME.format(balance=get_user_balance_value(ctx), prompts_url=PROMPTS_CHANNEL_URL)

def main_menu_kb() -> InlineKeyboardMarkup:
    vf = TOKEN_COSTS["veo_fast"]
    vq = TOKEN_COSTS["veo_quality"]
    vp = TOKEN_COSTS["veo_photo"]
    mj = TOKEN_COSTS["mj"]
    bn = TOKEN_COSTS["banana"]
rows = [
        [InlineKeyboardButton("🎬 Генерация видео (Veo)", callback_data="mode:veo_text")],
        [InlineKeyboardButton("🖼️ Генерация изображений (MJ)", callback_data="mode:mj_txt")],
        [InlineKeyboardButton("🍌 Редактор изображений (Banana)", callback_data="mode:banana")],
        [InlineKeyboardButton("📸 Оживить изображение (Veo)", callback_data="mode:veo_photo")],
        [InlineKeyboardButton(f"🎬 Генерация видео (Veo Fast) 💎{vf}", callback_data="mode:veo_text_fast")],
        [InlineKeyboardButton(f"🎬 Генерация видео (Veo Quality) 💎{vq}", callback_data="mode:veo_text_quality")],
        [InlineKeyboardButton(f"🖼️ Генерация изображений (MJ) 💎{mj}", callback_data="mode:mj_txt")],
        [InlineKeyboardButton(f"🍌 Редактор изображений (Banana) 💎{bn}", callback_data="mode:banana")],
        [InlineKeyboardButton(f"📸 Оживить изображение (Veo) 💎{vp}", callback_data="mode:veo_photo")],
[InlineKeyboardButton("🧠 Prompt-Master (ChatGPT)", callback_data="mode:prompt_master")],
[InlineKeyboardButton("💬 Обычный чат (ChatGPT)", callback_data="mode:chat")],
[
@@ -265,7 +283,7 @@ def build_card_text_veo(s: Dict[str, Any]) -> str:
if len(prompt_preview) > 900: prompt_preview = prompt_preview[:900] + "…"
has_prompt = "есть" if s.get("last_prompt") else "нет"
has_ref = "есть" if s.get("last_image_url") else "нет"
    model = "Fast" if s.get("model") == "veo3_fast" else ("Quality" if s.get("model") == "veo3" else "—")
    model = "Fast" if s.get("model") == "veo3_fast" else ("Quality" if s.get("model") else "—")
price = TOKEN_COSTS['veo_quality'] if s.get('model') == 'veo3' else TOKEN_COSTS['veo_fast']
lines = [
"🎬 *Карточка VEO*",
@@ -287,6 +305,8 @@ def card_keyboard_veo(s: Dict[str, Any]) -> InlineKeyboardMarkup:
rows: List[List[InlineKeyboardButton]] = []
rows.append([InlineKeyboardButton("🖼️ Добавить/Удалить фото", callback_data="card:toggle_photo"),
InlineKeyboardButton("✍️ Изменить промпт",       callback_data="card:edit_prompt")])
    # Prompt-Master кнопка прямо в карточке
    rows.append([InlineKeyboardButton("🧠 Prompt-Master (помочь с текстом)", callback_data="mode:prompt_master")])
rows.append(aspect_row(s.get("aspect") or "16:9"))
rows.append(model_row(s.get("model") or "veo3_fast"))
rows.append([InlineKeyboardButton("🚀 Сгенерировать", callback_data="card:generate")])
@@ -295,66 +315,20 @@ def card_keyboard_veo(s: Dict[str, Any]) -> InlineKeyboardMarkup:
rows.append([InlineKeyboardButton("💳 Пополнить баланс", callback_data="topup_open")])
return InlineKeyboardMarkup(rows)

def banana_kb(s: Dict[str, Any]) -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton("➕ Добавить ещё фото", callback_data="banana:add_more")],
        [InlineKeyboardButton("🧹 Сбросить фото", callback_data="banana:reset_imgs")],
        [InlineKeyboardButton("✍️ Изменить промпт", callback_data="banana:edit_prompt")],
        [InlineKeyboardButton("🚀 Начать генерацию (Banana)", callback_data="banana:start")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="back")]
    ]
    return InlineKeyboardMarkup(rows)

def mj_start_kb() -> InlineKeyboardMarkup:
return InlineKeyboardMarkup([[InlineKeyboardButton("🌆 Сгенерировать 16:9", callback_data="mj:ar:16:9")]])

# ==========================
#   Prompt-Master
# ==========================
PM_HINT = (
    "🧠 *Prompt-Master готов!*\n"
    "Чтобы получить лучший кинопромпт, напишите:\n"
    "• *Локацию*: где происходит сцена.\n"
    "• *Атмосферу*: свет, цвет, настроение.\n"
    "• *Действие*: что происходит в кадре.\n"
    "• *Камеру*: планы/движение (панорамы, трекинг, крены).\n"
    "• *Речь*: укажите реплики *в кавычках* (если есть).\n"
    "• *Детали*: одежда, реквизит, звук/музыка.\n\n"
    "Я соберу из этого один профессиональный кинопромпт."
)

async def oai_prompt_master(idea_text: str) -> Optional[str]:
    if openai is None or not OPENAI_API_KEY:
        return None
    system = (
        "You are a Prompt-Master for cinematic AI video generation (Veo-style). "
        "Return ONE multi-line prompt in ENGLISH following this exact structure and labels: "
        "High-quality cinematic 4K video (16:9).\n"
        "Scene: ...\nCamera: ...\nAction: ...\nDialogue: ...\nLip-sync: ...\nAudio: ...\n"
        "Lighting: ...\nWardrobe/props: ...\nFraming: ...\nConstraints: No subtitles. No on-screen text. No logos. "
        "Rules: keep 16:9; forbid legible text; be specific; keep it ~600–1100 chars; "
        "if user provides Russian dialogue, include it verbatim under Dialogue: and require frame-accurate lip sync by syllables."
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
def banana_ready_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton("🚀 Начать генерацию Banana", callback_data="banana:start")],
                                 [InlineKeyboardButton("🧹 Очистить фото", callback_data="banana:clear")]])

# ==========================
#   HTTP helpers (KIE)
#   HTTP helpers
# ==========================
def join_url(base: str, path: str) -> str:
    u = f"{base.rstrip('/')}/{path.lstrip('/')}"
    return u.replace("://", "§§").replace("//", "/").replace("§§", "://")

def _kie_headers_json() -> Dict[str, str]:
h = {"Content-Type": "application/json"}
token = (KIE_API_KEY or "").strip()
@@ -417,7 +391,7 @@ def add(u: str):
return urls

def _extract_result_url(data: Dict[str, Any]) -> Optional[str]:
    for key in ("originUrls", "resultUrls"):
    for key in ("originUrls", "resultUrls", "videoUrls"):
urls = _coerce_url_list(data.get(key))
if urls:
return urls[0]
@@ -439,12 +413,55 @@ def walk(x):
if r: return r
elif isinstance(x, str):
s = x.strip().split("?")[0].lower()
            if s.startswith("http") and s.endswith((".mp4", ".mov", ".webm")):
            if s.startswith("http") and s.endswith((".mp4", ".mov", ".webm", ".jpg", ".png", ".jpeg", ".webp")):
return x.strip()
return None
return walk(data)

# ---------- VEO ----------
# ---------- Upload helper (stream) ----------
def _upload_headers() -> Dict[str, str]:
    tok = (KIE_API_KEY or "").strip()
    if tok and not tok.lower().startswith("bearer "):
        tok = f"Bearer {tok}"
    return {"Authorization": tok} if tok else {}

def upload_image_stream(src_url: str, upload_path: str = "tg-uploads", timeout: int = 90) -> Optional[str]:
    try:
        rr = requests.get(src_url, stream=True, timeout=timeout); rr.raise_for_status()
        ct = (rr.headers.get("Content-Type") or "").lower()
        ext = ".jpg"
        if "png" in ct: ext = ".png"
        elif "webp" in ct: ext = ".webp"
        elif "jpeg" in ct: ext = ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
            for ch in rr.iter_content(256*1024):
                if ch: f.write(ch)
            local = f.name
    except Exception as e:
        log.warning("upload_stream_predownload_failed: %s", e)
        return None
    try:
        url = join_url(UPLOAD_BASE_URL, UPLOAD_STREAM_PATH)
        with open(local, "rb") as f:
            files = {"file": (os.path.basename(local), f)}
            data  = {"uploadPath": upload_path, "fileName": os.path.basename(local)}
            r = requests.post(url, headers=_upload_headers(), files=files, data=data, timeout=timeout)
        try: j = r.json()
        except Exception: j = {"error": r.text}
        if r.status_code == 200 and (j.get("code", 200) == 200):
            d = j.get("data") or {}
            u = d.get("downloadUrl") or d.get("fileUrl")
            if u: return u
    except Exception as e:
        log.warning("upload_stream_err: %s", e)
    finally:
        try: os.unlink(local)
        except Exception: pass
    return None

# ==========================
#   VEO
# ==========================
def _build_payload_for_veo(prompt: str, aspect: str, image_url: Optional[str], model_key: str) -> Dict[str, Any]:
payload: Dict[str, Any] = {
"prompt": prompt,
@@ -456,14 +473,18 @@ def _build_payload_for_veo(prompt: str, aspect: str, image_url: Optional[str], m
return payload

def submit_kie_veo(prompt: str, aspect: str, image_url: Optional[str], model_key: str) -> Tuple[bool, Optional[str], str]:
    payload = _build_payload_for_veo(prompt, aspect, image_url, model_key)
    img_for_kie = None
    if image_url:
        img_for_kie = upload_image_stream(image_url) or image_url
    payload = _build_payload_for_veo(prompt, aspect, img_for_kie, model_key)
status, j = _post_json(join_url(KIE_BASE_URL, KIE_VEO_GEN_PATH), payload)
code = j.get("code", status)
if status == 200 and code == 200:
tid = _extract_task_id(j)
if tid: return True, tid, "Задача создана."
return False, None, "Ответ KIE без taskId."
    return False, None, f"Ошибка VEO: {j}"
    msg = (j.get("msg") or j.get("message") or j.get("error") or "")
    return False, None, f"KIE error code={code}: {msg}"

def get_kie_veo_status(task_id: str) -> Tuple[bool, Optional[int], Optional[str], Optional[str]]:
status, j = _get_json(join_url(KIE_BASE_URL, KIE_VEO_STATUS_PATH), {"taskId": task_id})
@@ -475,7 +496,7 @@ def get_kie_veo_status(task_id: str) -> Tuple[bool, Optional[int], Optional[str]
except Exception: flag = None
msg = j.get("msg") or j.get("message")
return True, flag, msg, _extract_result_url(data)
    return False, None, f"Ошибка статуса VEO: {j}", None
    return False, None, f"KIE status error code={code}", None

def try_get_1080_url(task_id: str, attempts: int = 3, per_try_timeout: int = 15) -> Optional[str]:
last_err = None
@@ -485,8 +506,8 @@ def try_get_1080_url(task_id: str, attempts: int = 3, per_try_timeout: int = 15)
code = j.get("code", status)
if status == 200 and code == 200:
data = j.get("data") or {}
                u = (_nz(data.get("url")) or _extract_result_url(data))
                if _nz(u): return u
                u = (data.get("url") or _extract_result_url(data))
                if u: return u
last_err = "empty_1080"
else:
last_err = f"status={status}, code={code}"
@@ -496,17 +517,48 @@ def try_get_1080_url(task_id: str, attempts: int = 3, per_try_timeout: int = 15)
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
async def send_video_with_fallback(ctx: ContextTypes.DEFAULT_TYPE, chat_id: int, url: str) -> bool:
    try:
        await ctx.bot.send_video(chat_id=chat_id, video=url, supports_streaming=True)
        return True
    except Exception:
        pass
    # fallback: скачивание и отправка файлом отключено для краткости
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
            final_url = res_url
            if not expect_vertical:
                u1080 = await asyncio.to_thread(try_get_1080_url, task_id)
                if u1080: final_url = u1080
            await ctx.bot.send_message(chat_id, "🎞️ Рендер завершён — отправляю файл…")
            await send_video_with_fallback(ctx, chat_id, final_url)
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
#   MJ
# ==========================
def mj_generate(prompt: str) -> Tuple[bool, Optional[str], str]:
payload = {
"taskType": "mj_txt2img",
"prompt": prompt,
@@ -521,17 +573,15 @@ def mj_generate(prompt: str, ar: str) -> Tuple[bool, Optional[str], str]:
tid = _extract_task_id(j)
if tid: return True, tid, "MJ задача создана."
return False, None, "Ответ MJ без taskId."
    return False, None, _kie_error_message(status, j)
    return False, None, f"MJ error: code={code}"

def mj_status(task_id: str) -> Tuple[bool, Optional[int], Optional[Dict[str, Any]]]:
status, j = _get_json(join_url(KIE_BASE_URL, KIE_MJ_STATUS), {"taskId": task_id})
code = j.get("code", status)
if status == 200 and code == 200:
data = j.get("data") or {}
        try:
            flag = int(data.get("successFlag"))
        except Exception:
            flag = None
        try: flag = int(data.get("successFlag"))
        except Exception: flag = None
return True, flag, data
return False, None, None

@@ -545,377 +595,203 @@ def _extract_mj_image_urls(status_data: Dict[str, Any]) -> List[str]:
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
async def poll_mj_and_send_photos(chat_id: int, task_id: str, ctx: ContextTypes.DEFAULT_TYPE):
    s = state(ctx)
    start_ts = time.time()
    while True:
        ok, flag, data = await asyncio.to_thread(mj_status, task_id)
        if not ok:
            await ctx.bot.send_message(chat_id, "❌ MJ сейчас недоступен."); return
        if flag == 0:
            # throttle 40 сек
            now = time.time()
            if not s.get("mj_wait_sent") or (now - s.get("mj_wait_last_ts", 0)) >= 40:
                s["mj_wait_sent"] = True
                s["mj_wait_last_ts"] = now
                await ctx.bot.send_message(chat_id, f"🖼️✨ Рисую… {datetime.now().strftime('%H:%M:%S')}")
            if (now - start_ts) > 15*60:
                await ctx.bot.send_message(chat_id, "⌛ MJ долго не отвечает. Попробуйте позже."); return
            await asyncio.sleep(6)
            continue
        if flag in (2, 3):
            msg = (data or {}).get("errorMessage") or "No response from MidJourney."
            await ctx.bot.send_message(chat_id, f"❌ MJ: {msg}"); return
        if flag == 1:
            urls = _extract_mj_image_urls(data or {})
            if not urls:
                await ctx.bot.send_message(chat_id, "⚠️ Готово, но ссылки не найдены."); return
            if len(urls) == 1:
                await ctx.bot.send_photo(chat_id=chat_id, photo=urls[0])
            else:
                media = [InputMediaPhoto(u) for u in urls[:10]]
                await ctx.bot.send_media_group(chat_id=chat_id, media=media)
            await ctx.bot.send_message(chat_id, "✅ *Готово!*", parse_mode=ParseMode.MARKDOWN,
                                       reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🚀 Сгенерировать ещё", callback_data="start_new_cycle")]]))
            return

# ==========================
#   Sending video
#   Banana
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
def banana_generate(prompt: str, image_urls: List[str]) -> Tuple[bool, Optional[str], str]:
    payload = {
        "model": KIE_BANANA_MODEL,
        "input": {"prompt": prompt, "image_urls": image_urls, "output_format": "png", "image_size": "auto"}
    }
    status, j = _post_json(join_url(KIE_BASE_URL, KIE_BANANA_GENERATE), payload)
    code = j.get("code", status)
    if status == 200 and code == 200:
        tid = _extract_task_id(j)
        if tid: return True, tid, "Banana задача создана."
        return False, None, "Ответ без taskId."
    return False, None, f"Banana error code={code}"

# ==========================
#   Polling VEO
# ==========================
async def poll_veo_and_send(chat_id: int, task_id: str, gen_id: str, ctx: ContextTypes.DEFAULT_TYPE):
    s = state(ctx)
    s["generating"] = True
    s["generation_id"] = gen_id
    s["last_task_id"] = task_id
def banana_status(task_id: str) -> Tuple[bool, Optional[int], Optional[Dict[str, Any]]]:
    status, j = _get_json(join_url(KIE_BASE_URL, KIE_BANANA_STATUS), {"taskId": task_id})
    code = j.get("code", status)
    if status == 200 and code == 200:
        data = j.get("data") or {}
        try: flag = int(data.get("successFlag"))
        except Exception: flag = None
        return True, flag, data
    return False, None, None

    start_ts = time.time()
def _banana_result_urls(data: Dict[str, Any]) -> List[str]:
    urls = []
    d = data.get("data") or data
    rj = d.get("resultJson") or d.get("resultInfoJson") or {}
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
        if isinstance(rj, str):
            rj = json.loads(rj)
    except Exception:
        rj = {}
    arr = (rj.get("resultUrls") or [])
    return _coerce_url_list(arr)

# ==========================
#   MJ poll & send (anti-spam 40s)
# ==========================
async def poll_mj_and_send_photos(chat_id: int, task_id: str, ctx: ContextTypes.DEFAULT_TYPE):
    s = state(ctx)
    s["mj_wait_sent"] = False
    s["mj_last_wait_ts"] = 0.0
async def poll_banana_and_send(chat_id: int, task_id: str, ctx: ContextTypes.DEFAULT_TYPE):
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
    while True:
        ok, flag, data = await asyncio.to_thread(banana_status, task_id)
        if not ok:
            await ctx.bot.send_message(chat_id, "❌ Banana сейчас недоступен."); return
        if flag == 0:
            await asyncio.sleep(6)
            continue
        if flag in (2, 3):
            await ctx.bot.send_message(chat_id, "❌ Banana: ошибка генерации."); return
        if flag == 1:
            urls = _banana_result_urls({"data": data})
            if not urls:
                await ctx.bot.send_message(chat_id, "⚠️ Готово, но ссылки не найдены."); return
            if len(urls) == 1:
                await ctx.bot.send_photo(chat_id=chat_id, photo=urls[0])
            else:
                media = [InputMediaPhoto(u) for u in urls[:10]]
                await ctx.bot.send_media_group(chat_id=chat_id, media=media)
            await ctx.bot.send_message(chat_id, "✅ *Готово!*", parse_mode=ParseMode.MARKDOWN,
                                       reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🚀 Сгенерировать ещё", callback_data="start_new_cycle")]]))
            return
        if (time.time() - start_ts) > 15*60:
            await ctx.bot.send_message(chat_id, "⌛ Banana долго не отвечает. Попробуйте позже."); return

# ==========================
#   Handlers
#   Handlers: UI / Commands
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
    "• До 4 фото + промпт.\n"
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
    "⸻\n\n"
    "❓ *Частые вопросы*\n"
    "Сколько ждать?\n"
    "• Картинки / Banana — до 2 минут.\n"
    "• Видео Fast — 2–5 минут.\n"
    "• Видео Quality — 5–10 минут.\n\n"
    "Где пополнить баланс?\n"
    "— Через кнопку «Пополнить баланс» в меню.\n\n"
    "Можно ли писать на русском?\n"
    "— Да, все режимы понимают русский и английский.\n\n"
    "Если вопроса нет в списке — просто напишите сюда, я помогу."
)

def stars_topup_kb() -> InlineKeyboardMarkup:
rows: List[List[InlineKeyboardButton]] = []
    packs = {100:100, 200:200, 300:300, 400:400, 500:500}
    if DEV_MODE:
        packs = {1:1, **packs}
    for stars, tokens in sorted(packs.items()):
        cap = f"⭐ {stars} → 💎 {tokens}" + ("  (DEV)" if DEV_MODE and stars == 1 else "")
        rows.append([InlineKeyboardButton(cap, callback_data=f"buy:stars:{stars}")])
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
    uid = update.effective_user.id

    # Бонус новичка (проверяем в Redis, чтобы не повторялся)
    if redis_client and not has_signup_bonus(uid):
        # если у пользователя уже был баланс >0 (например, перенос) — не затираем, а лишь добавим при нуле
        if get_user_balance_value(ctx) == 0:
            set_user_balance_value(ctx, 10)
            await update.message.reply_text("🎁 Добро пожаловать! На баланс зачислено 10 бесплатных токенов 💎")
        set_signup_bonus(uid)

    await update.message.reply_text(render_welcome_for(uid, ctx),
    await update.message.reply_text(render_welcome_for(update.effective_user.id, ctx),
parse_mode=ParseMode.MARKDOWN,
reply_markup=main_menu_kb())

async def topup(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
await update.message.reply_text(
"💳 Пополнение токенов через *Telegram Stars*.\n"
        f"Если не хватает звёзд — купите их в официальном боте: {STARS_BUY_URL}",
        f"Если не хватает звёзд — купите их в {STARS_BUY_URL}",
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
            "• MJ: только 16:9; апскейл отключён.\n"
            "• Prompt-Master: возвращает кинопромпт (без авто-диалогов).\n"
            f"• Покупка Stars: {STARS_BUY_URL}",
            reply_markup=main_menu_kb(),
        ); return

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

    # Покупка Stars пакета
if data.startswith("buy:stars:"):
        stars = int(data.split(":")[-1])
        # отображаемые пакеты совпадают с меню (без логики на сервере)
        tokens = {1:1,100:100,200:200,300:300,400:400,500:500}.get(stars, 0) if DEV_MODE else {100:100,200:200,300:300,400:400,500:500}.get(stars,0)
        if tokens <= 0:
            await query.message.reply_text("⚠️ Такой пакет недоступен."); return

        title = f"{stars}⭐ → {tokens}💎"
        payload = json.dumps({"kind": "stars_pack", "stars": stars, "tokens": tokens})
        parts = data.split(":")
        stars = int(parts[2]); diamonds = int(parts[3])
        title = f"{stars}⭐ → {diamonds}💎"
        payload = json.dumps({"kind": "stars_pack", "stars": stars, "tokens": diamonds})
try:
await ctx.bot.send_invoice(
chat_id=update.effective_chat.id,
@@ -929,177 +805,161 @@ async def on_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
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
    # --- режимы
if data.startswith("mode:"):
_, mode = data.split(":", 1); s["mode"] = mode
        if mode == "veo_text":
        if mode == "veo_text_fast":
s["aspect"] = "16:9"; s["model"] = "veo3_fast"
            await query.message.reply_text("📝 Пришлите идею/промпт для видео."); await show_card_veo(update, ctx); return
            await query.message.reply_text("📝 Пришлите идею/промпт для видео (VEO Fast)."); await show_card_veo(update, ctx); return
        if mode == "veo_text_quality":
            s["aspect"] = "16:9"; s["model"] = "veo3"
            await query.message.reply_text("📝 Пришлите идею/промпт для видео (VEO Quality)."); await show_card_veo(update, ctx); return
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
            s["aspect"] = "16:9"; s["last_prompt"] = None
            await query.message.reply_text("🖼️ Пришлите текстовый prompt для картинки (16:9)."); return
if mode == "banana":
            s["banana_images"] = []; s["last_prompt"] = None
            s["banana_images"] = []; s["banana_prompt"] = None; s["banana_task_id"] = None
await query.message.reply_text(
                "🍌 Banana включён\n"
                "Пришлите до *4 фото* с подписью-промптом (или без — можно отдельно).\n"
                "Когда будут готовы — нажмите «🚀 Начать генерацию (Banana)».",
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=banana_kb(s)
                "🍌 Banana включён\nПришлите *до 4 фото одним сообщением* с подписью-промптом.\n"
                "После приёма появится кнопка «🚀 Начать генерацию».",
                parse_mode=ParseMode.MARKDOWN
); return
        if mode == "prompt_master":
            await query.message.reply_text(
                "🧠 *Prompt-Master готов!* Напишите в одном сообщении:\n"
                "• Идею сцены (1–2 предложения) и локацию.\n"
                "• Стиль/настроение, свет, ключевые предметы.\n"
                "• Действие в кадре и динамику камеры.\n"
                "• Реплики (если есть) — в кавычках.",
                parse_mode=ParseMode.MARKDOWN
            ); return
        if mode == "chat":
            await query.message.reply_text("✍️ Напишите вопрос для ChatGPT."); return

    # --- Banana UI callbacks
    if data.startswith("banana:"):
        action = data.split(":", 1)[1]
        if action == "add_more":
            await query.message.reply_text("➕ Пришлите ещё фото (всего до 4)."); return
        if action == "reset_imgs":
            s["banana_images"] = []
            await query.message.reply_text("🧹 Фото очищены. Пришлите новые.", reply_markup=banana_kb(s)); return
        if action == "edit_prompt":
            await query.message.reply_text("✍️ Пришлите новый промпт для Banana."); return
        if action == "start":
            imgs = s.get("banana_images") or []
            prompt = (s.get("last_prompt") or "").strip()
            if not imgs:
                await query.message.reply_text("⚠️ Сначала добавьте хотя бы одно фото."); return
            if not prompt:
                await query.message.reply_text("⚠️ Добавьте подпись-промпт."); return
            price = TOKEN_COSTS['banana']
            ok, rest = try_charge(ctx, price)
            if not ok:
                await query.message.reply_text(
                    f"💎 Недостаточно токенов для Banana: нужно {price}, на балансе {rest}.\n"
                    f"Пополните через Stars: {STARS_BUY_URL}", reply_markup=stars_topup_kb()
                ); return
            await query.message.reply_text("🍌 Запускаю Banana…")
            asyncio.create_task(_banana_run_and_send(update.effective_chat.id, ctx, imgs, prompt))
            return

    # --- VEO параметры
    # --- карточка VEO
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
                f"💎 Недостаточно токенов: нужно {price}, на балансе *{rest}*.\n"
                f"Пополните через Stars: {STARS_BUY_URL}", parse_mode=ParseMode.MARKDOWN, reply_markup=stars_topup_kb()
); return

        event("VEO_SUBMIT_REQ", aspect=s.get("aspect"), model=s.get("model"),
              with_image=bool(s.get("last_image_url")), prompt_len=len(s.get("last_prompt") or ""))

ok, task_id, msg = await asyncio.to_thread(
            submit_kie_veo, s["last_prompt"].strip(), s.get("aspect", "16:9"),
            s.get("last_image_url"), s.get("model", "veo3_fast")
            submit_kie_veo, s["last_prompt"].strip(), s.get("aspect", "16:9"), s.get("last_image_url"), s.get("model", "veo3_fast")
)
        event("VEO_SUBMIT_RESP", ok=ok, task_id=task_id, msg=msg)

if not ok or not task_id:
add_tokens(ctx, price)
await query.message.reply_text(f"❌ Не удалось создать VEO-задачу: {msg}\n💎 Токены возвращены."); return
        s["generating"] = True; s["last_task_id"] = task_id
        await query.message.reply_text(f"🚀 Задача отправлена. ⏳ Идёт процесс… {datetime.now().strftime('%H:%M:%S')}")
        await query.message.reply_text("🎥 Рендер запущен…")
        asyncio.create_task(poll_veo_and_send(update.effective_chat.id, task_id, ctx, expect_vertical=(s.get("aspect") == "9:16")))
        return

        gen_id = uuid.uuid4().hex[:12]
        s["generating"] = True; s["generation_id"] = gen_id; s["last_task_id"] = task_id
        mode_txt = "⚡ Fast" if s.get('model')=='veo3_fast' else "💎 Quality"
        await query.message.reply_text(
            f"🚀 Задача отправлена ({mode_txt}).\n🆔 taskId={task_id}\n"
            "🎛️ Подождите — подбираем кадры, свет и ритм…"
        )
        await query.message.reply_text("🎥 Рендер запущен… это может занять несколько минут.")
        asyncio.create_task(poll_veo_and_send(update.effective_chat.id, task_id, gen_id, ctx)); return

    # --- MJ запуск
    if data.startswith("mj:ar:"):
        ar = "16:9"
        prompt = s.get("last_prompt")
        if not prompt:
            await query.message.reply_text("⚠️ Сначала отправьте текстовый prompt."); return

        price = TOKEN_COSTS['mj']
    # --- Banana controls
    if data == "banana:start":
        imgs = s.get("banana_images") or []
        prompt = s.get("banana_prompt")
        if not imgs or not prompt:
            await query.message.reply_text("⚠️ Нужны фото (до 4) и подпись-промпт в одном сообщении."); return
        price = TOKEN_COSTS["banana"]
ok_balance, rest = try_charge(ctx, price)
if not ok_balance:
await query.message.reply_text(
                f"💎 Недостаточно токенов: нужно {price}, на балансе {rest}.\nПополните через Stars: {STARS_BUY_URL}",
                reply_markup=stars_topup_kb()
                f"💎 Недостаточно токенов: нужно {price}, на балансе *{rest}*.\n"
                f"Пополните через Stars: {STARS_BUY_URL}", parse_mode=ParseMode.MARKDOWN, reply_markup=stars_topup_kb()
); return

        await query.message.reply_text(f"🎨 Генерация фото запущена…\nФормат: *{ar}*\nPrompt: `{prompt}`",
                                       parse_mode=ParseMode.MARKDOWN)
        ok, task_id, msg = await asyncio.to_thread(mj_generate, prompt.strip(), ar)
        event("MJ_SUBMIT_RESP", ok=ok, task_id=task_id, msg=msg)
        ok, task_id, msg = await asyncio.to_thread(banana_generate, prompt, imgs)
if not ok or not task_id:
add_tokens(ctx, price)
            await query.message.reply_text(f"❌ Не удалось создать MJ-задачу: {msg}\n💎 Токены возвращены."); return
        await query.message.reply_text(f"🆔 MJ taskId: `{task_id}`\n🖌️ Рисую эскиз и детали…", parse_mode=ParseMode.MARKDOWN)
        asyncio.create_task(poll_mj_and_send_photos(update.effective_chat.id, task_id, ctx))
            await query.message.reply_text(f"❌ Не удалось создать Banana-задачу: {msg}\n💎 Токены возвращены."); return
        s["banana_task_id"] = task_id
        await query.message.reply_text(f"🚀 Banana запущен. ⏳ Идёт процесс… {datetime.now().strftime('%H:%M:%S')}")
        asyncio.create_task(poll_banana_and_send(update.effective_chat.id, task_id, ctx))
return
    if data == "banana:clear":
        s["banana_images"] = []; s["banana_prompt"] = None; s["banana_task_id"] = None
        await query.message.reply_text("🧹 Сессия Banana очищена."); return

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
        log.warning("show_card_veo failed: %s", e)

# ==========================
#   Message Handlers
# ==========================
async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
s = state(ctx)
text = (update.message.text or "").strip()
    mode = s.get("mode")

    # URL как картинка для VEO
low = text.lower()
if low.startswith(("http://", "https://")) and any(low.split("?")[0].endswith(ext) for ext in (".jpg",".jpeg",".png",".webp",".heic")):
        if mode == "banana":
            if len(s["banana_images"]) >= 4:
                await update.message.reply_text("⚠️ Достигнут лимит 4 фото. Нажмите «🚀 Начать генерацию (Banana)».", reply_markup=banana_kb(s)); return
            s["banana_images"].append(text.strip())
            await update.message.reply_text(f"📸 Фото принято ({len(s['banana_images'])}/4). Пришлите ещё или нажмите «🚀 Начать генерацию».", reply_markup=banana_kb(s))
            return
s["last_image_url"] = text.strip(); await update.message.reply_text("🧷 Ссылка на изображение принята."); await show_card_veo(update, ctx); return

    mode = s.get("mode")
if mode == "prompt_master":
        if len(text) == 0:
            await update.message.reply_text("✍️ Напишите идею по подсказке выше.")
            return
        if len(text) > 500:
            await update.message.reply_text("ℹ️ Урежу ввод до 500 символов для лучшего качества.")
        prompt = await oai_prompt_master(text[:500])
        if not prompt: await update.message.reply_text("⚠️ Prompt-Master недоступен или ответ пуст."); return
        s["last_prompt"] = prompt; await update.message.reply_text("🧠 Готово! Промпт добавлен в карточку."); await show_card_veo(update, ctx); return
        # Просто кладём текст в карточку
        s["last_prompt"] = text[:1000]
        await update.message.reply_text("🧠 Готово! Промпт добавлен в карточку."); await show_card_veo(update, ctx); return

    if mode == "mj_txt":
        s["last_prompt"] = text[:1000]
        await update.message.reply_text(
            f"✅ Prompt сохранён:\n\n`{s['last_prompt']}`\n\nНажмите ниже для запуска 16:9:",
            parse_mode=ParseMode.MARKDOWN, reply_markup=mj_start_kb()
        ); return

if mode == "chat":
if openai is None or not OPENAI_API_KEY:
@@ -1120,63 +980,12 @@ async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
await update.message.reply_text("⚠️ Ошибка запроса к ChatGPT.")
return

    if mode == "mj_txt":
        s["last_prompt"] = text
        await update.message.reply_text(
            f"✅ Prompt сохранён:\n\n`{text}`\n\nНажмите ниже для запуска 16:9:",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=mj_start_kb()
        )
        return

    if mode == "banana":
        s["last_prompt"] = text
        await update.message.reply_text("✍️ Промпт сохранён. Пришлите фото или нажмите «🚀 Начать генерацию (Banana)».", reply_markup=banana_kb(s))
        return

    # VEO по умолчанию
    s["last_prompt"] = text
    # по умолчанию — кладём текст в карточку VEO
    s["last_prompt"] = text[:2000]
await update.message.reply_text("🟦 *VEO — подготовка к рендеру*\nПроверь карточку ниже и жми «Сгенерировать».",
parse_mode=ParseMode.MARKDOWN)
await show_card_veo(update, ctx)

async def _banana_run_and_send(chat_id: int, ctx: ContextTypes.DEFAULT_TYPE, src_urls: List[str], prompt: str):
    try:
        task_id = await asyncio.to_thread(create_banana_task, prompt, src_urls, "png", "auto", None, None, 60)
        event("BANANA_SUBMIT_OK", task_id=task_id, imgs=len(src_urls))

        await ctx.bot.send_message(chat_id, f"🍌 Задача Banana создана.\n🆔 taskId={task_id}\nЖдём результат…")
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
@@ -1186,28 +995,28 @@ async def on_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
file = await ctx.bot.get_file(ph.file_id)
if not file.file_path:
await update.message.reply_text("⚠️ Не удалось получить путь к файлу Telegram."); return
        url = tg_direct_file_url(TELEGRAM_TOKEN, file.file_path)

        url = f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{file.file_path.lstrip('/')}"
if s.get("mode") == "banana":
            if len(s["banana_images"]) >= 4:
                await update.message.reply_text("⚠️ Достигнут лимит 4 фото. Нажмите «🚀 Начать генерацию (Banana)».", reply_markup=banana_kb(s)); return
            s["banana_images"].append(url)
            cap = (update.message.caption or "").strip()
            if cap:
                s["last_prompt"] = cap
            await update.message.reply_text(
                f"📸 Фото принято ({len(s['banana_images'])}/4). Пришлите ещё или нажмите «🚀 Начать генерацию».",
                reply_markup=banana_kb(s)
            )
            return

        s["last_image_url"] = url
        await update.message.reply_text("🖼️ Фото принято как референс."); await show_card_veo(update, ctx)
            imgs = s.get("banana_images") or []
            if len(imgs) >= 4:
                await update.message.reply_text("⚠️ Принято уже 4 фото. Нажмите «🚀 Начать генерацию»."); return
            imgs.append(url); s["banana_images"] = imgs
            # подпись как промпт
            if update.message.caption:
                s["banana_prompt"] = (update.message.caption or "").strip()
            await update.message.reply_text(f"📸 Фото принято ({len(imgs)}/4).", reply_markup=banana_ready_kb()); return
        else:
            s["last_image_url"] = url
            if update.message.caption:
                s["last_prompt"] = (update.message.caption or "").strip()
            await update.message.reply_text("🖼️ Фото принято как референс."); await show_card_veo(update, ctx)
except Exception as e:
log.exception("Get photo failed: %s", e)
await update.message.reply_text("⚠️ Не удалось обработать фото. Пришлите публичный URL картинки текстом.")

# ---------- Payments: Stars (XTR) ----------
# ==========================
#   Payments: Stars
# ==========================
async def precheckout_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
try:
await update.pre_checkout_query.answer(ok=True)
@@ -1223,67 +1032,96 @@ async def successful_payment_handler(update: Update, ctx: ContextTypes.DEFAULT_T
meta = json.loads(sp.invoice_payload)
except Exception:
meta = {}

stars = int(sp.total_amount)
    charge_id = getattr(sp, "telegram_payment_charge_id", None)
    if charge_id:
        # можно хранить последний charge_id в Redis при желании
        pass

if meta.get("kind") == "stars_pack":
tokens = int(meta.get("tokens", 0))
if tokens <= 0:
            # safety — рассчитать по звёздам
            map_dev = {1:1,100:100,200:200,300:300,400:400,500:500}
            map_prod = {100:100,200:200,300:300,400:400,500:500}
            tokens = (map_dev if DEV_MODE else map_prod).get(stars, 0)
            # safety fallback
            mapv = {s: d for (s, d, _) in STAR_PACKS}
            tokens = mapv.get(stars, stars)
add_tokens(ctx, tokens)
await update.message.reply_text(
            f"✅ Оплата получена: +{tokens} токенов.\nБаланс: {get_user_balance_value(ctx)} 💎"
            f"✅ Оплата получена: +*{tokens}* токенов.\nБаланс: *{get_user_balance_value(ctx)}* 💎",
            parse_mode=ParseMode.MARKDOWN
)
return
    await update.message.reply_text("✅ Оплата получена.")

    await update.message.reply_text("✅ Оплата получена. Баланс обновлён.")
# ==========================
#   Promo command
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
    # одноразово на пользователя: через Redis-флаг
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
#   Health / Errors
# ==========================
async def health(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    parts = [
        f"KIE_BASE: `{KIE_BASE_URL}`",
        f"VEO_GEN: `{KIE_VEO_GEN_PATH}`",
        f"VEO_STATUS: `{KIE_VEO_STATUS_PATH}`",
        f"MJ_GEN: `{KIE_MJ_GENERATE}`",
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

# ==========================
#   Entry
# ==========================
def main():
if not TELEGRAM_TOKEN: raise RuntimeError("TELEGRAM_TOKEN is not set")
    if not KIE_BASE_URL:   raise RuntimeError("KIE_BASE_URL is not set")
    if not KIE_API_KEY:    raise RuntimeError("KIE_API_KEY is not set")

    # ВАЖНО: удаляем вебхук перед polling — частая причина «бот не отвечает»
    # гарантированно снимаем вебхук для polling
try:
Bot(TELEGRAM_TOKEN).delete_webhook(drop_pending_updates=True)
        log.info("Webhook deleted (drop_pending_updates=True)")
    except Exception as e:
        log.warning("Delete webhook failed: %s", e)

    app = (ApplicationBuilder()
           .token(TELEGRAM_TOKEN)
           .rate_limiter(AIORateLimiter())
           .build())
    except Exception:
        pass

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).rate_limiter(AIORateLimiter()).build()
app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("health", health))
app.add_handler(CommandHandler("topup", topup))
    app.add_handler(CommandHandler("health", health))
    app.add_handler(CommandHandler("promo", promo_cmd))
app.add_handler(PreCheckoutQueryHandler(precheckout_callback))
app.add_handler(MessageHandler(filters.SUCCESSFUL_PAYMENT, successful_payment_handler))
app.add_handler(CallbackQueryHandler(on_callback))
app.add_handler(MessageHandler(filters.PHOTO, on_photo))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
app.add_error_handler(error_handler)

    log.info("Bot starting… (PTB polling, Redis=%s)", "on" if redis_client else "off")

    # run_polling блокирует поток до SIGTERM/SIGINT
    # stop_signals=None — не обязательно, но помогает на некоторых платформах.
    app.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
        stop_signals=None
    )
    log.info("Bot starting…")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
main()
