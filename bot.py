# -*- coding: utf-8 -*-
# Best VEO3 Bot — PTB 21.x
# Версия: 2025-09-12 (Redis balance + signup bonus + Banana 4 imgs + MJ prompt UI)
#
# Что внутри:
# - Redis-хранилище баланса/флагов → токены не теряются при деплоях.
# - Бонус новичка: +10 при первом /start (ENV SIGNUP_BONUS).
# - Цены: VEO Fast=50, Quality=200; MJ=15; Banana=5.
# - MJ: удобное меню (выбор 16:9/9:16, «✍️ Добавить описание», «🌆 Сгенерировать»), анти-спам «Рисую…».
# - Banana: сначала промпт → до 4 фото → запуск по кнопке.
# - Промокоды: /promo КОД (одноразово на пользователя), список в ENV PROMO_CODES.
# - Баланс в Welcome жирным.
# - Сообщения «⏳ запущено» перед рендером.

import os, json, time, uuid, asyncio, logging, tempfile, subprocess
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

# ==== ENV
load_dotenv()

def _env(k: str, d: str = "") -> str:
    v = os.getenv(k)
    return (v if v is not None else d).strip()

TELEGRAM_TOKEN      = _env("TELEGRAM_TOKEN")
PROMPTS_CHANNEL_URL = _env("PROMPTS_CHANNEL_URL", "https://t.me/bestveo3promts")
STARS_BUY_URL       = "https://t.me/PremiumBot"

# OpenAI (для Prompt-Master/чата — опционально)
OPENAI_API_KEY = _env("OPENAI_API_KEY")
try:
    import openai
    if OPENAI_API_KEY:
        openai.api_key = OPENAI_API_KEY
except Exception:
    openai = None

# ---- KIE
KIE_API_KEY  = _env("KIE_API_KEY")
KIE_BASE_URL = _env("KIE_BASE_URL", "https://api.kie.ai")
KIE_VEO_GEN_PATH    = _env("KIE_VEO_GEN_PATH",    _env("KIE_GEN_PATH",    "/api/v1/veo/generate"))
KIE_VEO_STATUS_PATH = _env("KIE_VEO_STATUS_PATH", _env("KIE_STATUS_PATH", "/api/v1/veo/record-info"))
KIE_VEO_1080_PATH   = _env("KIE_VEO_1080_PATH",   _env("KIE_HD_PATH",     "/api/v1/veo/get-1080p-video"))

# ---- MJ
KIE_MJ_GENERATE = _env("KIE_MJ_GENERATE", "/api/v1/mj/generate")
KIE_MJ_STATUS   = _env("KIE_MJ_STATUS",   "/api/v1/mj/record-info")

# ---- Upload API
UPLOAD_BASE_URL     = _env("UPLOAD_BASE_URL", "https://kieai.redpandaai.co")
UPLOAD_STREAM_PATH  = _env("UPLOAD_STREAM_PATH", "/api/file-stream-upload")
UPLOAD_URL_PATH     = _env("UPLOAD_URL_PATH",    "/api/file-url-upload")
UPLOAD_BASE64_PATH  = _env("UPLOAD_BASE64_PATH", "/api/file-base64-upload")

# ---- Видео-доставка
ENABLE_VERTICAL_NORMALIZE = _env("ENABLE_VERTICAL_NORMALIZE", "true").lower() == "true"
ALWAYS_FORCE_FHD          = _env("ALWAYS_FORCE_FHD", "true").lower() == "true"
FFMPEG_BIN                = _env("FFMPEG_BIN", "ffmpeg")
MAX_TG_VIDEO_MB           = int(_env("MAX_TG_VIDEO_MB", "48"))

POLL_INTERVAL_SECS = int(_env("POLL_INTERVAL_SECS", "6"))
POLL_TIMEOUT_SECS  = int(_env("POLL_TIMEOUT_SECS", str(20 * 60)))

# ---- Redis (персистентный баланс)
import redis
REDIS_URL    = _env("REDIS_URL")
REDIS_PREFIX = _env("REDIS_PREFIX", "veo3:prod")
SIGNUP_BONUS = int(_env("SIGNUP_BONUS", "10") or 0)

rdb = None
if REDIS_URL:
    try:
        rdb = redis.Redis.from_url(REDIS_URL, decode_responses=True, socket_timeout=5)
        rdb.ping()
    except Exception:
        rdb = None

def rk(*parts: str) -> str:
    return ":".join([REDIS_PREFIX, *[str(p) for p in parts]])

# ---- Промокоды
PROMO_CODES_ENV = _env("PROMO_CODES", "")

def parse_promo_map(s: str) -> Dict[str, int]:
    res: Dict[str,int] = {}
    for part in s.split(","):
        p = part.strip()
        if not p or "=" not in p: continue
        k, v = p.split("=", 1)
        try:
            res[k.strip().upper()] = int(v.strip())
        except: pass
    return res
PROMO_MAP = parse_promo_map(PROMO_CODES_ENV)

# ==== LOG
LOG_LEVEL = _env("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("veo3-bot")

# ==== Цены
TOKEN_COSTS = {
    "veo_fast": 50,
    "veo_quality": 200,
    "veo_photo": 50,
    "mj": 15,
    "banana": 5,
}

# ==== Helpers
def event(tag: str, **kw):
    try:
        log.info("EVT %s | %s", tag, json.dumps(kw, ensure_ascii=False))
    except Exception:
        log.info("EVT %s | %s", tag, kw)

def _nz(s: Optional[str]) -> Optional[str]:
    if not s: return None
    s2 = s.strip()
    return s2 if s2 else None

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
    for key in ("originUrls", "resultUrls"):
        urls = _coerce_url_list(data.get(key))
        if urls: return urls[0]
    for container in ("info", "response", "resultInfoJson"):
        v = data.get(container)
        if isinstance(v, dict):
            for key in ("originUrls", "resultUrls", "videoUrls"):
                urls = _coerce_url_list(v.get(key))
                if urls: return urls[0]
    def walk(x):
        if isinstance(x, dict):
            for vv in x.values():
                r = walk(vv)
                if r: return r
        elif isinstance(x, list):
            for vv in x:
                r = walk(vv)
                if r: return r
        elif isinstance(x, str):
            s = x.strip().split("?")[0].lower()
            if s.startswith("http") and s.endswith((".mp4", ".mov", ".webm")):
                return x.strip()
        return None
    return walk(data)

def tg_direct_file_url(bot_token: str, file_path: str) -> str:
    p = (file_path or "").strip()
    if p.startswith("http://") or p.startswith("https://"):
        return p
    return f"https://api.telegram.org/file/bot{bot_token}/{p.lstrip('/')}"

# ==== Redis balance helpers
def _uid(ctx): return ctx.user_data.get("uid")

def get_user_balance_value(ctx) -> int:
    if rdb and _uid(ctx):
        try:
            val = rdb.get(rk("balance", _uid(ctx)))
            return int(val or 0)
        except Exception as e:
            log.warning("redis get balance fail: %s", e)
    return int(ctx.user_data.get("balance", 0))

def set_user_balance_value(ctx, v: int):
    v = max(0, int(v))
    if rdb and _uid(ctx):
        try:
            rdb.set(rk("balance", _uid(ctx)), v)
            ctx.user_data["balance"] = v
            return
        except Exception as e:
            log.warning("redis set balance fail: %s", e)
    ctx.user_data["balance"] = v

def add_tokens(ctx, add: int):
    add = int(add)
    if add == 0: return
    if rdb and _uid(ctx):
        try:
            newv = rdb.incrby(rk("balance", _uid(ctx)), add)
            ctx.user_data["balance"] = max(0, int(newv)); return
        except Exception as e:
            log.warning("redis incr fail: %s", e)
    set_user_balance_value(ctx, get_user_balance_value(ctx) + add)

def try_charge(ctx, need: int) -> Tuple[bool,int]:
    need = int(need)
    if need <= 0: return True, get_user_balance_value(ctx)
    if rdb and _uid(ctx):
        LUA = """
        local k = KEYS[1]
        local need = tonumber(ARGV[1])
        local cur = tonumber(redis.call('GET', k) or '0')
        if cur >= need then
          local left = cur - need
          redis.call('SET', k, left)
          return {1, left}
        else
          return {0, cur}
        end
        """
        try:
            ok, left = rdb.eval(LUA, 1, rk("balance", _uid(ctx)), need)
            ctx.user_data["balance"] = int(left)
            return (int(ok)==1), int(left)
        except Exception as e:
            log.warning("redis charge fail: %s", e)
    bal = get_user_balance_value(ctx)
    if bal < need: return False, bal
    set_user_balance_value(ctx, bal - need)
    return True, bal - need

# ==== State/UI
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
    "chat_unlocked": True,  # чат бесплатный
    # MJ
    "mj_wait_sent": False,
    "mj_wait_last_ts": 0.0,
    "mj_aspect": "16:9",
    "awaiting_mj_prompt": False,
    # Banana
    "banana_photos": [],
    "banana_prompt": None,
    "banana_generating": False,
    "banana_task_id": None,
    "banana_wait_sent": False,
}

def state(ctx: ContextTypes.DEFAULT_TYPE) -> Dict[str, Any]:
    ud = ctx.user_data
    for k, v in DEFAULT_STATE.items():
        ud.setdefault(k, v)
    return ud

WELCOME = (
    "🎬 Veo 3 — съёмочная команда: опиши идею и получи готовый клип!\n"
    "🖌️ MJ — художник: нарисует изображение по твоему тексту.\n"
    "🍌 Banana — Редактор изображений из будущего\n"
    "🧠 Prompt-Master — вернёт профессиональный кинопромпт.\n"
    "💬 Обычный чат — общение с ИИ.\n"
    "💎 Ваш баланс: {balance}\n"
    "📈 Больше идей и примеров: {prompts_url}\n\n"
    "Выберите режим 👇"
)

def main_menu_kb() -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton("🎬 Генерация видео (Veo Fast) 💎50", callback_data="mode:veo_text_fast")],
        [InlineKeyboardButton("🎬 Генерация видео (Veo Quality) 💎200", callback_data="mode:veo_text_quality")],
        [InlineKeyboardButton("🖼️ Генерация изображений (MJ) 💎15", callback_data="mode:mj_txt")],
        [InlineKeyboardButton("🍌 Редактор изображений (Banana) 💎5", callback_data="mode:banana")],
        [InlineKeyboardButton("📸 Оживить изображение (Veo) 💎50", callback_data="mode:veo_photo")],
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

# ---- MJ mini UI
def mj_menu_kb(current_ar: str = "16:9") -> InlineKeyboardMarkup:
    ar = (current_ar or "16:9")
    row_ar = [
        InlineKeyboardButton(("16:9 ✅" if ar=="16:9" else "16:9"), callback_data="mj:set_ar:16:9"),
        InlineKeyboardButton(("9:16 ✅" if ar=="9:16" else "9:16"),  callback_data="mj:set_ar:9:16"),
    ]
    row_prompt = [InlineKeyboardButton("✍️ Добавить описание (промпт)", callback_data="mj:edit_prompt")]
    row_run = [InlineKeyboardButton("🌆 Сгенерировать", callback_data="mj:start")]
    row_back = [InlineKeyboardButton("⬅️ Назад", callback_data="back")]
    return InlineKeyboardMarkup([row_ar, row_prompt, row_run, row_back])

# ==== Prompt-Master (минимум)
async def oai_prompt_master(idea_text: str) -> Optional[str]:
    if openai is None or not OPENAI_API_KEY:
        return None
    system = (
        "You are a Prompt-Master for cinematic AI video (Veo-style). "
        "Return ONE prompt with labels:\n"
        "High-quality cinematic 4K video (16:9).\n"
        "Scene: ...\nCamera: ...\nAction: ...\nDialogue: ...\nLip-sync: ...\nAudio: ...\n"
        "Lighting: ...\nWardrobe/props: ...\nFraming: ...\nConstraints: No subtitles. No on-screen text. No logos."
    )
    try:
        user = idea_text.strip()[:800]
        resp = await asyncio.to_thread(
            openai.ChatCompletion.create,
            model="gpt-4o-mini",
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            temperature=0.8, max_tokens=800,
        )
        return (resp["choices"][0]["message"]["content"] or "").strip()[:1400]
    except Exception:
        return None

# ==== Upload helpers
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
    except Exception:
        return None
    try:
        url = join_url(UPLOAD_BASE_URL, UPLOAD_STREAM_PATH)
        with open(local, "rb") as f:
            files = {"file": (os.path.basename(local), f)}
            data  = {"uploadPath": upload_path, "fileName": os.path.basename(local)}
            r = requests.post(url, headers=_upload_headers(), files=files, data=data, timeout=timeout)
        j = r.json() if r.headers.get("content-type","").startswith("application/json") else {"code": r.status_code, "raw": r.text}
        if (r.status_code == 200) and (j.get("code",200)==200):
            d = j.get("data") or {}
            u = d.get("downloadUrl") or d.get("fileUrl")
            if _nz(u): return u
    finally:
        try: os.unlink(local)
        except Exception: pass
    return None

# ==== VEO API
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
    img_for_kie = None
    if _nz(image_url):
        img_for_kie = upload_image_stream(image_url) or image_url

    payload = _build_payload_for_veo(prompt, aspect, img_for_kie, model_key)
    status, j = _post_json(join_url(KIE_BASE_URL, KIE_VEO_GEN_PATH), payload)
    code = j.get("code", status)
    if status == 200 and code == 200:
        tid = _extract_task_id(j)
        if tid: return True, tid, "Задача создана."
        return False, None, "Ответ KIE без taskId."
    return False, None, (j.get("message") or j.get("error") or f"HTTP {status}")

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
    return False, None, (j.get("message") or j.get("error")), None

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

# ==== MJ API
def mj_generate(prompt: str, ar: str) -> Tuple[bool, Optional[str], str]:
    payload = {
        "taskType": "mj_txt2img",
        "prompt": prompt,
        "speed": "turbo",
        "aspectRatio": "9:16" if (ar or "").strip() == "9:16" else "16:9",
        "version": "7",
        "enableTranslation": True,
    }
    status, j = _post_json(join_url(KIE_BASE_URL, KIE_MJ_GENERATE), payload)
    code = j.get("code", status)
    if status == 200 and code == 200:
        tid = _extract_task_id(j)
        if tid: return True, tid, "MJ задача создана."
        return False, None, "Ответ MJ без taskId."
    return False, None, (j.get("message") or j.get("error") or f"HTTP {status}")

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

# ==== ffmpeg utils / sending video
def _ffmpeg_available() -> bool:
    from shutil import which
    return bool(which(FFMPEG_BIN))

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
    except Exception:
        return False

async def send_video_with_fallback(ctx: ContextTypes.DEFAULT_TYPE, chat_id: int, url: str) -> bool:
    try:
        await ctx.bot.send_video(chat_id=chat_id, video=url, supports_streaming=True)
        return True
    except Exception:
        pass

    tmp_path = None
    try:
        r = requests.get(url, stream=True, timeout=180); r.raise_for_status()
        ext = ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
            for c in r.iter_content(256*1024):
                if c: f.write(c)
            tmp_path = f.name

        if ALWAYS_FORCE_FHD and _ffmpeg_available():
            out = tmp_path + "_1080.mp4"
            if _ffmpeg_force_16x9_fhd(tmp_path, out, MAX_TG_VIDEO_MB):
                try:
                    with open(out, "rb") as f:
                        await ctx.bot.send_video(chat_id=chat_id, video=InputFile(f, filename="result_1080p.mp4"),
                                                 supports_streaming=True)
                    return True
                except Exception:
                    with open(out, "rb") as f:
                        await ctx.bot.send_document(chat_id=chat_id, document=InputFile(f, filename="result_1080p.mp4"))
                    return True

        with open(tmp_path, "rb") as f:
            await ctx.bot.send_video(chat_id=chat_id, video=InputFile(f, filename=f"result{ext}"),
                                     supports_streaming=True)
        return True
    except Exception:
        try:
            await ctx.bot.send_message(chat_id, f"🔗 Результат готов, но вложить файл не удалось. Ссылка:\n{url}")
            return True
        except Exception:
            return False
    finally:
        if tmp_path:
            try: os.unlink(tmp_path)
            except Exception: pass

# ==== Pollers
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
            if not ok:
                add_tokens(ctx, TOKEN_COSTS['veo_quality'] if s.get('model') == 'veo3' else TOKEN_COSTS['veo_fast'])
                await ctx.bot.send_message(chat_id, f"❌ Ошибка статуса: {msg or 'неизвестно'}\n💎 Токены возвращены."); break

            if _nz(res_url):
                final_url = res_url
                if (s.get("aspect") or "16:9") == "16:9":
                    u1080 = await asyncio.to_thread(try_get_1080_url, task_id)
                    if _nz(u1080): final_url = u1080
                await ctx.bot.send_message(chat_id, "🎞️ Рендер завершён — отправляю файл…")
                sent = await send_video_with_fallback(ctx, chat_id, final_url)
                s["last_result_url"] = final_url if sent else None
                await ctx.bot.send_message(
                    chat_id, "✅ *Готово!*", parse_mode=ParseMode.MARKDOWN,
                    reply_markup=InlineKeyboardMarkup(
                        [[InlineKeyboardButton("🚀 Сгенерировать ещё", callback_data="start_new_cycle")]]
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
    finally:
        if s.get("generation_id") == gen_id:
            s["generating"] = False
            s["generation_id"] = None

async def poll_mj_and_send_photos(chat_id: int, task_id: str, ctx: ContextTypes.DEFAULT_TYPE):
    s = state(ctx)
    start_ts = time.time()
    delay = 6
    last_ping = 0.0
    max_wait = 15 * 60
    try:
        while True:
            ok, flag, data = await asyncio.to_thread(mj_status, task_id)
            if not ok:
                add_tokens(ctx, TOKEN_COSTS["mj"])
                await ctx.bot.send_message(chat_id, "❌ MJ сейчас недоступен. 💎 Токены возвращены."); return

            if flag == 0:
                now = time.time()
                if (now - last_ping) >= 40:
                    await ctx.bot.send_message(chat_id, "🖼️✨ Рисую… Подождите немного.", disable_notification=True)
                    last_ping = now
                if (now - start_ts) > max_wait:
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
                    chat_id, "✅ *Готово!*", parse_mode=ParseMode.MARKDOWN,
                    reply_markup=InlineKeyboardMarkup(
                        [[InlineKeyboardButton("🚀 Сгенерировать ещё", callback_data="start_new_cycle")]]
                    ),
                )
                return
    except Exception:
        add_tokens(ctx, TOKEN_COSTS["mj"])
        try: await ctx.bot.send_message(chat_id, "💥 Внутренняя ошибка при опросе MJ. 💎 Токены возвращены.")
        except Exception: pass

# ==== Banana (через отдельный модуль)
from kie_banana import create_banana_task as banana_create, wait_for_banana_result as banana_wait

def banana_kb():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("➕ Добавить ещё фото", callback_data="banana:add_hint")],
        [InlineKeyboardButton("🧹 Очистить фото", callback_data="banana:reset")],
        [InlineKeyboardButton("✍️ Изменить промпт", callback_data="banana:edit_prompt")],
        [InlineKeyboardButton("🚀 Начать генерацию Banana", callback_data="banana:start")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="back")]
    ])

def banana_help_text(s):
    n = len(s.get("banana_photos") or [])
    has_p = "да" if s.get("banana_prompt") else "нет"
    return (
        "🍌 *Banana включён*\n"
        "Сначала пришлите *текст-промпт* (англ/рус). Затем добавьте до *4 фото*.\n"
        "Когда всё готово — нажмите «🚀 Начать генерацию Banana».\n\n"
        f"📸 Фото: *{n}/4*, Промпт: *{has_p}*"
    )

async def poll_banana_and_send(chat_id: int, task_id: str, ctx: ContextTypes.DEFAULT_TYPE):
    last_ping = 0.0
    try:
        while True:
            ok, state_name, result_url, _raw = await asyncio.to_thread(
                banana_wait, KIE_BASE_URL, _kie_headers_json(), task_id
            )
            if not ok:
                add_tokens(ctx, TOKEN_COSTS["banana"])
                await ctx.bot.send_message(chat_id, "❌ Banana сейчас недоступен. 💎 Токены возвращены.")
                break

            if state_name in ("waiting", "queuing", "generating"):
                now = time.time()
                if now - last_ping >= 40:
                    await ctx.bot.send_message(chat_id, "✨ Рисую… Подождите немного.", disable_notification=True)
                    last_ping = now
                await asyncio.sleep(6)
                continue

            if state_name == "success" and result_url:
                await ctx.bot.send_message(chat_id, "🎉 Готово! Отправляю файл…")
                try:
                    await ctx.bot.send_photo(chat_id=chat_id, photo=result_url)
                except Exception:
                    await ctx.bot.send_message(chat_id, f"🔗 Ссылка на результат:\n{result_url}")
                await ctx.bot.send_message(
                    chat_id, "✅ *Готово!*", parse_mode=ParseMode.MARKDOWN,
                    reply_markup=InlineKeyboardMarkup(
                        [[InlineKeyboardButton("🚀 Сгенерировать ещё", callback_data="start_new_cycle")]]
                    ),
                )
                break

            if state_name == "fail":
                add_tokens(ctx, TOKEN_COSTS["banana"])
                await ctx.bot.send_message(chat_id, "❌ Banana: ошибка рендера. 💎 Токены возвращены.")
                break
    except Exception:
        add_tokens(ctx, TOKEN_COSTS["banana"])
        try: await ctx.bot.send_message(chat_id, "💥 Ошибка опроса Banana. 💎 Токены возвращены.")
        except Exception: pass

# ==== Handlers
async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    ctx.user_data["uid"] = update.effective_user.id

    # бонус новичка (один раз)
    if rdb:
        try:
            if rdb.setnx(rk("signup_done", _uid(ctx)), "1"):
                if SIGNUP_BONUS > 0: add_tokens(ctx, SIGNUP_BONUS)
        except Exception as e:
            log.warning("signup bonus fail: %s", e)
    else:
        if not ctx.user_data.get("signup_done"):
            ctx.user_data["signup_done"] = True
            if SIGNUP_BONUS > 0: add_tokens(ctx, SIGNUP_BONUS)

    await update.message.reply_text(
        WELCOME.format(balance=f"*{get_user_balance_value(ctx)}*", prompts_url=PROMPTS_CHANNEL_URL),
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=main_menu_kb()
    )

async def promo_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    ctx.user_data["uid"] = update.effective_user.id
    parts = (update.message.text or "").split(maxsplit=1)
    if len(parts) < 2:
        await update.message.reply_text("Использование: `/promo КОД`", parse_mode=ParseMode.MARKDOWN)
        return
    code = parts[1].strip().upper()
    bonus = PROMO_MAP.get(code)
    if not bonus:
        await update.message.reply_text("❌ Неизвестный промокод."); return

    if rdb:
        try:
            if not rdb.setnx(rk("promo_used", code, _uid(ctx)), "1"):
                await update.message.reply_text("⚠️ Этот промокод уже был активирован."); return
        except Exception as e:
            log.warning("promo setnx fail: %s", e)
    else:
        used = ctx.user_data.setdefault("promo_used", set())
        if code in used:
            await update.message.reply_text("⚠️ Этот промокод уже был активирован."); return
        used.add(code)

    add_tokens(ctx, bonus)
    await update.message.reply_text(
        f"✅ Промокод активирован! +{bonus}💎\nБаланс: *{get_user_balance_value(ctx)}💎*",
        parse_mode=ParseMode.MARKDOWN
    )

async def topup(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "💳 Пополнение токенов через *Telegram Stars*.\n"
        f"Если не хватает звёзд — купите их в официальном боте: {STARS_BUY_URL}",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("⭐ 50 → 💎 50",  callback_data="buy:stars:50")],
            [InlineKeyboardButton("⭐ 100 → 💎 110", callback_data="buy:stars:100")],
            [InlineKeyboardButton("⭐ 200 → 💎 220", callback_data="buy:stars:200")],
            [InlineKeyboardButton("⭐ 300 → 💎 330", callback_data="buy:stars:300")],
            [InlineKeyboardButton("⭐ 400 → 💎 440", callback_data="buy:stars:400")],
            [InlineKeyboardButton("⭐ 500 → 💎 550", callback_data="buy:stars:500")],
            [InlineKeyboardButton("🛒 Где купить Stars", url=STARS_BUY_URL)],
            [InlineKeyboardButton("⬅️ Назад", callback_data="back")],
        ])
    )

def tokens_for_stars(stars: int) -> int:
    bonus_map = {50:50, 100:110, 200:220, 300:330, 400:440, 500:550}
    return int(bonus_map.get(int(stars), 0))

async def health(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🩺 OK")

async def error_handler(update: Optional[Update], context: ContextTypes.DEFAULT_TYPE):
    log.exception("Unhandled error: %s", context.error)
    try:
        if update and update.effective_chat:
            await context.bot.send_message(update.effective_chat.id, "⚠️ Системная ошибка. Попробуйте ещё раз.")
    except Exception:
        pass

# MJ / Banana / VEO callbacks
async def on_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    data = (query.data or "").strip()
    try: await query.answer()
    except Exception: pass
    s = state(ctx)

    if data == "faq":
        await query.message.reply_text(
            "📖 *FAQ — Вопросы и ответы*\n\n"
            "⭐ *Общая инфа*\n"
            "Что умеет бот? Видео (VEO), картинки (MJ), редактирование фото (Banana), Prompt-Master и чат.\n"
            "💎 Звёзды — внутренняя валюта бота. Покупаются через Telegram.\n\n"
            "🎬 *VEO*\n"
            "• Fast — 2–5 мин. Стоимость: 50💎\n"
            "• Quality — 5–10 мин. Стоимость: 200💎\n"
            "• Animate — оживление фото (50💎)\n\n"
            "🖼️ *MJ*\n"
            "• Стоимость: 15💎\n"
            "• Время: 30–90 сек\n"
            "• Форматы: 16:9 или 9:16\n\n"
            "🍌 *Banana*\n"
            "• Стоимость: 5💎\n"
            "• До 4 фото + промпт\n"
            "• Генерация только после нажатия «🚀 Начать»\n\n"
            "🧠 *Prompt-Master*\n"
            "• Бесплатно. Опишите локацию, стиль, настроение, свет, камеру — получите сильный промпт.\n\n"
            "💬 *ChatGPT*\n"
            "• Бесплатно.\n\n"
            "❓ *Сколько ждать?*\n"
            "• Banana/MJ — до 2 минут\n"
            "• VEO Fast — 2–5 минут\n"
            "• VEO Quality — 5–10 минут\n\n"
            "Где пополнить? — «Пополнить баланс». Русский/английский — оба подходят.",
            parse_mode=ParseMode.MARKDOWN
        ); return

    if data == "back":
        s.update({**DEFAULT_STATE})
        await query.message.reply_text("🏠 Главное меню:",
                                       reply_markup=main_menu_kb())
        return

    if data == "topup_open":
        await topup(update.effective_message, ctx); return

    # Покупка Stars (через инвойс XTR)
    if data.startswith("buy:stars:"):
        stars = int(data.split(":")[-1])
        tokens = tokens_for_stars(stars)
        if tokens <= 0:
            await query.message.reply_text("⚠️ Такой пакет недоступен."); return
        title = f"{stars}⭐ → {tokens}💎"
        try:
            await ctx.bot.send_invoice(
                chat_id=update.effective_chat.id,
                title=title,
                description="Пакет пополнения токенов",
                payload=json.dumps({"kind": "stars_pack", "stars": stars, "tokens": tokens}),
                provider_token="",
                currency="XTR",
                prices=[LabeledPrice(label=title, amount=stars)],
            )
        except Exception:
            await query.message.reply_text(
                f"Если счёт не открылся — купите 1⭐ в {STARS_BUY_URL} и попробуйте снова."
            )
        return

    # Режимы
    if data == "mode:veo_text_fast":
        s["mode"]="veo_text"; s["model"]="veo3_fast"; s["aspect"]="16:9"
        await query.message.reply_text("📝 Пришлите идею/промпт для видео (Fast).")
        await query.message.reply_text(build_card_text_veo(s), parse_mode=ParseMode.MARKDOWN, reply_markup=card_keyboard_veo(s)); return
    if data == "mode:veo_text_quality":
        s["mode"]="veo_text"; s["model"]="veo3"; s["aspect"]="16:9"
        await query.message.reply_text("📝 Пришлите идею/промпт для видео (Quality).")
        await query.message.reply_text(build_card_text_veo(s), parse_mode=ParseMode.MARKDOWN, reply_markup=card_keyboard_veo(s)); return
    if data == "mode:veo_photo":
        s["mode"]="veo_photo"; s["model"]="veo3_fast"; s["aspect"]="9:16"
        await query.message.reply_text("🖼️ Пришлите фото (подпись-промпт — по желанию).")
        await query.message.reply_text(build_card_text_veo(s), parse_mode=ParseMode.MARKDOWN, reply_markup=card_keyboard_veo(s)); return
    if data == "mode:prompt_master":
        await query.message.reply_text(
            "🧠 *Prompt-Master готов!*\n"
            "Напишите кратко идею сцены (1–2 предложения), локацию, стиль, настроение, свет/цвет, ключевые предметы, "
            "динамику камеры. Если есть реплики — добавьте в кавычках.",
            parse_mode=ParseMode.MARKDOWN
        ); return
    if data == "mode:chat":
        await query.message.reply_text("✍️ Напишите вопрос для ChatGPT."); return
    if data == "mode:mj_txt":
        s["mode"]="mj_txt"; s["mj_aspect"]=s.get("mj_aspect") or "16:9"; s["awaiting_mj_prompt"]=False
        await query.message.reply_text(
            "🖼️ Пришлите текстовый prompt для картинки или нажмите «✍️ Добавить описание (промпт)».",
            reply_markup=mj_menu_kb(s["mj_aspect"])
        ); return
    if data == "mode:banana":
        s["mode"]="banana"; s["banana_photos"]=[]; s["banana_prompt"]=None; s["banana_generating"]=False
        await query.message.reply_text("✍️ Пришлите текст-промпт для Banana.", reply_markup=banana_kb())
        await query.message.reply_text(banana_help_text(s), parse_mode=ParseMode.MARKDOWN, reply_markup=banana_kb())
        return

    # VEO параметры
    if data.startswith("aspect:"):
        _, val = data.split(":", 1); s["aspect"] = "9:16" if val.strip() == "9:16" else "16:9"
        await query.message.reply_text(build_card_text_veo(s), parse_mode=ParseMode.MARKDOWN, reply_markup=card_keyboard_veo(s)); return
    if data.startswith("model:"):
        _, val = data.split(":", 1); s["model"] = "veo3" if val.strip() == "veo3" else "veo3_fast"
        await query.message.reply_text(build_card_text_veo(s), parse_mode=ParseMode.MARKDOWN, reply_markup=card_keyboard_veo(s)); return

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
        await query.message.reply_text("🗂️ Карточка очищена.")
        await query.message.reply_text(build_card_text_veo(s), parse_mode=ParseMode.MARKDOWN, reply_markup=card_keyboard_veo(s)); return
    if data == "card:generate":
        if s.get("generating"): await query.message.reply_text("⏳ Уже рендерю это видео — подождите чуть-чуть."); return
        if not s.get("last_prompt"): await query.message.reply_text("✍️ Сначала укажите текст промпта."); return
        price = TOKEN_COSTS['veo_quality'] if s.get('model') == 'veo3' else TOKEN_COSTS['veo_fast']
        ok_balance, rest = try_charge(ctx, price)
        if not ok_balance:
            await query.message.reply_text(
                f"💎 Недостаточно токенов: нужно {price}, на балансе {rest}.\n"
                f"Пополните через Stars: {STARS_BUY_URL}", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("💳 Пополнить баланс", callback_data="topup_open")]])
            ); return
        ok, task_id, msg = await asyncio.to_thread(
            submit_kie_veo, (s["last_prompt"] or "").strip(), s.get("aspect","16:9"),
            s.get("last_image_url"), s.get("model","veo3_fast")
        )
        if not ok or not task_id:
            add_tokens(ctx, price)
            await query.message.reply_text(f"❌ Не удалось создать VEO-задачу: {msg}\n💎 Токены возвращены."); return
        gen_id = uuid.uuid4().hex[:12]
        s["generating"] = True; s["generation_id"]=gen_id; s["last_task_id"]=task_id
        await query.message.reply_text(
            f"🚀 Задача отправлена ({'⚡ Fast' if s.get('model')=='veo3_fast' else '💎 Quality'}).\n🆔 taskId={task_id}\n"
            "🎛️ Подождите — подбираем кадры, свет и ритм…"
        )
        await query.message.reply_text("🎥 Рендер запущен… ⏳")
        asyncio.create_task(poll_veo_and_send(update.effective_chat.id, task_id, gen_id, ctx)); return

    # MJ кнопки
    if data.startswith("mj:set_ar:"):
        _, _, ar = data.split(":", 2)
        s["mj_aspect"] = "9:16" if ar.strip()=="9:16" else "16:9"
        await query.message.reply_text(
            f"🖼️ Формат установлен: *{s['mj_aspect']}*.",
            parse_mode=ParseMode.MARKDOWN, reply_markup=mj_menu_kb(s["mj_aspect"])
        ); return
    if data == "mj:edit_prompt":
        s["awaiting_mj_prompt"] = True
        await query.message.reply_text(
            "✍️ Напишите описание/идею для изображения.\n"
            "Подсказка: укажите предмет/персонажа, стиль, цвет/свет, атмосферу, фон, ракурс.",
            reply_markup=mj_menu_kb(s.get("mj_aspect") or "16:9")
        ); return
    if data == "mj:start":
        prompt = (s.get("last_prompt") or "").strip()
        if not prompt:
            await query.message.reply_text(
                "⚠️ Сначала добавьте описание — нажмите «✍️ Добавить описание (промпт)».",
                reply_markup=mj_menu_kb(s.get("mj_aspect") or "16:9")
            ); return
        ar = s.get("mj_aspect") or "16:9"
        price = TOKEN_COSTS["mj"]
        ok_balance, rest = try_charge(ctx, price)
        if not ok_balance:
            await query.message.reply_text(
                f"💎 Недостаточно токенов: нужно {price}, на балансе {rest}.\n"
                f"Пополните через Stars: {STARS_BUY_URL}",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("💳 Пополнить баланс", callback_data="topup_open")]])
            ); return
        await query.message.reply_text(
            f"🎨 Генерация фото запущена…\nФормат: *{ar}*\nPrompt: {prompt}",
            parse_mode=ParseMode.MARKDOWN
        )
        s["mj_wait_sent"] = False; s["mj_wait_last_ts"] = 0.0
        ok, task_id, msg = await asyncio.to_thread(mj_generate, prompt, ar)
        if not ok or not task_id:
            add_tokens(ctx, price)
            await query.message.reply_text(f"❌ Не удалось создать MJ-задачу: {msg}\n💎 Токены возвращены."); return
        await query.message.reply_text(f"🆔 MJ taskId: `{task_id}`\n🖌️ Рисую эскиз и детали…", parse_mode=ParseMode.MARKDOWN)
        asyncio.create_task(poll_mj_and_send_photos(update.effective_chat.id, task_id, ctx)); return

    # Banana кнопки
    if data == "banana:add_hint":
        await query.message.reply_text("Пришлите ещё фото (до 4).", reply_markup=banana_kb()); return
    if data == "banana:reset":
        s["banana_photos"] = []; s["banana_prompt"]=None; s["banana_generating"]=False; s["banana_task_id"]=None
        await query.message.reply_text("🧹 Сессия Banana очищена.", reply_markup=banana_kb()); return
    if data == "banana:edit_prompt":
        await query.message.reply_text("✍️ Пришлите новый текст-промпта для Banana.", reply_markup=banana_kb()); return
    if data == "banana:start":
        if s.get("banana_generating"):
            await query.message.reply_text("⏳ Задача уже запущена. Подождите немного."); return
        photos = s.get("banana_photos") or []
        prompt = (s.get("banana_prompt") or "").strip()
        if not prompt:
            await query.message.reply_text("⚠️ Сначала пришлите текст-промпт.", reply_markup=banana_kb()); return
        if not photos:
            await query.message.reply_text("⚠️ Нужно минимум 1 фото.", reply_markup=banana_kb()); return
        ok_balance, rest = try_charge(ctx, TOKEN_COSTS["banana"])
        if not ok_balance:
            await query.message.reply_text(
                f"💎 Недостаточно токенов: нужно {TOKEN_COSTS['banana']}, на балансе {rest}.\nПополните через Stars: {STARS_BUY_URL}",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("💳 Пополнить баланс", callback_data="topup_open")]])
            ); return
        await query.message.reply_text("🍌 Запускаю Banana… ⏳", reply_markup=banana_kb())
        ok, task_id, err = await asyncio.to_thread(
            banana_create, KIE_BASE_URL, _kie_headers_json(), prompt, photos, "png", "auto", None
        )
        if not ok or not task_id:
            add_tokens(ctx, TOKEN_COSTS["banana"])
            await query.message.reply_text(f"❌ Не удалось создать Banana-задачу: {err or 'ошибка'}\n💎 Токены возвращены.", reply_markup=banana_kb()); return
        s["banana_generating"]=True; s["banana_task_id"]=task_id
        await query.message.reply_text(f"🆔 taskId=`{task_id}`\n🖌️ Ждём результат…", parse_mode=ParseMode.MARKDOWN, reply_markup=banana_kb())
        asyncio.create_task(poll_banana_and_send(update.effective_chat.id, task_id, ctx)); return

    if data == "start_new_cycle":
        s.update({**DEFAULT_STATE})
        await query.message.reply_text("Выберите режим:", reply_markup=main_menu_kb()); return

async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = state(ctx)
    text = (update.message.text or "").strip()

    # MJ промпт — если пользователь нажал «Добавить описание»
    if s.get("mode") == "mj_txt" and s.get("awaiting_mj_prompt"):
        s["last_prompt"] = text
        s["awaiting_mj_prompt"] = False
        await update.message.reply_text(
            f"✅ Описание сохранено.\n\n`{text}`\n\nПроверьте формат и жмите «🌆 Сгенерировать».",
            parse_mode=ParseMode.MARKDOWN, reply_markup=mj_menu_kb(s.get("mj_aspect") or "16:9")
        )
        return

    # Banana промпт
    if s.get("mode") == "banana" and not s.get("banana_prompt"):
        s["banana_prompt"] = text
        await update.message.reply_text("✍️ Промпт сохранён. Теперь пришлите до 4 фото.", reply_markup=banana_kb())
        return

    # Prompt-Master
    if s.get("mode") == "prompt_master":
        prompt = await oai_prompt_master(text[:500]) if text else None
        if not prompt:
            await update.message.reply_text("⚠️ Prompt-Master недоступен или вернул пусто."); return
        s["last_prompt"] = prompt
        await update.message.reply_text("🧠 Готово! Промпт добавлен в карточку.")
        return

    if s.get("mode") == "chat":
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
        except Exception:
            await update.message.reply_text("⚠️ Ошибка запроса к ChatGPT.")
        return

    # VEO/MJ общий текст как последний промпт
    s["last_prompt"] = text
    await update.message.reply_text("🟦 *VEO — подготовка к рендеру*\nПроверь карточку ниже и жми «Сгенерировать».",
                                    parse_mode=ParseMode.MARKDOWN)
    await update.message.reply_text(build_card_text_veo(s), parse_mode=ParseMode.MARKDOWN, reply_markup=card_keyboard_veo(s))

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
    except Exception:
        await update.message.reply_text("⚠️ Не удалось обработать фото."); return

    if s.get("mode") == "banana":
        s.setdefault("banana_photos", [])
        if len(s["banana_photos"]) >= 4:
            await update.message.reply_text("ℹ️ Уже 4/4 фото. Удалите лишнее или запускайте генерацию.", reply_markup=banana_kb()); return
        s["banana_photos"].append(url)
        await update.message.reply_text(f"📸 Фото принято (*{len(s['banana_photos'])}/4*).",
                                        parse_mode=ParseMode.MARKDOWN, reply_markup=banana_kb())
        return

    s["last_image_url"] = url
    await update.message.reply_text("🖼️ Фото принято как референс.")
    await update.message.reply_text(build_card_text_veo(s), parse_mode=ParseMode.MARKDOWN, reply_markup=card_keyboard_veo(s))

# Payments (минимально)
async def precheckout_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try: await update.pre_checkout_query.answer(ok=True)
    except Exception:
        await update.pre_checkout_query.answer(ok=False, error_message=f"Платёж отклонён. Пополните Stars в {STARS_BUY_URL}")

async def successful_payment_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    sp = update.message.successful_payment
    try: meta = json.loads(sp.invoice_payload)
    except Exception: meta = {}
    if meta.get("kind") == "stars_pack":
        tokens = int(meta.get("tokens") or 0)
        add_tokens(ctx, tokens)
        await update.message.reply_text(f"✅ Оплата получена: +{tokens} токенов.\nБаланс: *{get_user_balance_value(ctx)}💎*",
                                        parse_mode=ParseMode.MARKDOWN)
        return
    await update.message.reply_text("✅ Оплата получена.")

# ==== Entry
def main():
    if not TELEGRAM_TOKEN: raise RuntimeError("TELEGRAM_TOKEN is not set")
    if not KIE_BASE_URL:   raise RuntimeError("KIE_BASE_URL is not set")
    if not KIE_API_KEY:    raise RuntimeError("KIE_API_KEY is not set")

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).rate_limiter(AIORateLimiter()).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("health", health))
    app.add_handler(CommandHandler("topup", topup))
    app.add_handler(CommandHandler("promo", promo_cmd))
    app.add_handler(PreCheckoutQueryHandler(precheckout_callback))
    app.add_handler(MessageHandler(filters.SUCCESSFUL_PAYMENT, successful_payment_handler))
    app.add_handler(CallbackQueryHandler(on_callback))
    app.add_handler(MessageHandler(filters.PHOTO, on_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_error_handler(error_handler)

    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
