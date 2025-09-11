# -*- coding: utf-8 -*-
# Best VEO3 Bot — PTB 20.x/21.x
# Полный файл с оплатой токенами, тест-драйвом Stars (1⭐), Prompt-Master, обычным чатом и спиннерами.
# Версия: 2025-09-11

import os
import re
import json
import time
import uuid
import base64
import asyncio
import logging
import tempfile
import subprocess
import sqlite3
import secrets
import string
from typing import Dict, Any, Optional, List, Tuple

import requests
from dotenv import load_dotenv

from telegram import (
    Update, InlineKeyboardButton, InlineKeyboardMarkup,
    InputFile, InputMediaPhoto, LabeledPrice
)
from telegram.constants import ParseMode, ChatAction
from telegram.ext import (
    ApplicationBuilder, ContextTypes, CommandHandler,
    MessageHandler, CallbackQueryHandler, PreCheckoutQueryHandler,
    AIORateLimiter, filters
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
BOT_USERNAME        = _env("BOT_USERNAME", "")  # без @, нужен для диплинков (опционально)

# Тарифы (токены)
TOK_VEO_FAST    = int(_env("TOK_VEO_FAST", "65"))
TOK_VEO_QUALITY = int(_env("TOK_VEO_QUALITY", "110"))
TOK_MJ_GEN      = int(_env("TOK_MJ_GEN", "12"))
TOK_MJ_UPSCALE  = int(_env("TOK_MJ_UPSCALE", "12"))

# Пакеты (для будущих продаж; сейчас используем тест 1⭐)
TOKEN_PACKS     = [int(x) for x in _env("TOKEN_PACKS","200,500,1000").split(",") if x.strip().isdigit()]

# DB
DB_PATH = _env("DB_PATH", "bot.db")

# OpenAI (опционально для Prompt-Master и Chat)
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
KIE_VEO_GEN_PATH    = _env("KIE_VEO_GEN_PATH",    _env("KIE_GEN_PATH",    "/api/v1/veo/generate"))
KIE_VEO_STATUS_PATH = _env("KIE_VEO_STATUS_PATH", _env("KIE_STATUS_PATH", "/api/v1/veo/record-info"))
KIE_VEO_1080_PATH   = _env("KIE_VEO_1080_PATH",   _env("KIE_HD_PATH",     "/api/v1/veo/get-1080p-video"))

# ---- MJ (Midjourney Image via KIE)
KIE_MJ_GENERATE = _env("KIE_MJ_GENERATE", "/api/v1/mj/generate")
KIE_MJ_STATUS   = _env("KIE_MJ_STATUS",   "/api/v1/mj/record-info")
KIE_MJ_UPSCALE  = _env("KIE_MJ_UPSCALE",  "/api/v1/mj/generateVary")  # важно

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

LOG_LEVEL = _env("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("veo3-bot")

try:
    import telegram as _tg
    log.info("PTB version: %s", getattr(_tg, "__version__", "unknown"))
except Exception:
    _tg = None

# ==========================
#   DB (users, payments)
# ==========================
def db() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH)
    c.execute("PRAGMA journal_mode=WAL;")
    return c

def _db_init():
    with db() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS users(
            user_id INTEGER PRIMARY KEY,
            tokens  INTEGER NOT NULL DEFAULT 0
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS payments(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            currency TEXT,
            amount INTEGER,
            tokens_credited INTEGER,
            payload TEXT,
            charge_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")

def _ensure_user(uid: int):
    with db() as c:
        c.execute("INSERT OR IGNORE INTO users(user_id,tokens) VALUES(?,0)", (uid,))

def token_balance(uid: int) -> int:
    with db() as c:
        row = c.execute("SELECT tokens FROM users WHERE user_id=?", (uid,)).fetchone()
        return int(row[0]) if row else 0

def add_tokens(uid: int, amount: int):
    if amount <= 0: return
    _ensure_user(uid)
    with db() as c:
        c.execute("UPDATE users SET tokens=tokens+? WHERE user_id=?", (amount, uid))

def charge_tokens(uid: int, amount: int) -> bool:
    if amount <= 0: return True
    _ensure_user(uid)
    with db() as c:
        row = c.execute("SELECT tokens FROM users WHERE user_id=?", (uid,)).fetchone()
        cur = int(row[0]) if row else 0
        if cur < amount: return False
        c.execute("UPDATE users SET tokens=tokens-? WHERE user_id=?", (amount, uid))
        return True

# ==========================
#   Utils
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

# ==========================
#   State
# ==========================
DEFAULT_STATE = {
    "mode": None,          # 'veo_text' | 'veo_photo' | 'prompt_master' | 'chat' | 'mj_txt'
    "aspect": None,        # '16:9' | '9:16'
    "model": None,         # 'veo3_fast' | 'veo3'
    "last_prompt": None,
    "last_image_url": None,
    "generating": False,
    "generation_id": None,
    "last_ui_msg_id": None,
    "last_task_id": None,
    "last_result_url": None,
    "uid": None,
    "chat_id": None,
}

def state(ctx: ContextTypes.DEFAULT_TYPE) -> Dict[str, Any]:
    ud = ctx.user_data
    for k, v in DEFAULT_STATE.items():
        ud.setdefault(k, v)
    return ud

# ==========================
#   UI
# ==========================
WELCOME_TMPL = (
    "🎬 *Veo 3 — съёмочная команда*\n"
    "Опиши идею — и получишь готовый клип.\n\n"
    "🖌️ *MJ — художник*\n"
    "Нарисует изображение по твоему тексту.\n\n"
    "🧠 *ChatGPT — сценарист*\n"
    "Опиши идею, персонажа, текст озвучки, стиль голоса или локацию — верну профессиональный кинопромпт.\n\n"
    "💎 Баланс токенов: *{balance}*\n"
    "Тарифы: Fast — {t_fast} • Quality — {t_q} • MJ — {t_mj}\n\n"
    "✨ Больше идей: {prompts_channel}\n\n"
    "Выберите режим 👇"
)

def main_menu_kb() -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton("🎬 VEO по тексту", callback_data="mode:veo_text")],
        [InlineKeyboardButton("📸 Оживление фото (VEO)", callback_data="mode:veo_photo")],
        [InlineKeyboardButton("🖌️ MJ — картинка по тексту", callback_data="mode:mj_txt")],
        [InlineKeyboardButton("🧠 Промпт-мастер (ChatGPT)", callback_data="mode:prompt_master")],
        [InlineKeyboardButton("💬 Обычный чат", callback_data="mode:chat")],
        [
            InlineKeyboardButton("❓ FAQ", callback_data="faq"),
            InlineKeyboardButton("📈 Канал с промптами", url=PROMPTS_CHANNEL_URL),
        ],
        [InlineKeyboardButton("⭐ Тест-драйв 1⭐ для MJ", callback_data="stars:test_mj")],
        [InlineKeyboardButton("💳 Пополнить токены", callback_data="buy_tokens")],
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
    lines = [
        "🪄 *Карточка VEO*",
        "",
        "✍️ *Промпт:*",
        f"`{prompt_preview or '—'}`",
        "",
        "*📋 Параметры:*",
        f"• Aspect: *{s.get('aspect') or '—'}*",
        f"• Mode: *{model}*",
        f"• Промпт: *{has_prompt}*",
        f"• Референс: *{has_ref}*",
    ]
    cost = TOK_VEO_QUALITY if (s.get("model")=="veo3") else TOK_VEO_FAST
    lines.append(f"• Стоимость рендера: *{cost} ток.*")
    if s.get("uid"):
        lines.append(f"• Баланс: *{token_balance(s['uid'])} ток.*")
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
    rows.append([InlineKeyboardButton("💳 Пополнить токены", callback_data="buy_tokens")])
    return InlineKeyboardMarkup(rows)

# ---------- MJ UI (кнопки)
def mj_aspect_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🌆 16:9", callback_data="mj:ar:16:9"),
            InlineKeyboardButton("📱 9:16", callback_data="mj:ar:9:16"),
        ]
    ])

def mj_upscale_kb(task_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🔍 Повысить качество #1", callback_data=f"mj:up:{task_id}:1"),
            InlineKeyboardButton("🔍 Повысить качество #2", callback_data=f"mj:up:{task_id}:2"),
        ],
        [
            InlineKeyboardButton("🔍 Повысить качество #3", callback_data=f"mj:up:{task_id}:3"),
            InlineKeyboardButton("🔍 Повысить качество #4", callback_data=f"mj:up:{task_id}:4"),
        ],
    ])

# ==========================
#   Prompt-Master
# ==========================
def _extract_voiceover(txt: str) -> Optional[str]:
    pat = r"(?:текст\s*озвучки|озвучка|voice\s*over|voiceover|vo|говорит|текст)\s*[:\-–]\s*(.+)"
    m = re.search(pat, txt, flags=re.IGNORECASE | re.DOTALL)
    cand = m.group(1).strip() if m else None
    if not cand:
        m2 = re.search(r"[«\"](.+?)[»\"]", txt)
        if m2: cand = m2.group(1).strip()
    return cand if cand else None

async def oai_prompt_master(idea_text: str) -> Optional[str]:
    if openai is None or not OPENAI_API_KEY:
        return None
    vo = _extract_voiceover(idea_text or "")

    system = (
        "You are a Prompt-Master for cinematic AI video generation. "
        "Return EXACTLY ONE English prompt, 500–900 characters. "
        "Always start with the literal tag: (супер реалистичное видео). "
        "Write vivid but compact cinematic instructions: lens & camera movement, lighting, palette, texture, atmosphere, framing, depth cues, subtle audio cues. "
        "No lists, no prefaces, no metadata. "
        "If a voice-over text is provided, append at the end:\n"
        "Voice-over: «<KEEP TEXT AS IS>»\n"
        "If NO voice-over is provided, append at the end:\n"
        "Audio: instrumental soundtrack matching the mood — no voiceover."
    )

    user_msg = f"IDEA:\n{idea_text.strip()}"
    if vo:
        user_msg += f"\n\nVOICEOVER_TEXT (keep as is):\n{vo}"

    try:
        resp = await asyncio.to_thread(
            openai.ChatCompletion.create,
            model="gpt-4o-mini",
            messages=[{"role":"system","content":system},{"role":"user","content":user_msg}],
            temperature=0.85, max_tokens=750,
        )
        txt = (resp["choices"][0]["message"]["content"] or "").strip()
        if not txt.startswith("(супер реалистичное видео)"):
            txt = "(супер реалистичное видео) " + txt
        if vo and "Voice-over:" not in txt:
            txt = txt.rstrip() + f"\nVoice-over: «{vo}»"
        if (not vo) and "Audio:" not in txt:
            txt = txt.rstrip() + "\nAudio: instrumental soundtrack matching the mood — no voiceover."
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

def _kie_error_message(status_code: int, j: Dict[str, Any]) -> str:
    code = j.get("code", status_code)
    msg = j.get("msg") or j.get("message") or j.get("error") or ""
    mapping = {401: "Доступ запрещён (Bearer).", 402: "Недостаточно кредитов.",
               429: "Превышен лимит запросов.", 500: "Внутренняя ошибка KIE.",
               422: "Запрос отклонён модерацией.", 400: "Неверный запрос (400)."}
    base = mapping.get(code, f"KIE code {code}.")
    return f"{base} {('Сообщение: ' + msg) if msg else ''}".strip()

# ---------- Upload API (stream + base64 + url) ----------
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
        event("upload_stream_predownload_failed", err=str(e), url=src_url)
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
            if _nz(u):
                event("KIE_UPLOAD_STREAM_OK", url=u); return u
        event("upload_stream_failed", status=r.status_code, resp=j)
    except Exception as e:
        event("upload_stream_err", err=str(e))
    finally:
        try: os.unlink(local)
        except Exception: pass
    return None

def upload_image_base64(src_url: str, upload_path: str = "tg-uploads", timeout: int = 90) -> Optional[str]:
    try:
        rr = requests.get(src_url, stream=True, timeout=timeout); rr.raise_for_status()
        ct = (rr.headers.get("Content-Type") or "image/jpeg")
        data_url = f"data:{ct};base64,{base64.b64encode(rr.content).decode('utf-8')}"
    except Exception as e:
        event("upload_b64_predownload_failed", err=str(e)); return None
    try:
        url = join_url(UPLOAD_BASE_URL, UPLOAD_BASE64_PATH)
        payload = {"base64Data": data_url, "uploadPath": upload_path, "fileName": "tg-upload.jpg"}
        r = requests.post(url, json=payload, headers={**_upload_headers(), "Content-Type":"application/json"}, timeout=timeout)
        try: j = r.json()
        except Exception: j = {"error": r.text}
        if r.status_code == 200 and (j.get("code", 200) == 200):
            d = j.get("data") or {}
            u = d.get("downloadUrl") or d.get("fileUrl")
            if _nz(u):
                event("KIE_UPLOAD_B64_OK", url=u); return u
        event("upload_b64_failed", status=r.status_code, resp=j)
    except Exception as e:
        event("upload_b64_err", err=str(e))
    return None

def upload_image_url(file_url: str, upload_path: str = "tg-uploads", timeout: int = 60) -> Optional[str]:
    try:
        url = join_url(UPLOAD_BASE_URL, UPLOAD_URL_PATH)
        payload = {"fileUrl": file_url, "uploadPath": upload_path,
                   "fileName": os.path.basename(file_url.split("?")[0]) or "image.jpg"}
        r = requests.post(url, json=payload, headers={**_upload_headers(), "Content-Type":"application/json"}, timeout=timeout)
        try: j = r.json()
        except Exception: j = {"error": r.text}
        if r.status_code == 200 and (j.get("code", 200) == 200):
            d = j.get("data") or {}
            u = d.get("downloadUrl") or d.get("fileUrl")
            if _nz(u):
                event("KIE_UPLOAD_URL_OK", url=u); return u
        event("upload_url_failed", status=r.status_code, resp=j)
    except Exception as e:
        event("upload_url_err", err=str(e))
    return None

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
    img_for_kie = None
    if _nz(image_url):
        img_for_kie = (
            upload_image_stream(image_url) or
            upload_image_base64(image_url) or
            upload_image_url(image_url) or
            image_url
        )
    payload = _build_payload_for_veo(prompt, aspect, img_for_kie, model_key)
    status, j = _post_json(join_url(KIE_BASE_URL, KIE_VEO_GEN_PATH), payload)
    code = j.get("code", status)
    if status == 200 and code == 200:
        tid = _extract_task_id(j)
        if tid: return True, tid, "Задача создана."
        return False, None, "Ответ KIE без taskId."

    msg = (j.get("msg") or j.get("message") or j.get("error") or "").lower()
    if "image fetch failed" in msg or ("image" in msg and "failed" in msg):
        payload.pop("imageUrls", None)
        status2, j2 = _post_json(join_url(KIE_BASE_URL, KIE_VEO_GEN_PATH), payload)
        if status2 == 200 and (j2.get("code", 200) == 200):
            tid = _extract_task_id(j2)
            if tid: return True, tid, "Задача создана (без изображения)."
    return False, None, _kie_error_message(status, j)

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
    return False, None, _kie_error_message(status, j), None

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

# ---------- MJ (generate/status/upscale)
def mj_generate(prompt: str, ar: str) -> Tuple[bool, Optional[str], str]:
    payload = {
        "taskType": "mj_txt2img",
        "prompt": prompt,
        "speed": "turbo",
        "aspectRatio": "9:16" if ar == "9:16" else "16:9",
        "version": "7",
        "enableTranslation": True,  # русские промпты — ок
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

def mj_upscale(task_id: str, index: int) -> Tuple[bool, Optional[str], str]:
    payload = {"taskId": task_id, "imageIndex": int(index)}
    status, j = _post_json(join_url(KIE_BASE_URL, KIE_MJ_UPSCALE), payload)
    code = j.get("code", status)
    if status == 200 and code == 200:
        tid = _extract_task_id(j)
        if tid: return True, tid, "MJ апскейл создан."
    return False, None, _kie_error_message(status, j)

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
#   ffmpeg helpers
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
#   Spinners
# ==========================
class Spinner:
    def __init__(self, mid: int, task: asyncio.Task):
        self.mid = mid; self.task = task

async def start_spinner(ctx: ContextTypes.DEFAULT_TYPE, chat_id: int, header: str, frames: List[str], chat_action: Optional[str] = None) -> Spinner:
    await ctx.bot.send_message(chat_id, header)
    m = await ctx.bot.send_message(chat_id, frames[0])

    async def loop():
        i = 0
        try:
            while True:
                if chat_action:
                    try: await ctx.bot.send_chat_action(chat_id, chat_action)
                    except Exception: pass
                try:
                    await ctx.bot.edit_message_text(chat_id=chat_id, message_id=m.message_id, text=frames[i % len(frames)])
                except Exception:
                    pass
                i += 1
                await asyncio.sleep(1.8)
        except asyncio.CancelledError:
            try:
                await ctx.bot.edit_message_text(chat_id=chat_id, message_id=m.message_id, text="✅ Готово!")
            except Exception:
                pass
            raise

    return Spinner(m.message_id, asyncio.create_task(loop()))

async def stop_spinner(spinner: Spinner):
    try: spinner.task.cancel()
    except Exception: pass

# ==========================
#   Sending video (robust)
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
#   Polling VEO
# ==========================
async def poll_veo_and_send(chat_id: int, task_id: str, gen_id: str, ctx: ContextTypes.DEFAULT_TYPE):
    s = state(ctx)
    s["generating"] = True
    s["generation_id"] = gen_id
    s["last_task_id"] = task_id

    spinner = await start_spinner(ctx, chat_id, "🎬 Рендерим видео… это займёт пару минут", ["🎬","🎥","📽️","🎞️"], ChatAction.UPLOAD_VIDEO)

    start_ts = time.time()
    try:
        while True:
            if s.get("generation_id") != gen_id:
                break

            ok, flag, msg, res_url = await asyncio.to_thread(get_kie_veo_status, task_id)
            event("VEO_STATUS", task_id=task_id, flag=flag, has_url=bool(res_url), msg=msg)

            if not ok:
                await ctx.bot.send_message(chat_id, f"❌ Ошибка статуса: {msg or 'неизвестно'}"); break

            if _nz(res_url):
                final_url = res_url
                if (s.get("aspect") or "16:9") == "16:9":
                    u1080 = await asyncio.to_thread(try_get_1080_url, task_id)
                    if _nz(u1080):
                        final_url = u1080; event("VEO_1080_OK", task_id=task_id)
                    else:
                        event("VEO_1080_MISS", task_id=task_id)

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
                await ctx.bot.send_message(chat_id, f"❌ KIE не вернул ссылку на видео. Сообщение: {msg or 'нет сообщения'}")
                break

            if (time.time() - start_ts) > POLL_TIMEOUT_SECS:
                await ctx.bot.send_message(chat_id, "⏳ Таймаут ожидания результата VEO."); break

            await asyncio.sleep(POLL_INTERVAL_SECS)

    except Exception as e:
        log.exception("[VEO_POLL] crash: %s", e)
        try: await ctx.bot.send_message(chat_id, "❌ Внутренняя ошибка при опросе VEO.")
        except Exception: pass
    finally:
        try: await stop_spinner(spinner)
        except Exception: pass

        if s.get("generation_id") == gen_id:
            s["generating"] = False
            s["generation_id"] = None

# ==========================
#   MJ poll & send
# ==========================
async def poll_mj_and_send_photos(chat_id: int, task_id: str, ctx: ContextTypes.DEFAULT_TYPE):
    spinner = await start_spinner(ctx, chat_id, "🎨 Рисуем изображение…", ["🎨","🖌️","🖼️","✨"], ChatAction.TYPING)
    start_ts = time.time()
    try:
        while True:
            ok, flag, data = await asyncio.to_thread(mj_status, task_id)
            event("MJ_STATUS", task_id=task_id, flag=flag, has_data=bool(data))
            if not ok:
                await ctx.bot.send_message(chat_id, "❌ Ошибка статуса MJ."); return
            if flag == 0:
                if (time.time() - start_ts) > 10 * 60:
                    await ctx.bot.send_message(chat_id, "⏳ Таймаут ожидания результата MJ."); return
                await asyncio.sleep(10); continue
            if flag in (2, 3):
                msg = (data or {}).get("errorMessage") or "Генерация не удалась."
                await ctx.bot.send_message(chat_id, f"❌ MJ ошибка: {msg}"); return
            if flag == 1:
                urls = _extract_mj_image_urls(data or {})
                if not urls:
                    await ctx.bot.send_message(chat_id, "⚠️ Готово, но ссылки на изображения не найдены."); return
                if len(urls) == 1:
                    await ctx.bot.send_photo(chat_id=chat_id, photo=urls[0])
                else:
                    media = [InputMediaPhoto(u) for u in urls[:10]]
                    await ctx.bot.send_media_group(chat_id=chat_id, media=media)
                await ctx.bot.send_message(chat_id, "Выберите вариант для повышения качества:", reply_markup=mj_upscale_kb(task_id))
                return
    except Exception as e:
        log.exception("[MJ_POLL] crash: %s", e)
        try: await ctx.bot.send_message(chat_id, "❌ Внутренняя ошибка при опросе MJ.")
        except Exception: pass
    finally:
        try: await stop_spinner(spinner)
        except Exception: pass

# ==========================
#   Handlers
# ==========================
async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    _db_init()
    s = state(ctx); s.update({**DEFAULT_STATE})
    s["chat_id"] = update.effective_chat.id
    s["uid"] = update.effective_user.id
    _ensure_user(s["uid"])

    bal = token_balance(s["uid"])
    txt = WELCOME_TMPL.format(
        balance=bal, t_fast=TOK_VEO_FAST, t_q=TOK_VEO_QUALITY, t_mj=TOK_MJ_GEN,
        prompts_channel=PROMPTS_CHANNEL_URL
    )
    await update.message.reply_text(txt, parse_mode=ParseMode.MARKDOWN, reply_markup=main_menu_kb())

async def topup_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    bal = token_balance(update.effective_user.id)
    await update.message.reply_text(
        f"💳 Пополнение токенов.\nПока доступен тест-драйв Stars на 1⭐ (даёт {TOK_MJ_GEN} токенов — хватит на одну MJ).\n"
        f"Ваш баланс: *{bal} ток.*",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⭐ Тест-драйв 1⭐ (MJ)", callback_data="stars:test_mj")],
                                           [InlineKeyboardButton("⬅️ Назад", callback_data="back")]])
    )

async def health(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    parts = [
        f"PTB: `{getattr(_tg, '__version__', 'unknown')}`" if _tg else "PTB: `unknown`",
        f"KIEBASEURL: `{KIE_BASE_URL}`",
        f"GENPATH: `{KIE_VEO_GEN_PATH}`",
        f"STATUSPATH: `{KIE_VEO_STATUS_PATH}`",
        f"1080PATH: `{KIE_VEO_1080_PATH}`",
        f"MJ_GENERATE: `{KIE_MJ_GENERATE}`",
        f"MJ_STATUS:   `{KIE_MJ_STATUS}`",
        f"MJ_UPSCALE:  `{KIE_MJ_UPSCALE}`",
        f"UPLOADBASEURL: `{UPLOAD_BASE_URL}`",
        f"ENABLEVERTICALNORMALIZE: `{ENABLE_VERTICAL_NORMALIZE}`",
        f"ALWAYSFORCEFHD: `{ALWAYS_FORCE_FHD}`",
        f"FFMPEGBIN: `{FFMPEG_BIN}`",
        f"MAXTGVIDEOMB: `{MAX_TG_VIDEO_MB}`",
        f"TOKENS: FAST={TOK_VEO_FAST}, QUALITY={TOK_VEO_QUALITY}, MJ={TOK_MJ_GEN}, UPSCALE={TOK_MJ_UPSCALE}",
        f"PACKS: {TOKEN_PACKS}",
        f"DB: {DB_PATH}",
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

# ----------------- CALLBACKS -----------------
STAR_CENT = 100  # 1⭐ = 100 минимальных единиц

async def on_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    data = (query.data or "").strip()
    await query.answer()

    s = state(ctx)

    if data == "faq":
        await query.message.reply_text(
            "🧾 *Как это работает*\n"
            f"• Стоимость: Fast — {TOK_VEO_FAST}, Quality — {TOK_VEO_QUALITY}, MJ — {TOK_MJ_GEN} токенов.\n"
            "• Если не указана озвучка — добавим музыку по настроению (без голоса).\n"
            "• Вертикаль нормализуем до 1080×1920; 16:9 — тянем 1080p.\n"
            "• Тест-драйв Stars на 1⭐ доступен для MJ (даёт токены на 1 генерацию).",
            parse_mode=ParseMode.MARKDOWN, reply_markup=main_menu_kb()
        ); return

    if data == "back":
        s.update({**DEFAULT_STATE}); s["uid"] = update.effective_user.id; s["chat_id"] = update.effective_chat.id
        await query.message.reply_text("Главное меню:", reply_markup=main_menu_kb()); return

    if data == "start_new_cycle":
        s.update({**DEFAULT_STATE}); s["uid"] = update.effective_user.id; s["chat_id"] = update.effective_chat.id
        await query.message.reply_text("Выберите режим:", reply_markup=main_menu_kb()); return

    # Тест-драйв Stars 1⭐ → начислить TOK_MJ_GEN токенов
    if data == "stars:test_mj":
        title = "Тест-драйв: 1⭐ → токены для 1 MJ"
        desc  = f"За 1⭐ вы получаете {TOK_MJ_GEN} токенов — хватит на одну MJ-генерацию."
        payload = "stars_test_mj"
        prices = [LabeledPrice(label="1 star", amount=1*STAR_CENT)]
        try:
            await ctx.bot.send_invoice(
                chat_id=update.effective_chat.id,
                title=title, description=desc,
                payload=payload, provider_token="",  # для Stars — пусто
                currency="XTR", prices=prices
            )
            await query.message.reply_text(
                "После оплаты токены начислятся автоматически. Если инвойс не пришёл — у вашего аккаунта ещё нет Stars."
            )
        except Exception as e:
            await query.message.reply_text(f"❌ Не удалось отправить инвойс Stars: {e}")
        return

    # --- Режимы
    if data.startswith("mode:"):
        _, mode = data.split(":", 1); s["mode"] = mode
        if mode == "veo_text":
            s["aspect"] = "16:9"; s["model"] = "veo3_fast"
            await query.message.reply_text("VEO (текст): пришлите идею/промпт."); await show_card_veo(update, ctx); return
        if mode == "veo_photo":
            s["aspect"] = "9:16"; s["model"] = "veo3_fast"
            await query.message.reply_text("VEO (оживление фото): пришлите фото (подпись-промпт — опционально)."); await show_card_veo(update, ctx); return
        if mode == "prompt_master":
            await query.message.reply_text(
                "🧠 *Промпт-мастер*\n\n"
                "Пришлите описание (2–3 фразы), можно подробнее:\n"
                "• сюжет и настроение\n"
                "• персонажи и ключевые детали\n"
                "• локация / время суток / свет\n"
                "• *Текст озвучки* — если нужен голос\n\n"
                "Я верну один готовый кинопромпт на EN.",
                parse_mode=ParseMode.MARKDOWN
            ); return
        if mode == "chat":
            await query.message.reply_text("💬 Обычный чат: напишите вопрос или задачу."); return
        if mode == "mj_txt":
            s["aspect"] = None
            await query.message.reply_text(
                f"🖌️ MJ: пришлите текстовый prompt.\nСтоимость: *{TOK_MJ_GEN} ток.*",
                parse_mode=ParseMode.MARKDOWN
            ); return

    # --- VEO параметры
    if data.startswith("aspect:"):
        _, val = data.split(":", 1); s["aspect"] = "9:16" if val.strip() == "9:16" else "16:9"
        await show_card_veo(update, ctx, edit_only_markup=True); return

    if data.startswith("model:"):
        _, val = data.split(":", 1); s["model"] = "veo3" if val.strip() == "veo3" else "veo3_fast"
        await show_card_veo(update, ctx, edit_only_markup=True); return

    if data == "card:toggle_photo":
        if s.get("last_image_url"):
            s["last_image_url"] = None; await query.message.reply_text("Фото-референс удалён."); await show_card_veo(update, ctx)
        else:
            await query.message.reply_text("Пришлите фото вложением или публичный URL изображения.")
        return

    if data == "card:edit_prompt":
        await query.message.reply_text("Пришлите новый текст промпта."); return

    if data == "card:reset":
        keep_aspect = s.get("aspect") or "16:9"; keep_model = s.get("model") or "veo3_fast"
        s.update({**DEFAULT_STATE}); s["aspect"] = keep_aspect; s["model"] = keep_model
        await query.message.reply_text("Карточка очищена."); await show_card_veo(update, ctx); return

    if data == "card:generate":
        if s.get("generating"): await query.message.reply_text("⏳ Генерация уже идёт."); return
        if not s.get("last_prompt"): await query.message.reply_text("Сначала укажите текст промпта."); return

        uid = update.effective_user.id
        cost = TOK_VEO_QUALITY if s.get("model") == "veo3" else TOK_VEO_FAST
        if not charge_tokens(uid, cost):
            bal = token_balance(uid)
            await query.message.reply_text(
                f"⚠️ Недостаточно токенов: нужно {cost}, у вас {bal}.",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("⭐ Тест-драйв 1⭐ (MJ)", callback_data="stars:test_mj")],
                    [InlineKeyboardButton("💳 Пополнить токены", callback_data="buy_tokens")]
                ])
            ); return

        event("VEO_SUBMIT_REQ", aspect=s.get("aspect"), model=s.get("model"),
              with_image=bool(s.get("last_image_url")), prompt_len=len(s.get("last_prompt") or ""))

        ok, task_id, msg = await asyncio.to_thread(
            submit_kie_veo, s["last_prompt"].strip(), s.get("aspect", "16:9"),
            s.get("last_image_url"), s.get("model", "veo3_fast")
        )
        event("VEO_SUBMIT_RESP", ok=ok, task_id=task_id, msg=msg)

        if not ok or not task_id:
            add_tokens(uid, cost)  # вернуть
            await query.message.reply_text(f"❌ Не удалось создать VEO-задачу: {msg}"); return

        gen_id = uuid.uuid4().hex[:12]
        s["generating"] = True; s["generation_id"] = gen_id; s["last_task_id"] = task_id
        await query.message.reply_text(f"🚀 Задача отправлена ({'Fast' if s.get('model')=='veo3_fast' else 'Quality'}). taskId={task_id}")
        asyncio.create_task(poll_veo_and_send(update.effective_chat.id, task_id, gen_id, ctx)); return

    # --- MJ выбор формата
    if data.startswith("mj:ar:"):
        ar = data.split(":")[-1]
        prompt = s.get("last_prompt")
        if not prompt:
            await query.message.reply_text("⚠️ Сначала отправьте текстовый prompt."); return

        uid = update.effective_user.id
        if not charge_tokens(uid, TOK_MJ_GEN):
            await query.message.reply_text(
                f"⚠️ Нужно {TOK_MJ_GEN} токенов для MJ. Попробуйте тест-драйв 1⭐ или пополнение.",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("⭐ Тест-драйв 1⭐ (MJ)", callback_data="stars:test_mj")],
                    [InlineKeyboardButton("💳 Пополнить токены", callback_data="buy_tokens")]
                ])
            ); return

        await query.message.reply_text(f"🚀 Генерация фото запущена…\nФормат: *{ar}*\nPrompt: `{prompt}`",
                                       parse_mode=ParseMode.MARKDOWN)
        ok, task_id, msg = await asyncio.to_thread(mj_generate, prompt.strip(), ar)
        event("MJ_SUBMIT_RESP", ok=ok, task_id=task_id, msg=msg)
        if not ok or not task_id:
            add_tokens(uid, TOK_MJ_GEN)
            await query.message.reply_text(f"❌ Не удалось создать MJ-задачу: {msg}"); return
        await query.message.reply_text(f"🆔 MJ taskId: `{task_id}`", parse_mode=ParseMode.MARKDOWN)
        asyncio.create_task(poll_mj_and_send_photos(update.effective_chat.id, task_id, ctx))
        return

    # --- MJ апскейл
    if data.startswith("mj:up:"):
        try:
            _, _, task_id, idx = data.split(":")
            index = int(idx)
            assert 1 <= index <= 4
        except Exception:
            await query.message.reply_text("⚠️ Некорректная команда повышения качества."); return

        uid = update.effective_user.id
        if TOK_MJ_UPSCALE > 0:
            if not charge_tokens(uid, TOK_MJ_UPSCALE):
                await query.message.reply_text(f"⚠️ Нужны {TOK_MJ_UPSCALE} токенов для апскейла."); return

        await query.message.reply_text(f"🔍 Повышаю качество варианта #{index}…")
        ok, up_task, msg = await asyncio.to_thread(mj_upscale, task_id, index)
        if not ok or not up_task:
            if TOK_MJ_UPSCALE > 0: add_tokens(uid, TOK_MJ_UPSCALE)
            await query.message.reply_text(f"❌ Не удалось отправить апскейл: {msg}"); return

        start_ts = time.time()
        try:
            while True:
                ok2, flag2, data2 = await asyncio.to_thread(mj_status, up_task)
                if not ok2:
                    await asyncio.sleep(8); continue
                if flag2 == 0:
                    if (time.time() - start_ts) > 10 * 60:
                        await query.message.reply_text("⏳ Таймаут ожидания апскейла."); return
                    await asyncio.sleep(8); continue
                if flag2 in (2, 3):
                    msg2 = (data2 or {}).get("errorMessage") or "Апскейл не удался."
                    await query.message.reply_text(f"❌ MJ ошибка: {msg2}"); return
                if flag2 == 1:
                    urls2 = _extract_mj_image_urls(data2 or {})
                    if not urls2:
                        await query.message.reply_text("⚠️ Готово, но ссылка апскейла не получена."); return
                    await ctx.bot.send_photo(chat_id=update.effective_chat.id, photo=urls2[0])
                    await query.message.reply_text("✅ Готово! Можно повысить качество других вариантов или сгенерировать новый prompt.")
                    return
        except Exception as e:
            log.exception("[MJ_UPSCALE] crash: %s", e)
            await query.message.reply_text("❌ Внутренняя ошибка при апскейле.")
        return

    # Покупка (пока только тест-драйв 1⭐)
    if data == "buy_tokens":
        bal = token_balance(update.effective_user.id)
        await query.message.reply_text(
            f"💳 Пополнение токенов.\nПока доступен тест-драйв Stars на 1⭐ (даёт {TOK_MJ_GEN} токенов — хватит на одну MJ).\n"
            f"Ваш баланс: *{bal} ток.*",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⭐ Тест-драйв 1⭐ (MJ)", callback_data="stars:test_mj")],
                                               [InlineKeyboardButton("⬅️ Назад", callback_data="back")]])
        ); return

# ----------------- TEXT / PHOTO -----------------
async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = state(ctx)
    text_in = (update.message.text or "").strip()

    # Ссылка на картинку → референс
    low = text_in.lower()
    if low.startswith(("http://", "https://")) and any(low.split("?")[0].endswith(ext) for ext in (".jpg",".jpeg",".png",".webp",".heic")):
        s["last_image_url"] = text_in.strip(); await update.message.reply_text("✅ Ссылка на изображение принята."); await show_card_veo(update, ctx); return

    mode = s.get("mode")
    if mode == "prompt_master":
        prompt = await oai_prompt_master(text_in)
        if not prompt: await update.message.reply_text("⚠️ Prompt-мастер сейчас недоступен."); return
        s["last_prompt"] = prompt; await update.message.reply_text("🧠 Готово. Промпт добавлен в карточку VEO."); await show_card_veo(update, ctx); return

    if mode == "chat":
        if openai is None or not OPENAI_API_KEY:
            await update.message.reply_text("⚠️ Обычный чат временно недоступен (нет OPENAI_API_KEY)."); return
        try:
            resp = await asyncio.to_thread(
                openai.ChatCompletion.create,
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": "You are a helpful, concise assistant."},
                          {"role": "user", "content": text_in}],
                temperature=0.5, max_tokens=700,
            )
            answer = resp["choices"][0]["message"]["content"].strip()
            await update.message.reply_text(answer)
        except Exception as e:
            log.exception("Chat error: %s", e)
            await update.message.reply_text("⚠️ Ошибка запроса к ChatGPT.")
        return

    if mode == "mj_txt":
        s["last_prompt"] = text_in
        await update.message.reply_text(
            f"✅ Prompt для MJ сохранён:\n\n`{text_in}`\n\nВыберите формат:",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=mj_aspect_kb()
        )
        return

    # По умолчанию — VEO промпт
    s["last_prompt"] = text_in
    await update.message.reply_text("🟦 *VEO — подготовка к рендеру*\nПроверь карточку ниже и жми «Сгенерировать».",
                                    parse_mode=ParseMode.MARKDOWN)
    await show_card_veo(update, ctx)

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
        s["last_image_url"] = url
        await update.message.reply_text("🖼️ Фото принято как референс."); await show_card_veo(update, ctx)
    except Exception as e:
        log.exception("Get photo failed: %s", e)
        await update.message.reply_text("⚠️ Не удалось обработать фото. Пришлите публичный URL картинки текстом.")

# ----------------- PAYMENTS: Stars test 1⭐ -----------------
async def on_precheckout(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        await ctx.bot.answer_pre_checkout_query(update.pre_checkout_query.id, ok=True)
    except Exception as e:
        log.exception("pre_checkout error: %s", e)
        try:
            await ctx.bot.answer_pre_checkout_query(update.pre_checkout_query.id, ok=False, error_message="Ошибка предоплаты, попробуйте позже.")
        except Exception: pass

async def on_successful_payment(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    sp = update.message.successful_payment
    uid = update.effective_user.id
    _ensure_user(uid)
    payload = sp.invoice_payload or ""

    tokens = 0
    if payload == "stars_test_mj":
        tokens = TOK_MJ_GEN  # На одну MJ-генерацию

    if tokens > 0:
        add_tokens(uid, tokens)
        with db() as c:
            c.execute("INSERT INTO payments(user_id,currency,amount,tokens_credited,payload,charge_id) VALUES(?,?,?,?,?,?)",
                      (uid, "XTR", sp.total_amount, tokens, payload, sp.telegram_payment_charge_id))
        bal = token_balance(uid)
        await update.message.reply_text(
            f"✅ Оплата получена: +{tokens} токенов.\nБаланс: *{bal} ток.*\nМожно запускать MJ — хватит на одну генерацию.",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=main_menu_kb()
        )
    else:
        await update.message.reply_text("✅ Оплата получена.", reply_markup=main_menu_kb())

# ----------------- Entry -----------------
def main():
    if not TELEGRAM_TOKEN: raise RuntimeError("TELEGRAM_TOKEN is not set")
    if not KIE_BASE_URL:   raise RuntimeError("KIE_BASE_URL is not set")
    if not KIE_API_KEY:    raise RuntimeError("KIE_API_KEY is not set")

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).rate_limiter(AIORateLimiter()).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler(["topup","buy","tokens"], topup_cmd))
    app.add_handler(CommandHandler("health", health))
    # Тестовая команда для быстрой отправки Stars-инвойса на 1⭐
    async def stars_test_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        # как и кнопка stars:test_mj
        title = "Тест-драйв: 1⭐ → токены для 1 MJ"
        desc  = f"За 1⭐ вы получаете {TOK_MJ_GEN} токенов — хватит на одну MJ-генерацию."
        prices = [LabeledPrice(label="1 star", amount=1*STAR_CENT)]
        try:
            await ctx.bot.send_invoice(
                chat_id=update.effective_chat.id,
                title=title, description=desc,
                payload="stars_test_mj", provider_token="",
                currency="XTR", prices=prices
            )
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка XTR-инвойса: {e}")
    app.add_handler(CommandHandler("stars_test", stars_test_cmd))

    app.add_handler(CallbackQueryHandler(on_callback))
    app.add_handler(MessageHandler(filters.PHOTO, on_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_handler(PreCheckoutQueryHandler(on_precheckout))
    app.add_handler(MessageHandler(filters.SUCCESSFUL_PAYMENT, on_successful_payment))
    app.add_error_handler(error_handler)

    log.info(
        "Bot starting. PTB=%s | KIE_BASE=%s | GEN=%s | STATUS=%s | 1080=%s | MJ_GEN=%s | MJ_STATUS=%s | MJ_UPSCALE=%s | "
        "UPLOAD_BASE=%s | VERT_FIX=%s | FORCE_FHD=%s | MAX_MB=%s",
        getattr(_tg, '__version__', 'unknown') if _tg else 'unknown',
        KIE_BASE_URL, KIE_VEO_GEN_PATH, KIE_VEO_STATUS_PATH, KIE_VEO_1080_PATH,
        KIE_MJ_GENERATE, KIE_MJ_STATUS, KIE_MJ_UPSCALE,
        UPLOAD_BASE_URL, ENABLE_VERTICAL_NORMALIZE, ALWAYS_FORCE_FHD, MAX_TG_VIDEO_MB
    )

    # Если когда-то включался webhook — снимите:
    # https://api.telegram.org/bot<YOUR_TOKEN>/deleteWebhook
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
