# -*- coding: utf-8 -*-
# Best VEO3 + MJ Bot — PTB 20.7 (Extended UI, stable delivery)
# Версия: 2025-09-10-ext2
#
# ✅ VEO3: 16:9 / 9:16, надёжная отправка видео (URL → файл → документ)
# ✅ MJ: две кнопки — "Фото из текста (MJ)" и "Фото из селфи (MJ)"
# ✅ MJ: УБРАНЫ форматы 1:1 и 3:4 (остались 16:9 и 9:16)
# ✅ Подробные карточки, FAQ, health, логи

import os
import json
import time
import uuid
import asyncio
import logging
import tempfile
from typing import Dict, Any, Optional, List, Tuple, Union

import requests
from dotenv import load_dotenv

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputFile,
)
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    AIORateLimiter,
)

# ==========================
#   ENV / INIT
# ==========================
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()

# OpenAI (опционально)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
try:
    import openai  # type: ignore
    if OPENAI_API_KEY:
        openai.api_key = OPENAI_API_KEY
except Exception:
    openai = None

# ---- KIE core ----
KIE_API_KEY = os.getenv("KIE_API_KEY", "").strip()
KIE_BASE_URL = os.getenv("KIE_BASE_URL", "https://api.kie.ai").strip()

# VEO endpoints
KIE_VEO_GEN_PATH = os.getenv("KIE_VEO_GEN_PATH", os.getenv("KIE_GEN_PATH", "/api/v1/veo/generate"))
KIE_VEO_STATUS_PATH = os.getenv("KIE_VEO_STATUS_PATH", os.getenv("KIE_STATUS_PATH", "/api/v1/veo/record-info"))

# MJ endpoints (имена остаются прежними)
KIE_MJ_GEN_PATH = os.getenv("KIE_MJ_GEN_PATH", "/api/v1/mj/generate")
KIE_MJ_STATUS_PATH = os.getenv("KIE_MJ_STATUS_PATH", "/api/v1/mj/record-info")

PROMPTS_CHANNEL_URL = os.getenv("PROMPTS_CHANNEL_URL", "https://t.me/bestveo3promts").strip()
TOPUP_URL = os.getenv("TOPUP_URL", "https://t.me/bestveo3promts").strip()

POLL_INTERVAL_SECS = int(os.getenv("POLL_INTERVAL_SECS", "6"))
POLL_TIMEOUT_SECS  = int(os.getenv("POLL_TIMEOUT_SECS",  str(20 * 60)))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.getenv("LOG_FILE", "").strip()

fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
logging.basicConfig(level=LOG_LEVEL, format=fmt)
if LOG_FILE:
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(LOG_LEVEL)
    fh.setFormatter(logging.Formatter(fmt))
    logging.getLogger().addHandler(fh)

log = logging.getLogger("best-veo3-bot")

try:
    import telegram as _tg
    log.info("PTB version: %s", getattr(_tg, "__version__", "unknown"))
except Exception:
    pass


# ==========================
#   Utils
# ==========================
def join_url(base: str, path: str) -> str:
    u = f"{base.rstrip('/')}/{path.lstrip('/')}"
    return u.replace("://", "§§").replace("//", "/").replace("§§", "://")

def mask_secret(s: str, show: int = 6) -> str:
    if not s:
        return ""
    s = s.strip()
    return s if len(s) <= show else f"{'*'*(len(s)-show)}{s[-show:]}"

def pick_first_url(value: Union[str, List[str], None]) -> Optional[str]:
    if not value:
        return None
    if isinstance(value, str):
        v = value.strip()
        return v or None
    if isinstance(value, list):
        for v in value:
            if isinstance(v, str) and v.strip():
                return v.strip()
    return None

def tg_file_direct_url(bot_token: str, file_path: str) -> str:
    return f"https://api.telegram.org/file/bot{bot_token}/{file_path.lstrip('/')}"

def _nz(s: Optional[str]) -> Optional[str]:
    if s is None: return None
    s2 = s.strip()
    return s2 if s2 else None

def _short(o: Any, n: int = 300) -> str:
    try:
        s = o if isinstance(o, str) else json.dumps(o, ensure_ascii=False)
    except Exception:
        s = str(o)
    s = s.strip()
    return (s[:n] + "…") if len(s) > n else s


# ==========================
#   State
# ==========================
DEFAULT_STATE = {
    "mode": None,              # 'veo_text' | 'veo_photo' | 'prompt_master' | 'chat' | 'mj_text' | 'mj_face'
    # VEO
    "aspect": None,            # '16:9' | '9:16'
    "model": None,             # 'veo3_fast' | 'veo3'
    "last_prompt": None,
    "last_image_url": None,
    "generating": False,
    "generation_id": None,
    "last_ui_msg_id": None,
    "last_task_id": None,
    "last_result_url": None,
    # MJ shared
    "mj_aspect": "16:9",       # только 16:9 и 9:16
    "mj_speed": "relaxed",     # relaxed | fast | turbo
    "mj_version": "7",
    "mj_stylization": 50,
    "mj_weirdness": 0,
    "mj_variety": 5,
    # MJ text-to-image
    "mj_txt_prompt": None,
    "mj_txt_generating": False,
    "mj_txt_generation_id": None,
    "mj_txt_last_task_id": None,
    "mj_txt_last_images": None,
    # MJ selfie-to-image
    "mj_face_prompt": None,
    "mj_selfie_url": None,
    "mj_face_generating": False,
    "mj_face_generation_id": None,
    "mj_face_last_task_id": None,
    "mj_face_last_images": None,
}

def state(ctx: ContextTypes.DEFAULT_TYPE) -> Dict[str, Any]:
    ud = ctx.user_data
    for k, v in DEFAULT_STATE.items():
        ud.setdefault(k, v)
    return ud


# ==========================
#   UI
# ==========================
WELCOME = (
    "🎬 *Veo 3 — генерация видео*\n"
    "• Поддержка 16:9 и 9:16 (вертикальное возвращается корректно)\n"
    "• Референс фото (image-to-video)\n"
    "• Fast/Quality\n\n"
    "🧑‍🎨 *Midjourney (через Kie.ai)*\n"
    "• Фото из *текста*\n"
    "• Фото из *селфи*\n"
    "• Аспекты только 16:9 и 9:16\n\n"
    f"Идеи и промпты: {PROMPTS_CHANNEL_URL}\n"
    "Выберите режим:"
)

def main_menu_kb() -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton("🎬 VEO — по тексту", callback_data="mode:veo_text")],
        [InlineKeyboardButton("🖼️ VEO — по фото", callback_data="mode:veo_photo")],
        [InlineKeyboardButton("🧑‍🎨 MJ — Фото из текста", callback_data="mode:mj_text")],
        [InlineKeyboardButton("🧑‍🦰 MJ — Фото из селфи", callback_data="mode:mj_face")],
        [InlineKeyboardButton("🧠 Промпт-мастер (ChatGPT)", callback_data="mode:prompt_master")],
        [InlineKeyboardButton("💬 Обычный чат", callback_data="mode:chat")],
        [
            InlineKeyboardButton("❓ FAQ", callback_data="faq"),
            InlineKeyboardButton("📈 Канал с промптами", url=PROMPTS_CHANNEL_URL),
        ],
        [InlineKeyboardButton("💳 Пополнить баланс", url=TOPUP_URL)],
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
        return [InlineKeyboardButton("⚡ Fast", callback_data="model:veo3_fast"),
                InlineKeyboardButton("💎 Quality ✅", callback_data="model:veo3")]
    return [InlineKeyboardButton("⚡ Fast ✅", callback_data="model:veo3_fast"),
            InlineKeyboardButton("💎 Quality", callback_data="model:veo3")]

# MJ: только 16:9 и 9:16
def mj_aspect_row(current: str) -> List[InlineKeyboardButton]:
    opts = ["16:9", "9:16"]
    row: List[InlineKeyboardButton] = []
    for r in opts:
        mark = " ✅" if current == r else ""
        row.append(InlineKeyboardButton(f"{r}{mark}", callback_data=f"mj_aspect:{r}"))
    return row

def mj_speed_row(current: str) -> List[InlineKeyboardButton]:
    opts = ["relaxed", "fast", "turbo"]
    row: List[InlineKeyboardButton] = []
    for r in opts:
        mark = " ✅" if current == r else ""
        emoji = {"relaxed": "🐢", "fast": "⚡", "turbo": "🚀"}[r]
        row.append(InlineKeyboardButton(f"{emoji} {r}{mark}", callback_data=f"mj_speed:{r}"))
    return row

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
    return "\n".join(lines)

def build_card_text_mj_text(s: Dict[str, Any]) -> str:
    prompt_preview = (s.get("mj_txt_prompt") or "").strip()
    if len(prompt_preview) > 900: prompt_preview = prompt_preview[:900] + "…"
    lines = [
        "🪄 *MJ (фото из текста)*",
        "",
        "✍️ *Промпт:*",
        f"`{prompt_preview or '—'}`",
        "",
        "*📋 Параметры:*",
        f"• Aspect: *{s.get('mj_aspect','16:9')}*",
        f"• Speed: *{s.get('mj_speed','relaxed')}*",
        f"• Version: *{s.get('mj_version','7')}*",
        f"• Stylization: *{s.get('mj_stylization',50)}*",
        f"• Weirdness: *{s.get('mj_weirdness',0)}*",
        f"• Variety: *{s.get('mj_variety',5)}*",
    ]
    return "\n".join(lines)

def build_card_text_mj_face(s: Dict[str, Any]) -> str:
    prompt_preview = (s.get("mj_face_prompt") or "").strip()
    if len(prompt_preview) > 900: prompt_preview = prompt_preview[:900] + "…"
    has_selfie = "есть" if s.get("mj_selfie_url") else "нет"
    lines = [
        "🪄 *MJ (фото из селфи)*",
        "",
        "✍️ *Промпт:*",
        f"`{prompt_preview or '—'}`",
        "",
        "*📋 Параметры:*",
        f"• Aspect: *{s.get('mj_aspect','16:9')}*",
        f"• Speed: *{s.get('mj_speed','relaxed')}*",
        f"• Version: *{s.get('mj_version','7')}*",
        f"• Stylization: *{s.get('mj_stylization',50)}*",
        f"• Weirdness: *{s.get('mj_weirdness',0)}*",
        f"• Variety: *{s.get('mj_variety',5)}*",
        f"• Селфи: *{has_selfie}*",
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
    rows.append([InlineKeyboardButton("💳 Пополнить баланс", url=TOPUP_URL)])
    return InlineKeyboardMarkup(rows)

def card_keyboard_mj_text(s: Dict[str, Any]) -> InlineKeyboardMarkup:
    rows: List[List[InlineKeyboardButton]] = []
    rows.append([InlineKeyboardButton("✍️ Изменить промпт", callback_data="mjtxt:edit_prompt")])
    rows.append(mj_aspect_row(s.get("mj_aspect","16:9")))
    rows.append(mj_speed_row(s.get("mj_speed","relaxed")))
    rows.append([InlineKeyboardButton("🖼️ Сгенерировать фото", callback_data="mjtxt:generate")])
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="back")])
    rows.append([InlineKeyboardButton("💳 Пополнить баланс", url=TOPUP_URL)])
    return InlineKeyboardMarkup(rows)

def card_keyboard_mj_face(s: Dict[str, Any]) -> InlineKeyboardMarkup:
    rows: List[List[InlineKeyboardButton]] = []
    rows.append([InlineKeyboardButton("🧑‍🦰 Добавить/Удалить селфи", callback_data="mjface:toggle_selfie"),
                 InlineKeyboardButton("✍️ Изменить промпт",            callback_data="mjface:edit_prompt")])
    rows.append(mj_aspect_row(s.get("mj_aspect","16:9")))
    rows.append(mj_speed_row(s.get("mj_speed","relaxed")))
    rows.append([InlineKeyboardButton("🖼️ Сгенерировать фото", callback_data="mjface:generate")])
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="back")])
    rows.append([InlineKeyboardButton("💳 Пополнить баланс", url=TOPUP_URL)])
    return InlineKeyboardMarkup(rows)


# ==========================
#   Prompt-Master / Chat
# ==========================
async def oai_prompt_master(idea_text: str) -> Optional[str]:
    if openai is None or not OPENAI_API_KEY:
        return None
    system = (
        "You are Prompt-Master for cinematic AI video generation. "
        "Respond with EXACTLY ONE English prompt, 500–900 characters. "
        "No prefaces, no lists, no brand names or logos. "
        "Include: lens/optics, camera movement, lighting/palette, tiny sensory details, subtle audio cues."
    )
    try:
        resp = await asyncio.to_thread(
            openai.ChatCompletion.create,
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": idea_text.strip()}],
            temperature=0.9,
            max_tokens=700,
        )
        txt = resp["choices"][0]["message"]["content"].strip()
        return txt[:1200]
    except Exception as e:
        log.exception("Prompt-Master error: %s", e)
        return None


# ==========================
#   HTTP helpers (KIE)
# ==========================
def _kie_headers() -> Dict[str, str]:
    token = KIE_API_KEY
    headers = {"Content-Type": "application/json"}
    if token:
        if not token.lower().startswith("bearer "):
            token = f"Bearer {token}"
        headers["Authorization"] = token
    return headers

def _post_json(url: str, payload: Dict[str, Any], timeout: int = 40, req_id: str = "-") -> Tuple[int, Dict[str, Any]]:
    try:
        log.debug("HTTP POST -> %s | req_id=%s | payload=%s", url, req_id, _short(payload, 1200))
        r = requests.post(url, json=payload, headers=_kie_headers(), timeout=timeout)
        try: j = r.json()
        except Exception: j = {"error": r.text}
        log.debug("HTTP POST <- %s %s | req_id=%s | body=%s", r.status_code, url, req_id, _short(j, 1500))
        return r.status_code, j
    except Exception as e:
        log.exception("HTTP POST failed | req_id=%s | url=%s", req_id, url)
        return 599, {"error": str(e)}

def _get_json(url: str, params: Dict[str, Any], timeout: int = 40, req_id: str = "-") -> Tuple[int, Dict[str, Any]]:
    try:
        log.debug("HTTP GET -> %s | req_id=%s | params=%s", url, req_id, _short(params, 600))
        r = requests.get(url, params=params, headers=_kie_headers(), timeout=timeout)
        try: j = r.json()
        except Exception: j = {"error": r.text}
        log.debug("HTTP GET <- %s %s | req_id=%s | body=%s", r.status_code, url, req_id, _short(j, 1500))
        return r.status_code, j
    except Exception as e:
        log.exception("HTTP GET failed | req_id=%s | url=%s", req_id, url)
        return 599, {"error": str(e)}

def _extract_task_id(j: Dict[str, Any]) -> Optional[str]:
    data = j.get("data") or {}
    for k in ("taskId", "taskid", "id"):
        if j.get(k): return str(j[k])
        if data.get(k): return str(data[k])
    return None

def _parse_success_flag(j: Dict[str, Any]) -> Tuple[Optional[int], Optional[str], Dict[str, Any]]:
    data = j.get("data") or {}
    msg = j.get("msg") or j.get("message")
    flag = None
    for k in ("successFlag", "status", "state"):
        if k in data:
            try:
                flag = int(data[k])
                break
            except Exception:
                pass
    return flag, msg, data

def _coerce_url_list(value) -> List[str]:
    urls: List[str] = []
    def add(u: str):
        if isinstance(u, str):
            s = u.strip()
            if s.startswith("http"):
                urls.append(s)
    if not value:
        return urls
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
    # originUrls (для 9:16) приоритетно — затем resultUrls
    for key in ("originUrls", "resultUrls"):
        urls = _coerce_url_list(data.get(key))
        if urls: return urls[0]
    # внутри info/response/resultInfoJson
    for container in ("info", "response", "resultInfoJson"):
        v = data.get(container)
        if isinstance(v, dict):
            for key in ("originUrls", "resultUrls", "videoUrls"):
                urls = _coerce_url_list(v.get(key))
                if urls: return urls[0]
    # глубокий поиск http*.mp4/mov/webm
    def walk(x):
        if isinstance(x, dict):
            for vv in x.values():
                r = walk(vv);  if r: return r
        elif isinstance(x, list):
            for vv in x:
                r = walk(vv);  if r: return r
        elif isinstance(x, str):
            s = x.strip().split("?")[0].lower()
            if s.startswith("http") and s.endswith((".mp4", ".mov", ".webm")):
                return x.strip()
        return None
    return walk(data)

def _extract_result_urls_list(data: Dict[str, Any]) -> List[str]:
    urls: List[str] = []
    # стандартные ветки
    for key in ("resultUrls", "originUrls"):
        urls += _coerce_url_list(data.get(key))
    for container in ("info", "response", "resultInfoJson"):
        v = data.get(container)
        if isinstance(v, dict):
            for key in ("resultUrls", "originUrls", "imageUrls", "urls"):
                urls += _coerce_url_list(v.get(key))
    # одинокий url
    u = data.get("url")
    if isinstance(u, str) and u.strip(): urls.append(u.strip())
    # уникализируем, сохраняя порядок
    seen = set(); out = []
    for u in urls:
        if u not in seen:
            out.append(u); seen.add(u)
    return out

def _kie_error_message(status_code: int, j: Dict[str, Any]) -> str:
    code = j.get("code", status_code)
    msg = j.get("msg") or j.get("message") or j.get("error") or ""
    mapping = {
        401: "Доступ запрещён (Bearer).",
        402: "Недостаточно кредитов.",
        429: "Превышен лимит запросов.",
        451: "Ошибка загрузки изображения.",
        500: "Внутренняя ошибка KIE.",
        422: "Запрос отклонён модерацией.",
        400: "Неверный запрос (400).",
    }
    base = mapping.get(code, f"KIE code {code}.")
    return f"{base} {('Сообщение: ' + msg) if msg else ''}".strip()

# ---------- VEO payload
def _build_payload_for_veo(prompt: str, aspect: str, image_url: Optional[str], model_key: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "prompt": prompt,
        "aspectRatio": "9:16" if aspect == "9:16" else "16:9",
        "model": "veo3" if model_key == "veo3" else "veo3_fast",
        "enableFallback": aspect == "16:9",  # fallback только в 16:9 по доке
    }
    if image_url:
        payload["imageUrls"] = [image_url]
    return payload

# ---------- MJ payload (ТОЛЬКО 16:9 и 9:16)
ALLOWED_MJ_ASPECTS = {"16:9", "9:16"}
ALLOWED_MJ_SPEEDS = {"relaxed", "fast", "turbo"}

def _clamp_int(x: Any, lo: int, hi: int, default: int) -> int:
    try: v = int(x)
    except Exception: return default
    return max(lo, min(hi, v))

def build_payload_for_mj_txt2img(prompt: str, s: Dict[str, Any]) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    aspect = s.get("mj_aspect") or "16:9"
    if aspect not in ALLOWED_MJ_ASPECTS: aspect = "16:9"
    speed = s.get("mj_speed") or "relaxed"
    if speed not in ALLOWED_MJ_SPEEDS: speed = "relaxed"
    version = str(s.get("mj_version") or "7").strip() or "7"
    styl = _clamp_int(s.get("mj_stylization", 50), 0, 1000, 50)
    weird = _clamp_int(s.get("mj_weirdness", 0), 0, 3000, 0)
    var = _clamp_int(s.get("mj_variety", 5), 0, 100, 5)
    if not _nz(prompt):
        return False, "Нужен промпт для MJ.", {}
    payload: Dict[str, Any] = {
        "taskType": "mj_txt2img",
        "prompt": prompt,
        "aspectRatio": aspect,
        "version": version,
        "speed": speed,
        "stylization": styl,
        "weirdness": weird,
        "variety": var,
        "enableTranslation": False,
    }
    return True, None, payload

def build_payload_for_mj_img2img(prompt: str, selfie_url: str, s: Dict[str, Any]) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    if not _nz(selfie_url):
        return False, "Нужно селфи: файл или публичный URL.", {}
    aspect = s.get("mj_aspect") or "16:9"
    if aspect not in ALLOWED_MJ_ASPECTS: aspect = "16:9"
    speed = s.get("mj_speed") or "relaxed"
    if speed not in ALLOWED_MJ_SPEEDS: speed = "relaxed"
    version = str(s.get("mj_version") or "7").strip() or "7"
    styl = _clamp_int(s.get("mj_stylization", 50), 0, 1000, 50)
    weird = _clamp_int(s.get("mj_weirdness", 0), 0, 3000, 0)
    var = _clamp_int(s.get("mj_variety", 5), 0, 100, 5)
    if not _nz(prompt):
        return False, "Нужен промпт для MJ.", {}
    payload: Dict[str, Any] = {
        "taskType": "mj_img2img",
        "prompt": prompt,
        "fileUrls": [selfie_url],
        "aspectRatio": aspect,
        "version": version,
        "speed": speed,
        "stylization": styl,
        "weirdness": weird,
        "variety": var,
        "enableTranslation": False,
    }
    return True, None, payload


# ==========================
#   VEO API
# ==========================
def submit_kie_veo(prompt: str, aspect: str, image_url: Optional[str], model_key: str, req_id: str) -> Tuple[bool, Optional[str], str]:
    url = join_url(KIE_BASE_URL, KIE_VEO_GEN_PATH)
    payload = _build_payload_for_veo(prompt, aspect, image_url, model_key)
    status, j = _post_json(url, payload, req_id=req_id)
    code = j.get("code", status)
    if status == 200 and code == 200:
        task_id = _extract_task_id(j)
        if task_id:
            return True, task_id, "Задача создана."
        return False, None, "Ответ KIE без taskId."
    return False, None, _kie_error_message(status, j)

def get_kie_veo_status(task_id: str, req_id: str) -> Tuple[bool, Optional[int], Optional[str], Optional[str]]:
    url = join_url(KIE_BASE_URL, KIE_VEO_STATUS_PATH)
    status, j = _get_json(url, {"taskId": task_id}, req_id=req_id)
    code = j.get("code", status)
    if status == 200 and code == 200:
        flag, msg, data = _parse_success_flag(j)
        return True, flag, msg, _extract_result_url(data or {})
    return False, None, _kie_error_message(status, j), None


# ==========================
#   MJ API
# ==========================
def submit_kie_mj(payload: Dict[str, Any], req_id: str) -> Tuple[bool, Optional[str], str]:
    url = join_url(KIE_BASE_URL, KIE_MJ_GEN_PATH)
    status, j = _post_json(url, payload, req_id=req_id)
    code = j.get("code", status)
    if status == 200 and code == 200:
        task_id = _extract_task_id(j)
        if task_id:
            return True, task_id, "MJ задача создана."
        return False, None, "Ответ KIE (MJ) без taskId."
    return False, None, _kie_error_message(status, j)

def get_kie_mj_status(task_id: str, req_id: str) -> Tuple[bool, Optional[int], Optional[str], Dict[str, Any]]:
    url = join_url(KIE_BASE_URL, KIE_MJ_STATUS_PATH)
    status, j = _get_json(url, {"taskId": task_id}, req_id=req_id)
    code = j.get("code", status)
    if status == 200 and code == 200:
        flag, msg, data = _parse_success_flag(j)
        return True, flag, msg, (data or {})
    return False, None, _kie_error_message(status, j), {}


# ==========================
#   Sending video (robust)
# ==========================
async def send_video_with_fallback(ctx: ContextTypes.DEFAULT_TYPE, chat_id: int, url: str) -> bool:
    # 1) попытка по URL
    try:
        await ctx.bot.send_video(chat_id=chat_id, video=url, supports_streaming=True)
        return True
    except Exception as e:
        log.warning("Direct URL send failed: %s", e)

    # 2) скачиваем → пробуем как видео → при неудаче как документ
    tmp_path = None
    fname = "result.mp4"
    try:
        r = requests.get(url, stream=True, timeout=180)
        r.raise_for_status()
        ct = (r.headers.get("Content-Type") or "").lower()
        if ".mov" in url.lower() or "quicktime" in ct:
            fname = "result.mov"
        elif ".webm" in url.lower() or "webm" in ct:
            fname = "result.webm"

        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(fname)[1]) as f:
            for chunk in r.iter_content(chunk_size=256 * 1024):
                if chunk: f.write(chunk)
            tmp_path = f.name

        # как видео
        try:
            with open(tmp_path, "rb") as f:
                await ctx.bot.send_video(chat_id=chat_id, video=InputFile(f, filename=fname), supports_streaming=True)
            return True
        except Exception as e:
            log.warning("Send as video failed, fallback as document. %s", e)

        # как документ (гарантия доставки)
        with open(tmp_path, "rb") as f:
            await ctx.bot.send_document(chat_id=chat_id, document=InputFile(f, filename=fname))
        return True
    except Exception as e:
        log.exception("File send failed: %s", e)
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

    start_ts = time.time()
    req_id = f"veo:{gen_id}:{task_id}"
    log.info("VEO poll start | %s", req_id)

    try:
        while True:
            if s.get("generation_id") != gen_id:
                log.info("VEO poll stop (superseded) | %s", req_id)
                return

            ok, flag, msg, res_url = await asyncio.to_thread(get_kie_veo_status, task_id, req_id)
            if not ok:
                await ctx.bot.send_message(chat_id, f"❌ Ошибка статуса: {msg or 'неизвестно'}")
                break

            if flag == 0:
                if (time.time() - start_ts) > POLL_TIMEOUT_SECS:
                    await ctx.bot.send_message(chat_id, "⏳ Таймаут ожидания результата VEO.")
                    break
                await asyncio.sleep(POLL_INTERVAL_SECS)
                continue

            if flag == 1:
                if not res_url:
                    await ctx.bot.send_message(chat_id, "⚠️ Готово, но ссылка не найдена (ответ KIE без URL).")
                    break
                if s.get("generation_id") != gen_id:
                    return
                sent = await send_video_with_fallback(ctx, chat_id, res_url)
                s["last_result_url"] = res_url if sent else None
                await ctx.bot.send_message(
                    chat_id,
                    "✅ *Готово!*",
                    parse_mode=ParseMode.MARKDOWN,
                    reply_markup=InlineKeyboardMarkup(
                        [[InlineKeyboardButton("🚀 Сгенерировать ещё видео", callback_data="start_new_cycle")]]
                    ),
                )
                break

            if flag in (2, 3):
                await ctx.bot.send_message(chat_id, f"❌ Ошибка KIE: {msg or 'без сообщения'}")
                break

            await asyncio.sleep(POLL_INTERVAL_SECS)
    except Exception as e:
        log.exception("VEO poller crashed | %s | %s", req_id, e)
        try: await ctx.bot.send_message(chat_id, "❌ Внутренняя ошибка при опросе VEO.")
        except Exception: pass
    finally:
        if s.get("generation_id") == gen_id:
            s["generating"] = False
            s["generation_id"] = None
        log.info("VEO poll end | %s", req_id)


# ==========================
#   Polling MJ (общий для txt2img и img2img)
# ==========================
async def poll_mj_and_send(chat_id: int, task_id: str, gen_id: str, ctx: ContextTypes.DEFAULT_TYPE, mode: str):
    # mode: 'mj_text' или 'mj_face'
    s = state(ctx)
    if mode == "mj_text":
        s["mj_txt_generating"] = True
        s["mj_txt_generation_id"] = gen_id
        s["mj_txt_last_task_id"] = task_id
    else:
        s["mj_face_generating"] = True
        s["mj_face_generation_id"] = gen_id
        s["mj_face_last_task_id"] = task_id

    start_ts = time.time()
    req_id = f"{mode}:{gen_id}:{task_id}"
    log.info("MJ poll start | %s", req_id)

    try:
        while True:
            if (mode == "mj_text" and s.get("mj_txt_generation_id") != gen_id) or \
               (mode == "mj_face" and s.get("mj_face_generation_id") != gen_id):
                log.info("MJ poll stop (superseded) | %s", req_id)
                return

            ok, flag, msg, data = await asyncio.to_thread(get_kie_mj_status, task_id, req_id)
            if not ok:
                await ctx.bot.send_message(chat_id, f"❌ Ошибка статуса MJ: {msg or 'неизвестно'}")
                break

            if flag == 0:
                if (time.time() - start_ts) > POLL_TIMEOUT_SECS:
                    await ctx.bot.send_message(chat_id, "⏳ Таймаут ожидания результата MJ.")
                    break
                await asyncio.sleep(POLL_INTERVAL_SECS)
                continue

            if flag == 1:
                urls = _extract_result_urls_list(data or {})
                if not urls:
                    await ctx.bot.send_message(chat_id, "⚠️ MJ готово, но список изображений пуст.")
                    break
                if (mode == "mj_text" and s.get("mj_txt_generation_id") != gen_id) or \
                   (mode == "mj_face" and s.get("mj_face_generation_id") != gen_id):
                    return
                # запоминаем последнее
                if mode == "mj_text":
                    s["mj_txt_last_images"] = urls
                else:
                    s["mj_face_last_images"] = urls
                # отправляем до 4 изображений
                for u in urls[:4]:
                    try:
                        await ctx.bot.send_photo(chat_id=chat_id, photo=u)
                    except Exception as e:
                        log.warning("Send MJ photo failed | %s | url=%s | err=%s", req_id, _short(u, 200), e)
                await ctx.bot.send_message(
                    chat_id,
                    "✅ *Готово!*",
                    parse_mode=ParseMode.MARKDOWN,
                    reply_markup=InlineKeyboardMarkup(
                        [[InlineKeyboardButton("🚀 Ещё", callback_data=f"mode:{mode}")]]
                    ),
                )
                break

            if flag in (2, 3):
                reason = (data or {}).get("errorMessage") or msg or "generation failed"
                errc = (data or {}).get("errorCode")
                if errc not in (None, "", 0): reason = f"{reason} (code {errc})"
                await ctx.bot.send_message(chat_id, f"❌ Ошибка MJ (flag={flag}): {reason}")
                break

            await ctx.bot.send_message(chat_id, f"⚠️ Неизвестный статус MJ (flag={flag}).")
            break

    except Exception as e:
        log.exception("MJ poller crashed | %s | %s", req_id, e)
        try: await ctx.bot.send_message(chat_id, "❌ Внутренняя ошибка при опросе MJ.")
        except Exception: pass
    finally:
        if mode == "mj_text" and s.get("mj_txt_generation_id") == gen_id:
            s["mj_txt_generating"] = False
            s["mj_txt_generation_id"] = None
        if mode == "mj_face" and s.get("mj_face_generation_id") == gen_id:
            s["mj_face_generating"] = False
            s["mj_face_generation_id"] = None
        log.info("MJ poll end | %s", req_id)


# ==========================
#   Handlers
# ==========================
async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = state(ctx); s.update({**DEFAULT_STATE})
    await update.message.reply_text(WELCOME, parse_mode=ParseMode.MARKDOWN, reply_markup=main_menu_kb())

async def health(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    parts = [
        f"PTB: `{getattr(_tg, '__version__', 'unknown')}`",
        f"KIE_BASE_URL: `{KIE_BASE_URL}`",
        f"VEO_GEN: `{KIE_VEO_GEN_PATH}`",
        f"VEO_STATUS: `{KIE_VEO_STATUS_PATH}`",
        f"MJ_GEN: `{KIE_MJ_GEN_PATH}`",
        f"MJ_STATUS: `{KIE_MJ_STATUS_PATH}`",
        f"KIE key: `{'set' if KIE_API_KEY else 'missing'}`",
        f"OPENAI key: `{'set' if OPENAI_API_KEY else 'missing'}`",
        f"LOG_LEVEL: `{LOG_LEVEL}`",
        f"LOG_FILE: `{LOG_FILE or 'stdout'}`",
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
        log.warning("show_card_veo edit failed: %s", e)
        try:
            m = await ctx.bot.send_message(chat_id=chat_id, text=text, parse_mode=ParseMode.MARKDOWN,
                                           reply_markup=kb, disable_web_page_preview=True)
            s["last_ui_msg_id"] = m.message_id
        except Exception as e2:
            log.exception("show_card_veo send failed: %s", e2)

async def show_card_mj_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = state(ctx)
    text = build_card_text_mj_text(s)
    kb = card_keyboard_mj_text(s)
    try:
        await (update.callback_query.message.reply_text(text, parse_mode=ParseMode.MARKDOWN,
                                                        reply_markup=kb, disable_web_page_preview=True)
               if update.callback_query else
               update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN,
                                         reply_markup=kb, disable_web_page_preview=True))
    except Exception as e:
        log.exception("show_card_mj_text failed: %s", e)

async def show_card_mj_face(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = state(ctx)
    text = build_card_text_mj_face(s)
    kb = card_keyboard_mj_face(s)
    try:
        await (update.callback_query.message.reply_text(text, parse_mode=ParseMode.MARKDOWN,
                                                        reply_markup=kb, disable_web_page_preview=True)
               if update.callback_query else
               update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN,
                                         reply_markup=kb, disable_web_page_preview=True))
    except Exception as e:
        log.exception("show_card_mj_face failed: %s", e)

async def on_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    data = (query.data or "").strip()
    await query.answer()
    s = state(ctx)

    # Общие
    if data == "faq":
        await query.message.reply_text(
            "FAQ:\n"
            "• VEO: Fast/Quality, 16:9/9:16, фото-референс.\n"
            "• MJ: два режима — из текста и из селфи.\n"
            "• Видео всегда доедет: сначала по URL, иначе файлом/документом.",
            reply_markup=main_menu_kb(),
        ); return

    if data == "back":
        s.update({**DEFAULT_STATE})
        await query.message.reply_text("Главное меню:", reply_markup=main_menu_kb()); return

    if data == "start_new_cycle":
        s.update({**DEFAULT_STATE})
        await query.message.reply_text("Выберите режим:", reply_markup=main_menu_kb()); return

    # Режимы
    if data.startswith("mode:"):
        _, mode = data.split(":", 1); s["mode"] = mode
        if mode in ("veo_text", "veo_photo"):
            s["aspect"] = "16:9"; s["model"] = "veo3_fast"
            await query.message.reply_text("VEO: пришлите идею/промпт." if mode=="veo_text" else "VEO: пришлите фото (и при желании — подпись-промпт).")
            await show_card_veo(update, ctx); return
        if mode == "prompt_master":
            await query.message.reply_text("Промпт-мастер: пришлите идею (1–2 фразы)."); return
        if mode == "chat":
            await query.message.reply_text("Обычный чат: напишите вопрос."); return
        if mode == "mj_text":
            await query.message.reply_text("MJ (текст→фото): пришлите промпт.")
            await show_card_mj_text(update, ctx); return
        if mode == "mj_face":
            await query.message.reply_text("MJ (селфи→фото): пришлите селфи (или публичный URL), затем промпт.")
            await show_card_mj_face(update, ctx); return

    # VEO настройки
    if data.startswith("aspect:"):
        _, val = data.split(":", 1)
        s["aspect"] = "9:16" if val.strip()=="9:16" else "16:9"
        await show_card_veo(update, ctx, edit_only_markup=True); return

    if data.startswith("model:"):
        _, val = data.split(":", 1)
        s["model"] = "veo3" if val.strip()=="veo3" else "veo3_fast"
        await show_card_veo(update, ctx, edit_only_markup=True); return

    if data == "card:toggle_photo":
        if s.get("last_image_url"):
            s["last_image_url"] = None
            await query.message.reply_text("Фото-референс удалён."); await show_card_veo(update, ctx)
        else:
            await query.message.reply_text("Пришлите фото (или публичный URL).")
        return

    if data == "card:edit_prompt":
        await query.message.reply_text("Пришлите новый текст промпта."); return

    if data == "card:reset":
        keep_aspect = s.get("aspect") or "16:9"
        keep_model  = s.get("model") or "veo3_fast"
        s.update({**DEFAULT_STATE}); s["aspect"] = keep_aspect; s["model"] = keep_model
        await query.message.reply_text("Карточка очищена."); await show_card_veo(update, ctx); return

    if data == "card:generate":
        if s.get("generating"): await query.message.reply_text("⏳ Генерация уже идёт."); return
        if not s.get("last_prompt"): await query.message.reply_text("Сначала укажите текст промпта."); return
        gen_id = uuid.uuid4().hex[:12]
        req_id = f"veo:{gen_id}"
        ok, task_id, msg = await asyncio.to_thread(
            submit_kie_veo, s["last_prompt"].strip(), s.get("aspect", "16:9"),
            s.get("last_image_url"), s.get("model", "veo3_fast"), req_id
        )
        if not ok or not task_id:
            await query.message.reply_text(f"❌ Не удалось создать VEO-задачу: {msg}"); return
        s["generating"] = True; s["generation_id"] = gen_id; s["last_task_id"] = task_id
        await query.message.reply_text(f"🚀 Задача отправлена ({'Fast' if s.get('model')=='veo3_fast' else 'Quality'}). taskId={task_id}")
        await query.message.reply_text("⏳ Идёт рендеринг…")
        asyncio.create_task(poll_veo_and_send(update.effective_chat.id, task_id, gen_id, ctx)); return

    # MJ TEXT настройки
    if data.startswith("mjtxt:"):
        _, cmd = data.split(":", 1)
        if cmd == "edit_prompt":
            await query.message.reply_text("Пришлите новый промпт для MJ (текст→фото)."); return
        if cmd == "generate":
            if s.get("mj_txt_generating"): await query.message.reply_text("⏳ Генерация уже идёт."); return
            if not s.get("mj_txt_prompt"): await query.message.reply_text("Нужен промпт."); return
            ok, err, payload = build_payload_for_mj_txt2img(s["mj_txt_prompt"], s)
            if not ok:
                await query.message.reply_text(f"❌ {err}"); return
            gen_id = uuid.uuid4().hex[:12]; req_id = f"mjtxt:{gen_id}"
            ok, task_id, msg = await asyncio.to_thread(submit_kie_mj, payload, req_id)
            if not ok or not task_id:
                await query.message.reply_text(f"❌ Не удалось создать MJ-задачу: {msg}"); return
            s["mj_txt_generating"] = True; s["mj_txt_generation_id"] = gen_id; s["mj_txt_last_task_id"] = task_id
            await query.message.reply_text(f"🧑‍🎨 MJ (текст) задача отправлена. taskId={task_id}")
            await query.message.reply_text("⏳ Идёт рендеринг…")
            asyncio.create_task(poll_mj_and_send(update.effective_chat.id, task_id, gen_id, ctx, "mj_text")); return

    # MJ FACE настройки
    if data.startswith("mjface:"):
        _, cmd = data.split(":", 1)
        if cmd == "toggle_selfie":
            if s.get("mj_selfie_url"):
                s["mj_selfie_url"] = None
                await query.message.reply_text("Селфи удалено."); await show_card_mj_face(update, ctx)
            else:
                await query.message.reply_text("Пришлите селфи как фото или публичный URL.")
            return
        if cmd == "edit_prompt":
            await query.message.reply_text("Пришлите промпт для MJ (селфи→фото)."); return
        if cmd == "generate":
            if s.get("mj_face_generating"): await query.message.reply_text("⏳ Генерация уже идёт."); return
            if not s.get("mj_selfie_url"): await query.message.reply_text("Нужно селфи."); return
            if not s.get("mj_face_prompt"): await query.message.reply_text("Нужен промпт."); return
            ok, err, payload = build_payload_for_mj_img2img(s["mj_face_prompt"], s["mj_selfie_url"], s)
            if not ok:
                await query.message.reply_text(f"❌ {err}"); return
            gen_id = uuid.uuid4().hex[:12]; req_id = f"mjface:{gen_id}"
            ok, task_id, msg = await asyncio.to_thread(submit_kie_mj, payload, req_id)
            if not ok or not task_id:
                await query.message.reply_text(f"❌ Не удалось создать MJ-задачу: {msg}"); return
            s["mj_face_generating"] = True; s["mj_face_generation_id"] = gen_id; s["mj_face_last_task_id"] = task_id
            await query.message.reply_text(f"🧑‍🦰 MJ (селфи) задача отправлена. taskId={task_id}")
            await query.message.reply_text("⏳ Идёт рендеринг…")
            asyncio.create_task(poll_mj_and_send(update.effective_chat.id, task_id, gen_id, ctx, "mj_face")); return

    # MJ общие параметры
    if data.startswith("mj_aspect:"):
        _, val = data.split(":", 1)
        if val in ALLOWED_MJ_ASPECTS:
            s["mj_aspect"] = val
        # показываем соответствующую карточку
        if s.get("mode") == "mj_text":
            await show_card_mj_text(update, ctx)
        else:
            await show_card_mj_face(update, ctx)
        return

    if data.startswith("mj_speed:"):
        _, val = data.split(":", 1)
        if val in ALLOWED_MJ_SPEEDS:
            s["mj_speed"] = val
        if s.get("mode") == "mj_text":
            await show_card_mj_text(update, ctx)
        else:
            await show_card_mj_face(update, ctx)
        return


async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = state(ctx)
    text = (update.message.text or "").strip()
    mode = s.get("mode")

    # Если это публичный URL картинки
    low = text.lower()
    if low.startswith(("http://", "https://")) and any(low.split("?")[0].endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp")):
        if mode == "mj_face":
            s["mj_selfie_url"] = text.strip()
            await update.message.reply_text("✅ Селфи-URL принят (MJ).")
            await show_card_mj_face(update, ctx); return
        else:
            s["last_image_url"] = text.strip()
            await update.message.reply_text("✅ Ссылка на изображение принята.")
            await show_card_veo(update, ctx); return

    if mode == "prompt_master":
        prompt = await oai_prompt_master(text)
        if not prompt:
            await update.message.reply_text("⚠️ Prompt-Master недоступен или ответ пуст."); return
        s["last_prompt"] = prompt
        await update.message.reply_text("🧠 Готово! Промпт добавлен в карточку.")
        await show_card_veo(update, ctx); return

    if mode == "chat":
        if openai is None or not OPENAI_API_KEY:
            await update.message.reply_text("⚠️ ChatGPT недоступен (нет OPENAI_API_KEY)."); return
        try:
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

    if mode == "mj_text":
        s["mj_txt_prompt"] = text
        await update.message.reply_text("🟣 *MJ (текст→фото) — промпт установлен.*", parse_mode=ParseMode.MARKDOWN)
        await show_card_mj_text(update, ctx); return

    if mode == "mj_face":
        s["mj_face_prompt"] = text
        await update.message.reply_text("🟣 *MJ (селфи→фото) — промпт установлен.*", parse_mode=ParseMode.MARKDOWN)
        await show_card_mj_face(update, ctx); return

    # По умолчанию — это VEO промпт
    s["last_prompt"] = text
    await update.message.reply_text(
        "🟦 *VEO — подготовка к рендеру*\nПроверь карточку ниже и жми «Сгенерировать».",
        parse_mode=ParseMode.MARKDOWN,
    )
    await show_card_veo(update, ctx)

async def on_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = state(ctx)
    photos = update.message.photo
    if not photos: return
    ph = photos[-1]
    try:
        file = await ctx.bot.get_file(ph.file_id)
        file_path = file.file_path
        if not file_path:
            await update.message.reply_text("⚠️ Не удалось получить путь к файлу Telegram."); return
        url = tg_file_direct_url(TELEGRAM_TOKEN, file_path)
        log.info("Photo via TG path: ...%s", mask_secret(url, show=10))
        if s.get("mode") == "mj_face":
            s["mj_selfie_url"] = url
            await update.message.reply_text("🖼️ Селфи принято (MJ).")
            await show_card_mj_face(update, ctx)
        else:
            s["last_image_url"] = url
            await update.message.reply_text("🖼️ Фото принято как референс (VEO).")
            await show_card_veo(update, ctx)
    except Exception as e:
        log.exception("Get photo failed: %s", e)
        await update.message.reply_text("⚠️ Не удалось обработать фото. Пришлите публичный URL картинки текстом.")


# ==========================
#   Quick commands
# ==========================
async def cmd_veo(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = state(ctx); s.update({**DEFAULT_STATE})
    s["mode"] = "veo_text"; s["aspect"] = "16:9"; s["model"] = "veo3_fast"
    await update.message.reply_text("Режим VEO: пришлите идею или готовый промпт.")
    await show_card_veo(update, ctx)

async def cmd_mj(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = state(ctx); s.update({**DEFAULT_STATE})
    s["mode"] = "mj_text"
    await update.message.reply_text("MJ (текст→фото): пришлите промпт.")
    await show_card_mj_text(update, ctx)


# ==========================
#   Entry
# ==========================
def main():
    if not TELEGRAM_TOKEN: raise RuntimeError("TELEGRAM_TOKEN is not set")
    if not KIE_BASE_URL:   raise RuntimeError("KIE_BASE_URL is not set")

    app = (ApplicationBuilder().token(TELEGRAM_TOKEN).rate_limiter(AIORateLimiter()).build())

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("health", health))
    app.add_handler(CommandHandler("veo", cmd_veo))
    app.add_handler(CommandHandler("mj", cmd_mj))
    app.add_handler(CallbackQueryHandler(on_callback))
    app.add_handler(MessageHandler(filters.PHOTO, on_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_error_handler(error_handler)

    log.info(
        "Bot starting. PTB=20.7 | KIE_BASE=%s | VEO_GEN=%s | VEO_STATUS=%s | MJ_GEN=%s | MJ_STATUS=%s | LOG_LEVEL=%s | LOG_FILE=%s",
        KIE_BASE_URL, KIE_VEO_GEN_PATH, KIE_VEO_STATUS_PATH, KIE_MJ_GEN_PATH, KIE_MJ_STATUS_PATH, LOG_LEVEL, LOG_FILE or "stdout"
    )
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    # если когда-то был webhook — снимите:
    # https://api.telegram.org/bot<YOUR_TOKEN>/deleteWebhook
    main()
