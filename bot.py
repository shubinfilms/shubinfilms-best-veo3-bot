# -*- coding: utf-8 -*-
# Best VEO3 + MJ Bot — PTB 20.7 (фикс 400: Image fetch failed + доработки UI)
# Версия: 2025-09-08

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
#   Инициализация / ENV
# ==========================
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()

# OpenAI (старый SDK 0.28.1 — опционально для Prompt-Master/чата)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
try:
    import openai  # type: ignore
    if OPENAI_API_KEY:
        openai.api_key = OPENAI_API_KEY
except Exception:
    openai = None

# ---- KIE ----
KIE_API_KEY = os.getenv("KIE_API_KEY", "").strip()                 # токен без/с Bearer
KIE_BASE_URL = os.getenv("KIE_BASE_URL", "https://api.kie.ai")     # https://api.kie.ai

# VEO endpoints
KIE_VEO_GEN_PATH = os.getenv("KIE_GEN_PATH", "/api/v1/veo/generate")
KIE_VEO_STATUS_PATH = os.getenv("KIE_STATUS_PATH", "/api/v1/veo/record-info")

# MJ endpoints
KIE_MJ_GEN_PATH = "/api/v1/mj/generate"
KIE_MJ_STATUS_PATH = "/api/v1/mj/record-info"

# ⚠️ Новый upload-fallback для любых изображений (исправляет 400 Image fetch failed)
KIE_UPLOAD_PATH = os.getenv("KIE_UPLOAD_PATH", "/common-api/upload")  # если другой — укажи в ENV
ENABLE_KIE_UPLOAD = os.getenv("ENABLE_KIE_UPLOAD", "1").strip() not in ("0", "false", "False")

PROMPTS_CHANNEL_URL = os.getenv("PROMPTS_CHANNEL_URL", "https://t.me/bestveo3promts").strip()
TOPUP_URL = os.getenv("TOPUP_URL", "https://t.me/bestveo3promts").strip()

# Паузы
POLL_INTERVAL_SECS = int(os.getenv("POLL_INTERVAL_SECS", "6"))
POLL_TIMEOUT_SECS = int(os.getenv("POLL_TIMEOUT_SECS", str(20 * 60)))

# Логирование
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("best-veo3-bot")

# Версия PTB в логи
try:
    import telegram as _tg
    log.info("PTB version: %s", getattr(_tg, "__version__", "unknown"))
except Exception:
    pass


# ==========================
#   Утилиты
# ==========================
def join_url(base: str, path: str) -> str:
    u = f"{base.rstrip('/')}/{path.lstrip('/')}"
    return u.replace("://", "§§").replace("//", "/").replace("§§", "://")

def mask_secret(s: str, show: int = 6) -> str:
    if not s:
        return ""
    s = s.strip()
    if len(s) <= show:
        return "*" * len(s)
    return f"{'*' * (len(s) - show)}{s[-show:]}"

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
    # В URL присутствует токен — не логируем целиком
    return f"https://api.telegram.org/file/bot{bot_token}/{file_path.lstrip('/')}"

def _nz(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s2 = s.strip()
    return s2 if s2 else None


# ==========================
#   KIE upload-fallback
# ==========================
def _kie_headers_json() -> Dict[str, str]:
    token = KIE_API_KEY
    if token and not token.lower().startswith("bearer "):
        token = f"Bearer {token}"
    return {"Authorization": token or "", "Content-Type": "application/json"}

def _kie_headers_multipart() -> Dict[str, str]:
    token = KIE_API_KEY
    if token and not token.lower().startswith("bearer "):
        token = f"Bearer {token}"
    # multipart сам поставит boundary
    return {"Authorization": token or ""}

def _post_json(url: str, payload: Dict[str, Any], timeout: int = 40) -> Tuple[int, Dict[str, Any]]:
    r = requests.post(url, json=payload, headers=_kie_headers_json(), timeout=timeout)
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, {"error": r.text}

def _get_json(url: str, params: Dict[str, Any], timeout: int = 40) -> Tuple[int, Dict[str, Any]]:
    r = requests.get(url, params=params, headers=_kie_headers_json(), timeout=timeout)
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, {"error": r.text}

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

def _extract_result_url(data: Dict[str, Any]) -> Optional[str]:
    resp = data.get("response") or {}
    for key in ("resultUrls", "originUrls", "resultUrl", "originUrl"):
        url = pick_first_url(resp.get(key))
        if url:
            return url
    for key in ("resultUrls", "originUrls", "resultUrl", "originUrl"):
        url = pick_first_url(data.get(key))
        if url:
            return url
    return pick_first_url(data.get("url"))

def _extract_result_urls_list(data: Dict[str, Any]) -> List[str]:
    urls: List[str] = []
    ri = (data or {}).get("resultInfoJson") or {}
    arr = ri.get("resultUrls") or []
    if isinstance(arr, list):
        for item in arr:
            if isinstance(item, str) and item.strip():
                urls.append(item.strip())
            elif isinstance(item, dict):
                u = item.get("resultUrl") or item.get("url")
                if isinstance(u, str) and u.strip():
                    urls.append(u.strip())
    return urls

def _kie_error_message(status_code: int, j: Dict[str, Any]) -> str:
    code = j.get("code", status_code)
    msg = j.get("msg") or j.get("message") or j.get("error") or ""
    if code in (401, 403):
        base = "Доступ запрещён (KIE_API_KEY / Bearer)."
    elif code == 402:
        base = "Недостаточно кредитов (402)."
    elif code == 429:
        base = "Превышен лимит запросов (429)."
    elif code == 451:
        base = "Ошибка загрузки изображения (451)."
    elif code == 500:
        base = "Внутренняя ошибка KIE (500)."
    elif code == 422:
        base = "Запрос отклонён модерацией (422)."
    elif code == 400:
        base = "Неверный запрос (400) — проверьте параметры."
    else:
        base = f"KIE code {code}."
    return f"{base} {('Сообщение: ' + msg) if msg else ''}".strip()

def _kie_upload_bytes(fname: str, content: bytes) -> Optional[str]:
    """
    Загружает байты файла в KIE upload API и возвращает публичный URL.
    Ожидаемый ответ: {"code":200,"data":{"url":"https://..."}}
    """
    if not ENABLE_KIE_UPLOAD:
        return None
    try:
        url = join_url(KIE_BASE_URL, KIE_UPLOAD_PATH)
        files = {"file": (fname, content)}
        r = requests.post(url, headers=_kie_headers_multipart(), files=files, timeout=90)
        j = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        if r.ok and (j.get("code") == 200 or not j):
            data = j.get("data") or {}
            out = data.get("url") or data.get("fileUrl") or data.get("downloadUrl")
            if isinstance(out, str) and out.strip():
                return out.strip()
        log.warning("KIE upload response: %s %s", r.status_code, (j or r.text)[:400])
    except Exception as e:
        log.exception("KIE upload failed: %s", e)
    return None

def ensure_public_image_url(url_or_tg_url: str) -> str:
    """
    Принимает URL (включая Telegram file URL) и пытается получить постоянный публичный URL через upload-fallback.
    Если upload не удался — вернёт исходный URL.
    """
    try:
        r = requests.get(url_or_tg_url, timeout=60)
        r.raise_for_status()
        # извлечём имя
        fname = "image"
        ct = r.headers.get("content-type", "")
        if "jpeg" in ct: fname += ".jpg"
        elif "png" in ct: fname += ".png"
        elif "webp" in ct: fname += ".webp"
        else: fname += ".jpg"
        up = _kie_upload_bytes(fname, r.content)
        if up:
            log.info("Uploaded to KIE: %s", up)
            return up
    except Exception as e:
        log.warning("ensure_public_image_url download failed: %s", e)
    return url_or_tg_url


# ==========================
#   Состояние пользователя
# ==========================
DEFAULT_STATE = {
    "mode": None,              # 'veo_text' | 'veo_photo' | 'prompt_master' | 'chat' | 'mj_face'
    # VEO
    "aspect": None,            # '16:9' | '9:16' (устанавливаем после выбора режима)
    "model": None,             # 'veo3_fast' | 'veo3'
    "last_prompt": None,
    "last_image_url": None,
    "generating": False,
    "generation_id": None,
    "last_ui_msg_id": None,
    "last_task_id": None,
    "last_result_url": None,
    # MJ
    "mj_aspect": "1:1",
    "mj_speed": "relaxed",     # relaxed | fast | turbo
    "mj_version": "7",
    "mj_stylization": 35,      # дефолт ближе к реалистичности
    "mj_weirdness": 0,
    "mj_variety": 5,
    "mj_prompt": None,
    "mj_selfie_url": None,
    "mj_generating": False,
    "mj_generation_id": None,
    "mj_last_task_id": None,
    "mj_last_images": None,
}

def state(ctx: ContextTypes.DEFAULT_TYPE) -> Dict[str, Any]:
    ud = ctx.user_data
    for k, v in DEFAULT_STATE.items():
        ud.setdefault(k, v)
    return ud


# ==========================
#   Кнопки / UI
# ==========================
WELCOME = (
    "🎬 *Veo 3 — супер-генерация видео*\n"
    "Опиши идею — получишь готовый клип. Поддерживаются 16:9 и 9:16, Fast/Quality, фото-референс.\n\n"
    "• Промпт-мастер создаёт кинематографичный EN-промпт (500–900 знаков)\n"
    f"• Больше идей: {PROMPTS_CHANNEL_URL}\n\n"
    "Выберите режим ниже 👇"
)

def main_menu_kb() -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton("🎬 Сгенерировать по тексту (VEO)", callback_data="mode:veo_text")],
        [InlineKeyboardButton("🖼️ Сгенерировать по фото (VEO)",  callback_data="mode:veo_photo")],
        [InlineKeyboardButton("🧑‍🎨 Фото с вашим лицом (MJ)",    callback_data="mode:mj_face")],
        [InlineKeyboardButton("🧠 Промпт-мастер (ChatGPT)",       callback_data="mode:prompt_master")],
        [InlineKeyboardButton("💬 Обычный чат (ChatGPT)",         callback_data="mode:chat")],
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
        return [InlineKeyboardButton("⚡ Fast",      callback_data="model:veo3_fast"),
                InlineKeyboardButton("💎 Quality ✅", callback_data="model:veo3")]
    return [InlineKeyboardButton("⚡ Fast ✅", callback_data="model:veo3_fast"),
            InlineKeyboardButton("💎 Quality", callback_data="model:veo3")]

def mj_aspect_row(current: str) -> List[InlineKeyboardButton]:
    opts = ["1:1", "16:9", "9:16", "3:4"]
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
    if len(prompt_preview) > 900:
        prompt_preview = prompt_preview[:900] + "…"
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

def build_card_text_mj(s: Dict[str, Any]) -> str:
    prompt_preview = (s.get("mj_prompt") or "").strip()
    if len(prompt_preview) > 900:
        prompt_preview = prompt_preview[:900] + "…"
    has_selfie = "есть" if s.get("mj_selfie_url") else "нет"
    lines = [
        "🪄 *Карточка MJ (селфи → фото)*",
        "",
        "✍️ *Промпт:*",
        f"`{prompt_preview or '—'}`",
        "",
        "*📋 Параметры:*",
        f"• Aspect: *{s.get('mj_aspect','1:1')}*",
        f"• Speed: *{s.get('mj_speed','relaxed')}*",
        f"• Version: *{s.get('mj_version','7')}*",
        f"• Stylization: *{s.get('mj_stylization',35)}*",
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
    if s.get("last_prompt"):
        rows.append([InlineKeyboardButton("🚀 Сгенерировать", callback_data="card:generate")])
    if s.get("last_result_url"):
        rows.append([InlineKeyboardButton("🔁 Отправить ещё раз", callback_data="card:resend")])
    rows.append([InlineKeyboardButton("🔁 Начать заново", callback_data="card:reset"),
                 InlineKeyboardButton("⬅️ Назад",             callback_data="back")])
    rows.append([InlineKeyboardButton("💳 Пополнить баланс", url=TOPUP_URL)])
    return InlineKeyboardMarkup(rows)

def card_keyboard_mj(s: Dict[str, Any]) -> InlineKeyboardMarkup:
    rows: List[List[InlineKeyboardButton]] = []
    rows.append([InlineKeyboardButton("🧑‍🦰 Добавить/Удалить селфи", callback_data="mj:toggle_selfie"),
                 InlineKeyboardButton("✍️ Изменить промпт",          callback_data="mj:edit_prompt")])
    rows.append(mj_aspect_row(s.get("mj_aspect","1:1")))
    rows.append(mj_speed_row(s.get("mj_speed","relaxed")))
    rows.append([InlineKeyboardButton("🖼️ Сгенерировать фото", callback_data="mj:generate")])
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
        "Include: lens/optics (mm/anamorphic), camera movement (dolly, push-in, glide, rack focus), "
        "lighting/palette/atmosphere, tiny sensory details (dust, steam, lens flares, wind), "
        "subtle audio cues (music/ambience). Optionally one short hero line in quotes. "
        "Never add JSON or metadata. Output only the prompt text."
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
#   VEO payload / API
# ==========================
def _build_payload_for_veo(prompt: str, aspect: str, image_url: Optional[str], model_key: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "prompt": prompt,
        "aspectRatio": "9:16" if aspect == "9:16" else "16:9",
        "model": "veo3" if model_key == "veo3" else "veo3_fast",
        "enableFallback": aspect == "16:9",
    }
    if image_url:
        payload["imageUrls"] = [image_url]
    return payload

def submit_kie_veo(prompt: str, aspect: str, image_url: Optional[str], model_key: str) -> Tuple[bool, Optional[str], str]:
    url = join_url(KIE_BASE_URL, KIE_VEO_GEN_PATH)
    status, j = _post_json(url, _build_payload_for_veo(prompt, aspect, image_url, model_key))
    code = j.get("code", status)
    if status == 200 and code == 200:
        task_id = _extract_task_id(j)
        if task_id:
            return True, task_id, "Задача создана."
        return False, None, "Ответ KIE без taskId."
    return False, None, _kie_error_message(status, j)

def get_kie_veo_status(task_id: str) -> Tuple[bool, Optional[int], Optional[str], Optional[str]]:
    url = join_url(KIE_BASE_URL, KIE_VEO_STATUS_PATH)
    status, j = _get_json(url, {"taskId": task_id})
    code = j.get("code", status)
    if status == 200 and code == 200:
        flag, msg, data = _parse_success_flag(j)
        return True, flag, msg, _extract_result_url(data or {})
    return False, None, _kie_error_message(status, j), None


# ==========================
#   MJ payload / API
# ==========================
ALLOWED_MJ_ASPECTS = {"1:1", "16:9", "9:16", "3:4"}
ALLOWED_MJ_SPEEDS = {"relaxed", "fast", "turbo"}

def _clamp_int(x: Any, lo: int, hi: int, default: int) -> int:
    try:
        v = int(x)
    except Exception:
        return default
    return max(lo, min(hi, v))

def build_payload_for_mj_img2img(prompt: str, selfie_url: str, s: Dict[str, Any]) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    if not _nz(selfie_url):
        return False, "Нужно селфи: файл или публичный URL.", {}

    aspect = s.get("mj_aspect") or "1:1"
    if aspect not in ALLOWED_MJ_ASPECTS:
        aspect = "1:1"

    speed = s.get("mj_speed") or "relaxed"
    if speed not in ALLOWED_MJ_SPEEDS:
        speed = "relaxed"

    version = str(s.get("mj_version") or "7").strip() or "7"

    styl = _clamp_int(s.get("mj_stylization", 35), 0, 1000, 35)  # ближе к реализму
    weird = _clamp_int(s.get("mj_weirdness", 0), 0, 3000, 0)
    var = _clamp_int(s.get("mj_variety", 5), 0, 100, 5)

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

def submit_kie_mj_img2img(prompt: str, selfie_url: str, s: Dict[str, Any]) -> Tuple[bool, Optional[str], str]:
    ok, err, payload = build_payload_for_mj_img2img(prompt, selfie_url, s)
    if not ok:
        return False, None, err or "Invalid MJ payload."
    url = join_url(KIE_BASE_URL, KIE_MJ_GEN_PATH)
    status, j = _post_json(url, payload)
    code = j.get("code", status)
    if status == 200 and code == 200:
        task_id = _extract_task_id(j)
        if task_id:
            return True, task_id, "MJ задача создана."
        return False, None, "Ответ KIE (MJ) без taskId."
    return False, None, _kie_error_message(status, j)

def get_kie_mj_status(task_id: str) -> Tuple[bool, Optional[int], Optional[str], Dict[str, Any]]:
    url = join_url(KIE_BASE_URL, KIE_MJ_STATUS_PATH)
    status, j = _get_json(url, {"taskId": task_id})
    code = j.get("code", status)
    if status == 200 and code == 200:
        flag, msg, data = _parse_success_flag(j)
        return True, flag, msg, (data or {})
    return False, None, _kie_error_message(status, j), {}


# ==========================
#   Отправка медиа
# ==========================
async def send_video_with_fallback(ctx: ContextTypes.DEFAULT_TYPE, chat_id: int, url: str) -> bool:
    try:
        await ctx.bot.send_video(chat_id=chat_id, video=url, supports_streaming=True)
        return True
    except Exception as e:
        log.warning("Direct URL send failed, try download. %s", e)

    tmp_path = None
    try:
        r = requests.get(url, stream=True, timeout=180)
        r.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
            for chunk in r.iter_content(chunk_size=256 * 1024):
                if chunk:
                    f.write(chunk)
            tmp_path = f.name
        with open(tmp_path, "rb") as f:
            await ctx.bot.send_video(chat_id=chat_id, video=InputFile(f, filename="result.mp4"), supports_streaming=True)
        return True
    except Exception as e:
        log.exception("File send failed: %s", e)
        return False
    finally:
        if tmp_path:
            try: os.unlink(tmp_path)
            except Exception: pass


# ==========================
#   Поллинг VEO
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
        log.exception("VEO poller crashed: %s", e)
        try:
            await ctx.bot.send_message(chat_id, "❌ Внутренняя ошибка при опросе VEO.")
        except Exception:
            pass
    finally:
        if s.get("generation_id") == gen_id:
            s["generating"] = False
            s["generation_id"] = None


# ==========================
#   Поллинг MJ
# ==========================
async def poll_mj_and_send(chat_id: int, task_id: str, gen_id: str, ctx: ContextTypes.DEFAULT_TYPE):
    s = state(ctx)
    s["mj_generating"] = True
    s["mj_generation_id"] = gen_id
    s["mj_last_task_id"] = task_id

    start_ts = time.time()
    try:
        while True:
            if s.get("mj_generation_id") != gen_id:
                return

            ok, flag, msg, data = await asyncio.to_thread(get_kie_mj_status, task_id)

            try:
                log.debug("MJ status raw | ok=%s flag=%s msg=%s | %s", ok, flag, msg, json.dumps(data, ensure_ascii=False)[:1500])
            except Exception:
                pass

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
                if s.get("mj_generation_id") != gen_id:
                    return
                s["mj_last_images"] = urls
                for u in urls[:4]:
                    try:
                        await ctx.bot.send_photo(chat_id=chat_id, photo=u)
                    except Exception:
                        pass
                await ctx.bot.send_message(
                    chat_id,
                    "✅ *Готово!*",
                    parse_mode=ParseMode.MARKDOWN,
                    reply_markup=InlineKeyboardMarkup(
                        [[InlineKeyboardButton("🚀 Сгенерировать ещё фото", callback_data="mode:mj_face")]]
                    ),
                )
                break

            if flag in (2, 3):
                reason = (data or {}).get("errorMessage") or msg or "generation failed"
                errc = (data or {}).get("errorCode")
                if errc not in (None, "", 0):
                    reason = f"{reason} (code {errc})"
                await ctx.bot.send_message(chat_id, f"❌ Ошибка MJ (flag={flag}): {reason}")
                break

            await ctx.bot.send_message(chat_id, f"⚠️ Неизвестный статус MJ (flag={flag}).")
            break

    except Exception as e:
        log.exception("MJ poller crashed: %s", e)
        try:
            await ctx.bot.send_message(chat_id, "❌ Внутренняя ошибка при опросе MJ.")
        except Exception:
            pass
    finally:
        if s.get("mj_generation_id") == gen_id:
            s["mj_generating"] = False
            s["mj_generation_id"] = None


# ==========================
#   Хэндлеры
# ==========================
async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = state(ctx)
    s.update({**DEFAULT_STATE})
    await update.message.reply_text(WELCOME, parse_mode=ParseMode.MARKDOWN, reply_markup=main_menu_kb())

async def health(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    parts = [
        f"PTB: `{getattr(_tg, '__version__', 'unknown')}`",
        f"KIE_BASE_URL: `{KIE_BASE_URL}`",
        f"KIE key: `{'set' if KIE_API_KEY else 'missing'}`",
        f"OPENAI key: `{'set' if OPENAI_API_KEY else 'missing'}`",
        f"UPLOAD: `{'on' if ENABLE_KIE_UPLOAD else 'off'}` path=`{KIE_UPLOAD_PATH}`",
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
                await ctx.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=last_id,
                    text=text,
                    parse_mode=ParseMode.MARKDOWN,
                    reply_markup=kb,
                    disable_web_page_preview=True,
                )
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

async def show_card_mj(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = state(ctx)
    text = build_card_text_mj(s)
    kb = card_keyboard_mj(s)
    try:
        await (update.callback_query.message.reply_text(text, parse_mode=ParseMode.MARKDOWN,
                                                        reply_markup=kb, disable_web_page_preview=True)
               if update.callback_query else
               update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN,
                                         reply_markup=kb, disable_web_page_preview=True))
    except Exception as e:
        log.exception("show_card_mj failed: %s", e)

async def on_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    data = (query.data or "").strip()
    await query.answer()

    s = state(ctx)

    # ===== общие
    if data == "faq":
        await query.message.reply_text(
            "FAQ:\n"
            "• VEO: Fast/Quality, 16:9/9:16, фото-референс.\n"
            "• MJ: селфи + промпт → 4 фото.\n"
            "• Если не стримится видео — отправлю файл.",
            reply_markup=main_menu_kb(),
        )
        return

    if data == "back":
        s.update({**DEFAULT_STATE})
        await query.message.reply_text("Главное меню:", reply_markup=main_menu_kb())
        return

    if data == "start_new_cycle":
        s.update({**DEFAULT_STATE})
        await query.message.reply_text("Выберите режим:", reply_markup=main_menu_kb())
        return

    # ===== выбор режима
    if data.startswith("mode:"):
        _, mode = data.split(":", 1)
        s["mode"] = mode

        if mode in ("veo_text", "veo_photo"):
            s["aspect"] = "16:9"
            s["model"] = "veo3_fast"
            if mode == "veo_text":
                await query.message.reply_text("Режим VEO: пришлите идею или готовый промпт.")
            else:
                await query.message.reply_text("Режим VEO: пришлите фото (и при желании — подпись-промпт).")
            await show_card_veo(update, ctx)
            return

        if mode == "prompt_master":
            await query.message.reply_text("Промпт-мастер: пришлите идею (1–2 фразы). Верну EN-кинопромпт.")
            return

        if mode == "chat":
            await query.message.reply_text("Обычный чат: напишите вопрос.")
            return

        if mode == "mj_face":
            await query.message.reply_text("MJ: пришлите селфи (или публичный URL картинки). Затем — промпт.")
            await show_card_mj(update, ctx)
            return

    # ===== VEO параметры
    if data.startswith("aspect:"):
        if not s.get("aspect"):
            s["aspect"] = "16:9"
        _, val = data.split(":", 1)
        s["aspect"] = "9:16" if val.strip() == "9:16" else "16:9"
        await show_card_veo(update, ctx, edit_only_markup=True)
        return

    if data.startswith("model:"):
        if not s.get("model"):
            s["model"] = "veo3_fast"
        _, val = data.split(":", 1)
        s["model"] = "veo3" if val.strip() == "veo3" else "veo3_fast"
        await show_card_veo(update, ctx, edit_only_markup=True)
        return

    if data == "card:toggle_photo":
        if s.get("last_image_url"):
            s["last_image_url"] = None
            await query.message.reply_text("Фото-референс удалён.")
            await show_card_veo(update, ctx)
        else:
            await query.message.reply_text(
                "Пришлите фото. Я возьму ссылку Telegram и, при необходимости, перезалью в KIE (надёжнее), "
                "либо пришлите публичный URL."
            )
        return

    if data == "card:edit_prompt":
        await query.message.reply_text("Пришлите новый текст промпта (или идею для Prompt-Мастера).")
        return

    if data == "card:reset":
        keep_aspect = s.get("aspect") or "16:9"
        keep_model = s.get("model") or "veo3_fast"
        s.update({**DEFAULT_STATE})
        s["aspect"] = keep_aspect
        s["model"] = keep_model
        await query.message.reply_text("Карточка очищена.")
        await show_card_veo(update, ctx)
        return

    if data == "card:resend":
        if not s.get("last_result_url"):
            await query.message.reply_text("Нет последнего результата для повторной отправки.")
            return
        ok = await send_video_with_fallback(ctx, update.effective_chat.id, s["last_result_url"])
        await query.message.reply_text("✅ Готово!" if ok else "⚠️ Не получилось отправить видео.")
        return

    if data == "card:generate":
        if s.get("generating"):
            await query.message.reply_text("⏳ Генерация уже идёт. Дождитесь завершения.")
            return
        if not s.get("last_prompt"):
            await query.message.reply_text("Сначала укажите текст промпта.")
            return

        # если есть референс — убедимся, что ссылка публичная (upload-fallback)
        img_url = s.get("last_image_url")
        if img_url:
            safe_url = ensure_public_image_url(img_url)
            if safe_url != img_url:
                s["last_image_url"] = safe_url

        ok, task_id, msg = await asyncio.to_thread(
            submit_kie_veo, s["last_prompt"].strip(), s.get("aspect", "16:9"),
            s.get("last_image_url"), s.get("model", "veo3_fast")
        )
        if not ok or not task_id:
            await query.message.reply_text(f"❌ Не удалось создать VEO-задачу: {msg}")
            return

        gen_id = uuid.uuid4().hex[:12]
        s["generating"] = True
        s["generation_id"] = gen_id
        s["last_task_id"] = task_id
        log.info("VEO submitted: chat=%s task=%s gen=%s model=%s", update.effective_chat.id, task_id, gen_id, s.get("model"))
        await query.message.reply_text(f"🚀 Задача отправлена ({'Fast' if s.get('model')=='veo3_fast' else 'Quality'}). taskId={task_id}")
        await query.message.reply_text("⏳ Идёт рендеринг…")
        asyncio.create_task(poll_veo_and_send(update.effective_chat.id, task_id, gen_id, ctx))
        return

    # ===== MJ параметры
    if data.startswith("mj_aspect:"):
        _, val = data.split(":", 1)
        if val in ALLOWED_MJ_ASPECTS:
            s["mj_aspect"] = val
        await show_card_mj(update, ctx)
        return

    if data.startswith("mj_speed:"):
        _, val = data.split(":", 1)
        if val in ALLOWED_MJ_SPEEDS:
            s["mj_speed"] = val
        await show_card_mj(update, ctx)
        return

    if data == "mj:toggle_selfie":
        if s.get("mj_selfie_url"):
            s["mj_selfie_url"] = None
            await query.message.reply_text("Селфи удалено.")
            await show_card_mj(update, ctx)
        else:
            await query.message.reply_text("Пришлите селфи как фото или публичный URL изображения.")
        return

    if data == "mj:edit_prompt":
        await query.message.reply_text("Пришлите новый текст промпта для MJ.")
        return

    if data == "mj:generate":
        if s.get("mj_generating"):
            await query.message.reply_text("⏳ Генерация уже идёт. Дождитесь завершения.")
            return
        if not s.get("mj_selfie_url"):
            await query.message.reply_text("Нужно селфи. Пришлите фото или публичный URL.")
            return
        if not s.get("mj_prompt"):
            await query.message.reply_text("Нужен промпт. Пришлите описание сцены/стиля.")
            return

        # upload-fallback для селфи
        s["mj_selfie_url"] = ensure_public_image_url(s["mj_selfie_url"])

        ok, task_id, msg = await asyncio.to_thread(
            submit_kie_mj_img2img, s["mj_prompt"].strip(), s["mj_selfie_url"], s
        )
        if not ok or not task_id:
            await query.message.reply_text(f"❌ Не удалось создать MJ-задачу: {msg}")
            return

        gen_id = uuid.uuid4().hex[:12]
        s["mj_generating"] = True
        s["mj_generation_id"] = gen_id
        s["mj_last_task_id"] = task_id
        log.info("MJ submitted: chat=%s task=%s gen=%s", update.effective_chat.id, task_id, gen_id)
        await query.message.reply_text(f"🧑‍🎨 MJ задача отправлена. taskId={task_id}")
        await query.message.reply_text("⏳ Идёт рендеринг…")
        asyncio.create_task(poll_mj_and_send(update.effective_chat.id, task_id, gen_id, ctx))
        return


async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = state(ctx)
    text = (update.message.text or "").strip()

    # Публичный URL изображения?
    low = text.lower()
    if low.startswith(("http://", "https://")) and any(low.split("?")[0].endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp")):
        if s.get("mode") == "mj_face":
            s["mj_selfie_url"] = ensure_public_image_url(text.strip())
            await update.message.reply_text("✅ Селфи-URL принят (MJ).")
            await show_card_mj(update, ctx)
            return
        else:
            s["last_image_url"] = ensure_public_image_url(text.strip())
            await update.message.reply_text("✅ Ссылка на изображение принята.")
            await show_card_veo(update, ctx)
            return

    mode = s.get("mode")

    if mode == "prompt_master":
        prompt = await oai_prompt_master(text)
        if not prompt:
            await update.message.reply_text("⚠️ Prompt-Master недоступен или ответ пуст. Попробуйте ещё раз.")
            return
        s["last_prompt"] = prompt
        await update.message.reply_text("🧠 Готово! Промпт добавлен в карточку.")
        await show_card_veo(update, ctx)
        return

    if mode == "chat":
        if openai is None or not OPENAI_API_KEY:
            await update.message.reply_text("⚠️ ChatGPT недоступен (нет OPENAI_API_KEY).")
            return
        try:
            resp = await asyncio.to_thread(
                openai.ChatCompletion.create,
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": "You are a helpful, concise assistant."},
                          {"role": "user", "content": text}],
                temperature=0.5,
                max_tokens=700,
            )
            answer = resp["choices"][0]["message"]["content"].strip()
            await update.message.reply_text(answer)
        except Exception as e:
            log.exception("Chat error: %s", e)
            await update.message.reply_text("⚠️ Ошибка запроса к ChatGPT.")
        return

    if mode == "mj_face":
        s["mj_prompt"] = text
        await update.message.reply_text(
            "🟣 *MJ — подготовка к рендеру*\nНужны селфи и промпт.",
            parse_mode=ParseMode.MARKDOWN,
        )
        await show_card_mj(update, ctx)
        return

    # VEO по умолчанию — это промпт
    s["last_prompt"] = text
    await update.message.reply_text(
        "🟦 *VEO — подготовка к рендеру*\nПроверь карточку ниже и жми «Сгенерировать».",
        parse_mode=ParseMode.MARKDOWN,
    )
    await show_card_veo(update, ctx)

async def on_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = state(ctx)
    photos = update.message.photo
    if not photos:
        return
    ph = photos[-1]
    try:
        file = await ctx.bot.get_file(ph.file_id)
        file_path = file.file_path
        if not file_path:
            await update.message.reply_text("⚠️ Не удалось получить путь к файлу Telegram.")
            return
        tg_url = tg_file_direct_url(TELEGRAM_TOKEN, file_path)
        log.info("Photo via TG path: ...%s", mask_secret(tg_url, show=10))
        safe_url = ensure_public_image_url(tg_url)
        if s.get("mode") == "mj_face":
            s["mj_selfie_url"] = safe_url
            await update.message.reply_text("🖼️ Селфи принято как референс (MJ).")
            await show_card_mj(update, ctx)
        else:
            s["last_image_url"] = safe_url
            await update.message.reply_text("🖼️ Фото принято как референс.")
            await show_card_veo(update, ctx)
    except Exception as e:
        log.exception("Get photo failed: %s", e)
        await update.message.reply_text("⚠️ Не удалось обработать фото. Пришлите публичный URL картинки текстом.")


# ==========================
#   Entry point
# ==========================
def main():
    if not TELEGRAM_TOKEN:
        raise RuntimeError("TELEGRAM_TOKEN is not set")
    if not KIE_BASE_URL:
        raise RuntimeError("KIE_BASE_URL is not set")

    app = (
        ApplicationBuilder()
        .token(TELEGRAM_TOKEN)
        .rate_limiter(AIORateLimiter())
        .build()
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("health", health))
    app.add_handler(CallbackQueryHandler(on_callback))
    app.add_handler(MessageHandler(filters.PHOTO, on_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_error_handler(error_handler)

    log.info(
        "Bot starting. PTB=20.7 | KIE_BASE=%s VEO_GEN=%s VEO_STATUS=%s MJ_GEN=%s MJ_STATUS=%s UPLOAD=%s %s",
        KIE_BASE_URL, KIE_VEO_GEN_PATH, KIE_VEO_STATUS_PATH, KIE_MJ_GEN_PATH, KIE_MJ_STATUS_PATH,
        "on" if ENABLE_KIE_UPLOAD else "off", KIE_UPLOAD_PATH
    )
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    # если когда-то был webhook — снимите:
    # https://api.telegram.org/bot<YOUR_TOKEN>/deleteWebhook
    main()
