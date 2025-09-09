# -*- coding: utf-8 -*-
# Best VEO3 + MJ Bot — PTB 20.7
# Версия: 2025-09-10 (VEO вертикаль фикс, MJ возвращён, 1:1 и 3:4 удалены)
# Логи подробные, надёжная отправка видео/картинок в Telegram, безопасный ре-хост фото.

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

# ---- OpenAI (не обязателен; используется только для Prompt-Master / Chat) ----
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
KIE_VEO_GEN_PATH    = os.getenv("KIE_VEO_GEN_PATH", "/api/v1/veo/generate")
KIE_VEO_STATUS_PATH = os.getenv("KIE_VEO_STATUS_PATH", "/api/v1/veo/record-info")

# ---- KIE upload (официальный файловый сервис) ----
KIE_UPLOAD_BASE          = os.getenv("KIE_UPLOAD_BASE", "https://kieai.redpandaai.co").strip()
KIE_UPLOAD_DIR           = os.getenv("KIE_UPLOAD_DIR", "images/user-uploads").strip()
KIE_STREAM_UPLOAD_PATH   = os.getenv("KIE_STREAM_UPLOAD_PATH", "/api/file-stream-upload")
KIE_URL_UPLOAD_PATH      = os.getenv("KIE_URL_UPLOAD_PATH", "/api/file-url-upload")

# ---- MJ (Midjourney) API — укажите ваши реальные эндпоинты через ENV ----
# Пример для Kie.ai (переопределите, если у вас другой провайдер):
MJ_BASE_URL        = os.getenv("MJ_BASE_URL", "https://api.kie.ai").strip()
MJ_TEXT2IMG_PATH   = os.getenv("MJ_TEXT2IMG_PATH", "/api/v1/mj/generate")         # POST
MJ_STATUS_PATH     = os.getenv("MJ_STATUS_PATH",   "/api/v1/mj/record-info")      # GET ?taskId=
MJ_UPSCALE_PATH    = os.getenv("MJ_UPSCALE_PATH",  "/api/v1/mj/upscale")          # POST (опц.)
MJ_VARIATION_PATH  = os.getenv("MJ_VARIATION_PATH","/api/v1/mj/variation")        # POST (опц.)

PROMPTS_CHANNEL_URL = os.getenv("PROMPTS_CHANNEL_URL", "https://t.me/bestveo3promts").strip()
TOPUP_URL           = os.getenv("TOPUP_URL", "https://t.me/bestveo3promts").strip()

POLL_INTERVAL_SECS = int(os.getenv("POLL_INTERVAL_SECS", "6"))
POLL_TIMEOUT_SECS  = int(os.getenv("POLL_TIMEOUT_SECS",  str(20 * 60)))  # 20 минут

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("best-bot")

try:
    import telegram as _tg
    log.info("PTB version: %s", getattr(_tg, "__version__", "unknown"))
except Exception:
    pass


# ==========================
#   Helpers
# ==========================
def join_url(base: str, path: str) -> str:
    u = f"{base.rstrip('/')}/{path.lstrip('/')}"
    # двойные слэши в середине пути не допускаем
    return u.replace("://", "§§").replace("//", "/").replace("§§", "://")

def _bearer(token: str) -> str:
    if not token:
        return ""
    return token if token.lower().startswith("bearer ") else f"Bearer {token}"

def _headers_json(token: str) -> Dict[str, str]:
    return {"Authorization": _bearer(token), "Content-Type": "application/json"}

def _headers_upload(token: str) -> Dict[str, str]:
    return {"Authorization": _bearer(token)}

def _post_json(url: str, payload: Dict[str, Any], headers: Dict[str, str], timeout: int = 60) -> Tuple[int, Dict[str, Any]]:
    r = requests.post(url, json=payload, headers=headers, timeout=timeout)
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, {"error": r.text}

def _get_json(url: str, params: Dict[str, Any], headers: Dict[str, str], timeout: int = 60) -> Tuple[int, Dict[str, Any]]:
    r = requests.get(url, params=params, headers=headers, timeout=timeout)
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, {"error": r.text}

def event(tag: str, **kw):
    log.info("EVT %s | %s", tag, json.dumps(kw, ensure_ascii=False))


# ==========================
#   Safe public rehost (Telegraph → KIE Upload fallback)
# ==========================
TELEGRAPH_UPLOAD = "https://telegra.ph/upload"

def _upload_bytes_to_telegraph(data: bytes, filename: str) -> Optional[str]:
    files = {"file": (filename, data)}
    try:
        r = requests.post(TELEGRAPH_UPLOAD, files=files, timeout=60)
        r.raise_for_status()
        arr = r.json()
        if isinstance(arr, list) and arr and "src" in arr[0]:
            return "https://telegra.ph" + arr[0]["src"]
    except Exception as e:
        log.warning("Telegraph upload failed: %s", e)
    return None

def _kie_upload_bytes(data: bytes, filename: str, mime: str = "image/jpeg") -> Optional[str]:
    url = join_url(KIE_UPLOAD_BASE, KIE_STREAM_UPLOAD_PATH)
    files = {"file": (filename, data, mime)}
    form  = {"uploadPath": KIE_UPLOAD_DIR, "fileName": filename}
    try:
        r = requests.post(url, headers=_headers_upload(KIE_API_KEY), files=files, data=form, timeout=120)
        r.raise_for_status()
        j = r.json()
        if j.get("success") and j.get("code") == 200:
            d = j.get("data") or {}
            # API может вернуть downloadUrl или fileUrl
            return (d.get("downloadUrl") or d.get("fileUrl") or "").strip() or None
    except Exception as e:
        log.warning("KIE stream upload failed: %s", e)
    return None

def _kie_upload_from_url(file_url: str, filename: Optional[str] = None) -> Optional[str]:
    url = join_url(KIE_UPLOAD_BASE, KIE_URL_UPLOAD_PATH)
    payload = {"fileUrl": file_url, "uploadPath": KIE_UPLOAD_DIR}
    if filename:
        payload["fileName"] = filename
    try:
        r = requests.post(url, json=payload, headers={**_headers_upload(KIE_API_KEY), "Content-Type": "application/json"}, timeout=90)
        r.raise_for_status()
        j = r.json()
        if j.get("success") and j.get("code") == 200:
            d = j.get("data") or {}
            return (d.get("downloadUrl") or d.get("fileUrl") or "").strip() or None
    except Exception as e:
        log.warning("KIE url upload failed: %s", e)
    return None

async def ensure_public_url_from_tg_or_http(ctx: ContextTypes.DEFAULT_TYPE, *, file_id: Optional[str], http_url: Optional[str]) -> Optional[str]:
    """Строим безопасный публичный URL для imageUrls (VEO/MJ) из Telegram photo или уже публичного URL."""
    try:
        if file_id:
            tg_file = await ctx.bot.get_file(file_id)
            raw = bytes(await tg_file.download_as_bytearray())
            name = os.path.basename(tg_file.file_path or "photo.jpg")

            # 1) Быстро: Telegraph
            tele = _upload_bytes_to_telegraph(raw, name)
            if tele:
                return tele

            # 2) Надёжно: KIE Upload
            ext = os.path.splitext(name)[1].lower()
            mime = "image/jpeg" if ext in (".jpg", ".jpeg", "") else ("image/png" if ext == ".png" else "image/webp")
            return _kie_upload_bytes(raw, name if ext else (name + ".jpg"), mime=mime)

        if http_url:
            # 1) Попробуем KIE url-upload
            u_name = os.path.basename(http_url.split("?")[0]) or "image.jpg"
            url_up = _kie_upload_from_url(http_url, filename=u_name)
            if url_up:
                return url_up

            # 2) Скачаем и закинем потоково
            r = requests.get(http_url, timeout=60)
            r.raise_for_status()
            raw = r.content
            name = u_name if os.path.splitext(u_name)[1] else (u_name + ".jpg")
            return _kie_upload_bytes(raw, name, mime="image/jpeg")
    except Exception as e:
        log.exception("ensure_public_url_from_tg_or_http failed: %s", e)
    return None


# ==========================
#   State
# ==========================
DEFAULT_STATE = {
    "mode": None,          # 'veo_text' | 'veo_photo' | 'mj' | 'prompt_master' | 'chat'
    # VEO
    "aspect": "16:9",      # '16:9' | '9:16'
    "model": "veo3_fast",  # 'veo3_fast' | 'veo3'
    "last_prompt": None,
    "last_image_url": None,
    # MJ
    "mj_aspect": "16:9",   # '16:9' | '9:16'  (1:1 и 3:4 УБРАНЫ)
    "mj_style": "relaxed", # подсказка (если провайдер поддерживает)
    "generating": False,
    "generation_id": None,
    "last_ui_msg_id": None,
    "last_task_id": None,
    "last_result_url": None,
}

def S(ctx: ContextTypes.DEFAULT_TYPE) -> Dict[str, Any]:
    ud = ctx.user_data
    for k, v in DEFAULT_STATE.items():
        ud.setdefault(k, v)
    return ud


# ==========================
#   UI
# ==========================
WELCOME = (
    "🤖 *Best VEO3 bot*\n"
    "— Google Veo3 (видео) и MJ (фото). Поддержка только *16:9* и *9:16*.\n"
    "— Я сам перезалью ваши фото на публичный хост, чтобы API их видел.\n\n"
    f"📚 Примеры и идеи: {PROMPTS_CHANNEL_URL}\n"
    "Выберите режим:"
)

def main_menu_kb() -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton("🎬 VEO — текст → видео", callback_data="mode:veo_text")],
        [InlineKeyboardButton("🖼️ VEO — фото → видео", callback_data="mode:veo_photo")],
        [InlineKeyboardButton("🎨 MJ — фото из текста", callback_data="mode:mj")],
        [
            InlineKeyboardButton("🧠 Промпт-мастер", callback_data="mode:prompt_master"),
            InlineKeyboardButton("💬 Чат", callback_data="mode:chat"),
        ],
        [InlineKeyboardButton("📈 Канал с промптами", url=PROMPTS_CHANNEL_URL)],
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

def mj_aspect_row(current: str) -> List[InlineKeyboardButton]:
    # Форматы 1:1 и 3:4 — УДАЛЕНЫ по вашему требованию
    if current == "9:16":
        return [InlineKeyboardButton("16:9", callback_data="mj_aspect:16:9"),
                InlineKeyboardButton("9:16 ✅", callback_data="mj_aspect:9:16")]
    return [InlineKeyboardButton("16:9 ✅", callback_data="mj_aspect:16:9"),
            InlineKeyboardButton("9:16",     callback_data="mj_aspect:9:16")]

def card_text_veo(s: Dict[str, Any]) -> str:
    p = (s.get("last_prompt") or "").strip()
    p = (p[:1000] + "…") if len(p) > 1000 else p
    model_label = "Fast" if s.get("model") == "veo3_fast" else "Quality"
    has_img = "есть" if s.get("last_image_url") else "нет"
    return (
        "🪄 *Карточка VEO*\n\n"
        "✍️ *Промпт:*\n"
        f"`{p or '—'}`\n\n"
        "*📋 Параметры:*\n"
        f"• Aspect: *{s.get('aspect')}*\n"
        f"• Model: *{model_label}*\n"
        f"• Референс: *{has_img}*\n"
    )

def card_text_mj(s: Dict[str, Any]) -> str:
    p = (s.get("last_prompt") or "").strip()
    p = (p[:1000] + "…") if len(p) > 1000 else p
    return (
        "🎨 *MJ — карточка*\n\n"
        "✍️ *Промпт:*\n"
        f"`{p or '—'}`\n\n"
        "*📋 Параметры:*\n"
        f"• Aspect: *{s.get('mj_aspect')}*\n"
        f"• Style: *{s.get('mj_style')}*\n"
    )

def kb_card_veo(s: Dict[str, Any]) -> InlineKeyboardMarkup:
    rows: List[List[InlineKeyboardButton]] = []
    rows.append([InlineKeyboardButton("🖼️ Добавить/удалить фото", callback_data="card:toggle_photo"),
                 InlineKeyboardButton("✍️ Изменить промпт", callback_data="card:edit_prompt")])
    rows.append(aspect_row(s.get("aspect")))
    rows.append(model_row(s.get("model")))
    rows.append([InlineKeyboardButton("🚀 Сгенерировать (VEO)", callback_data="card:generate_veo")])
    rows.append([InlineKeyboardButton("🔁 Начать заново", callback_data="card:reset"),
                 InlineKeyboardButton("⬅️ Назад", callback_data="back")])
    rows.append([InlineKeyboardButton("💳 Пополнить баланс", url=TOPUP_URL)])
    return InlineKeyboardMarkup(rows)

def kb_card_mj(s: Dict[str, Any]) -> InlineKeyboardMarkup:
    rows: List[List[InlineKeyboardButton]] = []
    rows.append([InlineKeyboardButton("✍️ Изменить промпт", callback_data="card:edit_prompt"),
                 InlineKeyboardButton("🎛️ Стиль", callback_data="mj:style")])
    rows.append(mj_aspect_row(s.get("mj_aspect")))
    rows.append([InlineKeyboardButton("🎨 Сгенерировать (MJ)", callback_data="card:generate_mj")])
    rows.append([InlineKeyboardButton("🔁 Начать заново", callback_data="card:reset"),
                 InlineKeyboardButton("⬅️ Назад", callback_data="back")])
    rows.append([InlineKeyboardButton("💳 Пополнить баланс", url=TOPUP_URL)])
    return InlineKeyboardMarkup(rows)


# ==========================
#   OpenAI Prompt-Master / Chat (как было)
# ==========================
async def oai_prompt_master(idea_text: str) -> Optional[str]:
    if openai is None or not OPENAI_API_KEY:
        return None
    system = (
        "You are Prompt-Master for cinematic AI video generation (Veo3). "
        "Return EXACTLY ONE English filmic prompt, 500–900 chars: lens/optics, camera moves, light/color palette, "
        "tiny tactile/sensory details, subtle audio cues. No lists, no prefaces."
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
#   VEO (Kie.ai)
# ==========================
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

    if not value:
        return urls

    if isinstance(value, str):
        s = value.strip()
        if s.startswith("["):
            try:
                arr = json.loads(s)
                if isinstance(arr, list):
                    for v in arr:
                        if isinstance(v, str):
                            add(v)
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
    # Приоритет для 9:16 — originUrls, затем resultUrls
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

def _kie_error_message(status_code: int, j: Dict[str, Any]) -> str:
    code = j.get("code", status_code)
    msg = j.get("msg") or j.get("message") or j.get("error") or ""
    mapping = {
        401: "Доступ запрещён (Bearer).",
        402: "Недостаточно кредитов.",
        429: "Превышен лимит запросов.",
        500: "Внутренняя ошибка KIE.",
        422: "Запрос отклонён модерацией.",
        400: "Неверный запрос (400).",
    }
    base = mapping.get(code, f"KIE code {code}.")
    return f"{base} {('Сообщение: ' + msg) if msg else ''}".strip()

def _veo_payload(prompt: str, aspect: str, image_url: Optional[str], model_key: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "prompt": prompt,
        "aspectRatio": "9:16" if aspect == "9:16" else "16:9",
        "model": "veo3" if model_key == "veo3" else "veo3_fast",
        # по доке fallback только при 16:9
        "enableFallback": (aspect == "16:9"),
    }
    if image_url:
        payload["imageUrls"] = [image_url]
    return payload

def veo_submit(prompt: str, aspect: str, image_url: Optional[str], model_key: str) -> Tuple[bool, Optional[str], str]:
    url = join_url(KIE_BASE_URL, KIE_VEO_GEN_PATH)
    status, j = _post_json(url, _veo_payload(prompt, aspect, image_url, model_key), _headers_json(KIE_API_KEY))
    code = j.get("code", status)
    if status == 200 and code == 200:
        task_id = _extract_task_id(j)
        if task_id: return True, task_id, "OK"
        return False, None, "Ответ без taskId."
    return False, None, _kie_error_message(status, j)

def veo_status(task_id: str) -> Tuple[bool, Optional[int], Optional[str], Optional[str]]:
    url = join_url(KIE_BASE_URL, KIE_VEO_STATUS_PATH)
    status, j = _get_json(url, {"taskId": task_id}, _headers_json(KIE_API_KEY))
    code = j.get("code", status)
    if status == 200 and code == 200:
        data = (j.get("data") or {})
        flag = data.get("successFlag")
        try: flag = int(flag)
        except Exception: flag = None
        msg = j.get("msg") or j.get("message")
        return True, flag, msg, _extract_result_url(data)
    return False, None, _kie_error_message(status, j), None


# Надёжная отправка видео в Telegram
async def send_video_with_fallback(ctx: ContextTypes.DEFAULT_TYPE, chat_id: int, url: str) -> bool:
    # 1) попробовать стримом
    try:
        await ctx.bot.send_video(chat_id=chat_id, video=url, supports_streaming=True)
        return True
    except Exception as e:
        log.warning("Send video by URL failed: %s", e)

    # 2) скачать → как видео → при ошибке как документ
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
                if chunk:
                    f.write(chunk)
            tmp_path = f.name

        try:
            with open(tmp_path, "rb") as f:
                await ctx.bot.send_video(chat_id=chat_id, video=InputFile(f, filename=fname), supports_streaming=True)
            return True
        except Exception as e:
            log.warning("Send as video failed, fallback to document. %s", e)

        with open(tmp_path, "rb") as f:
            await ctx.bot.send_document(chat_id=chat_id, document=InputFile(f, filename=fname))
        return True
    except Exception as e:
        log.exception("Video send failed: %s", e)
        return False
    finally:
        if tmp_path:
            try: os.unlink(tmp_path)
            except Exception: pass


async def poll_veo_and_send(chat_id: int, task_id: str, gen_id: str, ctx: ContextTypes.DEFAULT_TYPE):
    s = S(ctx)
    s["generating"] = True
    s["generation_id"] = gen_id
    s["last_task_id"] = task_id

    start = time.time()
    try:
        while True:
            if s.get("generation_id") != gen_id:
                return

            ok, flag, msg, res_url = await asyncio.to_thread(veo_status, task_id)
            event("VEO_STATUS", task_id=task_id, flag=flag, url=bool(res_url))

            if not ok:
                await ctx.bot.send_message(chat_id, f"❌ Статус VEO: {msg or 'неизвестно'}")
                break

            if flag == 0:
                if (time.time() - start) > POLL_TIMEOUT_SECS:
                    await ctx.bot.send_message(chat_id, "⏳ Таймаут ожидания VEO.")
                    break
                await asyncio.sleep(POLL_INTERVAL_SECS)
                continue

            if flag == 1:
                if not res_url:
                    await ctx.bot.send_message(chat_id, "⚠️ Готово, но URL не найден.")
                    break
                sent = await send_video_with_fallback(ctx, chat_id, res_url)
                s["last_result_url"] = res_url if sent else None
                await ctx.bot.send_message(
                    chat_id,
                    "✅ *Готово!*",
                    parse_mode=ParseMode.MARKDOWN,
                    reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🚀 Сгенерировать ещё", callback_data="start_new")]])
                )
                break

            if flag in (2, 3):
                await ctx.bot.send_message(chat_id, f"❌ Ошибка генерации VEO: {msg or 'без сообщения'}")
                break

            await asyncio.sleep(POLL_INTERVAL_SECS)
    finally:
        if s.get("generation_id") == gen_id:
            s["generating"] = False
            s["generation_id"] = None


# ==========================
#   MJ (Midjourney-like)
# ==========================
def _mj_payload(prompt: str, aspect: str, style: str) -> Dict[str, Any]:
    # Подгоните поля под ваш фактический MJ API.
    # Здесь нейтральная схема с явными полями.
    return {
        "prompt": prompt,
        "aspectRatio": "9:16" if aspect == "9:16" else "16:9",
        "style": style,  # например: "relaxed"|"fast"|"turbo" — зависит от провайдера
    }

def mj_submit(prompt: str, aspect: str, style: str) -> Tuple[bool, Optional[str], str]:
    url = join_url(MJ_BASE_URL, MJ_TEXT2IMG_PATH)
    status, j = _post_json(url, _mj_payload(prompt, aspect, style), _headers_json(KIE_API_KEY))
    code = j.get("code", status)
    if status == 200 and code == 200:
        tid = _extract_task_id(j)
        if tid: return True, tid, "OK"
        return False, None, "Ответ без taskId."
    # более общий текст ошибки
    msg = j.get("msg") or j.get("message") or j.get("error") or f"HTTP {status}"
    return False, None, f"MJ ошибка: {msg}"

def mj_status(task_id: str) -> Tuple[bool, Optional[int], Optional[str], Optional[str]]:
    url = join_url(MJ_BASE_URL, MJ_STATUS_PATH)
    status, j = _get_json(url, {"taskId": task_id}, _headers_json(KIE_API_KEY))
    code = j.get("code", status)
    if status == 200 and code == 200:
        data = j.get("data") or {}
        # successFlag: 0 — в работе, 1 — готово, 2/3 — ошибки (держим ту же семантику)
        flag = data.get("successFlag")
        try: flag = int(flag)
        except Exception: flag = None

        # результат может быть как строкой URL, так и массивом
        url_field = data.get("resultUrls") or data.get("imageUrls") or data.get("urls")
        urls = _coerce_url_list(url_field)
        res = urls[0] if urls else None
        msg = j.get("msg") or j.get("message")
        return True, flag, msg, res
    msg = j.get("msg") or j.get("message") or j.get("error") or f"HTTP {status}"
    return False, None, msg, None

async def send_image(ctx: ContextTypes.DEFAULT_TYPE, chat_id: int, url: str) -> bool:
    # Сначала простая отправка по URL (Telegram сам скачает превью)
    try:
        await ctx.bot.send_photo(chat_id=chat_id, photo=url)
        return True
    except Exception as e:
        log.warning("Send photo by URL failed: %s", e)

    # Скачаем и отправим файлом
    tmp_path = None
    fname = "result.jpg"
    try:
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        ext = ".png" if "png" in (r.headers.get("Content-Type") or "") else ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
            for chunk in r.iter_content(chunk_size=128 * 1024):
                if chunk:
                    f.write(chunk)
            tmp_path = f.name

        with open(tmp_path, "rb") as f:
            await ctx.bot.send_photo(chat_id=chat_id, photo=InputFile(f, filename=os.path.basename(tmp_path)))
        return True
    except Exception as e:
        log.exception("Image send failed: %s", e)
        return False
    finally:
        if tmp_path:
            try: os.unlink(tmp_path)
            except Exception: pass

async def poll_mj_and_send(chat_id: int, task_id: str, gen_id: str, ctx: ContextTypes.DEFAULT_TYPE):
    s = S(ctx)
    s["generating"] = True
    s["generation_id"] = gen_id
    s["last_task_id"] = task_id

    start = time.time()
    try:
        while True:
            if s.get("generation_id") != gen_id:
                return

            ok, flag, msg, res_url = await asyncio.to_thread(mj_status, task_id)
            event("MJ_STATUS", task_id=task_id, flag=flag, url=bool(res_url))

            if not ok:
                await ctx.bot.send_message(chat_id, f"❌ Статус MJ: {msg or 'неизвестно'}")
                break

            if flag == 0:
                if (time.time() - start) > POLL_TIMEOUT_SECS:
                    await ctx.bot.send_message(chat_id, "⏳ Таймаут ожидания MJ.")
                    break
                await asyncio.sleep(POLL_INTERVAL_SECS)
                continue

            if flag == 1:
                if not res_url:
                    await ctx.bot.send_message(chat_id, "⚠️ Готово, но URL не найден.")
                    break
                sent = await send_image(ctx, chat_id, res_url)
                s["last_result_url"] = res_url if sent else None
                await ctx.bot.send_message(
                    chat_id,
                    "✅ *Готово!*",
                    parse_mode=ParseMode.MARKDOWN,
                    reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🎨 Сгенерировать ещё", callback_data="start_new")]])
                )
                break

            if flag in (2, 3):
                await ctx.bot.send_message(chat_id, f"❌ Ошибка генерации MJ: {msg or 'без сообщения'}")
                break

            await asyncio.sleep(POLL_INTERVAL_SECS)
    finally:
        if s.get("generation_id") == gen_id:
            s["generating"] = False
            s["generation_id"] = None


# ==========================
#   Handlers
# ==========================
async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = S(ctx); s.update({**DEFAULT_STATE})
    await update.message.reply_text(WELCOME, parse_mode=ParseMode.MARKDOWN, reply_markup=main_menu_kb())

async def health(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    parts = [
        f"PTB: `{getattr(_tg, '__version__', 'unknown')}`",
        f"KIE_BASE_URL: `{KIE_BASE_URL}`",
        f"KIE_UPLOAD_BASE: `{KIE_UPLOAD_BASE}`",
        f"MJ_BASE_URL: `{MJ_BASE_URL}`",
        f"KIE key: `{'set' if KIE_API_KEY else 'missing'}`",
        f"OPENAI key: `{'set' if OPENAI_API_KEY else 'missing'}`",
    ]
    await update.message.reply_text("🩺 *Health*\n" + "\n".join(parts), parse_mode=ParseMode.MARKDOWN)

async def error_handler(update: Optional[Update], context: ContextTypes.DEFAULT_TYPE):
    log.exception("Unhandled error: %s", context.error)
    try:
        if update and update.effective_chat:
            await context.bot.send_message(update.effective_chat.id, "⚠️ Системная ошибка. Попробуйте ещё раз.")
    except Exception:
        pass

async def show_card(update: Update, ctx: ContextTypes.DEFAULT_TYPE, what: str, edit_only_markup: bool = False):
    s = S(ctx)
    text = card_text_veo(s) if what == "veo" else card_text_mj(s)
    kb   = kb_card_veo(s) if what == "veo" else kb_card_mj(s)
    chat_id = update.effective_chat.id
    last_id = s.get("last_ui_msg_id")

    try:
        if last_id:
            if edit_only_markup:
                await ctx.bot.edit_message_reply_markup(chat_id, last_id, reply_markup=kb)
            else:
                await ctx.bot.edit_message_text(chat_id=chat_id, message_id=last_id,
                                                text=text, parse_mode=ParseMode.MARKDOWN,
                                                reply_markup=kb, disable_web_page_preview=True)
        else:
            m = await (update.callback_query.message.reply_text(text, parse_mode=ParseMode.MARKDOWN,
                                                                reply_markup=kb, disable_web_page_preview=True)
                       if update.callback_query else
                       update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN,
                                                 reply_markup=kb, disable_web_page_preview=True))
            s["last_ui_msg_id"] = m.message_id
    except Exception as e:
        log.warning("show_card edit failed: %s", e)
        try:
            m = await ctx.bot.send_message(chat_id, text=text, parse_mode=ParseMode.MARKDOWN,
                                           reply_markup=kb, disable_web_page_preview=True)
            s["last_ui_msg_id"] = m.message_id
        except Exception as e2:
            log.exception("show_card send failed: %s", e2)

async def on_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    data = (q.data or "").strip()
    await q.answer()
    s = S(ctx)

    if data == "back":
        s.update({**DEFAULT_STATE})
        await q.message.reply_text("Главное меню:", reply_markup=main_menu_kb()); return

    if data == "start_new":
        s.update({**DEFAULT_STATE})
        await q.message.reply_text("Выберите режим:", reply_markup=main_menu_kb()); return

    if data.startswith("mode:"):
        _, mode = data.split(":", 1)
        s["mode"] = mode
        if mode == "veo_text":
            s["aspect"] = s.get("aspect") or "16:9"
            s["model"]  = s.get("model")  or "veo3_fast"
            await q.message.reply_text("VEO: пришлите текст промпта (или используйте Промпт-мастер).")
            await show_card(update, ctx, "veo"); return
        if mode == "veo_photo":
            s["aspect"] = s.get("aspect") or "16:9"
            s["model"]  = s.get("model")  or "veo3_fast"
            await q.message.reply_text("VEO: пришлите фото (и при желании подпись-промпт). Я загружу в KIE Upload.")
            await show_card(update, ctx, "veo"); return
        if mode == "mj":
            s["mj_aspect"] = s.get("mj_aspect") or "16:9"
            await q.message.reply_text("MJ: пришлите текстовый промпт на английском (лучше).")
            await show_card(update, ctx, "mj"); return
        if mode == "prompt_master":
            await q.message.reply_text("Промпт-мастер: пришлите идею (1–2 фразы), верну EN кинопромпт."); return
        if mode == "chat":
            await q.message.reply_text("Чат: задайте вопрос."); return

    # Переключатели VEO/MJ
    if data.startswith("aspect:"):
        _, v = data.split(":", 1)
        s["aspect"] = "9:16" if v == "9:16" else "16:9"
        await show_card(update, ctx, "veo", edit_only_markup=True); return

    if data.startswith("model:"):
        _, v = data.split(":", 1)
        s["model"] = "veo3" if v == "veo3" else "veo3_fast"
        await show_card(update, ctx, "veo", edit_only_markup=True); return

    if data.startswith("mj_aspect:"):
        _, v = data.split(":", 1)
        s["mj_aspect"] = "9:16" if v == "9:16" else "16:9"
        await show_card(update, ctx, "mj", edit_only_markup=True); return

    if data == "mj:style":
        # Для простоты циклим relaxed→fast→turbo (если провайдер поддерживает)
        seq = ["relaxed", "fast", "turbo"]
        cur = s.get("mj_style") or "relaxed"
        s["mj_style"] = seq[(seq.index(cur) + 1) % len(seq)] if cur in seq else "relaxed"
        await show_card(update, ctx, "mj", edit_only_markup=True); return

    if data == "card:toggle_photo":
        if s.get("last_image_url"):
            s["last_image_url"] = None
            await q.message.reply_text("Фото-референс удалён."); await show_card(update, ctx, "veo")
        else:
            await q.message.reply_text("Пришлите фото (или публичный URL). Я загружу в KIE Upload.")
        return

    if data == "card:edit_prompt":
        await q.message.reply_text("Пришлите новый текст промпта."); return

    if data == "card:reset":
        keep = {"aspect": s.get("aspect"), "model": s.get("model"), "mj_aspect": s.get("mj_aspect")}
        s.update({**DEFAULT_STATE})
        s.update({k: v for k, v in keep.items() if v})
        await q.message.reply_text("Карточка очищена."); 
        await show_card(update, ctx, "veo" if (s.get("mode") or "").startswith("veo") else "mj"); 
        return

    if data == "card:generate_veo":
        if s.get("generating"):
            await q.message.reply_text("⏳ Генерация уже идёт."); return
        if not s.get("last_prompt"):
            await q.message.reply_text("Сначала укажите текст промпта."); return
        ok, task_id, msg = await asyncio.to_thread(
            veo_submit, s["last_prompt"].strip(), s.get("aspect", "16:9"),
            s.get("last_image_url"), s.get("model", "veo3_fast")
        )
        if not ok or not task_id:
            await q.message.reply_text(f"❌ Не удалось создать VEO-задачу: {msg}"); return
        gen_id = uuid.uuid4().hex[:12]
        S(ctx)["generating"] = True; S(ctx)["generation_id"] = gen_id; S(ctx)["last_task_id"] = task_id
        event("VEO_SUBMIT", chat=update.effective_chat.id, task_id=task_id, aspect=s.get("aspect"), model=s.get("model"))
        await q.message.reply_text(f"🚀 Задача отправлена (VEO). taskId={task_id}")
        await q.message.reply_text("⏳ Идёт рендеринг…")
        asyncio.create_task(poll_veo_and_send(update.effective_chat.id, task_id, gen_id, ctx)); return

    if data == "card:generate_mj":
        if s.get("generating"):
            await q.message.reply_text("⏳ Генерация уже идёт."); return
        if not s.get("last_prompt"):
            await q.message.reply_text("Сначала укажите текст промпта."); return
        ok, task_id, msg = await asyncio.to_thread(
            mj_submit, s["last_prompt"].strip(), s.get("mj_aspect", "16:9"), s.get("mj_style", "relaxed")
        )
        if not ok or not task_id:
            await q.message.reply_text(f"❌ Не удалось создать MJ-задачу: {msg}"); return
        gen_id = uuid.uuid4().hex[:12]
        S(ctx)["generating"] = True; S(ctx)["generation_id"] = gen_id; S(ctx)["last_task_id"] = task_id
        event("MJ_SUBMIT", chat=update.effective_chat.id, task_id=task_id, aspect=s.get("mj_aspect"))
        await q.message.reply_text(f"🎨 Задача отправлена (MJ). taskId={task_id}")
        await q.message.reply_text("⏳ Рисую…")
        asyncio.create_task(poll_mj_and_send(update.effective_chat.id, task_id, gen_id, ctx)); return

async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = S(ctx)
    text = (update.message.text or "").strip()

    # Если пришёл URL картинки — сразу делаем безопасный публичный URL (для VEO ref или MJ референсов, если понадобится)
    low = text.lower()
    if low.startswith(("http://", "https://")) and any(low.split("?")[0].endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp", ".heic")):
        pub = await ensure_public_url_from_tg_or_http(ctx, file_id=None, http_url=text)
        if not pub:
            await update.message.reply_text("⚠️ Не удалось подготовить изображение. Пришлите фото файлом.")
            return
        s["last_image_url"] = pub
        await update.message.reply_text("✅ Ссылка на изображение принята.")
        await show_card(update, ctx, "veo" if (s.get("mode") or "").startswith("veo") else "mj")
        return

    mode = s.get("mode")
    if mode == "prompt_master":
        prompt = await oai_prompt_master(text)
        if not prompt:
            await update.message.reply_text("⚠️ Prompt-Master недоступен.")
            return
        s["last_prompt"] = prompt
        await update.message.reply_text("🧠 Готово! Промпт добавлен в карточку.")
        await show_card(update, ctx, "veo")
        return

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

    # По умолчанию — сохраняем как промпт для выбранного режима
    s["last_prompt"] = text
    if (mode or "").startswith("veo") or not mode:
        await update.message.reply_text("🟦 *VEO — подготовка к рендеру*\nПроверь карточку и жми «Сгенерировать (VEO)».", parse_mode=ParseMode.MARKDOWN)
        await show_card(update, ctx, "veo")
    elif mode == "mj":
        await update.message.reply_text("🎨 *MJ — подготовка*\nПроверь карточку и жми «Сгенерировать (MJ)».", parse_mode=ParseMode.MARKDOWN)
        await show_card(update, ctx, "mj")

async def on_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = S(ctx)
    photos = update.message.photo
    if not photos: return
    ph = photos[-1]
    try:
        public_url = await ensure_public_url_from_tg_or_http(ctx, file_id=ph.file_id, http_url=None)
        if not public_url:
            await update.message.reply_text("⚠️ Не удалось подготовить фото. Попробуйте ещё раз.")
            return
        s["last_image_url"] = public_url
        await update.message.reply_text("🖼️ Фото принято (перезалил на публичный хост).")
        await show_card(update, ctx, "veo" if (s.get("mode") or "").startswith("veo") else "mj")
    except Exception as e:
        log.exception("Get photo failed: %s", e)
        await update.message.reply_text("⚠️ Не удалось обработать фото. Пришлите публичный URL картинкой.")

# ==========================
#   Entry
# ==========================
def main():
    if not TELEGRAM_TOKEN: raise RuntimeError("TELEGRAM_TOKEN is not set")
    if not KIE_BASE_URL:   raise RuntimeError("KIE_BASE_URL is not set")
    if not KIE_API_KEY:    raise RuntimeError("KIE_API_KEY is not set")

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).rate_limiter(AIORateLimiter()).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("health", health))
    app.add_handler(CallbackQueryHandler(on_callback))
    app.add_handler(MessageHandler(filters.PHOTO, on_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_error_handler(error_handler)

    log.info("Bot starting. PTB=%s | KIE=%s | UPLOAD=%s | MJ=%s | VEO_GEN=%s | VEO_STATUS=%s",
             getattr(_tg, "__version__", 'unknown'),
             KIE_BASE_URL, KIE_UPLOAD_BASE, MJ_BASE_URL, KIE_VEO_GEN_PATH, KIE_VEO_STATUS_PATH)

    # ВАЖНО: для режима long-polling не должен быть установлен вебхук
    # Проверь/сбрось:
    #   https://api.telegram.org/bot<token>/getWebhookInfo
    #   https://api.telegram.org/bot<token>/deleteWebhook?drop_pending_updates=true
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
