# -*- coding: utf-8 -*-
# Best VEO3 Bot — PTB 20.7
# Версия: 2025-09-09 (NO-MJ, vertical video fix, robust URL extraction)

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
KIE_VEO_GEN_PATH = os.getenv("KIE_GEN_PATH", "/api/v1/veo/generate")
KIE_VEO_STATUS_PATH = os.getenv("KIE_STATUS_PATH", "/api/v1/veo/record-info")

# ---- KIE official upload host (для imageUrls референсов) ----
KIE_UPLOAD_BASE = os.getenv("KIE_UPLOAD_BASE", "https://kieai.redpandaai.co").strip()
KIE_UPLOAD_DIR  = os.getenv("KIE_UPLOAD_DIR",  "images/user-uploads").strip()
KIE_STREAM_UPLOAD_PATH = "/api/file-stream-upload"   # multipart
KIE_URL_UPLOAD_PATH    = "/api/file-url-upload"      # JSON

PROMPTS_CHANNEL_URL = os.getenv("PROMPTS_CHANNEL_URL", "https://t.me/bestveo3promts").strip()
TOPUP_URL = os.getenv("TOPUP_URL", "https://t.me/bestveo3promts").strip()

POLL_INTERVAL_SECS = int(os.getenv("POLL_INTERVAL_SECS", "6"))
POLL_TIMEOUT_SECS  = int(os.getenv("POLL_TIMEOUT_SECS",  str(20 * 60)))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("veo3-bot")

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

def _nz(s: Optional[str]) -> Optional[str]:
    if s is None: return None
    s2 = s.strip()
    return s2 if s2 else None

def event(tag: str, **kw):
    log.info("EVT %s | %s", tag, json.dumps(kw, ensure_ascii=False))


# ==========================
#   Public image rehost
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
        log.exception("Telegraph upload failed: %s", e)
    return None

def _kie_headers_upload() -> Dict[str, str]:
    token = KIE_API_KEY
    if token and not token.lower().startswith("bearer "):
        token = f"Bearer {token}"
    return {"Authorization": token or ""}

def _kie_upload_bytes(data: bytes, filename: str, mime: str = "image/jpeg") -> Optional[str]:
    url = join_url(KIE_UPLOAD_BASE, KIE_STREAM_UPLOAD_PATH)
    files = {"file": (filename, data, mime)}
    form  = {"uploadPath": KIE_UPLOAD_DIR, "fileName": filename}
    try:
        r = requests.post(url, headers=_kie_headers_upload(), files=files, data=form, timeout=120)
        r.raise_for_status()
        j = r.json()
        if j.get("success") and j.get("code") == 200:
            d = j.get("data") or {}
            return (d.get("downloadUrl") or d.get("fileUrl") or "").strip() or None
    except Exception as e:
        log.exception("KIE stream upload failed: %s", e)
    return None

def _kie_upload_from_url(file_url: str, filename: Optional[str] = None) -> Optional[str]:
    url = join_url(KIE_UPLOAD_BASE, KIE_URL_UPLOAD_PATH)
    payload = {"fileUrl": file_url, "uploadPath": KIE_UPLOAD_DIR}
    if filename: payload["fileName"] = filename
    try:
        r = requests.post(url, json=payload, headers={**_kie_headers_upload(), "Content-Type": "application/json"}, timeout=60)
        r.raise_for_status()
        j = r.json()
        if j.get("success") and j.get("code") == 200:
            d = j.get("data") or {}
            return (d.get("downloadUrl") or d.get("fileUrl") or "").strip() or None
    except Exception as e:
        log.exception("KIE url upload failed: %s", e)
    return None

async def ensure_public_url_from_tg_or_http(ctx: ContextTypes.DEFAULT_TYPE, *, file_id: Optional[str], http_url: Optional[str]) -> Optional[str]:
    try:
        if file_id:
            tg_file = await ctx.bot.get_file(file_id)
            raw = bytes(await tg_file.download_as_bytearray())
            name = os.path.basename(tg_file.file_path or "photo.jpg")
            # быстрый вариант
            tele = _upload_bytes_to_telegraph(raw, name)
            if tele: return tele
            # надёжный — через KIE upload
            ext = os.path.splitext(name)[1].lower()
            mime = "image/jpeg" if ext in (".jpg", ".jpeg", "") else ("image/png" if ext==".png" else "image/webp")
            return _kie_upload_bytes(raw, name if ext else (name + ".jpg"), mime=mime)

        if http_url:
            # сначала попробуем KIE url-upload
            u_name = os.path.basename(http_url.split("?")[0]) or "image.jpg"
            url_up = _kie_upload_from_url(http_url, filename=u_name)
            if url_up: return url_up
            # скачиваем и грузим потоково
            r = requests.get(http_url, timeout=60); r.raise_for_status()
            raw = r.content
            name = u_name if os.path.splitext(u_name)[1] else (u_name + ".jpg")
            return _kie_upload_bytes(raw, name, mime="image/jpeg")
    except Exception as e:
        log.exception("ensure_public_url_from_tg_or_http failed: %s", e)
    return None


# ==========================
#   State (NO MJ)
# ==========================
DEFAULT_STATE = {
    "mode": None,          # 'veo_text' | 'veo_photo' | 'prompt_master' | 'chat'
    # VEO
    "aspect": None,        # '16:9' | '9:16'
    "model": None,         # 'veo3_fast' | 'veo3'
    "last_prompt": None,
    "last_image_url": None,
    "generating": False,
    "generation_id": None,
    "last_ui_msg_id": None,
    "last_task_id": None,
    "last_result_url": None,
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
    "🎬 VEO — опиши идею и получи клип. Поддержка 16:9 и 9:16.\n"
    "🧠 Prompt-Master/Chat — по желанию.\n"
    f"• Идеи: {PROMPTS_CHANNEL_URL}\n"
    "Выберите режим:"
)

def main_menu_kb() -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton("🎬 Сгенерировать по тексту (VEO)", callback_data="mode:veo_text")],
        [InlineKeyboardButton("🖼️ Сгенерировать по фото (VEO)",  callback_data="mode:veo_photo")],
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
        return [InlineKeyboardButton("⚡ Fast",    callback_data="model:veo3_fast"),
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
    return "\n".join(lines)

def card_keyboard_veo(s: Dict[str, Any]) -> InlineKeyboardMarkup:
    rows: List[List[InlineKeyboardButton]] = []
    rows.append([InlineKeyboardButton("🖼️ Добавить/Удалить фото", callback_data="card:toggle_photo"),
                 InlineKeyboardButton("✍️ Изменить промпт",       callback_data="card:edit_prompt")])
    rows.append(aspect_row(s.get("aspect") or "16:9"))
    rows.append(model_row(s.get("model") or "veo3_fast"))
    rows.append([InlineKeyboardButton("🚀 Сгенерировать", callback_data="card:generate")])
    rows.append([InlineKeyboardButton("🔁 Начать заново", callback_data="card:reset"),
                 InlineKeyboardButton("⬅️ Назад",             callback_data="back")])
    rows.append([InlineKeyboardButton("💳 Пополнить баланс", url=TOPUP_URL)])
    return InlineKeyboardMarkup(rows)


# ==========================
#   Prompt-Master / Chat (опционально)
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
#   HTTP helpers (KIE VEO)
# ==========================
def _kie_headers_json() -> Dict[str, str]:
    token = KIE_API_KEY
    if token and not token.lower().startswith("bearer "):
        token = f"Bearer {token}"
    return {"Authorization": token or "", "Content-Type": "application/json"}

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

# ---- ключевой блок: корректно достаём ссылку из статус-ответа ----
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
    # 1) приоритетно — originUrls (для 9:16), затем resultUrls
    for key in ("originUrls", "resultUrls"):
        urls = _coerce_url_list(data.get(key))
        if urls: return urls[0]

    # 2) иногда внутри info/response/resultInfoJson
    for container in ("info", "response", "resultInfoJson"):
        v = data.get(container)
        if isinstance(v, dict):
            for key in ("originUrls", "resultUrls", "videoUrls"):
                urls = _coerce_url_list(v.get(key))
                if urls: return urls[0]

    # 3) глубокий поиск любой http*.mp4/mov/webm
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

# ---------- VEO payload
def _build_payload_for_veo(prompt: str, aspect: str, image_url: Optional[str], model_key: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "prompt": prompt,
        "aspectRatio": "9:16" if aspect == "9:16" else "16:9",
        "model": "veo3" if model_key == "veo3" else "veo3_fast",
        "enableFallback": aspect == "16:9",  # разрешаем fallback только для 16:9, как в доке
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
        if task_id: return True, task_id, "Задача создана."
        return False, None, "Ответ KIE без taskId."
    return False, None, _kie_error_message(status, j)

def get_kie_veo_status(task_id: str) -> Tuple[bool, Optional[int], Optional[str], Optional[str]]:
    url = join_url(KIE_BASE_URL, KIE_VEO_STATUS_PATH)
    status, j = _get_json(url, {"taskId": task_id})
    code = j.get("code", status)
    if status == 200 and code == 200:
        data = (j.get("data") or {})
        flag = data.get("successFlag")
        try: flag = int(flag)
        except Exception: flag = None
        msg = j.get("msg") or j.get("message")
        return True, flag, msg, _extract_result_url(data)
    return False, None, _kie_error_message(status, j), None


# ==========================
#   Sending video (robust)
# ==========================
async def send_video_with_fallback(ctx: ContextTypes.DEFAULT_TYPE, chat_id: int, url: str) -> bool:
    # 1) пробуем стримить по URL
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

        # 2a) как видео
        try:
            with open(tmp_path, "rb") as f:
                await ctx.bot.send_video(chat_id=chat_id, video=InputFile(f, filename=fname), supports_streaming=True)
            return True
        except Exception as e:
            log.warning("Send as video failed (codec/container). %s", e)

        # 2b) как документ (гарантированно доезжает)
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
    try:
        while True:
            if s.get("generation_id") != gen_id:
                return

            ok, flag, msg, res_url = await asyncio.to_thread(get_kie_veo_status, task_id)
            event("VEO_STATUS", task_id=task_id, flag=flag, url=bool(res_url))

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
        try: await ctx.bot.send_message(chat_id, "❌ Внутренняя ошибка при опросе VEO.")
        except Exception: pass
    finally:
        if s.get("generation_id") == gen_id:
            s["generating"] = False
            s["generation_id"] = None


# ==========================
#   Handlers
# ==========================
def card_keyboard(s): return card_keyboard_veo(s)

async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = state(ctx); s.update({**DEFAULT_STATE})
    await update.message.reply_text(WELCOME, parse_mode=ParseMode.MARKDOWN, reply_markup=main_menu_kb())

async def health(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    parts = [
        f"PTB: `{getattr(_tg, '__version__', 'unknown')}`",
        f"KIE_BASE_URL: `{KIE_BASE_URL}`",
        f"KIE_UPLOAD_BASE: `{KIE_UPLOAD_BASE}`",
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
                    chat_id=chat_id, message_id=last_id, text=text,
                    parse_mode=ParseMode.MARKDOWN, reply_markup=kb, disable_web_page_preview=True
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

async def on_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    data = (query.data or "").strip()
    await query.answer()

    s = state(ctx)

    if data == "faq":
        await query.message.reply_text(
            "FAQ:\n"
            "• Поддержка VEO 16:9 и 9:16.\n"
            "• Если Telegram не принимает контейнер — отправлю как документ.\n",
            reply_markup=main_menu_kb(),
        ); return

    if data == "back":
        s.update({**DEFAULT_STATE})
        await query.message.reply_text("Главное меню:", reply_markup=main_menu_kb()); return

    if data == "start_new_cycle":
        s.update({**DEFAULT_STATE})
        await query.message.reply_text("Выберите режим:", reply_markup=main_menu_kb()); return

    if data.startswith("mode:"):
        _, mode = data.split(":", 1); s["mode"] = mode
        if mode in ("veo_text", "veo_photo"):
            s["aspect"] = "16:9"; s["model"] = "veo3_fast"
            await query.message.reply_text(
                "VEO: пришлите идею/промпт"
                if mode == "veo_text"
                else "VEO: пришлите фото (и при желании — подпись-промпт)."
            )
            await show_card_veo(update, ctx); return
        if mode == "prompt_master":
            await query.message.reply_text("Промпт-мастер: пришлите идею (1–2 фразы)."); return
        if mode == "chat":
            await query.message.reply_text("Обычный чат: напишите вопрос."); return

    if data.startswith("aspect:"):
        _, val = data.split(":", 1)
        s["aspect"] = "9:16" if val.strip() == "9:16" else "16:9"
        await show_card_veo(update, ctx, edit_only_markup=True); return

    if data.startswith("model:"):
        _, val = data.split(":", 1)
        s["model"] = "veo3" if val.strip() == "veo3" else "veo3_fast"
        await show_card_veo(update, ctx, edit_only_markup=True); return

    if data == "card:toggle_photo":
        if s.get("last_image_url"):
            s["last_image_url"] = None
            await query.message.reply_text("Фото-референс удалён."); await show_card_veo(update, ctx)
        else:
            await query.message.reply_text("Пришлите фото. Я сделаю безопасную публичную ссылку (KIE upload) или пришлите уже публичный URL.")
        return

    if data == "card:edit_prompt":
        await query.message.reply_text("Пришлите новый текст промпта."); return

    if data == "card:reset":
        keep_aspect = s.get("aspect") or "16:9"
        keep_model  = s.get("model") or "veo3_fast"
        s.update({**DEFAULT_STATE}); s["aspect"] = keep_aspect; s["model"] = keep_model
        await query.message.reply_text("Карточка очищена."); await show_card_veo(update, ctx); return

    if data == "card:generate":
        if s.get("generating"):
            await query.message.reply_text("⏳ Генерация уже идёт. Дождитесь завершения."); return
        if not s.get("last_prompt"):
            await query.message.reply_text("Сначала укажите текст промпта."); return

        ok, task_id, msg = await asyncio.to_thread(
            submit_kie_veo, s["last_prompt"].strip(), s.get("aspect", "16:9"),
            s.get("last_image_url"), s.get("model", "veo3_fast")
        )
        if not ok or not task_id:
            await query.message.reply_text(f"❌ Не удалось создать VEO-задачу: {msg}"); return

        gen_id = uuid.uuid4().hex[:12]
        s["generating"] = True; s["generation_id"] = gen_id; s["last_task_id"] = task_id
        event("VEO_SUBMIT", chat=update.effective_chat.id, task_id=task_id, model=s.get("model"), aspect=s.get("aspect"))
        await query.message.reply_text(f"🚀 Задача отправлена ({'Fast' if s.get('model')=='veo3_fast' else 'Quality'}). taskId={task_id}")
        await query.message.reply_text("⏳ Идёт рендеринг…")
        asyncio.create_task(poll_veo_and_send(update.effective_chat.id, task_id, gen_id, ctx)); return


async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = state(ctx)
    text = (update.message.text or "").strip()

    # Если это URL картинки — подготовим публичную ссылку для imageUrls
    low = text.lower()
    if low.startswith(("http://", "https://")) and any(low.split("?")[0].endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp", ".heic")):
        pub = await ensure_public_url_from_tg_or_http(ctx, file_id=None, http_url=text)
        if not pub:
            await update.message.reply_text("⚠️ Не удалось подготовить изображение. Пришлите фото файлом."); return
        s["last_image_url"] = pub
        await update.message.reply_text("✅ Ссылка на изображение принята.")
        await show_card_veo(update, ctx); return

    mode = s.get("mode")
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

    # По умолчанию — VEO промпт
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
        public_url = await ensure_public_url_from_tg_or_http(ctx, file_id=ph.file_id, http_url=None)
        if not public_url:
            await update.message.reply_text("⚠️ Не удалось подготовить фото. Попробуйте ещё раз."); return
        s["last_image_url"] = public_url
        await update.message.reply_text("🖼️ Фото принято как референс.")
        await show_card_veo(update, ctx)
    except Exception as e:
        log.exception("Get photo failed: %s", e)
        await update.message.reply_text("⚠️ Не удалось обработать фото. Пришлите публичный URL картинки текстом.")


# ==========================
#   Entry
# ==========================
def main():
    if not TELEGRAM_TOKEN: raise RuntimeError("TELEGRAM_TOKEN is not set")
    if not KIE_BASE_URL:   raise RuntimeError("KIE_BASE_URL is not set")
    if not KIE_API_KEY:    raise RuntimeError("KIE_API_KEY is not set")

    app = (ApplicationBuilder().token(TELEGRAM_TOKEN).rate_limiter(AIORateLimiter()).build())

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("health", health))
    app.add_handler(CallbackQueryHandler(on_callback))
    app.add_handler(MessageHandler(filters.PHOTO, on_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_error_handler(error_handler)

    log.info("Bot starting. PTB=20.7 | KIE_BASE=%s | KIE_UPLOAD_BASE=%s | VEO_GEN=%s | VEO_STATUS=%s",
             KIE_BASE_URL, KIE_UPLOAD_BASE, KIE_VEO_GEN_PATH, KIE_VEO_STATUS_PATH)

    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    # если когда-то был webhook — снимите:
    # https://api.telegram.org/bot<YOUR_TOKEN>/deleteWebhook
    main()
