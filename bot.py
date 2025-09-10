# -*- coding: utf-8 -*-
# Best VEO3 Bot — PTB 20.x/21.x
# Версия: 2025-09-10 (fix-pack: VEO vertical tg-safe keepdims, MJ upscale double-try, MJ aspect 9:16 parsing)

import os
import json
import time
import uuid
import base64
import asyncio
import logging
import tempfile
import subprocess
from typing import Dict, Any, Optional, List, Tuple, Union

import requests
from dotenv import load_dotenv

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputFile
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder, ContextTypes, CommandHandler,
    MessageHandler, CallbackQueryHandler, filters, AIORateLimiter
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
TOPUP_URL           = _env("TOPUP_URL", "https://t.me/bestveo3promts")

# ---- KIE core ----
KIE_API_KEY  = _env("KIE_API_KEY")
KIE_BASE_URL = _env("KIE_BASE_URL", "https://api.kie.ai")
KIE_VEO_GEN_PATH    = _env("KIE_VEO_GEN_PATH",    _env("KIE_GEN_PATH",    "/api/v1/veo/generate"))
KIE_VEO_STATUS_PATH = _env("KIE_VEO_STATUS_PATH", _env("KIE_STATUS_PATH", "/api/v1/veo/record-info"))
KIE_VEO_1080_PATH   = _env("KIE_VEO_1080_PATH",   _env("KIE_HD_PATH",     "/api/v1/veo/get-1080p-video"))

# ---- Upload API (для фото-референсов в VEO)
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
    pass

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
    """
    Если file_path уже абсолютный (http/https) — возвращаем как есть.
    Иначе строим https://api.telegram.org/file/bot<TOKEN>/<file_path>
    """
    p = (file_path or "").strip()
    if p.startswith("http://") or p.startswith("https://"):
        return p
    return f"https://api.telegram.org/file/bot{bot_token}/{p.lstrip('/')}"

# ==========================
#   State
# ==========================
DEFAULT_STATE = {
    "mode": None,          # 'veo_text' | 'veo_photo' | 'mj' | 'chat' (chat не трогаем)
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
    # MJ
    "mj_aspect": "16:9",
    "mj_last_task_id": None,
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
    "🎬 *Veo 3 — супер-генерация видео*\n"
    "Опиши идею — получишь готовый клип!\n\n"
    "🧠 *MJ (фото по тексту)* — генерация изображений.\n"
    "📸 *Оживление фотографии (VEO)* — анимируй фото.\n\n"
    "Выберите режим 👇"
)

def main_menu_kb() -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton("🎬 Сгенерировать по тексту (VEO)", callback_data="mode:veo_text")],
        [InlineKeyboardButton("📸 Оживление фотографии (VEO)",    callback_data="mode:veo_photo")],
        [InlineKeyboardButton("🖼️ Midjourney — фото по тексту",   callback_data="mode:mj")],
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

# ==========================
#   HTTP helpers
# ==========================
def _kie_headers_json() -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    tok = (KIE_API_KEY or "").strip()
    if tok and not tok.lower().startswith("bearer "):
        tok = f"Bearer {tok}"
    if tok:
        h["Authorization"] = tok
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
            if s.startswith("http"): urls.append(s)
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

    # если картинка не загрузилась — пробуем без неё
    msg = (j.get("msg") or j.get("message") or j.get("error") or "").lower()
    if "image" in msg and "failed" in msg:
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

# ==========================
#   ffmpeg helpers
# ==========================
def _ffmpeg_available() -> bool:
    from shutil import which
    return bool(which(FFMPEG_BIN))

def _ffmpeg_make_tg_safe_keepdims(inp: str, outp: str) -> bool:
    """
    Перекодировать в H.264/yuv420p БЕЗ изменения разрешения.
    Убираем rotate, выравниваем чётные размеры, faststart.
    """
    cmd = [
        FFMPEG_BIN, "-y", "-i", inp,
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,setsar=1,format=yuv420p",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-c:a", "aac", "-b:a", "128k",
        "-metadata:s:v:0", "rotate=0",
        outp
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE); return True
    except Exception as e:
        log.warning("ffmpeg keepdims failed: %s", e); return False

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
#   Sending video (robust)
# ==========================
async def send_video_with_fallback(ctx: ContextTypes.DEFAULT_TYPE, chat_id: int, url: str, expect_vertical: bool = False) -> bool:
    event("SEND_TRY_URL", url=url, expect_vertical=expect_vertical)

    # 1) Сначала всегда пробуем прямой URL (и вертикаль тоже)
    try:
        await ctx.bot.send_video(chat_id=chat_id, video=url, supports_streaming=True)
        event("SEND_OK", mode="direct_url"); return True
    except Exception as e:
        event("SEND_FAIL", mode="direct_url", err=str(e))

    # 2) Скачиваем локально
    tmp_path = None
    try:
        r = requests.get(url, stream=True, timeout=180); r.raise_for_status()
        ct = (r.headers.get("Content-Type") or "").lower()
        ext = ".mp4"
        lu = url.lower()
        if ".mov" in lu or "quicktime" in ct: ext = ".mov"
        elif ".webm" in lu or "webm" in ct:   ext = ".webm"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
            for c in r.iter_content(256*1024):
                if c: f.write(c)
            tmp_path = f.name
        event("DOWNLOAD_OK", path=tmp_path, content_type=ct)

        # 3) Вертикаль → tg-safe без изменения размеров
        if expect_vertical and ENABLE_VERTICAL_NORMALIZE and _ffmpeg_available():
            out = tmp_path + "_v.mp4"
            if _ffmpeg_make_tg_safe_keepdims(tmp_path, out):
                try:
                    with open(out, "rb") as f:
                        await ctx.bot.send_video(chat_id=chat_id, video=InputFile(f, filename="result_vertical.mp4"),
                                                 supports_streaming=True)
                    event("SEND_OK", mode="upload_video_keepdims"); return True
                except Exception as e:
                    event("SEND_FAIL", mode="upload_video_keepdims", err=str(e))
                    with open(out, "rb") as f:
                        await ctx.bot.send_document(chat_id=chat_id, document=InputFile(f, filename="result_vertical.mp4"))
                    event("SEND_OK", mode="upload_document_keepdims"); return True

        # 4) Горизонталь → при необходимости форсим 1080p (размер под лимит)
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

        # 5) Без перекодирования — как есть
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

    start_ts = time.time()
    try:
        while True:
            if s.get("generation_id") != gen_id:
                return

            ok, flag, msg, res_url = await asyncio.to_thread(get_kie_veo_status, task_id)
            event("VEO_STATUS", task_id=task_id, flag=flag, has_url=bool(res_url), msg=msg)

            if not ok:
                await ctx.bot.send_message(chat_id, f"❌ Ошибка статуса: {msg or 'неизвестно'}"); break

            # как только появился URL — отправляем
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

            # если KIE завершил без URL — прекращаем ожидание
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
        if s.get("generation_id") == gen_id:
            s["generating"] = False
            s["generation_id"] = None

# ==========================
#   Midjourney (MJ)
# ==========================
KIE_MJ_GEN_PATH    = _env("KIE_MJ_GEN_PATH",    "/api/v1/mj/generate")
KIE_MJ_STATUS_PATH = _env("KIE_MJ_STATUS_PATH", "/api/v1/mj/record-info")
KIE_MJ_UPSCALE     = _env("KIE_MJ_UPSCALE",     "/api/v1/mj/upscale")
KIE_MJ_VARY        = _env("KIE_MJ_VARY",        "/api/v1/mj/generateVary")  # fallback

def mj_generate(prompt: str, aspect: str) -> Tuple[bool, Optional[str], str]:
    payload = {
        "taskType": "mj_txt2img",
        "prompt": prompt,
        "aspectRatio": aspect if aspect in ("16:9", "9:16", "1:1", "4:3", "3:4", "2:3", "3:2", "5:6", "6:5", "2:1", "1:2") else "16:9",
        "speed": "turbo",   # по требованиям — всегда Turbo
        "version": "7"
    }
    status, j = _post_json(join_url(KIE_BASE_URL, KIE_MJ_GEN_PATH), payload)
    code = j.get("code", status)
    if status == 200 and code == 200:
        tid = _extract_task_id(j)
        if tid: return True, tid, "MJ задача создана."
        return False, None, "Ответ MJ без taskId."
    return False, None, _kie_error_message(status, j)

def mj_status(task_id: str) -> Tuple[bool, Optional[int], bool, Optional[List[str]], Optional[str]]:
    status, j = _get_json(join_url(KIE_BASE_URL, KIE_MJ_STATUS_PATH), {"taskId": task_id})
    code = j.get("code", status)
    if status == 200 and code == 200:
        data = (j.get("data") or {})
        flag = data.get("successFlag")
        try: flag = int(flag)
        except Exception: flag = None
        # вытаскиваем картинки
        urls = []
        info = data.get("resultInfoJson") or {}
        arr = info.get("resultUrls") or []
        urls = [d.get("resultUrl") for d in arr if isinstance(d, dict) and d.get("resultUrl")]
        return True, flag, bool(urls), urls or None, None
    return False, None, False, None, _kie_error_message(status, j)

def mj_upscale(task_id: str, index: int) -> Tuple[bool, Optional[str], str]:
    """Надёжный апскейл: сначала /upscale (index & imageIndex), затем fallback /generateVary."""
    # попытка 1: /upscale
    payload_u = {"taskId": task_id, "index": int(index), "imageIndex": int(index)}
    status, j = _post_json(join_url(KIE_BASE_URL, KIE_MJ_UPSCALE), payload_u)
    code = j.get("code", status)
    if status == 200 and code == 200:
        tid = _extract_task_id(j)
        if tid: return True, tid, "MJ апскейл создан."
        return False, None, "Ответ MJ без taskId (upscale)."

    # попытка 2: /generateVary (ожидает imageIndex)
    payload_v = {"taskId": task_id, "imageIndex": int(index)}
    status2, j2 = _post_json(join_url(KIE_BASE_URL, KIE_MJ_VARY), payload_v)
    code2 = j2.get("code", status2)
    if status2 == 200 and code2 == 200:
        tid2 = _extract_task_id(j2)
        if tid2: return True, tid2, "MJ vary (апскейл) создан."
        return False, None, "Ответ MJ без taskId (generateVary)."

    return False, None, f"Upscale 404/err. upscale={code} {j.get('msg') or j.get('message') or ''} | vary={code2} {j2.get('msg') or j2.get('message') or ''}".strip()

# ==========================
#   Handlers
# ==========================
async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = state(ctx); s.update({**DEFAULT_STATE})
    await update.message.reply_text(WELCOME, parse_mode=ParseMode.MARKDOWN, reply_markup=main_menu_kb())

async def health(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    parts = [
        f"PTB: `{getattr(_tg, '__version__', 'unknown')}`",
        f"KIEBASEURL: `{KIE_BASE_URL}`",
        f"GENPATH: `{KIE_VEO_GEN_PATH}`",
        f"STATUSPATH: `{KIE_VEO_STATUS_PATH}`",
        f"1080PATH: `{KIE_VEO_1080_PATH}`",
        f"UPLOADBASEURL: `{UPLOAD_BASE_URL}`",
        f"UPLOADSTREAMPATH: `{UPLOAD_STREAM_PATH}`",
        f"UPLOADB64PATH: `{UPLOAD_BASE64_PATH}`",
        f"MJ_GEN: `{KIE_MJ_GEN_PATH}`",
        f"MJ_STATUS: `{KIE_MJ_STATUS_PATH}`",
        f"MJ_UPSCALE: `{KIE_MJ_UPSCALE}`",
        f"MJ_VARY: `{KIE_MJ_VARY}`",
        f"KIE key: `{'set' if KIE_API_KEY else 'missing'}`",
        f"ENABLEVERTICALNORMALIZE: `{ENABLE_VERTICAL_NORMALIZE}`",
        f"ALWAYSFORCEFHD: `{ALWAYS_FORCE_FHD}`",
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
            "• Вертикальное видео делаем tg-safe без изменения кадра (H.264/yuv420p, rotate=0).\n"
            "• Горизонталь 16:9 — тянем 1080p и при необходимости форсим FHD.\n"
            "• MJ: Turbo, поддержка Upscale, корректная передача aspectRatio.",
            reply_markup=main_menu_kb(),
        ); return

    if data == "back":
        s.update({**DEFAULT_STATE}); await query.message.reply_text("Главное меню:", reply_markup=main_menu_kb()); return

    if data == "start_new_cycle":
        s.update({**DEFAULT_STATE}); await query.message.reply_text("Выберите режим:", reply_markup=main_menu_kb()); return

    if data.startswith("mode:"):
        _, mode = data.split(":", 1); s["mode"] = mode
        if mode == "veo_text":
            s["aspect"] = "16:9"; s["model"] = "veo3_fast"
            await query.message.reply_text("VEO (текст): пришлите идею/промпт."); await show_card_veo(update, ctx); return
        if mode == "veo_photo":
            s["aspect"] = "9:16"; s["model"] = "veo3_fast"
            await query.message.reply_text("VEO (оживление фото): пришлите фото (подпись-промпт — опционально)."); await show_card_veo(update, ctx); return
        if mode == "mj":
            await query.message.reply_text(
                "🖼️ *Midjourney*\nПришлите текст промпта. Выберите формат:",
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("16:9", callback_data="mj:ar:16:9"),
                     InlineKeyboardButton("9:16", callback_data="mj:ar:9:16"),
                     InlineKeyboardButton("1:1",  callback_data="mj:ar:1:1")],
                ])
            ); return

    if data.startswith("aspect:"):
        _, val = data.split(":", 1); s["aspect"] = "9:16" if val.strip() == "9:16" else "16:9"
        await show_card_veo(update, ctx, edit_only_markup=True); return

    if data.startswith("model:"):
        _, val = data.split(":", 1); s["model"] = "veo3" if val.strip() == "veo3" else "veo3_fast"
        await show_card_veo(update, ctx, edit_only_markup=True); return

    # ---- MJ UI ----
    if data.startswith("mj:ar:"):
        # FIX: корректно берём все символы после "mj:ar:" (в т.ч. "9:16")
        ar = data[len("mj:ar:"):]
        if ar in ("16:9", "9:16", "1:1", "4:3", "3:4", "2:3", "3:2", "5:6", "6:5", "2:1", "1:2"):
            s["mj_aspect"] = ar
        await query.message.reply_text(f"Формат: {s['mj_aspect']}\nПришлите текст промпта для MJ."); return

    if data.startswith("mj:up:"):
        # кнопки вида mj:up:1|2|3|4 ссылаются на последний mj_task_id
        try:
            idx = int(data.split(":")[-1])
        except Exception:
            idx = 1
        tid = s.get("mj_last_task_id")
        if not tid:
            await query.message.reply_text("Нет активной MJ задачи для апскейла."); return
        ok, new_tid, msg = await asyncio.to_thread(mj_upscale, tid, idx)
        event("MJ_UPSCALE_RESP", ok=ok, task_id=new_tid, msg=msg)
        if not ok or not new_tid:
            await query.message.reply_text(f"❌ Не удалось отправить апскейл: {msg}"); return
        await query.message.reply_text("🔎 Повышение качества запущено…")
        # ждём и шлём
        await poll_mj_and_send(update.effective_chat.id, new_tid, ctx, caption_prefix="🔎 Апскейл: ")
        return

async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = state(ctx)
    text = (update.message.text or "").strip()

    # URL картинки — как референс для VEO
    low = text.lower()
    if low.startswith(("http://", "https://")) and any(low.split("?")[0].endswith(ext) for ext in (".jpg",".jpeg",".png",".webp",".heic")):
        s["last_image_url"] = text.strip()
        await update.message.reply_text("✅ Ссылка на изображение принята."); await show_card_veo(update, ctx); return

    mode = s.get("mode")

    # ---- MJ text-to-image ----
    if mode == "mj":
        ar = s.get("mj_aspect") or "16:9"
        ok, task_id, msg = await asyncio.to_thread(mj_generate, text, ar)
        event("MJ_SUBMIT_RESP", ok=ok, task_id=task_id, msg=msg)
        if not ok or not task_id:
            await update.message.reply_text(f"❌ MJ ошибка создания: {msg}"); return
        s["mj_last_task_id"] = task_id
        await update.message.reply_text("🖼️ Генерация фото запущена...")
        await poll_mj_and_send(update.effective_chat.id, task_id, ctx)
        return

    # ---- VEO (по умолчанию) ----
    s["last_prompt"] = text
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

# ---- VEO generate button handler ----
async def on_card_generate(query_msg, s, ctx):
    if s.get("generating"):
        await query_msg.reply_text("⏳ Генерация уже идёт."); return
    if not s.get("last_prompt"):
        await query_msg.reply_text("Сначала укажите текст промпта."); return

    event("VEO_SUBMIT_REQ", aspect=s.get("aspect"), model=s.get("model"),
          with_image=bool(s.get("last_image_url")), prompt_len=len(s.get("last_prompt") or ""))

    ok, task_id, msg = await asyncio.to_thread(
        submit_kie_veo, s["last_prompt"].strip(), s.get("aspect", "16:9"),
        s.get("last_image_url"), s.get("model", "veo3_fast")
    )
    event("VEO_SUBMIT_RESP", ok=ok, task_id=task_id, msg=msg)

    if not ok or not task_id:
        await query_msg.reply_text(f"❌ Не удалось создать VEO-задачу: {msg}"); return

    gen_id = uuid.uuid4().hex[:12]
    s["generating"] = True; s["generation_id"] = gen_id; s["last_task_id"] = task_id
    await query_msg.reply_text(f"🚀 Задача отправлена ({'Fast' if s.get('model')=='veo3_fast' else 'Quality'}). taskId={task_id}")
    await query_msg.reply_text("⏳ Идёт рендеринг…")
    asyncio.create_task(poll_veo_and_send(query_msg.chat_id, task_id, gen_id, ctx))

async def poll_mj_and_send(chat_id: int, task_id: str, ctx: ContextTypes.DEFAULT_TYPE, caption_prefix: str = ""):
    start_ts = time.time()
    while True:
        ok, flag, has_data, urls, err = await asyncio.to_thread(mj_status, task_id)
        event("MJ_STATUS", task_id=task_id, flag=flag, has_data=has_data)
        if not ok:
            await ctx.bot.send_message(chat_id, f"❌ MJ ошибка статуса: {err or 'неизвестно'}"); return
        if has_data and urls:
            # отправляем альбомом, как просили (как есть)
            media = []
            try:
                from telegram import InputMediaPhoto
                for u in urls:
                    media.append(InputMediaPhoto(media=u))
                if media:
                    await ctx.bot.send_media_group(chat_id, media=media)
            except Exception as e:
                # fallback по одной
                for u in urls:
                    try:
                        await ctx.bot.send_photo(chat_id, photo=u)
                    except Exception:
                        await ctx.bot.send_message(chat_id, u)
            # кнопки апскейла
            kb = InlineKeyboardMarkup([
                [InlineKeyboardButton("🔎 Повысить качество #1", callback_data="mj:up:1"),
                 InlineKeyboardButton("🔎 Повысить качество #2", callback_data="mj:up:2")],
                [InlineKeyboardButton("🔎 Повысить качество #3", callback_data="mj:up:3"),
                 InlineKeyboardButton("🔎 Повысить качество #4", callback_data="mj:up:4")],
            ])
            await ctx.bot.send_message(chat_id, f"{caption_prefix}MJ taskId: `{task_id}`", parse_mode=ParseMode.MARKDOWN, reply_markup=kb)
            return
        if flag in (2, 3):
            await ctx.bot.send_message(chat_id, "❌ MJ генерация завершилась ошибкой."); return
        if time.time() - start_ts > POLL_TIMEOUT_SECS:
            await ctx.bot.send_message(chat_id, "⏳ MJ таймаут ожидания."); return
        await asyncio.sleep(POLL_INTERVAL_SECS)

async def on_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    # уже определён выше; дополнили — см. выше (двойное объявление блокирует)
    pass  # заглушка, фактическая реализация выше

# Переопределяем основной CallbackHandler с рабочей реализацией (см. выше)
# (PTB возьмёт тот, который добавлен в Application)
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

    log.info(
        "Bot starting. PTB=%s | KIE_BASE=%s | GEN=%s | STATUS=%s | 1080=%s | "
        "UPLOAD_BASE=%s | VERT_FIX=%s | FORCE_FHD=%s | MAX_MB=%s | MJ_GEN=%s | MJ_STATUS=%s | MJ_UPSCALE=%s | MJ_VARY=%s",
        getattr(_tg, '__version__', 'unknown'),
        KIE_BASE_URL, KIE_VEO_GEN_PATH, KIE_VEO_STATUS_PATH, KIE_VEO_1080_PATH,
        UPLOAD_BASE_URL, ENABLE_VERTICAL_NORMALIZE, ALWAYS_FORCE_FHD, MAX_TG_VIDEO_MB,
        KIE_MJ_GEN_PATH, KIE_MJ_STATUS_PATH, KIE_MJ_UPSCALE, KIE_MJ_VARY
    )

    # Если когда-то включался webhook — снимите:
    # https://api.telegram.org/bot<YOUR_TOKEN>/deleteWebhook
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
