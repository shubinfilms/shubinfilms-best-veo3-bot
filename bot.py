# -*- coding: utf-8 -*-
"""Best VEO3 Bot main module (legacy 2025-09-11 layout).

This version keeps the simple state-machine based flow that the production
bot relied on before the large refactor.  It supports:

* VEO video generation (text or photo reference)
* Midjourney image generation (via KIE)
* Prompt-Master helper powered by OpenAI
* ChatGPT mini-chat (optional, paid unlock)
* Banana image editor mode (KIE google/nano-banana-edit)
* Telegram Stars based balance top-up

The file is intentionally monolithic to match the previous structure so the
existing deployment scripts and expectations keep working without the more
complex plugin/handler architecture that caused regressions.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import tempfile
import time
import uuid
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputFile,
    InputMediaPhoto,
    LabeledPrice,
    Update,
)
from telegram.constants import ParseMode
from telegram.ext import (
    AIORateLimiter,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    PreCheckoutQueryHandler,
    filters,
)

from kie_banana import KieBananaError, create_banana_task, wait_for_banana_result

load_dotenv()


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------
def _env(key: str, default: str = "") -> str:
    value = os.getenv(key)
    return (value if value is not None else default).strip()


TELEGRAM_TOKEN = _env("TELEGRAM_TOKEN")
PROMPTS_CHANNEL_URL = _env("PROMPTS_CHANNEL_URL", "https://t.me/bestveo3promts")
STARS_BUY_URL = _env("STARS_BUY_URL", "https://t.me/PremiumBot")
DEV_MODE = _env("DEV_MODE", "true").lower() == "true"

OPENAI_API_KEY = _env("OPENAI_API_KEY")
try:  # pragma: no cover - optional dependency
    import openai  # type: ignore

    if OPENAI_API_KEY:
        openai.api_key = OPENAI_API_KEY
except Exception:  # pragma: no cover - keep working without openai
    openai = None  # type: ignore

KIE_API_KEY = _env("KIE_API_KEY")
KIE_BASE_URL = _env("KIE_BASE_URL", "https://api.kie.ai")
KIE_VEO_GEN_PATH = _env("KIE_VEO_GEN_PATH", _env("KIE_GEN_PATH", "/api/v1/veo/generate"))
KIE_VEO_STATUS_PATH = _env("KIE_VEO_STATUS_PATH", _env("KIE_STATUS_PATH", "/api/v1/veo/record-info"))
KIE_VEO_1080_PATH = _env("KIE_VEO_1080_PATH", _env("KIE_HD_PATH", "/api/v1/veo/get-1080p-video"))
KIE_MJ_GENERATE = _env("KIE_MJ_GENERATE", "/api/v1/mj/generate")
KIE_MJ_STATUS = _env("KIE_MJ_STATUS", "/api/v1/mj/record-info")
UPLOAD_BASE_URL = _env("UPLOAD_BASE_URL", "https://kieai.redpandaai.co")
UPLOAD_STREAM_PATH = _env("UPLOAD_STREAM_PATH", "/api/file-stream-upload")
UPLOAD_URL_PATH = _env("UPLOAD_URL_PATH", "/api/file-url-upload")
UPLOAD_BASE64_PATH = _env("UPLOAD_BASE64_PATH", "/api/file-base64-upload")

ENABLE_VERTICAL_NORMALIZE = _env("ENABLE_VERTICAL_NORMALIZE", "true").lower() == "true"
ALWAYS_FORCE_FHD = _env("ALWAYS_FORCE_FHD", "true").lower() == "true"
FFMPEG_BIN = _env("FFMPEG_BIN", "ffmpeg")
MAX_TG_VIDEO_MB = int(_env("MAX_TG_VIDEO_MB", "48"))
POLL_INTERVAL_SECS = int(_env("POLL_INTERVAL_SECS", "6"))
POLL_TIMEOUT_SECS = int(_env("POLL_TIMEOUT_SECS", str(20 * 60)))

logging.basicConfig(
    level=_env("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("veo3-bot")

try:  # pragma: no cover - optional info only
    import telegram as _tg  # type: ignore

    log.info("PTB version: %s", getattr(_tg, "__version__", "unknown"))
except Exception:  # pragma: no cover
    _tg = None  # type: ignore


# ---------------------------------------------------------------------------
# Token accounting
# ---------------------------------------------------------------------------
TOKEN_COSTS = {
    "veo_fast": 1,
    "veo_quality": 1,
    "veo_photo": 1,
    "mj": 1,
    "banana": 1,
}

CHAT_UNLOCK_PRICE = 50


def get_user_balance(ctx: ContextTypes.DEFAULT_TYPE) -> int:
    return int(ctx.user_data.get("balance", 0))


def set_user_balance(ctx: ContextTypes.DEFAULT_TYPE, value: int) -> None:
    ctx.user_data["balance"] = max(0, int(value))


def add_tokens(ctx: ContextTypes.DEFAULT_TYPE, amount: int) -> None:
    set_user_balance(ctx, get_user_balance(ctx) + int(amount))


def try_charge(ctx: ContextTypes.DEFAULT_TYPE, amount: int) -> Tuple[bool, int]:
    balance = get_user_balance(ctx)
    if balance < amount:
        return False, balance
    set_user_balance(ctx, balance - amount)
    return True, balance - amount


STARS_PACKS: Dict[int, int] = {100: 100, 200: 200, 300: 300, 400: 400, 500: 500}
if DEV_MODE:
    STARS_PACKS = {1: 1, **STARS_PACKS}


def tokens_for_stars(stars: int) -> int:
    return int(STARS_PACKS.get(int(stars), 0))


LAST_STAR_CHARGE_BY_USER: Dict[int, str] = {}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _nz(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    stripped = text.strip()
    return stripped if stripped else None


def join_url(base: str, path: str) -> str:
    joined = f"{base.rstrip('/')}/{path.lstrip('/')}"
    return joined.replace("://", "§§").replace("//", "/").replace("§§", "://")


def event(tag: str, **kwargs: Any) -> None:
    try:
        log.info("EVT %s | %s", tag, json.dumps(kwargs, ensure_ascii=False))
    except Exception:
        log.info("EVT %s | %s", tag, kwargs)


def tg_direct_file_url(bot_token: str, file_path: str) -> str:
    clean = (file_path or "").strip()
    if clean.startswith("http://") or clean.startswith("https://"):
        return clean
    return f"https://api.telegram.org/file/bot{bot_token}/{clean.lstrip('/')}"


# ---------------------------------------------------------------------------
# Conversation state
# ---------------------------------------------------------------------------
DEFAULT_STATE: Dict[str, Any] = {
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
    "chat_unlocked": False,
}


def state(ctx: ContextTypes.DEFAULT_TYPE) -> Dict[str, Any]:
    data = ctx.user_data
    for key, value in DEFAULT_STATE.items():
        data.setdefault(key, value)
    return data


WELCOME = (
    "🎬 *Veo 3 — съёмочная команда* — опиши идею и получи готовый клип!\n"
    "🖌️ *MJ — художник* — нарисует изображение по твоему тексту (только 16:9).\n"
    "🍌 *Banana — редактор изображений из будущего*.\n"
    "🧠 *Prompt-Master* — верну профессиональный кинопромпт.\n"
    "💬 *Обычный чат* — общение с ИИ.\n\n"
    "💎 *Ваш баланс:* {balance}\n"
    "📈 Больше идей и примеров: {prompts_url}\n\n"
    "Выберите режим 👇"
)


def render_welcome(ctx: ContextTypes.DEFAULT_TYPE) -> str:
    return WELCOME.format(balance=get_user_balance(ctx), prompts_url=PROMPTS_CHANNEL_URL)


def main_menu_kb() -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton("🎬 Генерация видео (Veo) 💎1", callback_data="mode:veo_text")],
        [InlineKeyboardButton("🖼️ Генерация изображений (MJ) 💎1", callback_data="mode:mj_txt")],
        [InlineKeyboardButton("🍌 Редактор изображений (Banana) 💎1", callback_data="mode:banana")],
        [InlineKeyboardButton("📸 Оживить изображение (Veo) 💎1", callback_data="mode:veo_photo")],
        [InlineKeyboardButton("🧠 Prompt-Master (ChatGPT)", callback_data="mode:prompt_master")],
        [InlineKeyboardButton("💬 Обычный чат (ChatGPT) 💎10", callback_data="mode:chat")],
        [
            InlineKeyboardButton("❓ FAQ", callback_data="faq"),
            InlineKeyboardButton("📈 Канал с промптами", url=PROMPTS_CHANNEL_URL),
        ],
        [InlineKeyboardButton("💳 Пополнить баланс", callback_data="topup_open")],
    ]
    return InlineKeyboardMarkup(rows)


def aspect_row(current: str) -> List[InlineKeyboardButton]:
    if current == "9:16":
        return [
            InlineKeyboardButton("16:9", callback_data="aspect:16:9"),
            InlineKeyboardButton("9:16 ✅", callback_data="aspect:9:16"),
        ]
    return [
        InlineKeyboardButton("16:9 ✅", callback_data="aspect:16:9"),
        InlineKeyboardButton("9:16", callback_data="aspect:9:16"),
    ]


def model_row(current: str) -> List[InlineKeyboardButton]:
    if current == "veo3":
        return [
            InlineKeyboardButton("⚡ Fast", callback_data="model:veo3_fast"),
            InlineKeyboardButton("💎 Quality ✅", callback_data="model:veo3"),
        ]
    return [
        InlineKeyboardButton("⚡ Fast ✅", callback_data="model:veo3_fast"),
        InlineKeyboardButton("💎 Quality", callback_data="model:veo3"),
    ]


def build_card_text_veo(s: Dict[str, Any]) -> str:
    prompt_preview = (s.get("last_prompt") or "").strip()
    if len(prompt_preview) > 900:
        prompt_preview = prompt_preview[:900] + "…"
    has_prompt = "есть" if s.get("last_prompt") else "нет"
    has_ref = "есть" if s.get("last_image_url") else "нет"
    model = "Fast" if s.get("model") == "veo3_fast" else ("Quality" if s.get("model") else "—")
    price = TOKEN_COSTS["veo_quality"] if s.get("model") == "veo3" else TOKEN_COSTS["veo_fast"]
    parts = [
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
    return "\n".join(parts)


def card_keyboard_veo(s: Dict[str, Any]) -> InlineKeyboardMarkup:
    rows: List[List[InlineKeyboardButton]] = []
    rows.append(
        [
            InlineKeyboardButton("🖼️ Добавить/Удалить фото", callback_data="card:toggle_photo"),
            InlineKeyboardButton("✍️ Изменить промпт", callback_data="card:edit_prompt"),
        ]
    )
    rows.append(aspect_row(s.get("aspect") or "16:9"))
    rows.append(model_row(s.get("model") or "veo3_fast"))
    rows.append([InlineKeyboardButton("🚀 Сгенерировать", callback_data="card:generate")])
    rows.append(
        [
            InlineKeyboardButton("🔁 Начать заново", callback_data="card:reset"),
            InlineKeyboardButton("⬅️ Назад", callback_data="back"),
        ]
    )
    rows.append([InlineKeyboardButton("💳 Пополнить баланс", callback_data="topup_open")])
    return InlineKeyboardMarkup(rows)


def mj_start_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton("🌆 Сгенерировать 16:9", callback_data="mj:ar:16:9")]])


# ---------------------------------------------------------------------------
# Prompt-Master helper
# ---------------------------------------------------------------------------
async def oai_prompt_master(idea_text: str) -> Optional[str]:
    if openai is None or not OPENAI_API_KEY:
        return None

    system_prompt = (
        "You are a Prompt-Master for cinematic AI video generation (Veo-style). "
        "Return ONE multi-line prompt in ENGLISH following this exact structure and labels: "
        "High-quality cinematic 4K video (16:9).\n"
        "Scene: ...\nCamera: ...\nAction: ...\nDialogue: ...\nLip-sync: ...\nAudio: ...\n"
        "Lighting: ...\nWardrobe/props: ...\nFraming: ...\nConstraints: No subtitles. No on-screen text. No logos. "
        "Rules: keep 16:9; forbid legible text; be specific; keep it 600–1100 chars; no lists, no bullets beyond labels; "
        "if the user mentions Russian speech, place it inside quotes and note that lip sync must match every syllable."
    )

    try:
        user = idea_text.strip()
        if len(user) > 800:
            user = user[:800] + "..."
        response = await asyncio.to_thread(
            openai.ChatCompletion.create,
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user}],
            temperature=0.8,
            max_tokens=800,
        )
        text = response["choices"][0]["message"]["content"].strip()
        return text[:1400]
    except Exception as exc:  # pragma: no cover - network failures
        log.exception("Prompt-Master error: %s", exc)
        return None


# ---------------------------------------------------------------------------
# HTTP helpers for KIE APIs
# ---------------------------------------------------------------------------
def _kie_headers_json() -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    token = (KIE_API_KEY or "").strip()
    if token and not token.lower().startswith("bearer "):
        token = f"Bearer {token}"
    if token:
        headers["Authorization"] = token
    return headers


def _post_json(url: str, payload: Dict[str, Any], timeout: int = 40) -> Tuple[int, Dict[str, Any]]:
    response = requests.post(url, json=payload, headers=_kie_headers_json(), timeout=timeout)
    try:
        return response.status_code, response.json()
    except Exception:
        return response.status_code, {"error": response.text}


def _get_json(url: str, params: Dict[str, Any], timeout: int = 40) -> Tuple[int, Dict[str, Any]]:
    response = requests.get(url, params=params, headers=_kie_headers_json(), timeout=timeout)
    try:
        return response.status_code, response.json()
    except Exception:
        return response.status_code, {"error": response.text}


def _extract_task_id(payload: Dict[str, Any]) -> Optional[str]:
    data = payload.get("data") or {}
    for key in ("taskId", "taskid", "id"):
        if payload.get(key):
            return str(payload[key])
        if data.get(key):
            return str(data[key])
    return None


def _coerce_url_list(value: Any) -> List[str]:
    urls: List[str] = []

    def add(item: str) -> None:
        if isinstance(item, str):
            stripped = item.strip()
            if stripped.startswith("http"):
                urls.append(stripped)

    if not value:
        return urls

    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            try:
                arr = json.loads(stripped)
                if isinstance(arr, list):
                    for val in arr:
                        if isinstance(val, str):
                            add(val)
                return urls
            except Exception:
                add(stripped)
                return urls
        add(stripped)
        return urls

    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                add(item)
            elif isinstance(item, dict):
                candidate = item.get("resultUrl") or item.get("originUrl") or item.get("url")
                if isinstance(candidate, str):
                    add(candidate)
        return urls

    if isinstance(value, dict):
        for key in ("resultUrl", "originUrl", "url"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                add(candidate)
        return urls

    return urls


def _extract_result_url(data: Dict[str, Any]) -> Optional[str]:
    for key in ("originUrls", "resultUrls"):
        urls = _coerce_url_list(data.get(key))
        if urls:
            return urls[0]

    for container in ("info", "response", "resultInfoJson"):
        nested = data.get(container)
        if isinstance(nested, dict):
            for key in ("originUrls", "resultUrls", "videoUrls"):
                urls = _coerce_url_list(nested.get(key))
                if urls:
                    return urls[0]

    def walk(value: Any) -> Optional[str]:
        if isinstance(value, dict):
            for candidate in value.values():
                found = walk(candidate)
                if found:
                    return found
        elif isinstance(value, list):
            for candidate in value:
                found = walk(candidate)
                if found:
                    return found
        elif isinstance(value, str):
            clean = value.strip().split("?")[0].lower()
            if clean.startswith("http") and clean.endswith((".mp4", ".mov", ".webm")):
                return value.strip()
        return None

    return walk(data)


def _upload_headers() -> Dict[str, str]:
    token = (KIE_API_KEY or "").strip()
    if token and not token.lower().startswith("bearer "):
        token = f"Bearer {token}"
    return {"Authorization": token} if token else {}


def upload_image_stream(src_url: str, upload_path: str = "tg-uploads", timeout: int = 90) -> Optional[str]:
    try:
        response = requests.get(src_url, stream=True, timeout=timeout)
        response.raise_for_status()
        content_type = (response.headers.get("Content-Type") or "").lower()
        ext = ".jpg"
        if "png" in content_type:
            ext = ".png"
        elif "webp" in content_type:
            ext = ".webp"
        elif "jpeg" in content_type:
            ext = ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            for chunk in response.iter_content(256 * 1024):
                if chunk:
                    tmp.write(chunk)
            local_path = tmp.name
    except Exception as exc:
        event("upload_stream_predownload_failed", err=str(exc), url=src_url)
        return None

    try:
        url = join_url(UPLOAD_BASE_URL, UPLOAD_STREAM_PATH)
        with open(local_path, "rb") as fh:
            files = {"file": (os.path.basename(local_path), fh)}
            data = {"uploadPath": upload_path, "fileName": os.path.basename(local_path)}
            response = requests.post(url, headers=_upload_headers(), files=files, data=data, timeout=timeout)
        try:
            payload = response.json()
        except Exception:
            payload = {"error": response.text}
        if response.status_code == 200 and (payload.get("code", 200) == 200):
            data = payload.get("data") or {}
            result_url = data.get("downloadUrl") or data.get("fileUrl")
            if _nz(result_url):
                event("KIE_UPLOAD_STREAM_OK", url=result_url)
                return result_url
        event("upload_stream_failed", status=response.status_code, resp=payload)
    except Exception as exc:
        event("upload_stream_err", err=str(exc))
    finally:
        try:
            os.unlink(local_path)
        except Exception:
            pass
    return None


def upload_image_base64(src_url: str, upload_path: str = "tg-uploads", timeout: int = 90) -> Optional[str]:
    try:
        response = requests.get(src_url, stream=True, timeout=timeout)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type") or "image/jpeg"
        data_url = f"data:{content_type};base64,{base64.b64encode(response.content).decode('utf-8')}"
    except Exception as exc:
        event("upload_b64_predownload_failed", err=str(exc))
        return None

    try:
        url = join_url(UPLOAD_BASE_URL, UPLOAD_BASE64_PATH)
        payload = {"base64Data": data_url, "uploadPath": upload_path, "fileName": "tg-upload.jpg"}
        response = requests.post(
            url,
            json=payload,
            headers={**_upload_headers(), "Content-Type": "application/json"},
            timeout=timeout,
        )
        try:
            data = response.json()
        except Exception:
            data = {"error": response.text}
        if response.status_code == 200 and (data.get("code", 200) == 200):
            payload = data.get("data") or {}
            result_url = payload.get("downloadUrl") or payload.get("fileUrl")
            if _nz(result_url):
                event("KIE_UPLOAD_B64_OK", url=result_url)
                return result_url
        event("upload_b64_failed", status=response.status_code, resp=data)
    except Exception as exc:
        event("upload_b64_err", err=str(exc))
    return None


def upload_image_url(file_url: str, upload_path: str = "tg-uploads", timeout: int = 60) -> Optional[str]:
    try:
        url = join_url(UPLOAD_BASE_URL, UPLOAD_URL_PATH)
        payload = {
            "fileUrl": file_url,
            "uploadPath": upload_path,
            "fileName": os.path.basename(file_url.split("?")[0]) or "image.jpg",
        }
        response = requests.post(
            url,
            json=payload,
            headers={**_upload_headers(), "Content-Type": "application/json"},
            timeout=timeout,
        )
        try:
            data = response.json()
        except Exception:
            data = {"error": response.text}
        if response.status_code == 200 and (data.get("code", 200) == 200):
            payload = data.get("data") or {}
            result_url = payload.get("downloadUrl") or payload.get("fileUrl")
            if _nz(result_url):
                event("KIE_UPLOAD_URL_OK", url=result_url)
                return result_url
        event("upload_url_failed", status=response.status_code, resp=data)
    except Exception as exc:
        event("upload_url_err", err=str(exc))
    return None


# ---------------------------------------------------------------------------
# VEO helper logic
# ---------------------------------------------------------------------------
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
    image_for_kie = None
    if _nz(image_url):
        image_for_kie = (
            upload_image_stream(image_url)
            or upload_image_base64(image_url)
            or upload_image_url(image_url)
            or image_url
        )

    payload = _build_payload_for_veo(prompt, aspect, image_for_kie, model_key)
    status, data = _post_json(join_url(KIE_BASE_URL, KIE_VEO_GEN_PATH), payload)
    code = data.get("code", status)
    if status == 200 and code == 200:
        task_id = _extract_task_id(data)
        if task_id:
            return True, task_id, "Задача создана."
        return False, None, "Ответ KIE без taskId."

    message = (data.get("msg") or data.get("message") or data.get("error") or "").lower()
    if "image fetch failed" in message or ("image" in message and "failed" in message):
        payload.pop("imageUrls", None)
        status2, data2 = _post_json(join_url(KIE_BASE_URL, KIE_VEO_GEN_PATH), payload)
        if status2 == 200 and (data2.get("code", 200) == 200):
            task_id = _extract_task_id(data2)
            if task_id:
                return True, task_id, "Задача создана (без изображения)."

    return False, None, _kie_error_message(status, data)


def get_kie_veo_status(task_id: str) -> Tuple[bool, Optional[int], Optional[str], Optional[str]]:
    status, payload = _get_json(join_url(KIE_BASE_URL, KIE_VEO_STATUS_PATH), {"taskId": task_id})
    code = payload.get("code", status)
    if status == 200 and code == 200:
        data = payload.get("data") or {}
        flag = data.get("successFlag")
        try:
            flag = int(flag)
        except Exception:
            flag = None
        message = payload.get("msg") or payload.get("message")
        return True, flag, message, _extract_result_url(data)
    return False, None, _kie_error_message(status, payload), None


def try_get_1080_url(task_id: str, attempts: int = 3, per_try_timeout: int = 15) -> Optional[str]:
    last_error: Optional[str] = None
    for idx in range(attempts):
        try:
            status, payload = _get_json(
                join_url(KIE_BASE_URL, KIE_VEO_1080_PATH),
                {"taskId": task_id},
                timeout=per_try_timeout,
            )
            code = payload.get("code", status)
            if status == 200 and code == 200:
                data = payload.get("data") or {}
                result = _nz(data.get("url")) or _extract_result_url(data)
                if _nz(result):
                    return result
                last_error = "empty_1080"
            else:
                last_error = f"status={status}, code={code}"
        except Exception as exc:
            last_error = str(exc)
        time.sleep(1 + idx)
    if last_error:
        log.warning("1080p fetch retries failed: %s", last_error)
    return None


# ---------------------------------------------------------------------------
# MJ helper logic
# ---------------------------------------------------------------------------
def _kie_error_message(status_code: int, payload: Dict[str, Any]) -> str:
    code = payload.get("code", status_code)
    message = payload.get("msg") or payload.get("message") or payload.get("error") or ""
    mapping = {
        401: "Доступ запрещён (Bearer).",
        402: "Недостаточно кредитов.",
        429: "Превышен лимит запросов.",
        500: "Внутренняя ошибка KIE.",
        422: "Запрос отклонён модерацией.",
        400: "Неверный запрос (400).",
    }
    base = mapping.get(code, f"KIE code {code}.")
    return f"{base} {('Сообщение: ' + message) if message else ''}".strip()


def mj_generate(prompt: str, aspect_ratio: str) -> Tuple[bool, Optional[str], str]:
    payload = {
        "taskType": "mj_txt2img",
        "prompt": prompt,
        "speed": "turbo",
        "aspectRatio": "16:9",
        "version": "7",
        "enableTranslation": True,
    }
    status, data = _post_json(join_url(KIE_BASE_URL, KIE_MJ_GENERATE), payload)
    code = data.get("code", status)
    if status == 200 and code == 200:
        task_id = _extract_task_id(data)
        if task_id:
            return True, task_id, "MJ задача создана."
        return False, None, "Ответ MJ без taskId."
    return False, None, _kie_error_message(status, data)


def mj_status(task_id: str) -> Tuple[bool, Optional[int], Optional[Dict[str, Any]]]:
    status, payload = _get_json(join_url(KIE_BASE_URL, KIE_MJ_STATUS), {"taskId": task_id})
    code = payload.get("code", status)
    if status == 200 and code == 200:
        data = payload.get("data") or {}
        try:
            flag = int(data.get("successFlag"))
        except Exception:
            flag = None
        return True, flag, data
    return False, None, None


def _extract_mj_image_urls(status_data: Dict[str, Any]) -> List[str]:
    result: List[str] = []
    info = status_data.get("resultInfoJson") or {}
    urls = _coerce_url_list(info.get("resultUrls") or [])
    for url in urls:
        if isinstance(url, str) and url.startswith("http"):
            result.append(url)
    return result


# ---------------------------------------------------------------------------
# ffmpeg helpers & delivery
# ---------------------------------------------------------------------------
def _ffmpeg_available() -> bool:
    from shutil import which

    return bool(which(FFMPEG_BIN))


def _ffmpeg_normalize_vertical(inp: str, outp: str) -> bool:
    command = [
        FFMPEG_BIN,
        "-y",
        "-i",
        inp,
        "-vf",
        "scale=1080:1920:force_original_aspect_ratio=decrease,"
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2,format=yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-metadata:s:v:0",
        "rotate=0",
        outp,
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except Exception as exc:
        log.warning("ffmpeg vertical failed: %s", exc)
        return False


def _ffmpeg_force_16x9_fhd(inp: str, outp: str, target_mb: int) -> bool:
    target_bytes = max(8, int(target_mb)) * 1024 * 1024
    command = [
        FFMPEG_BIN,
        "-y",
        "-i",
        inp,
        "-vf",
        "scale=1920:1080:force_original_aspect_ratio=decrease,"
        "pad=1920:1080:(ow-iw)/2:(oh-ih)/2,format=yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-fs",
        str(target_bytes),
        outp,
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except Exception as exc:
        log.warning("ffmpeg 16x9 FHD failed: %s", exc)
        return False


async def send_video_with_fallback(
    ctx: ContextTypes.DEFAULT_TYPE, chat_id: int, url: str, expect_vertical: bool = False
) -> bool:
    event("SEND_TRY_URL", url=url, expect_vertical=expect_vertical)
    if not expect_vertical:
        try:
            await ctx.bot.send_video(chat_id=chat_id, video=url, supports_streaming=True)
            event("SEND_OK", mode="direct_url")
            return True
        except Exception as exc:
            event("SEND_FAIL", mode="direct_url", err=str(exc))

    tmp_path: Optional[str] = None
    try:
        response = requests.get(url, stream=True, timeout=180)
        response.raise_for_status()
        content_type = (response.headers.get("Content-Type") or "").lower()
        ext = ".mp4"
        if ".mov" in url.lower() or "quicktime" in content_type:
            ext = ".mov"
        elif ".webm" in url.lower() or "webm" in content_type:
            ext = ".webm"

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            for chunk in response.iter_content(256 * 1024):
                if chunk:
                    tmp.write(chunk)
            tmp_path = tmp.name
        event("DOWNLOAD_OK", path=tmp_path, content_type=content_type)

        if expect_vertical and ENABLE_VERTICAL_NORMALIZE and _ffmpeg_available():
            normalized = tmp_path + "_v.mp4"
            if _ffmpeg_normalize_vertical(tmp_path, normalized):
                try:
                    with open(normalized, "rb") as fh:
                        await ctx.bot.send_video(
                            chat_id=chat_id,
                            video=InputFile(fh, filename="result_vertical.mp4"),
                            supports_streaming=True,
                        )
                    event("SEND_OK", mode="upload_video_norm")
                    return True
                except Exception as exc:
                    event("SEND_FAIL", mode="upload_video_norm", err=str(exc))
                    with open(normalized, "rb") as fh:
                        await ctx.bot.send_document(
                            chat_id=chat_id,
                            document=InputFile(fh, filename="result_vertical.mp4"),
                        )
                    event("SEND_OK", mode="upload_document_norm")
                    return True

        if (not expect_vertical) and ALWAYS_FORCE_FHD and _ffmpeg_available():
            normalized = tmp_path + "_1080.mp4"
            if _ffmpeg_force_16x9_fhd(tmp_path, normalized, MAX_TG_VIDEO_MB):
                try:
                    with open(normalized, "rb") as fh:
                        await ctx.bot.send_video(
                            chat_id=chat_id,
                            video=InputFile(fh, filename="result_1080p.mp4"),
                            supports_streaming=True,
                        )
                    event("SEND_OK", mode="upload_16x9_forced")
                    return True
                except Exception as exc:
                    event("SEND_FAIL", mode="upload_16x9_forced", err=str(exc))
                    with open(normalized, "rb") as fh:
                        await ctx.bot.send_document(
                            chat_id=chat_id,
                            document=InputFile(fh, filename="result_1080p.mp4"),
                        )
                    event("SEND_OK", mode="upload_document_16x9_forced")
                    return True

        try:
            with open(tmp_path, "rb") as fh:
                await ctx.bot.send_video(
                    chat_id=chat_id,
                    video=InputFile(fh, filename=f"result{ext}"),
                    supports_streaming=True,
                )
            event("SEND_OK", mode="upload_video_raw")
            return True
        except Exception as exc:
            event("SEND_FAIL", mode="upload_video_raw", err=str(exc))
            with open(tmp_path, "rb") as fh:
                await ctx.bot.send_document(
                    chat_id=chat_id,
                    document=InputFile(fh, filename=f"result{ext}"),
                )
            event("SEND_OK", mode="upload_document_raw")
            return True

    except Exception as exc:  # pragma: no cover - network failure handling
        log.exception("File send failed: %s", exc)
        try:
            await ctx.bot.send_message(chat_id, f"🔗 Результат готов, но вложить файл не удалось. Ссылка:\n{url}")
            event("SEND_OK", mode="link_fallback_on_error")
            return True
        except Exception:
            return False
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Polling tasks
# ---------------------------------------------------------------------------
async def poll_veo_and_send(chat_id: int, task_id: str, gen_id: str, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    s = state(ctx)
    s["generating"] = True
    s["generation_id"] = gen_id
    s["last_task_id"] = task_id

    start_ts = time.time()
    try:
        while True:
            if s.get("generation_id") != gen_id:
                return

            ok, flag, message, result_url = await asyncio.to_thread(get_kie_veo_status, task_id)
            event("VEO_STATUS", task_id=task_id, flag=flag, has_url=bool(result_url), msg=message)

            if not ok:
                add_tokens(
                    ctx,
                    TOKEN_COSTS["veo_quality"] if s.get("model") == "veo3" else TOKEN_COSTS["veo_fast"],
                )
                await ctx.bot.send_message(
                    chat_id,
                    f"❌ Ошибка статуса: {message or 'неизвестно'}\n💎 Токены возвращены.",
                )
                break

            if _nz(result_url):
                final_url = result_url
                if (s.get("aspect") or "16:9") == "16:9":
                    url_1080 = await asyncio.to_thread(try_get_1080_url, task_id)
                    if _nz(url_1080):
                        final_url = url_1080
                        event("VEO_1080_OK", task_id=task_id)
                    else:
                        event("VEO_1080_MISS", task_id=task_id)

                await ctx.bot.send_message(chat_id, "🎞️ Рендер завершён — отправляю файл…")
                sent = await send_video_with_fallback(
                    ctx,
                    chat_id,
                    final_url,
                    expect_vertical=(s.get("aspect") == "9:16"),
                )
                s["last_result_url"] = final_url if sent else None
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
                add_tokens(
                    ctx,
                    TOKEN_COSTS["veo_quality"] if s.get("model") == "veo3" else TOKEN_COSTS["veo_fast"],
                )
                await ctx.bot.send_message(
                    chat_id,
                    f"❌ KIE не вернул ссылку на видео.\nℹ️ Сообщение: {message or 'нет'}\n💎 Токены возвращены.",
                )
                break

            if (time.time() - start_ts) > POLL_TIMEOUT_SECS:
                add_tokens(
                    ctx,
                    TOKEN_COSTS["veo_quality"] if s.get("model") == "veo3" else TOKEN_COSTS["veo_fast"],
                )
                await ctx.bot.send_message(chat_id, "⌛ Превышено время ожидания VEO.\n💎 Токены возвращены.")
                break

            await asyncio.sleep(POLL_INTERVAL_SECS)

    except Exception as exc:  # pragma: no cover - network/telegram failures
        log.exception("[VEO_POLL] crash: %s", exc)
        add_tokens(
            ctx,
            TOKEN_COSTS["veo_quality"] if s.get("model") == "veo3" else TOKEN_COSTS["veo_fast"],
        )
        try:
            await ctx.bot.send_message(chat_id, "💥 Внутренняя ошибка при опросе VEO.\n💎 Токены возвращены.")
        except Exception:
            pass
    finally:
        if s.get("generation_id") == gen_id:
            s["generating"] = False
            s["generation_id"] = None


async def poll_mj_and_send_photos(chat_id: int, task_id: str, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    start_ts = time.time()
    delay = 6
    max_wait = 15 * 60
    try:
        while True:
            ok, flag, data = await asyncio.to_thread(mj_status, task_id)
            event("MJ_STATUS", task_id=task_id, flag=flag, has_data=bool(data))

            if not ok:
                add_tokens(ctx, TOKEN_COSTS["mj"])
                await ctx.bot.send_message(chat_id, "❌ MJ сейчас недоступен. 💎 Токены возвращены.")
                return

            if flag == 0:
                if (time.time() - start_ts) > max_wait:
                    add_tokens(ctx, TOKEN_COSTS["mj"])
                    await ctx.bot.send_message(
                        chat_id, "⌛ MJ долго не отвечает. Попробуйте позже. 💎 Токены возвращены."
                    )
                    return
                await ctx.bot.send_message(chat_id, "🖼️✨ Рисую… Подождите немного.", disable_notification=True)
                await asyncio.sleep(delay)
                delay = min(delay * 1.5, 20)
                continue

            if flag in (2, 3):
                message = (data or {}).get("errorMessage") or "No response from MidJourney Official Website."
                add_tokens(ctx, TOKEN_COSTS["mj"])
                await ctx.bot.send_message(chat_id, f"❌ MJ: {message}\n💎 Токены возвращены.")
                return

            if flag == 1:
                urls = _extract_mj_image_urls(data or {})
                if not urls:
                    add_tokens(ctx, TOKEN_COSTS["mj"])
                    await ctx.bot.send_message(
                        chat_id, "⚠️ Готово, но ссылки на изображения не найдены. 💎 Токены возвращены."
                    )
                    return
                if len(urls) == 1:
                    await ctx.bot.send_photo(chat_id=chat_id, photo=urls[0])
                else:
                    media = [InputMediaPhoto(u) for u in urls[:10]]
                    await ctx.bot.send_media_group(chat_id=chat_id, media=media)
                await ctx.bot.send_message(chat_id, "✅ Готово! (апскейл отключён по умолчанию)")
                return
    except Exception as exc:  # pragma: no cover - network/telegram failures
        log.exception("[MJ_POLL] crash: %s", exc)
        add_tokens(ctx, TOKEN_COSTS["mj"])
        try:
            await ctx.bot.send_message(chat_id, "💥 Внутренняя ошибка при опросе MJ. 💎 Токены возвращены.")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------
def stars_topup_kb() -> InlineKeyboardMarkup:
    rows: List[List[InlineKeyboardButton]] = []
    for stars in sorted(STARS_PACKS.keys()):
        tokens = STARS_PACKS[stars]
        label = f"⭐ {stars} → 💎 {tokens}" + ("  (DEV)" if DEV_MODE and stars == 1 else "")
        rows.append([InlineKeyboardButton(label, callback_data=f"buy:stars:{stars}")])
    rows.append([InlineKeyboardButton("🛒 Где купить Stars", url=STARS_BUY_URL)])
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="back")])
    return InlineKeyboardMarkup(rows)


async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    s = state(ctx)
    s.update({**DEFAULT_STATE})
    await update.message.reply_text(
        render_welcome(ctx),
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=main_menu_kb(),
    )


async def topup(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "💳 Пополнение токенов через *Telegram Stars*.\n"
        f"Если не хватает звёзд — купите их в официальном боте: {STARS_BUY_URL}",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=stars_topup_kb(),
    )


async def health(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    parts = [
        f"PTB: `{getattr(_tg, '__version__', 'unknown')}`" if _tg else "PTB: `unknown`",
        f"KIEBASEURL: `{KIE_BASE_URL}`",
        f"OPENAI key: `{'set' if OPENAI_API_KEY else 'missing'}`",
        f"KIE key: `{'set' if KIE_API_KEY else 'missing'}`",
        f"DEV_MODE: `{DEV_MODE}`",
        f"FFMPEGBIN: `{FFMPEG_BIN}`",
        f"MAXTGVIDEOMB: `{MAX_TG_VIDEO_MB}`",
    ]
    await update.message.reply_text("🩺 *Health*\n" + "\n".join(parts), parse_mode=ParseMode.MARKDOWN)


async def error_handler(update: Optional[Update], context: ContextTypes.DEFAULT_TYPE) -> None:
    log.exception("Unhandled error: %s", context.error)
    try:
        if update and update.effective_chat:
            await context.bot.send_message(update.effective_chat.id, "⚠️ Системная ошибка. Попробуйте ещё раз.")
    except Exception:
        pass


async def show_card_veo(update: Update, ctx: ContextTypes.DEFAULT_TYPE, edit_only_markup: bool = False) -> None:
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
            if update.callback_query:
                msg = await update.callback_query.message.reply_text(
                    text,
                    parse_mode=ParseMode.MARKDOWN,
                    reply_markup=kb,
                    disable_web_page_preview=True,
                )
            else:
                msg = await update.message.reply_text(
                    text,
                    parse_mode=ParseMode.MARKDOWN,
                    reply_markup=kb,
                    disable_web_page_preview=True,
                )
            s["last_ui_msg_id"] = msg.message_id
    except Exception as exc:
        log.warning("show_card_veo edit/send failed: %s", exc)
        try:
            msg = await ctx.bot.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=kb,
                disable_web_page_preview=True,
            )
            s["last_ui_msg_id"] = msg.message_id
        except Exception as exc2:  # pragma: no cover - Telegram errors
            log.exception("show_card_veo send failed: %s", exc2)


async def on_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    data = (query.data or "").strip()
    await query.answer()

    s = state(ctx)

    if data == "faq":
        await query.message.reply_text(
            "FAQ:\n"
            "• VEO: 9:16 нормализуем локально в 1080×1920, 16:9 тянем до 1080p.\n"
            "• MJ: только 16:9 (вертикаль отключена из-за ошибок API).\n"
            "• Banana: пришлите фото с подписью (промпт) — вернётся отредактированное изображение.\n"
            f"• Пополнение Stars: {STARS_BUY_URL}",
            reply_markup=main_menu_kb(),
        )
        return

    if data == "back":
        s.update({**DEFAULT_STATE})
        await query.message.reply_text("🏠 Главное меню:", reply_markup=main_menu_kb())
        return

    if data == "start_new_cycle":
        s.update({**DEFAULT_STATE})
        await query.message.reply_text("Выберите режим:", reply_markup=main_menu_kb())
        return

    if data == "topup_open":
        await query.message.reply_text(
            f"💳 Пополнение через Telegram Stars. Если звёзд не хватает — купите в {STARS_BUY_URL}",
            reply_markup=stars_topup_kb(),
        )
        return

    if data.startswith("buy:stars:"):
        stars = int(data.split(":")[-1])
        tokens = tokens_for_stars(stars)
        if tokens <= 0:
            await query.message.reply_text("⚠️ Такой пакет недоступен.")
            return

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
                need_name=False,
                need_phone_number=False,
                need_email=False,
                need_shipping_address=False,
                is_flexible=False,
            )
        except Exception as exc:  # pragma: no cover - Telegram invoice errors
            event("STARS_INVOICE_ERR", err=str(exc))
            await query.message.reply_text(
                "Если счёт не открылся — у аккаунта могут быть не активированы Stars.\n"
                f"Купите 1⭐ в {STARS_BUY_URL} и попробуйте снова."
            )
        return

    if data.startswith("mode:"):
        _, mode = data.split(":", 1)
        s["mode"] = mode
        if mode == "veo_text":
            s["aspect"] = "16:9"
            s["model"] = "veo3_fast"
            await query.message.reply_text("📝 Пришлите идею/промпт для видео.")
            await show_card_veo(update, ctx)
            return
        if mode == "veo_photo":
            s["aspect"] = "9:16"
            s["model"] = "veo3_fast"
            await query.message.reply_text("🖼️ Пришлите фото (подпись-промпт — по желанию).")
            await show_card_veo(update, ctx)
            return
        if mode == "prompt_master":
            await query.message.reply_text("🧠 Пришлите идею (1–2 фразы). Я верну кинопромпт в формате из примера.")
            return
        if mode == "chat":
            if not s.get("chat_unlocked"):
                ok, rest = try_charge(ctx, CHAT_UNLOCK_PRICE)
                if not ok:
                    await query.message.reply_text(
                        f"🔐 Для доступа к «Обычному чату» нужна разовая разблокировка: *{CHAT_UNLOCK_PRICE}💎*.\n"
                        f"На балансе: {rest}.\nНажмите «Пополнить баланс».",
                        parse_mode=ParseMode.MARKDOWN,
                        reply_markup=stars_topup_kb(),
                    )
                    return
                s["chat_unlocked"] = True
                await query.message.reply_text("✅ Чат разблокирован навсегда для этого аккаунта!")
            await query.message.reply_text("✍️ Напишите вопрос для ChatGPT.")
            return
        if mode == "mj_txt":
            s["aspect"] = "16:9"
            await query.message.reply_text("🖼️ Пришлите текстовый prompt для картинки (формат 16:9).")
            return
        if mode == "banana":
            await query.message.reply_text(
                "🍌 *Banana включён*\nПришлите фото *с подписью-промптом* (англ/рус не важно).\n"
                "Пример: _turn this photo into a character figure on a round base..._",
                parse_mode=ParseMode.MARKDOWN,
            )
            return

    if data.startswith("aspect:"):
        _, value = data.split(":", 1)
        s["aspect"] = "9:16" if value.strip() == "9:16" else "16:9"
        await show_card_veo(update, ctx, edit_only_markup=True)
        return

    if data.startswith("model:"):
        _, value = data.split(":", 1)
        s["model"] = "veo3" if value.strip() == "veo3" else "veo3_fast"
        await show_card_veo(update, ctx, edit_only_markup=True)
        return

    if data == "card:toggle_photo":
        if s.get("last_image_url"):
            s["last_image_url"] = None
            await query.message.reply_text("🧹 Фото-референс удалён.")
            await show_card_veo(update, ctx)
        else:
            await query.message.reply_text("📎 Пришлите фото вложением или публичный URL изображения.")
        return

    if data == "card:edit_prompt":
        await query.message.reply_text("✍️ Пришлите новый текст промпта.")
        return

    if data == "card:reset":
        keep_aspect = s.get("aspect") or "16:9"
        keep_model = s.get("model") or "veo3_fast"
        s.update({**DEFAULT_STATE})
        s["aspect"] = keep_aspect
        s["model"] = keep_model
        await query.message.reply_text("🗂️ Карточка очищена.")
        await show_card_veo(update, ctx)
        return

    if data == "card:generate":
        if s.get("generating"):
            await query.message.reply_text("⏳ Уже рендерю это видео — подождите чуть-чуть.")
            return
        if not s.get("last_prompt"):
            await query.message.reply_text("✍️ Сначала укажите текст промпта.")
            return

        price = TOKEN_COSTS["veo_quality"] if s.get("model") == "veo3" else TOKEN_COSTS["veo_fast"]
        ok_balance, rest = try_charge(ctx, price)
        if not ok_balance:
            await query.message.reply_text(
                f"💎 Недостаточно токенов: нужно {price}, на балансе {rest}.\n"
                f"Пополните через Stars: {STARS_BUY_URL}",
                reply_markup=stars_topup_kb(),
            )
            return

        event(
            "VEO_SUBMIT_REQ",
            aspect=s.get("aspect"),
            model=s.get("model"),
            with_image=bool(s.get("last_image_url")),
            prompt_len=len(s.get("last_prompt") or ""),
        )

        ok, task_id, message = await asyncio.to_thread(
            submit_kie_veo,
            s["last_prompt"].strip(),
            s.get("aspect", "16:9"),
            s.get("last_image_url"),
            s.get("model", "veo3_fast"),
        )
        event("VEO_SUBMIT_RESP", ok=ok, task_id=task_id, msg=message)

        if not ok or not task_id:
            add_tokens(ctx, price)
            await query.message.reply_text(f"❌ Не удалось создать VEO-задачу: {message}\n💎 Токены возвращены.")
            return

        gen_id = uuid.uuid4().hex[:12]
        s["generating"] = True
        s["generation_id"] = gen_id
        s["last_task_id"] = task_id
        await query.message.reply_text(
            f"🚀 Задача отправлена ({'⚡ Fast' if s.get('model') == 'veo3_fast' else '💎 Quality'}).\n"
            f"🆔 taskId={task_id}\n🎛️ Подождите — подбираем кадры, свет и ритм…"
        )
        await query.message.reply_text("🎥 Рендер запущен… это может занять несколько минут.")
        asyncio.create_task(poll_veo_and_send(update.effective_chat.id, task_id, gen_id, ctx))
        return

    if data.startswith("mj:ar:"):
        ar = "16:9"
        prompt = s.get("last_prompt")
        if not prompt:
            await query.message.reply_text("⚠️ Сначала отправьте текстовый prompt.")
            return

        price = TOKEN_COSTS["mj"]
        ok_balance, rest = try_charge(ctx, price)
        if not ok_balance:
            await query.message.reply_text(
                f"💎 Недостаточно токенов: нужно {price}, на балансе {rest}.\nПополните через Stars: {STARS_BUY_URL}",
                reply_markup=stars_topup_kb(),
            )
            return

        await query.message.reply_text(
            f"🎨 Генерация фото запущена…\nФормат: *{ar}*\nPrompt: `{prompt}`",
            parse_mode=ParseMode.MARKDOWN,
        )
        ok, task_id, message = await asyncio.to_thread(mj_generate, prompt.strip(), ar)
        event("MJ_SUBMIT_RESP", ok=ok, task_id=task_id, msg=message)
        if not ok or not task_id:
            add_tokens(ctx, price)
            await query.message.reply_text(f"❌ Не удалось создать MJ-задачу: {message}\n💎 Токены возвращены.")
            return
        await query.message.reply_text(
            f"🆔 MJ taskId: `{task_id}`\n🖌️ Рисую эскиз и детали…",
            parse_mode=ParseMode.MARKDOWN,
        )
        asyncio.create_task(poll_mj_and_send_photos(update.effective_chat.id, task_id, ctx))
        return


async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    s = state(ctx)
    text = (update.message.text or "").strip()
    mode = s.get("mode")

    lower = text.lower()
    if lower.startswith(("http://", "https://")) and any(
        lower.split("?")[0].endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp", ".heic")
    ):
        if mode == "banana":
            s["last_image_url"] = text.strip()
            await update.message.reply_text(
                "🧷 Ссылка на изображение принята. Теперь пришлите фото вложением с подписью — или нажмите /banana_go для запуска."
            )
            return
        s["last_image_url"] = text.strip()
        await update.message.reply_text("🧷 Ссылка на изображение принята.")
        await show_card_veo(update, ctx)
        return

    if mode == "prompt_master":
        if len(text) > 500:
            await update.message.reply_text("ℹ️ Prompt-Master: урежу ввод до 500 символов для лучшего качества.")
        prompt = await oai_prompt_master(text[:500])
        if not prompt:
            await update.message.reply_text("⚠️ Prompt-Master недоступен или ответ пуст.")
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
            await update.message.reply_text("💬 Думаю над ответом…")
            response = await asyncio.to_thread(
                openai.ChatCompletion.create,
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful, concise assistant."},
                    {"role": "user", "content": text},
                ],
                temperature=0.5,
                max_tokens=700,
            )
            answer = response["choices"][0]["message"]["content"].strip()
            await update.message.reply_text(answer)
        except Exception as exc:
            log.exception("Chat error: %s", exc)
            await update.message.reply_text("⚠️ Ошибка запроса к ChatGPT.")
        return

    if mode == "mj_txt":
        s["last_prompt"] = text
        await update.message.reply_text(
            f"✅ Prompt сохранён:\n\n`{text}`\n\nНажмите ниже для запуска 16:9:",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=mj_start_kb(),
        )
        return

    s["last_prompt"] = text
    await update.message.reply_text(
        "🟦 *VEO — подготовка к рендеру*\nПроверь карточку ниже и жми «Сгенерировать».",
        parse_mode=ParseMode.MARKDOWN,
    )
    await show_card_veo(update, ctx)


async def _banana_run_and_send(chat_id: int, ctx: ContextTypes.DEFAULT_TYPE, src_url: str, prompt: str) -> None:
    try:
        task_id = await asyncio.to_thread(create_banana_task, prompt, [src_url], "png", "auto", None, None, 60)
        event("BANANA_SUBMIT_OK", task_id=task_id)

        await ctx.bot.send_message(chat_id, f"🍌 Задача Banana создана.\n🆔 taskId={task_id}\nЖдём результат…")
        urls = await asyncio.to_thread(wait_for_banana_result, task_id, 8 * 60, 3)

        if not urls:
            await ctx.bot.send_message(chat_id, "⚠️ Banana вернула пустой результат. 💎 Токены возвращены.")
            add_tokens(ctx, TOKEN_COSTS["banana"])
            return

        first = urls[0]
        try:
            await ctx.bot.send_photo(chat_id=chat_id, photo=first, caption="✅ Banana готово")
        except Exception:
            try:
                response = requests.get(first, timeout=180)
                response.raise_for_status()
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp.write(response.content)
                    path = tmp.name
                with open(path, "rb") as fh:
                    await ctx.bot.send_document(
                        chat_id=chat_id,
                        document=InputFile(fh, filename="banana.png"),
                        caption="✅ Banana готово",
                    )
            finally:
                try:
                    os.unlink(path)
                except Exception:
                    pass

    except KieBananaError as exc:
        add_tokens(ctx, TOKEN_COSTS["banana"])
        await ctx.bot.send_message(chat_id, f"❌ Banana ошибка: {exc}\n💎 Токены возвращены.")
    except Exception as exc:
        add_tokens(ctx, TOKEN_COSTS["banana"])
        log.exception("BANANA unexpected: %s", exc)
        await ctx.bot.send_message(chat_id, "💥 Внутренняя ошибка Banana. 💎 Токены возвращены.")


async def on_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    s = state(ctx)
    photos = update.message.photo
    if not photos:
        return
    photo = photos[-1]
    try:
        file = await ctx.bot.get_file(photo.file_id)
        if not file.file_path:
            await update.message.reply_text("⚠️ Не удалось получить путь к файлу Telegram.")
            return
        url = tg_direct_file_url(TELEGRAM_TOKEN, file.file_path)

        if s.get("mode") == "banana":
            prompt = (update.message.caption or "").strip()
            if not prompt:
                prompt = "Enhance and retouch photo in a stylish, realistic way"

            ok, rest = try_charge(ctx, TOKEN_COSTS["banana"])
            if not ok:
                await update.message.reply_text(
                    f"💎 Недостаточно токенов для Banana: нужно {TOKEN_COSTS['banana']}, на балансе {rest}.\n"
                    f"Пополните через Stars: {STARS_BUY_URL}",
                    reply_markup=stars_topup_kb(),
                )
                return

            await update.message.reply_text("🍌 Запускаю Banana…")
            asyncio.create_task(_banana_run_and_send(update.effective_chat.id, ctx, url, prompt))
            return

        s["last_image_url"] = url
        await update.message.reply_text("🖼️ Фото принято как референс.")
        await show_card_veo(update, ctx)
    except Exception as exc:
        log.exception("Get photo failed: %s", exc)
        await update.message.reply_text("⚠️ Не удалось обработать фото. Пришлите публичный URL картинки текстом.")


# ---------------------------------------------------------------------------
# Payments via Stars
# ---------------------------------------------------------------------------
async def precheckout_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        await update.pre_checkout_query.answer(ok=True)
    except Exception:
        await update.pre_checkout_query.answer(
            ok=False,
            error_message=f"Платёж отклонён. Проверьте баланс Stars или пополните в {STARS_BUY_URL}",
        )


async def successful_payment_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    payment = update.message.successful_payment
    try:
        meta = json.loads(payment.invoice_payload)
    except Exception:
        meta = {}

    stars = int(payment.total_amount)
    charge_id = getattr(payment, "telegram_payment_charge_id", None)
    if charge_id:
        LAST_STAR_CHARGE_BY_USER[update.effective_user.id] = charge_id

    if meta.get("kind") == "stars_pack":
        tokens = int(meta.get("tokens") or tokens_for_stars(stars))
        add_tokens(ctx, tokens)
        await update.message.reply_text(f"✅ Оплата получена: +{tokens} токенов.\nБаланс: {get_user_balance(ctx)} 💎")
        event("STARS_TOPUP_OK", user=update.effective_user.id, stars=stars, tokens=tokens, charge_id=charge_id)
        return

    await update.message.reply_text("✅ Оплата получена. Баланс обновлён.")


async def refund_last(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id
    charge_id = LAST_STAR_CHARGE_BY_USER.get(uid)
    if not charge_id:
        await update.message.reply_text("Нет последнего платежа для возврата.")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/refundStarPayment"
        response = requests.post(
            url,
            json={"user_id": uid, "telegram_payment_charge_id": charge_id},
            timeout=20,
        )
        data = response.json()
        if data.get("ok"):
            await update.message.reply_text("🔄 Возврат Stars инициирован. Проверьте баланс звёзд.")
            event("STARS_REFUND_OK", user=uid, charge_id=charge_id)
        else:
            await update.message.reply_text(f"⚠️ Не удалось вернуть Stars: {data.get('description') or 'ошибка'}")
            event("STARS_REFUND_FAIL", user=uid, charge_id=charge_id, resp=data)
    except Exception as exc:
        await update.message.reply_text("⚠️ Ошибка при возврате Stars.")
        event("STARS_REFUND_ERR", err=str(exc))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    if not TELEGRAM_TOKEN:
        raise RuntimeError("TELEGRAM_TOKEN is not set")
    if not KIE_BASE_URL:
        raise RuntimeError("KIE_BASE_URL is not set")
    if not KIE_API_KEY:
        raise RuntimeError("KIE_API_KEY is not set")

    application = (
        ApplicationBuilder()
        .token(TELEGRAM_TOKEN)
        .rate_limiter(AIORateLimiter())
        .build()
    )

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("health", health))
    application.add_handler(CommandHandler("topup", topup))
    application.add_handler(CommandHandler("refund_last", refund_last))
    application.add_handler(PreCheckoutQueryHandler(precheckout_callback))
    application.add_handler(MessageHandler(filters.SUCCESSFUL_PAYMENT, successful_payment_handler))
    application.add_handler(CallbackQueryHandler(on_callback))
    application.add_handler(MessageHandler(filters.PHOTO, on_photo))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    application.add_error_handler(error_handler)

    log.info("Bot starting with Banana enabled.")
    application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
