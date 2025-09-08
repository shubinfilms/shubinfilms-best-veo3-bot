# -*- coding: utf-8 -*-
# Best VEO3 + Midjourney (MJ) Bot — PTB 20.7
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
    Bot,
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

# OpenAI (опционально для Prompt-Master/чата)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
try:
    import openai  # type: ignore
    if OPENAI_API_KEY:
        openai.api_key = OPENAI_API_KEY
except Exception:
    openai = None

# ---- KIE ----
KIE_API_KEY = os.getenv("KIE_API_KEY", "").strip()
KIE_BASE_URL = os.getenv("KIE_BASE_URL", "https://api.kie.ai").strip()

# VEO
KIE_VEO_GEN_PATH = os.getenv("KIE_GEN_PATH", "/api/v1/veo/generate")
KIE_VEO_STATUS_PATH = os.getenv("KIE_STATUS_PATH", "/api/v1/veo/record-info")

# MJ
KIE_MJ_GEN_PATH = "/api/v1/mj/generate"
KIE_MJ_STATUS_PATH = "/api/v1/mj/record-info"

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

# Версия PTB в лог
try:
    import telegram
    log.info("PTB version: %s", getattr(telegram, "__version__", "unknown"))
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
    return f"https://api.telegram.org/file/bot{bot_token}/{file_path.lstrip('/')}"


# ==========================
#   Состояние пользователя
# ==========================
DEFAULT_STATE = {
    # режимы: 'veo_text', 'veo_photo', 'mj_face', 'prompt_master', 'chat'
    "mode": None,

    # Общие параметры UI
    "last_ui_msg_id": None,

    # VEO
    "veo_aspect": "16:9",      # 16:9 | 9:16
    "veo_model": "veo3_fast",  # veo3_fast | veo3
    "veo_last_prompt": None,
    "veo_last_image_url": None,
    "veo_generating": False,
    "veo_generation_id": None,
    "veo_last_result_url": None,

    # MJ (лица/фото)
    "mj_aspect": "1:1",        # 1:1 | 16:9 | 9:16 | 3:4
    "mj_speed": "relaxed",     # relaxed | fast | turbo
    "mj_version": "7",         # '7' | '6.1' | '6' | '5.2' | '5.1' | 'niji6'
    "mj_last_prompt": None,
    "mj_last_selfie_url": None,
    "mj_generating": False,
    "mj_generation_id": None,
    "mj_last_task_id": None,
    "mj_last_images": [],      # список URL (последний успешный результат)
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

def start_menu_kb() -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton("🎬 Сгенерировать по тексту (VEO)", callback_data="mode:veo_text")],
        [InlineKeyboardButton("🖼️ Сгенерировать по фото (VEO)", callback_data="mode:veo_photo")],
        [InlineKeyboardButton("👤 Фото с вашим лицом (MJ)", callback_data="mode:mj_face")],
        [InlineKeyboardButton("🧠 Промпт-мастер (ChatGPT)", callback_data="mode:prompt_master")],
        [InlineKeyboardButton("💬 Обычный чат (ChatGPT)", callback_data="mode:chat")],
        [
            InlineKeyboardButton("❓ FAQ", callback_data="faq"),
            InlineKeyboardButton("📈 Канал с промптами", url=PROMPTS_CHANNEL_URL),
        ],
        [InlineKeyboardButton("💳 Пополнить баланс", url=TOPUP_URL)],
    ]
    return InlineKeyboardMarkup(rows)

def aspect_row_veo(current: str) -> List[InlineKeyboardButton]:
    # только 16:9 и 9:16
    if current == "9:16":
        return [InlineKeyboardButton("16:9", callback_data="veo_aspect:16:9"),
                InlineKeyboardButton("9:16 ✅", callback_data="veo_aspect:9:16")]
    return [InlineKeyboardButton("16:9 ✅", callback_data="veo_aspect:16:9"),
            InlineKeyboardButton("9:16", callback_data="veo_aspect:9:16")]

def model_row_veo(current: str) -> List[InlineKeyboardButton]:
    if current == "veo3":
        return [InlineKeyboardButton("⚡ Fast", callback_data="veo_model:veo3_fast"),
                InlineKeyboardButton("💎 Quality ✅", callback_data="veo_model:veo3")]
    return [InlineKeyboardButton("⚡ Fast ✅", callback_data="veo_model:veo3_fast"),
            InlineKeyboardButton("💎 Quality", callback_data="veo_model:veo3")]

def aspect_row_mj(current: str) -> List[InlineKeyboardButton]:
    order = ["1:1", "16:9", "9:16", "3:4"]
    row: List[InlineKeyboardButton] = []
    for r in order:
        label = f"{r} ✅" if r == current else r
        row.append(InlineKeyboardButton(label, callback_data=f"mj_aspect:{r}"))
    # разложим на две строки по 2
    return row

def speed_row_mj(current: str) -> List[InlineKeyboardButton]:
    order = [("relaxed", "🐢 relaxed"), ("fast", "⚡ fast"), ("turbo", "🚀 turbo")]
    row = []
    for key, label in order:
        row.append(InlineKeyboardButton(label + (" ✅" if key == current else ""), callback_data=f"mj_speed:{key}"))
    return row

def build_card_text_veo(s: Dict[str, Any]) -> str:
    prompt_preview = (s.get("veo_last_prompt") or "").strip()
    if len(prompt_preview) > 900:
        prompt_preview = prompt_preview[:900] + "…"
    has_prompt = "есть" if s.get("veo_last_prompt") else "нет"
    has_ref = "есть" if s.get("veo_last_image_url") else "нет"
    model = "Fast" if s.get("veo_model") == "veo3_fast" else "Quality"
    lines = [
        "🪄 *Карточка VEO*",
        "",
        "✍️ *Промпт:*",
        f"`{prompt_preview or '—'}`",
        "",
        "*📋 Параметры:*",
        f"• Aspect: *{s.get('veo_aspect','16:9')}*",
        f"• Speed: *{model}*",
        f"• Промпт: *{has_prompt}*",
        f"• Референс: *{has_ref}*",
    ]
    return "\n".join(lines)

def card_keyboard_veo(s: Dict[str, Any]) -> InlineKeyboardMarkup:
    rows: List[List[InlineKeyboardButton]] = []
    rows.append([InlineKeyboardButton("🖼️ Добавить/Удалить фото", callback_data="veo_card:toggle_photo"),
                 InlineKeyboardButton("✍️ Изменить промпт", callback_data="veo_card:edit_prompt")])
    ar = aspect_row_veo(s["veo_aspect"])
    rows.append(ar[:1] + ar[1:2])  # 2 кнопки на строке
    rows.append(model_row_veo(s["veo_model"]))
    if s.get("veo_last_prompt"):
        rows.append([InlineKeyboardButton("🚀 Сгенерировать", callback_data="veo_card:generate")])
    if s.get("veo_last_result_url"):
        rows.append([InlineKeyboardButton("🔁 Отправить ещё раз", callback_data="veo_card:resend")])
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="back")])
    rows.append([InlineKeyboardButton("💳 Пополнить баланс", url=TOPUP_URL)])
    return InlineKeyboardMarkup(rows)

def build_card_text_mj(s: Dict[str, Any]) -> str:
    prompt_preview = (s.get("mj_last_prompt") or "").strip()
    if len(prompt_preview) > 900:
        prompt_preview = prompt_preview[:900] + "…"
    has_prompt = "есть" if s.get("mj_last_prompt") else "нет"
    has_selfie = "есть" if s.get("mj_last_selfie_url") else "нет"
    lines = [
        "🪄 *Карточка MJ (фото с вашим лицом)*",
        "",
        "✍️ *Промпт:*",
        f"`{prompt_preview or '—'}`",
        "",
        "*📋 Параметры:*",
        f"• Aspect: *{s.get('mj_aspect','1:1')}*",
        f"• Speed: *{s.get('mj_speed','relaxed')}*",
        f"• Version: *{s.get('mj_version','7')}*",
        f"• Промпт: *{has_prompt}*",
        f"• Селфи: *{has_selfie}*",
    ]
    return "\n".join(lines)

def card_keyboard_mj(s: Dict[str, Any]) -> InlineKeyboardMarkup:
    rows: List[List[InlineKeyboardButton]] = []
    rows.append([InlineKeyboardButton("🖼️ Добавить/Удалить селфи", callback_data="mj_card:toggle_selfie"),
                 InlineKeyboardButton("✍️ Изменить промпт", callback_data="mj_card:edit_prompt")])
    # аспекты — 2 строки по 2
    order = ["1:1", "16:9", "9:16", "3:4"]
    first = order[:2]
    second = order[2:]
    rows.append([InlineKeyboardButton((a + (" ✅" if a == s["mj_aspect"] else "")), callback_data=f"mj_aspect:{a}") for a in first])
    rows.append([InlineKeyboardButton((a + (" ✅" if a == s["mj_aspect"] else "")), callback_data=f"mj_aspect:{a}") for a in second])
    rows.append(speed_row_mj(s["mj_speed"]))
    if s.get("mj_last_prompt") and s.get("mj_last_selfie_url"):
        rows.append([InlineKeyboardButton("🖼️ Сгенерировать фото", callback_data="mj_card:generate")])
    if s.get("mj_last_images"):
        rows.append([InlineKeyboardButton("🔁 Отправить ещё раз", callback_data="mj_card:resend")])
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
#   KIE API helpers
# ==========================
def _kie_headers() -> Dict[str, str]:
    token = KIE_API_KEY
    if token and not token.lower().startswith("bearer "):
        token = f"Bearer {token}"
    return {"Content-Type": "application/json", "Authorization": token or ""}

def _post_json(url: str, payload: Dict[str, Any], timeout: int = 40) -> Tuple[int, Dict[str, Any]]:
    r = requests.post(url, json=payload, headers=_kie_headers(), timeout=timeout)
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, {"error": r.text}

def _get_json(url: str, params: Dict[str, Any], timeout: int = 40) -> Tuple[int, Dict[str, Any]]:
    r = requests.get(url, params=params, headers=_kie_headers(), timeout=timeout)
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
    # для MJ
    ri = data.get("resultInfoJson") or {}
    out: List[str] = []
    for item in ri.get("resultUrls") or []:
        u = item.get("resultUrl") if isinstance(item, dict) else item
        if isinstance(u, str) and u.strip():
            out.append(u.strip())
    return out

def _kie_error_message(status_code: int, j: Dict[str, Any]) -> str:
    code = j.get("code", status_code)
    msg = j.get("msg") or j.get("message") or j.get("error") or ""
    if code in (401, 403):
        base = "Доступ запрещён (проверьте KIE_API_KEY/Bearer)."
    elif code == 402:
        base = "Недостаточно кредитов."
    elif code == 429:
        base = "Превышен лимит запросов (429)."
    elif code == 451:
        base = "Ошибка загрузки изображения (451)."
    elif code == 500:
        base = "Внутренняя ошибка KIE (500)."
    elif code == 422:
        base = "Запрос отклонён модерацией/валидацией параметров (422)."
    elif code == 400:
        base = "Неверный запрос (400)."
    else:
        base = f"KIE code {code}."
    return f"{base} {('Сообщение: ' + msg) if msg else ''}".strip()


# ==========================
#   Payload builders
# ==========================
def _veo_build_payload(prompt: str, aspect: str, image_url: Optional[str], model_key: str) -> Dict[str, Any]:
    return {
        "prompt": prompt,
        "aspectRatio": aspect,
        "model": "veo3" if model_key == "veo3" else "veo3_fast",
        "enableFallback": aspect == "16:9",  # fallback только для 16:9
        **({"imageUrls": [image_url]} if image_url else {}),
    }

def _mj_build_payload(prompt: str, selfie_url: Optional[str], aspect: str, speed: str, version: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "taskType": "mj_img2img" if selfie_url else "mj_txt2img",
        "prompt": prompt,
        "speed": speed,         # relaxed | fast | turbo
        "version": version,     # '7' | '6.1' | '6' | '5.2' | '5.1' | 'niji6'
        "aspectRatio": aspect,  # '1:1' | '16:9' | '9:16' | '3:4'
        # реализм/стабильность лиц:
        "stylization": 50,      # 0..1000 — меньше стилизации → реалистичнее
        "weirdness": 0,         # 0..3000 — без «странностей»
        "variety": 5,           # 0..100 — стабильнее образ
    }
    if selfie_url:
        payload["fileUrls"] = [selfie_url]
    return payload


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
async def poll_kie_veo_and_send(chat_id: int, task_id: str, gen_id: str, ctx: ContextTypes.DEFAULT_TYPE):
    s = state(ctx)
    s["veo_generating"] = True
    s["veo_generation_id"] = gen_id

    start_ts = time.time()
    try:
        while True:
            if s.get("veo_generation_id") != gen_id:
                return

            status, j = await asyncio.to_thread(_get_json, join_url(KIE_BASE_URL, KIE_VEO_STATUS_PATH), {"taskId": task_id})
            code = j.get("code", status)
            if status != 200 or code != 200:
                await ctx.bot.send_message(chat_id, f"❌ Ошибка статуса VEO: {_kie_error_message(status, j)}")
                break

            flag, msg, data = _parse_success_flag(j)
            if flag == 0:
                if (time.time() - start_ts) > POLL_TIMEOUT_SECS:
                    await ctx.bot.send_message(chat_id, "⏳ Таймаут ожидания результата VEO.")
                    break
                await asyncio.sleep(POLL_INTERVAL_SECS)
                continue

            if flag == 1:
                url = _extract_result_url(data or {}) or ""
                if not url:
                    await ctx.bot.send_message(chat_id, "⚠️ Готово, но ссылка не найдена.")
                    break
                if s.get("veo_generation_id") != gen_id:
                    return
                sent = await send_video_with_fallback(ctx, chat_id, url)
                s["veo_last_result_url"] = url if sent else None
                await ctx.bot.send_message(chat_id, "✅ *Готово!*", parse_mode=ParseMode.MARKDOWN,
                                           reply_markup=InlineKeyboardMarkup(
                                               [[InlineKeyboardButton("🚀 Сгенерировать ещё видео", callback_data="mode:veo_text")]]
                                           ))
                break

            if flag in (2, 3):
                await ctx.bot.send_message(chat_id, f"❌ Ошибка KIE VEO: {msg or 'без сообщения'}")
                break

            await asyncio.sleep(POLL_INTERVAL_SECS)
    except Exception as e:
        log.exception("VEO poller crashed: %s", e)
        try:
            await ctx.bot.send_message(chat_id, "❌ Внутренняя ошибка при опросе статуса VEO.")
        except Exception:
            pass
    finally:
        if s.get("veo_generation_id") == gen_id:
            s["veo_generating"] = False
            s["veo_generation_id"] = None


# ==========================
#   Поллинг MJ
# ==========================
async def poll_kie_mj_and_send(chat_id: int, task_id: str, gen_id: str, ctx: ContextTypes.DEFAULT_TYPE):
    s = state(ctx)
    s["mj_generating"] = True
    s["mj_generation_id"] = gen_id
    s["mj_last_task_id"] = task_id

    start_ts = time.time()
    try:
        while True:
            if s.get("mj_generation_id") != gen_id:
                return

            status, j = await asyncio.to_thread(_get_json, join_url(KIE_BASE_URL, KIE_MJ_STATUS_PATH), {"taskId": task_id})
            code = j.get("code", status)
            if status != 200 or code != 200:
                await ctx.bot.send_message(chat_id, f"❌ Ошибка статуса MJ: {_kie_error_message(status, j)}")
                break

            flag, msg, data = _parse_success_flag(j)
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
                # отправим первое изображение (потом можно все или альбомом)
                for u in urls[:4]:
                    try:
                        await ctx.bot.send_photo(chat_id=chat_id, photo=u)
                    except Exception:
                        pass
                await ctx.bot.send_message(chat_id, "✅ *Готово!*", parse_mode=ParseMode.MARKDOWN,
                                           reply_markup=InlineKeyboardMarkup(
                                               [[InlineKeyboardButton("🚀 Сгенерировать ещё фото", callback_data="mode:mj_face")]]
                                           ))
                break

            if flag in (2, 3):
                await ctx.bot.send_message(chat_id, f"❌ Ошибка KIE MJ: {msg or 'без сообщения'}")
                break

            await asyncio.sleep(POLL_INTERVAL_SECS)
    except Exception as e:
        log.exception("MJ poller crashed: %s", e)
        try:
            await ctx.bot.send_message(chat_id, "❌ Внутренняя ошибка при опросе статуса MJ.")
        except Exception:
            pass
    finally:
        if s.get("mj_generation_id") == gen_id:
            s["mj_generating"] = False
            s["mj_generation_id"] = None


# ==========================
#   Вьюхи карточек
# ==========================
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
                                                parse_mode=ParseMode.MARKDOWN, reply_markup=kb, disable_web_page_preview=True)
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

async def show_card_mj(update: Update, ctx: ContextTypes.DEFAULT_TYPE, edit_only_markup: bool = False):
    s = state(ctx)
    text = build_card_text_mj(s)
    kb = card_keyboard_mj(s)
    chat_id = update.effective_chat.id
    last_id = s.get("last_ui_msg_id")
    try:
        if last_id:
            if edit_only_markup:
                await ctx.bot.edit_message_reply_markup(chat_id, last_id, reply_markup=kb)
            else:
                await ctx.bot.edit_message_text(chat_id=chat_id, message_id=last_id, text=text,
                                                parse_mode=ParseMode.MARKDOWN, reply_markup=kb, disable_web_page_preview=True)
        else:
            m = await (update.callback_query.message.reply_text(text, parse_mode=ParseMode.MARKDOWN,
                                                               reply_markup=kb, disable_web_page_preview=True)
                       if update.callback_query else
                       update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN,
                                                 reply_markup=kb, disable_web_page_preview=True))
            s["last_ui_msg_id"] = m.message_id
    except Exception as e:
        log.warning("show_card_mj edit failed: %s", e)
        try:
            m = await ctx.bot.send_message(chat_id=chat_id, text=text, parse_mode=ParseMode.MARKDOWN,
                                           reply_markup=kb, disable_web_page_preview=True)
            s["last_ui_msg_id"] = m.message_id
        except Exception as e2:
            log.exception("show_card_mj send failed: %s", e2)


# ==========================
#   Хэндлеры
# ==========================
async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = state(ctx)
    s.clear()
    s.update({**DEFAULT_STATE})
    await update.message.reply_text(WELCOME, parse_mode=ParseMode.MARKDOWN, reply_markup=start_menu_kb())

async def health(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    parts = [
        f"PTB: {getattr(telegram, '__version__', 'unknown')}",
        f"KIE_BASE_URL: {KIE_BASE_URL}",
        f"VEO_GEN: {KIE_VEO_GEN_PATH}",
        f"VEO_STATUS: {KIE_VEO_STATUS_PATH}",
        f"MJ_GEN: {KIE_MJ_GEN_PATH}",
        f"MJ_STATUS: {KIE_MJ_STATUS_PATH}",
        f"KIE_API_KEY: {('ok '+mask_secret(KIE_API_KEY)) if KIE_API_KEY else '—'}",
        f"OPENAI: {'ok' if OPENAI_API_KEY else '—'}",
    ]
    await update.message.reply_text("🩺 *Health:*\n" + "\n".join(parts), parse_mode=ParseMode.MARKDOWN)

async def error_handler(update: Optional[Update], context: ContextTypes.DEFAULT_TYPE):
    log.exception("Unhandled error: %s", context.error)
    try:
        if update and update.effective_chat:
            await context.bot.send_message(update.effective_chat.id, "⚠️ Системная ошибка. Попробуйте ещё раз.")
    except Exception:
        pass

async def on_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    data = (query.data or "").strip()
    await query.answer()

    s = state(ctx)

    # ===== ГЛАВНОЕ МЕНЮ — выбор режимов
    if data.startswith("mode:"):
        _, mode = data.split(":", 1)
        s["mode"] = mode
        if mode == "veo_text":
            s["veo_last_image_url"] = None
            await query.message.reply_text("Режим: VEO по тексту. Пришлите идею или готовый EN-промпт.")
            await show_card_veo(update, ctx)
        elif mode == "veo_photo":
            await query.message.reply_text("Режим: VEO по фото. Пришлите фото (и при желании — подпись-промпт).")
            await show_card_veo(update, ctx)
        elif mode == "mj_face":
            await query.message.reply_text("Режим: MJ — фото с вашим лицом. Пришлите *селфи* и *текст-промпт*.", parse_mode=ParseMode.MARKDOWN)
            await show_card_mj(update, ctx)
        elif mode == "prompt_master":
            await query.message.reply_text("Промпт-мастер: пришлите идею (1–2 фразы). Верну EN-кинопромпт.")
        elif mode == "chat":
            await query.message.reply_text("Обычный чат: напишите вопрос — отвечу через ChatGPT.")
        return

    if data == "faq":
        await query.message.reply_text(
            "FAQ:\n"
            "• VEO: 16:9 / 9:16, Fast/Quality, фото-референс.\n"
            "• MJ: селфи + промпт, реалистичные лица (stylization=50, weirdness=0, variety=5).\n"
            "• Если видео/фото не приходит — смотрите логи и /health.",
            reply_markup=start_menu_kb(),
        )
        return

    if data == "back":
        s.clear(); s.update({**DEFAULT_STATE})
        await query.message.reply_text("Главное меню:", reply_markup=start_menu_kb())
        return

    # ===== VEO карточка
    if data.startswith("veo_aspect:"):
        s["veo_aspect"] = data.split(":", 1)[1]
        await show_card_veo(update, ctx, edit_only_markup=True); return

    if data.startswith("veo_model:"):
        s["veo_model"] = data.split(":", 1)[1]
        await show_card_veo(update, ctx, edit_only_markup=True); return

    if data == "veo_card:toggle_photo":
        if s.get("veo_last_image_url"):
            s["veo_last_image_url"] = None
            await query.message.reply_text("Фото-референс удалён.")
            await show_card_veo(update, ctx)
        else:
            await query.message.reply_text("Пришлите фото. Я возьму прямую ссылку Telegram (если KIE примет) "
                                           "или пришлите публичный URL картинки текстом.")
        return

    if data == "veo_card:edit_prompt":
        await query.message.reply_text("Пришлите новый EN-промпт (или идею для Prompt-Мастера)."); return

    if data == "veo_card:generate":
        if not s.get("veo_last_prompt"):
            await query.message.reply_text("Сначала укажите текст промпта.")
            return
        if s.get("veo_generating"):
            await query.message.reply_text("⏳ Генерация уже идёт. Дождитесь завершения.")
            return
        # submit
        payload = _veo_build_payload(s["veo_last_prompt"].strip(), s["veo_aspect"], s.get("veo_last_image_url"), s["veo_model"])
        status, j = await asyncio.to_thread(_post_json, join_url(KIE_BASE_URL, KIE_VEO_GEN_PATH), payload)
        ok = status == 200 and j.get("code", status) == 200
        if not ok:
            await query.message.reply_text(f"❌ Не удалось создать VEO-задачу: {_kie_error_message(status, j)}")
            return
        task_id = _extract_task_id(j)
        if not task_id:
            await query.message.reply_text("⚠️ Ответ VEO без taskId.")
            return
        gen_id = uuid.uuid4().hex[:12]
        s["veo_generating"] = True
        s["veo_generation_id"] = gen_id
        log.info("VEO submitted: chat=%s task=%s", update.effective_chat.id, task_id)
        await query.message.reply_text(f"🚀 Задача отправлена (VEO {('Fast' if s['veo_model']=='veo3_fast' else 'Quality')}). taskId={task_id}\n⌛ Идёт рендеринг…")
        asyncio.create_task(poll_kie_veo_and_send(update.effective_chat.id, task_id, gen_id, ctx))
        return

    if data == "veo_card:resend":
        url = s.get("veo_last_result_url")
        if not url:
            await query.message.reply_text("Нет последнего результата.")
            return
        sent = await send_video_with_fallback(ctx, update.effective_chat.id, url)
        await query.message.reply_text("✅ Готово!" if sent else "⚠️ Не получилось отправить видео.")
        return

    # ===== MJ карточка
    if data.startswith("mj_aspect:"):
        s["mj_aspect"] = data.split(":", 1)[1]
        await show_card_mj(update, ctx, edit_only_markup=True); return

    if data.startswith("mj_speed:"):
        s["mj_speed"] = data.split(":", 1)[1]
        await show_card_mj(update, ctx, edit_only_markup=True); return

    if data == "mj_card:toggle_selfie":
        if s.get("mj_last_selfie_url"):
            s["mj_last_selfie_url"] = None
            await query.message.reply_text("Селфи удалено.")
            await show_card_mj(update, ctx)
        else:
            await query.message.reply_text("Пришлите *селфи*. Возьму прямую ссылку Telegram (или пришлите публичный URL).", parse_mode=ParseMode.MARKDOWN)
        return

    if data == "mj_card:edit_prompt":
        await query.message.reply_text("Пришлите новый промпт для MJ (EN/RU — можно, мы не переводим автоматически).")
        return

    if data == "mj_card:generate":
        if not s.get("mj_last_prompt") or not s.get("mj_last_selfie_url"):
            await query.message.reply_text("Нужны *селфи* и *промпт*.", parse_mode=ParseMode.MARKDOWN)
            return
        if s.get("mj_generating"):
            await query.message.reply_text("⏳ Генерация уже идёт. Дождитесь завершения.")
            return
        payload = _mj_build_payload(
            s["mj_last_prompt"].strip(),
            s["mj_last_selfie_url"],
            s["mj_aspect"],
            s["mj_speed"],
            s["mj_version"],
        )
        status, j = await asyncio.to_thread(_post_json, join_url(KIE_BASE_URL, KIE_MJ_GEN_PATH), payload)
        ok = status == 200 and j.get("code", status) == 200
        if not ok:
            await query.message.reply_text(f"❌ Не удалось создать MJ-задачу: {_kie_error_message(status, j)}")
            return
        task_id = _extract_task_id(j)
        if not task_id:
            await query.message.reply_text("⚠️ Ответ MJ без taskId.")
            return
        gen_id = uuid.uuid4().hex[:12]
        s["mj_generating"] = True
        s["mj_generation_id"] = gen_id
        s["mj_last_task_id"] = task_id
        log.info("MJ submitted: chat=%s task=%s", update.effective_chat.id, task_id)
        await query.message.reply_text(f"🖼️ MJ задача отправлена. taskId={task_id}\n⌛ Идёт рендеринг…")
        asyncio.create_task(poll_kie_mj_and_send(update.effective_chat.id, task_id, gen_id, ctx))
        return

    if data == "mj_card:resend":
        imgs: List[str] = s.get("mj_last_images") or []
        if not imgs:
            await query.message.reply_text("Нет последнего результата.")
            return
        for u in imgs[:4]:
            try:
                await ctx.bot.send_photo(chat_id=update.effective_chat.id, photo=u)
            except Exception:
                pass
        await query.message.reply_text("✅ Готово!")
        return


async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = state(ctx)
    text = (update.message.text or "").strip()

    # Публичный URL изображения?
    low = text.lower()
    if low.startswith(("http://", "https://")) and low.split("?")[0].endswith((".jpg", ".jpeg", ".png", ".webp")):
        # Куда записывать — завися от режима
        if s.get("mode") == "mj_face":
            s["mj_last_selfie_url"] = text.strip()
            await update.message.reply_text("✅ Ссылка на селфи принята.")
            await show_card_mj(update, ctx)
        else:
            s["veo_last_image_url"] = text.strip()
            await update.message.reply_text("✅ Ссылка на изображение принята.")
            await show_card_veo(update, ctx)
        return

    mode = s.get("mode")
    if mode == "prompt_master":
        prompt = await oai_prompt_master(text)
        if not prompt:
            await update.message.reply_text("⚠️ Prompt-Master недоступен или ответ пуст. Попробуйте ещё раз.")
            return
        # пишем в VEO-поле, чаще всего для видео
        s["veo_last_prompt"] = prompt
        await update.message.reply_text("🧠 Готово! Промпт добавлен в карточку VEO.")
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

    # По умолчанию — считаем это промптом для активной карточки
    if mode == "mj_face":
        s["mj_last_prompt"] = text
        await update.message.reply_text("🟪 *MJ — подготовка к рендеру*\nНужны селфи и промпт.", parse_mode=ParseMode.MARKDOWN)
        await show_card_mj(update, ctx)
    else:
        s["veo_last_prompt"] = text
        await update.message.reply_text("🟦 *VEO — подготовка к рендеру*\nПроверь карточку ниже и жми «Сгенерировать».", parse_mode=ParseMode.MARKDOWN)
        await show_card_veo(update, ctx)

async def on_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = state(ctx)
    photos = update.message.photo
    if not photos:
        return
    ph = photos[-1]
    try:
        file = await ctx.bot.get_file(ph.file_id)
        if not file.file_path:
            await update.message.reply_text("⚠️ Не удалось получить путь к файлу Telegram.")
            return
        url = tg_file_direct_url(TELEGRAM_TOKEN, file.file_path)
        log.info("Photo via TG path: ...%s", mask_secret(url, show=10))
        if s.get("mode") == "mj_face":
            s["mj_last_selfie_url"] = url
            await update.message.reply_text("🖼️ Селфи принято как референс (MJ).")
            await show_card_mj(update, ctx)
        else:
            s["veo_last_image_url"] = url
            await update.message.reply_text("🖼️ Фото принято как референс (VEO).")
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
    if not (KIE_BASE_URL and KIE_VEO_GEN_PATH and KIE_VEO_STATUS_PATH):
        raise RuntimeError("KIE_* env vars are not properly set")

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
        "Bot starting. PTB=20.7 | KIE_BASE=%s VEO_GEN=%s VEO_STATUS=%s MJ_GEN=%s MJ_STATUS=%s",
        KIE_BASE_URL, KIE_VEO_GEN_PATH, KIE_VEO_STATUS_PATH, KIE_MJ_GEN_PATH, KIE_MJ_STATUS_PATH
    )
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    # если был webhook — снимите:
    # https://api.telegram.org/bot<YOUR_TOKEN>/deleteWebhook
    main()
