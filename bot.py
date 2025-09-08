# -*- coding: utf-8 -*-
# Best VEO3 + Midjourney Face Bot — PTB 20.7
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

# OpenAI (старый SDK 0.28.1 — опционально)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
try:
    import openai  # type: ignore
    if OPENAI_API_KEY:
        openai.api_key = OPENAI_API_KEY
except Exception:
    openai = None

# ---- KIE ----
KIE_API_KEY = os.getenv("KIE_API_KEY", "").strip()               # токен без/с Bearer
KIE_BASE_URL = os.getenv("KIE_BASE_URL", "https://api.kie.ai")
# VEO
KIE_GEN_PATH = os.getenv("KIE_GEN_PATH", "/api/v1/veo/generate")
KIE_STATUS_PATH = os.getenv("KIE_STATUS_PATH", "/api/v1/veo/record-info")
# MJ
MJ_GEN_PATH = "/api/v1/mj/generate"
MJ_STATUS_PATH = "/api/v1/mj/record-info"

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
        if v:
            return v
        return None
    if isinstance(value, list):
        for v in value:
            if isinstance(v, str) and v.strip():
                return v.strip()
    return None

def tg_file_direct_url(bot_token: str, file_path: str) -> str:
    return f"https://api.telegram.org/file/bot{bot_token}/{file_path.lstrip('/')}"

def _clamp_int(val: Optional[int], lo: int, hi: int, default: int) -> int:
    try:
        v = int(val if val is not None else default)
        return max(lo, min(hi, v))
    except Exception:
        return default

# ==========================
#   Состояние пользователя
# ==========================
DEFAULT_STATE = {
    # общий
    "mode": None,              # 'gen_text'|'gen_photo'|'prompt_master'|'chat'|'mj_face'
    "last_ui_msg_id": None,

    # VEO
    "aspect": "16:9",
    "model": "veo3_fast",      # 'veo3_fast' | 'veo3'
    "last_prompt": None,
    "last_image_url": None,    # референс для VEO / селфи для MJ (в режиме зависит)
    "generating": False,
    "generation_id": None,
    "last_task_id": None,
    "last_result_url": None,

    # MJ (Face/Image)
    "mj_aspect": "1:1",        # только 1:1,16:9,9:16,3:4
    "mj_speed": "fast",        # relaxed | fast | turbo
    "mj_version": "7",
    "mj_stylization": 500,     # 0..1000
    "mj_weirdness": 0,         # 0..3000
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
        [InlineKeyboardButton("🎬 Сгенерировать по тексту (VEO)", callback_data="mode:gen_text")],
        [InlineKeyboardButton("🖼️ Сгенерировать по фото (VEO)",  callback_data="mode:gen_photo")],
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
        return [InlineKeyboardButton("⚡ Fast",         callback_data="model:veo3_fast"),
                InlineKeyboardButton("💎 Quality ✅",   callback_data="model:veo3")]
    return [InlineKeyboardButton("⚡ Fast ✅", callback_data="model:veo3_fast"),
            InlineKeyboardButton("💎 Quality", callback_data="model:veo3")]

def mj_aspect_row(current: str) -> List[InlineKeyboardButton]:
    allowed = ["1:1", "16:9", "9:16", "3:4"]
    if current not in allowed:
        current = "1:1"
    return [
        InlineKeyboardButton(f"1:1{' ✅' if current=='1:1' else ''}",   callback_data="mj_aspect:1:1"),
        InlineKeyboardButton(f"16:9{' ✅' if current=='16:9' else ''}", callback_data="mj_aspect:16:9"),
        InlineKeyboardButton(f"9:16{' ✅' if current=='9:16' else ''}", callback_data="mj_aspect:9:16"),
        InlineKeyboardButton(f"3:4{' ✅' if current=='3:4' else ''}",   callback_data="mj_aspect:3:4"),
    ]

def mj_speed_row(current: str) -> List[InlineKeyboardButton]:
    return [
        InlineKeyboardButton(f"🐢 relaxed{' ✅' if current=='relaxed' else ''}", callback_data="mj_speed:relaxed"),
        InlineKeyboardButton(f"⚡ fast{' ✅' if current=='fast' else ''}",        callback_data="mj_speed:fast"),
        InlineKeyboardButton(f"🚀 turbo{' ✅' if current=='turbo' else ''}",      callback_data="mj_speed:turbo"),
    ]

def build_card_text(s: Dict[str, Any]) -> str:
    prompt_preview = (s.get("last_prompt") or "").strip()
    if len(prompt_preview) > 900:
        prompt_preview = prompt_preview[:900] + "…"
    has_prompt = "есть" if s.get("last_prompt") else "нет"
    has_ref = "есть" if s.get("last_image_url") else "нет"

    if s.get("mode") == "mj_face":
        lines = [
            "🪄 *Промпт:*",
            f"`{prompt_preview or '—'}`",
            "",
            "*📋 Параметры:*",
            f"• Aspect: *{s.get('mj_aspect','1:1')}*",
            f"• Speed: *{s.get('mj_speed','fast')}*",
            f"• Version: *{s.get('mj_version','7')}*",
            f"• Референс: *{has_ref}*",
        ]
        return "\n".join(lines)

    model = "Fast" if s.get("model") == "veo3_fast" else "Quality"
    lines = [
        "🪄 *Промпт:*",
        f"`{prompt_preview or '—'}`",
        "",
        "*📋 Параметры:*",
        f"• Aspect: *{s.get('aspect','16:9')}*",
        f"• Speed: *{model}*",
        f"• Промпт: *{has_prompt}*",
        f"• Референс: *{has_ref}*",
    ]
    return "\n".join(lines)

def card_keyboard(s: Dict[str, Any]) -> InlineKeyboardMarkup:
    rows: List[List[InlineKeyboardButton]] = []
    if s.get("mode") == "mj_face":
        rows.append([InlineKeyboardButton("🧑‍🦱 Добавить/Удалить селфи", callback_data="card:toggle_photo"),
                     InlineKeyboardButton("✍️ Изменить промпт",            callback_data="card:edit_prompt")])
        rows.append(mj_aspect_row(s.get("mj_aspect","1:1")))
        rows.append(mj_speed_row(s.get("mj_speed","fast")))
        if s.get("last_prompt") and s.get("last_image_url"):
            rows.append([InlineKeyboardButton("🖼️ Сгенерировать фото", callback_data="mj:generate")])
        rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="back")])
        rows.append([InlineKeyboardButton("💳 Пополнить баланс", url=TOPUP_URL)])
        return InlineKeyboardMarkup(rows)

    # VEO
    rows.append([InlineKeyboardButton("🖼️ Добавить/Удалить фото", callback_data="card:toggle_photo"),
                 InlineKeyboardButton("✍️ Изменить промпт",       callback_data="card:edit_prompt")])
    rows.append(aspect_row(s["aspect"]))
    rows.append(model_row(s["model"]))
    if s.get("last_prompt"):
        rows.append([InlineKeyboardButton("🚀 Сгенерировать", callback_data="card:generate")])
        if s.get("last_task_id"):
            rows.append([InlineKeyboardButton("📦 Отправить ещё раз", callback_data="card:resend")])
    rows.append([InlineKeyboardButton("🔁 Начать заново", callback_data="card:reset"),
                 InlineKeyboardButton("⬅️ Назад",         callback_data="back")])
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
#   HTTP helpers
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
        base = "Запрос отклонён модерацией / enableFallback (422)."
    else:
        base = f"KIE code {code}."
    return f"{base} {('Сообщение: ' + msg) if msg else ''}".strip()

# ==========================
#   VEO helpers
# ==========================
def _extract_result_url_from_data(data: Dict[str, Any]) -> Optional[str]:
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

def _build_payload_for_veo(prompt: str, aspect: str, image_url: Optional[str], model_key: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "prompt": prompt,
        "aspectRatio": aspect,
        "model": "veo3" if model_key == "veo3" else "veo3_fast",
        "enableFallback": aspect == "16:9",
    }
    if image_url:
        payload["imageUrls"] = [image_url]
    return payload

def submit_kie_generation(prompt: str, aspect: str, image_url: Optional[str], model_key: str) -> Tuple[bool, Optional[str], str]:
    url = join_url(KIE_BASE_URL, KIE_GEN_PATH)
    status, j = _post_json(url, _build_payload_for_veo(prompt, aspect, image_url, model_key))
    code = j.get("code", status)
    if status == 200 and code == 200:
        task_id = _extract_task_id(j)
        if task_id:
            return True, task_id, "Задача создана."
        return False, None, "Ответ KIE без taskId."
    return False, None, _kie_error_message(status, j)

def get_kie_task_status(task_id: str) -> Tuple[bool, Optional[int], Optional[str], Optional[str]]:
    url = join_url(KIE_BASE_URL, KIE_STATUS_PATH)
    status, j = _get_json(url, {"taskId": task_id})
    code = j.get("code", status)
    if status == 200 and code == 200:
        data = j.get("data") or {}
        flag = None
        for k in ("successFlag", "status", "state"):
            if k in data:
                try:
                    flag = int(data[k]); break
                except Exception:
                    pass
        msg = j.get("msg") or j.get("message")
        return True, flag, msg, _extract_result_url_from_data(data)
    return False, None, _kie_error_message(status, j), None

# ==========================
#   MJ helpers
# ==========================
def _build_payload_for_mj(prompt: str,
                          aspect: str,
                          file_url: Optional[str],
                          speed: str,
                          version: str,
                          stylization_opt: Optional[int] = None,
                          weirdness_opt: Optional[int] = None,
                          enable_translation: bool = False) -> Dict[str, Any]:
    aspect_allowed = {"1:1", "16:9", "9:16", "3:4"}
    if aspect not in aspect_allowed:
        aspect = "1:1"
    speed = speed if speed in ("relaxed", "fast", "turbo") else "fast"
    version = version if version in ("7", "6.1", "6", "5.2", "5.1", "niji6") else "7"

    stylization = _clamp_int(stylization_opt, 0, 1000, 500)
    weirdness   = _clamp_int(weirdness_opt,   0, 3000, 0)

    payload: Dict[str, Any] = {
        "taskType": "mj_txt2img" if not file_url else "mj_img2img",
        "prompt": prompt,
        "speed": speed,
        "aspectRatio": aspect,
        "version": version,
        "stylization": stylization,
        "weirdness": weirdness,
        "enableTranslation": bool(enable_translation),
    }
    if file_url:
        payload["fileUrls"] = [file_url]
    return payload

def submit_mj_generation(prompt: str,
                         aspect: str,
                         selfie_url: Optional[str],
                         speed: str,
                         version: str,
                         stylization_opt: Optional[int],
                         weirdness_opt: Optional[int]) -> Tuple[bool, Optional[str], str]:
    url = join_url(KIE_BASE_URL, MJ_GEN_PATH)
    payload = _build_payload_for_mj(
        prompt=prompt,
        aspect=aspect,
        file_url=selfie_url,
        speed=speed,
        version=version,
        stylization_opt=stylization_opt,
        weirdness_opt=weirdness_opt,
        enable_translation=False,
    )
    status, j = _post_json(url, payload)
    code = j.get("code", status)
    if status == 200 and code == 200:
        task_id = _extract_task_id(j)
        if task_id:
            return True, task_id, "Задача создана."
        return False, None, "Ответ MJ без taskId."
    return False, None, _kie_error_message(status, j)

def get_mj_task_status(task_id: str) -> Tuple[bool, Optional[int], Optional[str], Optional[List[str]]]:
    url = join_url(KIE_BASE_URL, MJ_STATUS_PATH)
    status, j = _get_json(url, {"taskId": task_id})
    code = j.get("code", status)
    if status == 200 and code == 200:
        data = j.get("data") or {}
        flag = data.get("successFlag")
        msg = j.get("msg") or j.get("message")
        res = []
        info = data.get("resultInfoJson") or {}
        for item in info.get("resultUrls") or []:
            url = item.get("resultUrl")
            if url:
                res.append(url)
        return True, flag, msg, res
    return False, None, _kie_error_message(status, j), None

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
async def poll_kie_and_send(chat_id: int, task_id: str, gen_id: str, ctx: ContextTypes.DEFAULT_TYPE):
    s = state(ctx)
    s["generating"] = True
    s["generation_id"] = gen_id
    s["last_task_id"] = task_id

    start_ts = time.time()
    log.info("VEO Polling start: chat=%s task=%s gen=%s", chat_id, task_id, gen_id)

    try:
        while True:
            if s.get("generation_id") != gen_id:
                log.info("Polling cancelled — superseded by newer job.")
                return

            ok, flag, msg, res_url = await asyncio.to_thread(get_kie_task_status, task_id)
            if not ok:
                await ctx.bot.send_message(chat_id, f"❌ Ошибка статуса: {msg or 'неизвестно'}")
                break

            if flag == 0:
                if (time.time() - start_ts) > POLL_TIMEOUT_SECS:
                    await ctx.bot.send_message(chat_id, "⏳ Таймаут ожидания результата. Попробуйте позже.")
                    break
                await asyncio.sleep(POLL_INTERVAL_SECS)
                continue

            if flag == 1:
                if not res_url:
                    await ctx.bot.send_message(chat_id, "⚠️ Готово, но ссылка не найдена (ответ KIE без URL).")
                    break
                if s.get("generation_id") != gen_id:
                    log.info("Ready but superseded — skip send.")
                    return
                sent = await send_video_with_fallback(ctx, chat_id, res_url)
                s["last_result_url"] = res_url
                await ctx.bot.send_message(chat_id, "✅ Готово!" if sent else "⚠️ Не получилось отправить видео.")
                await ctx.bot.send_message(chat_id, "🚀 Сгенерировать ещё видео", reply_markup=InlineKeyboardMarkup(
                    [[InlineKeyboardButton("🚀 Сгенерировать ещё видео", callback_data="start_new")]]
                ))
                break

            if flag in (2, 3):
                await ctx.bot.send_message(chat_id, f"❌ Ошибка KIE: {msg or 'без сообщения'}")
                break

            await asyncio.sleep(POLL_INTERVAL_SECS)
    except Exception as e:
        log.exception("Poller crashed: %s", e)
        try:
            await ctx.bot.send_message(chat_id, "❌ Внутренняя ошибка при опросе статуса.")
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
    s["generating"] = True
    s["generation_id"] = gen_id
    s["last_task_id"] = task_id

    start_ts = time.time()
    log.info("MJ Polling start: chat=%s task=%s gen=%s", chat_id, task_id, gen_id)

    try:
        while True:
            if s.get("generation_id") != gen_id:
                log.info("MJ Poll cancelled — superseded.")
                return

            ok, flag, msg, urls = await asyncio.to_thread(get_mj_task_status, task_id)
            if not ok:
                await ctx.bot.send_message(chat_id, f"❌ Ошибка статуса MJ: {msg or 'неизвестно'}")
                break

            if flag == 0:
                if (time.time() - start_ts) > POLL_TIMEOUT_SECS:
                    await ctx.bot.send_message(chat_id, "⏳ Таймаут ожидания MJ. Попробуйте позже.")
                    break
                await asyncio.sleep(POLL_INTERVAL_SECS)
                continue

            if flag == 1:
                if not urls:
                    await ctx.bot.send_message(chat_id, "⚠️ Готово, но ссылки не найдены.")
                    break
                # отправим все изображения
                for u in urls:
                    try:
                        await ctx.bot.send_photo(chat_id=chat_id, photo=u)
                    except Exception:
                        pass
                await ctx.bot.send_message(chat_id, "✅ Готово!")
                await ctx.bot.send_message(chat_id, "🚀 Сгенерировать ещё фото (MJ)", reply_markup=InlineKeyboardMarkup(
                    [[InlineKeyboardButton("🚀 Сгенерировать ещё фото (MJ)", callback_data="start_new_mj")]]
                ))
                break

            if flag in (2, 3):
                await ctx.bot.send_message(chat_id, f"❌ Ошибка MJ: {msg or 'без сообщения'}")
                break

            await asyncio.sleep(POLL_INTERVAL_SECS)
    except Exception as e:
        log.exception("MJ Poller crashed: %s", e)
        try:
            await ctx.bot.send_message(chat_id, "❌ Внутренняя ошибка при опросе MJ статуса.")
        except Exception:
            pass
    finally:
        if s.get("generation_id") == gen_id:
            s["generating"] = False
            s["generation_id"] = None

# ==========================
#   Хэндлеры
# ==========================
async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = state(ctx)
    s.update({**DEFAULT_STATE})
    await update.message.reply_text(WELCOME, parse_mode=ParseMode.MARKDOWN, reply_markup=main_menu_kb())

async def health(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    parts = [
        f"PTB: {getattr(telegram, '__version__', 'unknown')}",
        f"KIE_BASE_URL: {KIE_BASE_URL}",
        f"KIE key: {('ok ' + mask_secret(KIE_API_KEY)) if KIE_API_KEY else '—'}",
        f"OPENAI key: {('ok ' + mask_secret(OPENAI_API_KEY)) if OPENAI_API_KEY else '—'}",
    ]
    await update.message.reply_text("🩺 /health\n" + "\n".join(parts))

async def error_handler(update: Optional[Update], context: ContextTypes.DEFAULT_TYPE):
    log.exception("Unhandled error: %s", context.error)
    try:
        if update and update.effective_chat:
            await context.bot.send_message(update.effective_chat.id, "⚠️ Системная ошибка. Попробуйте ещё раз.")
    except Exception:
        pass

async def show_card(update: Update, ctx: ContextTypes.DEFAULT_TYPE, edit_only_markup: bool = False):
    s = state(ctx)
    text = build_card_text(s)
    kb = card_keyboard(s)
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
        log.warning("show_card edit failed: %s", e)
        try:
            m = await ctx.bot.send_message(chat_id=chat_id, text=text, parse_mode=ParseMode.MARKDOWN,
                                           reply_markup=kb, disable_web_page_preview=True)
            s["last_ui_msg_id"] = m.message_id
        except Exception as e2:
            log.exception("show_card send failed: %s", e2)

async def on_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    data = (query.data or "").strip()
    await query.answer()
    s = state(ctx)

    # Старт нового круга после готовности
    if data == "start_new":
        keep_aspect = s.get("aspect", "16:9")
        keep_model = s.get("model", "veo3_fast")
        s.update({**DEFAULT_STATE})
        s["aspect"] = keep_aspect
        s["model"] = keep_model
        s["mode"] = "gen_text"
        await query.message.reply_text("Новый цикл VEO. Пришлите идею или готовый промпт.")
        await show_card(update, ctx)
        return

    if data == "start_new_mj":
        s.update({**DEFAULT_STATE})
        s["mode"] = "mj_face"
        await query.message.reply_text("Новый цикл MJ. Пришлите селфи и промпт.")
        await show_card(update, ctx)
        return

    # Переключение аспектов VEO
    if data.startswith("aspect:"):
        _, val = data.split(":", 1)
        s["aspect"] = "9:16" if val.strip() == "9:16" else "16:9"
        await show_card(update, ctx, edit_only_markup=True)
        return

    # Переключение модели VEO
    if data.startswith("model:"):
        _, val = data.split(":", 1)
        s["model"] = "veo3" if val.strip() == "veo3" else "veo3_fast"
        await show_card(update, ctx, edit_only_markup=True)
        return

    # Переключения MJ
    if data.startswith("mj_aspect:"):
        _, val = data.split(":", 1)
        s["mj_aspect"] = val if val in ("1:1","16:9","9:16","3:4") else "1:1"
        await show_card(update, ctx, edit_only_markup=True)
        return

    if data.startswith("mj_speed:"):
        _, val = data.split(":", 1)
        s["mj_speed"] = val if val in ("relaxed","fast","turbo") else "fast"
        await show_card(update, ctx, edit_only_markup=True)
        return

    # Режимы
    if data.startswith("mode:"):
        _, mode = data.split(":", 1)
        s["mode"] = mode
        if mode == "gen_text":
            await query.message.reply_text("Режим: VEO — генерация по тексту. Пришлите идею или готовый промпт.")
        elif mode == "gen_photo":
            await query.message.reply_text("Режим: VEO — генерация по фото. Пришлите фото (и при желании — подпись-промпт).")
        elif mode == "mj_face":
            await query.message.reply_text(
                "Режим: Midjourney — фото с вашим лицом.\n"
                "1) Пришлите селфи (или ссылку на фото)\n"
                "2) Пришлите короткий промпт\n"
                "3) Проверьте параметры и жмите «Сгенерировать фото»"
            )
        elif mode == "prompt_master":
            await query.message.reply_text("Промпт-мастер: пришлите идею (1–2 фразы). Верну EN-кинопромпт.")
        elif mode == "chat":
            await query.message.reply_text("Обычный чат: напишите вопрос — отвечу через ChatGPT.")
        await show_card(update, ctx)
        return

    if data == "faq":
        await query.message.reply_text(
            "FAQ:\n"
            "• VEO: Fast/Quality, 16:9 / 9:16. Видео придёт сюда.\n"
            "• MJ: Фото по селфи. Форматы 1:1 / 16:9 / 9:16 / 3:4. Stylization 0–1000.\n"
            "• Ошибки 401/403/422/451/500 — проверь ключ и права KIE.",
            reply_markup=main_menu_kb(),
        )
        return

    if data == "back":
        s.update({**DEFAULT_STATE})
        await query.message.reply_text("Главное меню:", reply_markup=main_menu_kb())
        return

    if data == "card:toggle_photo":
        if s.get("last_image_url"):
            s["last_image_url"] = None
            await query.message.reply_text("Фото-референс удалён.")
            await show_card(update, ctx)
        else:
            if s.get("mode") == "mj_face":
                await query.message.reply_text("Пришлите селфи. Я возьму прямую ссылку Telegram (если KIE примет) или пришлите публичный URL.")
            else:
                await query.message.reply_text("Пришлите фото. Я возьму прямую ссылку Telegram (если KIE примет) или пришлите публичный URL.")
        return

    if data == "card:edit_prompt":
        await query.message.reply_text("Пришлите новый текст промпта (или идею для Prompt-Мастера).")
        return

    if data == "card:reset":
        keep = {
            "aspect": s.get("aspect","16:9"),
            "model": s.get("model","veo3_fast"),
            "mj_aspect": s.get("mj_aspect","1:1"),
            "mj_speed": s.get("mj_speed","fast"),
            "mj_version": s.get("mj_version","7"),
            "mode": s.get("mode"),
        }
        s.update({**DEFAULT_STATE})
        s.update(keep)
        await query.message.reply_text("Карточка очищена.")
        await show_card(update, ctx)
        return

    if data == "card:resend":
        if not s.get("last_prompt"):
            await query.message.reply_text("Сначала укажите текст промпта.")
            return
        if s.get("generating"):
            await query.message.reply_text("⏳ Генерация уже идёт. Дождитесь завершения.")
            return
        ok, task_id, msg = await asyncio.to_thread(
            submit_kie_generation, s["last_prompt"].strip(), s.get("aspect", "16:9"),
            s.get("last_image_url"), s.get("model", "veo3_fast")
        )
        if not ok or not task_id:
            await query.message.reply_text(f"❌ Не удалось создать задачу: {msg}")
            return
        gen_id = uuid.uuid4().hex[:12]
        s["generating"] = True
        s["generation_id"] = gen_id
        s["last_task_id"] = task_id
        log.info("Re-Submitted: chat=%s task=%s gen=%s", update.effective_chat.id, task_id, gen_id)
        await query.message.reply_text(f"🚀 Задача отправлена (Fast/Quality). taskId={task_id}\n⏳ Идёт рендеринг…")
        asyncio.create_task(poll_kie_and_send(update.effective_chat.id, task_id, gen_id, ctx))
        return

    if data == "card:generate":
        if not s.get("last_prompt"):
            await query.message.reply_text("Сначала укажите текст промпта.")
            return
        if s.get("generating"):
            await query.message.reply_text("⏳ Генерация уже идёт. Дождитесь завершения.")
            return

        ok, task_id, msg = await asyncio.to_thread(
            submit_kie_generation, s["last_prompt"].strip(), s.get("aspect", "16:9"),
            s.get("last_image_url"), s.get("model", "veo3_fast")
        )
        if not ok or not task_id:
            await query.message.reply_text(f"❌ Не удалось создать задачу: {msg}")
            return

        gen_id = uuid.uuid4().hex[:12]
        s["generating"] = True
        s["generation_id"] = gen_id
        s["last_task_id"] = task_id
        log.info("Submitted: chat=%s task=%s gen=%s model=%s", update.effective_chat.id, task_id, gen_id, s.get("model"))
        await query.message.reply_text(f"🚀 Задача отправлена ({'Fast' if s.get('model')=='veo3_fast' else 'Quality'}). taskId={task_id}\n⏳ Идёт рендеринг…")

        asyncio.create_task(poll_kie_and_send(update.effective_chat.id, task_id, gen_id, ctx))
        return

    if data == "mj:generate":
        if not s.get("last_image_url") or not s.get("last_prompt"):
            await query.message.reply_text("Нужны селфи и промпт.")
            return
        if s.get("generating"):
            await query.message.reply_text("⏳ Генерация уже идёт. Дождитесь завершения.")
            return

        ok, task_id, msg = await asyncio.to_thread(
            submit_mj_generation,
            s.get("last_prompt") or "",
            s.get("mj_aspect","1:1"),
            s.get("last_image_url"),
            s.get("mj_speed","fast"),
            s.get("mj_version","7"),
            s.get("mj_stylization",500),
            s.get("mj_weirdness",0),
        )
        if not ok or not task_id:
            await query.message.reply_text(f"❌ Не удалось создать MJ-задачу: {msg}")
            return

        gen_id = uuid.uuid4().hex[:12]
        s["generating"] = True
        s["generation_id"] = gen_id
        s["last_task_id"] = task_id
        log.info("MJ Submitted: chat=%s task=%s gen=%s", update.effective_chat.id, task_id, gen_id)
        await query.message.reply_text(f"🖼️ MJ задача отправлена. taskId={task_id}\n⏳ Идёт рендеринг…")

        asyncio.create_task(poll_mj_and_send(update.effective_chat.id, task_id, gen_id, ctx))
        return

async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = state(ctx)
    text = (update.message.text or "").strip()

    # Публичный URL изображения?
    low = text.lower()
    if low.startswith(("http://", "https://")) and low.split("?")[0].endswith((".jpg", ".jpeg", ".png", ".webp")):
        s["last_image_url"] = text.strip()
        await update.message.reply_text("✅ Ссылка на изображение принята.")
        await show_card(update, ctx)
        return

    mode = s.get("mode")
    if mode == "prompt_master":
        prompt = await oai_prompt_master(text)
        if not prompt:
            await update.message.reply_text("⚠️ Prompt-Master недоступен или ответ пуст. Попробуйте ещё раз.")
            return
        s["last_prompt"] = prompt
        await update.message.reply_text("🧠 Готово! Промпт добавлен в карточку.")
        await show_card(update, ctx)
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

    # По умолчанию — считаем это промптом для текущего режима
    s["last_prompt"] = text
    if mode == "mj_face" and not s.get("last_image_url"):
        await update.message.reply_text("🟪 *Подготовка к рендеру (MJ)*\nНужны селфи и промпт.", parse_mode=ParseMode.MARKDOWN)
    else:
        await update.message.reply_text(
            "🟦 *VEO — подготовка к рендеру*\nПроверь карточку ниже и жми «Сгенерировать».",
            parse_mode=ParseMode.MARKDOWN,
        )
    await show_card(update, ctx)

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
        url = tg_file_direct_url(TELEGRAM_TOKEN, file_path)
        log.info("Photo via TG path: ...%s", mask_secret(url, show=10))
        s["last_image_url"] = url
        await update.message.reply_text("🖼️ Фото принято как референс.")
        await show_card(update, ctx)
    except Exception as e:
        log.exception("Get photo failed: %s", e)
        await update.message.reply_text("⚠️ Не удалось обработать фото. Пришлите публичный URL картинки текстом.")

# ==========================
#   Entry point
# ==========================
def main():
    if not TELEGRAM_TOKEN:
        raise RuntimeError("TELEGRAM_TOKEN is not set")
    if not (KIE_BASE_URL and KIE_GEN_PATH and KIE_STATUS_PATH):
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
        KIE_BASE_URL, KIE_GEN_PATH, KIE_STATUS_PATH, MJ_GEN_PATH, MJ_STATUS_PATH
    )
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    # Если когда-то был webhook — снимите:
    # https://api.telegram.org/bot<YOUR_TOKEN>/deleteWebhook
    main()
