# -*- coding: utf-8 -*-
# Best VEO3 + Midjourney — PTB 20.7
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

# ---- KIE (VEO + MJ) ----
KIE_API_KEY = os.getenv("KIE_API_KEY", "").strip()
KIE_BASE_URL = os.getenv("KIE_BASE_URL", "https://api.kie.ai")

# VEO
KIE_GEN_PATH = os.getenv("KIE_GEN_PATH", "/api/v1/veo/generate")
KIE_STATUS_PATH = os.getenv("KIE_STATUS_PATH", "/api/v1/veo/record-info")
KIE_HD_PATH = os.getenv("KIE_HD_PATH", "/api/v1/veo/get-1080p-video")

# Midjourney
MJ_GEN_PATH = os.getenv("MJ_GEN_PATH", "/api/v1/mj/generate")
MJ_STATUS_PATH = os.getenv("MJ_STATUS_PATH", "/api/v1/mj/record-info")
MJ_UPSCALE_PATH = os.getenv("MJ_UPSCALE_PATH", "/api/v1/mj/generateUpscale")

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
    import telegram  # type: ignore
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


# ==========================
#   Состояние пользователя
# ==========================
DEFAULT_STATE = {
    # общий
    "mode": None,                    # 'gen_text' | 'gen_photo' | 'prompt_master' | 'chat' | 'mj'
    "last_ui_msg_id": None,
    "generating": False,
    "generation_id": None,
    "last_task_id": None,

    # VEO
    "aspect": "16:9",
    "model": "veo3_fast",            # 'veo3_fast' | 'veo3'
    "last_prompt": None,
    "last_image_url": None,          # VEO ref

    # VEO last result (для «Отправить ещё раз»)
    "last_result_url": None,
    "last_result_kind": None,        # 'video' | 'photo'

    # MJ
    "mj_prompt": None,
    "mj_aspect": "1:1",              # 1:1 | 16:9 | 9:16 | 3:4
    "mj_speed": "relaxed",           # relaxed | fast | turbo
    "mj_version": "7",
    "mj_selfie_url": None,
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
        [InlineKeyboardButton("🧑‍🎨 Фото с вашим лицом (MJ)",     callback_data="mode:mj")],
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
    # VEO формат
    return [
        InlineKeyboardButton(f"16:9 {'✅' if current=='16:9' else ''}".strip(), callback_data="aspect:16:9"),
        InlineKeyboardButton(f"9:16 {'✅' if current=='9:16' else ''}".strip(),  callback_data="aspect:9:16"),
    ]

def model_row(current: str) -> List[InlineKeyboardButton]:
    return [
        InlineKeyboardButton(f"⚡ Fast {'✅' if current=='veo3_fast' else ''}".strip(), callback_data="model:veo3_fast"),
        InlineKeyboardButton(f"💎 Quality {'✅' if current=='veo3' else ''}".strip(),   callback_data="model:veo3"),
    ]

def mj_aspect_row(current: str) -> List[List[InlineKeyboardButton]]:
    # Только 1:1 16:9 9:16 3:4
    def b(txt, key): return InlineKeyboardButton(f"{txt} {'✅' if current==key else ''}".strip(), callback_data=f"mj_aspect:{key}")
    return [
        [b("1:1", "1:1"), b("16:9", "16:9"), b("9:16", "9:16"), b("3:4", "3:4")],
    ]

def mj_speed_row(current: str) -> List[InlineKeyboardButton]:
    return [
        InlineKeyboardButton(f"🐢 relaxed {'✅' if current=='relaxed' else ''}".strip(), callback_data="mj_speed:relaxed"),
        InlineKeyboardButton(f"⚡ fast {'✅' if current=='fast' else ''}".strip(),       callback_data="mj_speed:fast"),
        InlineKeyboardButton(f"🚀 turbo {'✅' if current=='turbo' else ''}".strip(),     callback_data="mj_speed:turbo"),
    ]

def build_card_text(s: Dict[str, Any]) -> str:
    mode = s.get("mode")
    if mode == "mj":
        prompt_preview = (s.get("mj_prompt") or "").strip()
        if len(prompt_preview) > 900:
            prompt_preview = prompt_preview[:900] + "…"
        has_ref = "есть" if s.get("mj_selfie_url") else "нет"
        lines = [
            "🧑‍🎨 *MJ карточка (селфи → фото)*",
            "",
            "✍️ *Промпт:*",
            f"`{prompt_preview or '—'}`",
            "",
            "*📋 Параметры:*",
            f"• Aspect: *{s.get('mj_aspect','1:1')}*",
            f"• Speed: *{s.get('mj_speed','relaxed')}*",
            f"• Version: *{s.get('mj_version','7')}*",
            f"• Референс (селфи): *{has_ref}*",
        ]
        return "\n".join(lines)

    # VEO карточка
    prompt_preview = (s.get("last_prompt") or "").strip()
    if len(prompt_preview) > 900:
        prompt_preview = prompt_preview[:900] + "…"
    has_prompt = "есть" if s.get("last_prompt") else "нет"
    has_ref = "есть" if s.get("last_image_url") else "нет"
    model = "Fast" if s.get("model") == "veo3_fast" else "Quality"
    lines = [
        "🪄 *Карточка генерации (VEO)*",
        "",
        "✍️ *Промпт:*",
        f"`{prompt_preview or '—'}`",
        "",
        "*📋 Параметры:*",
        f"• Формат: *{s.get('aspect','16:9')}*",
        f"• Режим: *{model}*",
        f"• Промпт: *{has_prompt}*",
        f"• Референс: *{has_ref}*",
    ]
    return "\n".join(lines)

def card_keyboard(s: Dict[str, Any]) -> InlineKeyboardMarkup:
    mode = s.get("mode")
    rows: List[List[InlineKeyboardButton]] = []

    if mode == "mj":
        rows.append([InlineKeyboardButton("🖼️ Добавить/Удалить селфи", callback_data="mj:toggle_selfie"),
                     InlineKeyboardButton("✍️ Изменить промпт",           callback_data="mj:edit_prompt")])
        rows.extend(mj_aspect_row(s.get("mj_aspect","1:1")))
        rows.append(mj_speed_row(s.get("mj_speed","relaxed")))
        if s.get("mj_prompt") and s.get("mj_selfie_url"):
            rows.append([InlineKeyboardButton("🖼️ Сгенерировать фото", callback_data="mj:generate")])
        rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="back")])
        rows.append([InlineKeyboardButton("💳 Пополнить баланс", url=TOPUP_URL)])
        return InlineKeyboardMarkup(rows)

    # VEO UI
    rows.append([InlineKeyboardButton("🖼️ Добавить/Удалить фото", callback_data="card:toggle_photo"),
                 InlineKeyboardButton("✍️ Изменить промпт",       callback_data="card:edit_prompt")])
    rows.append(aspect_row(s["aspect"]))
    rows.append(model_row(s["model"]))
    if s.get("last_prompt"):
        rows.append([InlineKeyboardButton("🚀 Сгенерировать", callback_data="card:generate")])
    if s.get("last_result_url"):
        rows.append([InlineKeyboardButton("📤 Отправить ещё раз", callback_data="resend:last")])
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
#   KIE API helpers (общие)
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

def _build_payload_for_kie(prompt: str, aspect: str, image_url: Optional[str], model_key: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "prompt": prompt,
        "aspectRatio": aspect,
        "model": "veo3" if model_key == "veo3" else "veo3_fast",
        "enableFallback": aspect == "16:9",  # fallback доступен только в 16:9
    }
    if image_url:
        payload["imageUrls"] = [image_url]
    return payload

def submit_kie_generation(prompt: str, aspect: str, image_url: Optional[str], model_key: str) -> Tuple[bool, Optional[str], str]:
    url = join_url(KIE_BASE_URL, KIE_GEN_PATH)
    status, j = _post_json(url, _build_payload_for_kie(prompt, aspect, image_url, model_key))
    code = j.get("code", status)
    if status == 200 and code == 200:
        task_id = _extract_task_id(j)
        if task_id:
            return True, task_id, "Задача создана."
        return False, None, "Ответ KIE без taskId."
    return False, None, _kie_error_message(status, j)

def get_kie_task_status(task_id: str) -> Tuple[bool, Optional[int], Optional[str], Optional[str]]:
    """Возвращает (ok, success_flag, status_message, result_url)."""
    url = join_url(KIE_BASE_URL, KIE_STATUS_PATH)
    status, j = _get_json(url, {"taskId": task_id})
    code = j.get("code", status)
    if status == 200 and code == 200:
        flag, msg, data = _parse_success_flag(j)
        return True, flag, msg, _extract_result_url(data or {})
    return False, None, _kie_error_message(status, j), None


# ==========================
#   MJ helpers
# ==========================
def _mj_build_payload(prompt: str, selfie_url: str, aspect: str, speed: str, version: str) -> Dict[str, Any]:
    # mj_img2img (селфи + промпт), можно и txt2img если нет селфи
    payload = {
        "taskType": "mj_img2img" if selfie_url else "mj_txt2img",
        "prompt": prompt,
        "speed": speed,
        "version": version,
        "aspectRatio": aspect,
    }
    if selfie_url:
        payload["fileUrls"] = [selfie_url]
    return payload

def submit_mj_generation(prompt: str, selfie_url: str, aspect: str, speed: str, version: str) -> Tuple[bool, Optional[str], str]:
    url = join_url(KIE_BASE_URL, MJ_GEN_PATH)
    status, j = _post_json(url, _mj_build_payload(prompt, selfie_url, aspect, speed, version))
    code = j.get("code", status)
    if status == 200 and code == 200:
        task_id = _extract_task_id(j)
        if task_id:
            return True, task_id, "Задача создана."
        return False, None, "Ответ MJ без taskId."
    return False, None, _kie_error_message(status, j)

def get_mj_task_status(task_id: str) -> Tuple[bool, Optional[int], Optional[str], Optional[List[str]]]:
    """
    Возвращает: ok, successFlag, human_msg, result_urls
    """
    url = join_url(KIE_BASE_URL, MJ_STATUS_PATH)
    status, j = _get_json(url, {"taskId": task_id})
    code = j.get("code", status)
    if status == 200 and code == 200:
        data = j.get("data") or {}
        flag = data.get("successFlag")
        err_msg = data.get("errorMessage")
        err_code = data.get("errorCode")
        base_msg = j.get("msg") or j.get("message")
        human_msg = None
        if err_msg:
            human_msg = str(err_msg)
            if err_code:
                human_msg = f"{human_msg} (code {err_code})"
        else:
            human_msg = base_msg

        urls: List[str] = []
        info = data.get("resultInfoJson") or {}
        for item in info.get("resultUrls") or []:
            u = item.get("resultUrl")
            if u:
                urls.append(u)
        return True, flag, human_msg, urls

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
                log.info("VEO Polling cancelled — superseded by newer job.")
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
                if sent:
                    s["last_result_url"] = res_url
                    s["last_result_kind"] = "video"
                await ctx.bot.send_message(
                    chat_id,
                    "✅ Готово!",
                    reply_markup=InlineKeyboardMarkup(
                        [[InlineKeyboardButton("🚀 Сгенерировать ещё видео (VEO)", callback_data="start_new_veo")]]
                    ),
                )
                break

            if flag in (2, 3):
                await ctx.bot.send_message(chat_id, f"❌ Ошибка KIE: {msg or 'без сообщения'}")
                break

            await asyncio.sleep(POLL_INTERVAL_SECS)
    except Exception as e:
        log.exception("VEO Poller crashed: %s", e)
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

            ok, flag, human_msg, urls = await asyncio.to_thread(get_mj_task_status, task_id)
            if not ok:
                await ctx.bot.send_message(chat_id, f"❌ Ошибка статуса MJ: {human_msg or 'неизвестно'}")
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
                # отправляем 1–4 изображения
                for u in urls:
                    try:
                        await ctx.bot.send_photo(chat_id=chat_id, photo=u)
                        s["last_result_url"] = u
                        s["last_result_kind"] = "photo"
                    except Exception:
                        pass
                await ctx.bot.send_message(
                    chat_id,
                    "✅ Готово!",
                    reply_markup=InlineKeyboardMarkup(
                        [[InlineKeyboardButton("🚀 Сгенерировать ещё фото (MJ)", callback_data="start_new_mj")]]
                    ),
                )
                break

            if flag in (2, 3):
                tip = ""
                if (s.get("mj_selfie_url") or "").startswith("https://api.telegram.org/"):
                    tip = "\n💡 Похоже, ссылка на селфи из Telegram недоступна для MJ. Пришлите публичный URL (CDN/хостинг)."
                await ctx.bot.send_message(chat_id, f"❌ Ошибка MJ: {human_msg or 'неизвестно'}{tip}")
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
    # простая диагностика
    checks = []
    checks.append(f"TELEGRAM_TOKEN: {'OK' if TELEGRAM_TOKEN else 'MISSING'}")
    checks.append(f"KIE_API_KEY: {'OK' if KIE_API_KEY else 'MISSING'}")
    checks.append(f"KIE_BASE_URL: {KIE_BASE_URL}")
    checks.append(f"VEO paths: {KIE_GEN_PATH}, {KIE_STATUS_PATH}")
    checks.append(f"MJ paths: {MJ_GEN_PATH}, {MJ_STATUS_PATH}")
    try:
        import telegram as tg
        checks.append(f"PTB: {getattr(tg, '__version__', 'unknown')}")
    except Exception:
        checks.append("PTB: unknown")
    if openai and OPENAI_API_KEY:
        checks.append("OpenAI: OK")
    else:
        checks.append("OpenAI: disabled")
    await update.message.reply_text("🔎 /health\n" + "\n".join(checks))

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

    # ——— Глобальные действия
    if data == "back":
        s.update({**DEFAULT_STATE})
        await query.message.reply_text("Главное меню:", reply_markup=main_menu_kb())
        return

    if data == "resend:last":
        url = s.get("last_result_url")
        kind = s.get("last_result_kind")
        if not url:
            await query.message.reply_text("Нет последнего результата.")
            return
        if kind == "video":
            ok = await send_video_with_fallback(ctx, query.message.chat_id, url)
            await query.message.reply_text("✅ Отправлено ещё раз." if ok else "⚠️ Не удалось отправить.")
        else:
            try:
                await ctx.bot.send_photo(chat_id=query.message.chat_id, photo=url)
                await query.message.reply_text("✅ Отправлено ещё раз.")
            except Exception:
                await query.message.reply_text("⚠️ Не удалось отправить.")
        return

    if data == "start_new_veo":
        keep_aspect = s.get("aspect", "16:9")
        keep_model = s.get("model", "veo3_fast")
        s.update({**DEFAULT_STATE})
        s["mode"] = "gen_text"
        s["aspect"] = keep_aspect
        s["model"] = keep_model
        await query.message.reply_text("VEO — подготовка к рендеру\nПришлите идею/промпт.")
        await show_card(update, ctx)
        return

    if data == "start_new_mj":
        s.update({**DEFAULT_STATE})
        s["mode"] = "mj"
        await query.message.reply_text("MJ — селфи + промпт.\nПришлите селфи и текст.")
        await show_card(update, ctx)
        return

    # ——— Режимы
    if data.startswith("mode:"):
        _, mode = data.split(":", 1)
        s["mode"] = mode
        if mode == "gen_text":
            await query.message.reply_text("Режим VEO: генерация по тексту. Пришлите идею или готовый промпт.")
        elif mode == "gen_photo":
            await query.message.reply_text("Режим VEO: генерация по фото. Пришлите фото (и при желании — подпись-промпт).")
        elif mode == "prompt_master":
            await query.message.reply_text("Промпт-мастер: пришлите идею (1–2 фразы). Верну EN-кинопромпт.")
        elif mode == "chat":
            await query.message.reply_text("Обычный чат: напишите вопрос — отвечу через ChatGPT.")
        elif mode == "mj":
            await query.message.reply_text("MJ: пришлите селфи (или публичный URL) и промпт. Поддержка 1:1, 16:9, 9:16, 3:4; speeds: relaxed/fast/turbo.")
        await show_card(update, ctx)
        return

    if data == "faq":
        await query.message.reply_text(
            "FAQ:\n"
            "• VEO: Fast/Quality, 16:9 / 9:16. Видео придёт сюда.\n"
            "• MJ: селфи + промпт, аспекты 1:1 / 16:9 / 9:16 / 3:4.\n"
            "• Если ссылка Telegram не принимается MJ — загрузите фото на публичный хостинг и пришлите URL.",
            reply_markup=main_menu_kb(),
        )
        return

    # ——— VEO переключатели
    if data.startswith("aspect:"):
        _, val = data.split(":", 1)
        s["aspect"] = "9:16" if val.strip() == "9:16" else "16:9"
        await show_card(update, ctx, edit_only_markup=True)
        return

    if data.startswith("model:"):
        _, val = data.split(":", 1)
        s["model"] = "veo3" if val.strip() == "veo3" else "veo3_fast"
        await show_card(update, ctx, edit_only_markup=True)
        return

    if data == "card:toggle_photo":
        if s.get("last_image_url"):
            s["last_image_url"] = None
            await query.message.reply_text("Фото-референс удалён.")
            await show_card(update, ctx)
        else:
            await query.message.reply_text(
                "Пришлите фото. Я возьму прямую ссылку Telegram (если KIE примет)\n"
                "или пришлите публичный URL картинки текстом."
            )
        return

    if data == "card:edit_prompt":
        await query.message.reply_text("Пришлите новый текст промпта (или идею для Prompt-Мастера).")
        return

    if data == "card:reset":
        keep_aspect = s.get("aspect", "16:9")
        keep_model = s.get("model", "veo3_fast")
        s.update({**DEFAULT_STATE})
        s["mode"] = "gen_text"
        s["aspect"] = keep_aspect
        s["model"] = keep_model
        await query.message.reply_text("Карточка очищена.")
        await show_card(update, ctx)
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
        log.info("VEO Submitted: chat=%s task=%s gen=%s model=%s", update.effective_chat.id, task_id, gen_id, s.get("model"))
        await query.message.reply_text(f"🚀 Задача отправлена ({'Fast' if s.get('model')=='veo3_fast' else 'Quality'}). taskId={task_id}")
        await query.message.reply_text("⏳ Идёт рендеринг…")
        asyncio.create_task(poll_kie_and_send(update.effective_chat.id, task_id, gen_id, ctx))
        return

    # ——— MJ переключатели/кнопки
    if data.startswith("mj_aspect:"):
        _, val = data.split(":", 1)
        if val in ("1:1", "16:9", "9:16", "3:4"):
            s["mj_aspect"] = val
        await show_card(update, ctx, edit_only_markup=True)
        return

    if data.startswith("mj_speed:"):
        _, val = data.split(":", 1)
        if val in ("relaxed", "fast", "turbo"):
            s["mj_speed"] = val
        await show_card(update, ctx, edit_only_markup=True)
        return

    if data == "mj:toggle_selfie":
        if s.get("mj_selfie_url"):
            s["mj_selfie_url"] = None
            await query.message.reply_text("Селфи удалено.")
            await show_card(update, ctx)
        else:
            await query.message.reply_text(
                "Пришлите селфи. Я возьму прямую ссылку Telegram (если MJ примет)\n"
                "или пришлите *публичный URL* картинки текстом."
            )
        return

    if data == "mj:edit_prompt":
        await query.message.reply_text("Пришлите новый текст промпта для MJ.")
        return

    if data == "mj:generate":
        if not s.get("mj_prompt"):
            await query.message.reply_text("Нужен промпт для MJ.")
            return
        if not s.get("mj_selfie_url"):
            await query.message.reply_text("Нужно селфи/референс (URL).")
            return
        if s.get("generating"):
            await query.message.reply_text("⏳ Генерация уже идёт. Дождитесь завершения.")
            return

        ok, task_id, msg = await asyncio.to_thread(
            submit_mj_generation, s["mj_prompt"].strip(), s.get("mj_selfie_url",""),
            s.get("mj_aspect","1:1"), s.get("mj_speed","relaxed"), s.get("mj_version","7")
        )
        if not ok or not task_id:
            await query.message.reply_text(f"❌ Не удалось создать MJ-задачу: {msg}")
            return

        gen_id = uuid.uuid4().hex[:12]
        s["generating"] = True
        s["generation_id"] = gen_id
        log.info("MJ Submitted: chat=%s task=%s gen=%s", update.effective_chat.id, task_id, gen_id)
        await query.message.reply_text(f"🖼️ MJ задача отправлена. taskId={task_id}")
        await query.message.reply_text("⏳ Идёт рендеринг…")
        asyncio.create_task(poll_mj_and_send(update.effective_chat.id, task_id, gen_id, ctx))
        return


async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = state(ctx)
    text = (update.message.text or "").strip()

    # Публичный URL изображения?
    low = text.lower()
    if low.startswith(("http://", "https://")) and any(low.split("?")[0].endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp")):
        # Если мы в MJ — считаем это селфи, иначе — VEO референс
        if s.get("mode") == "mj":
            s["mj_selfie_url"] = text.strip()
            await update.message.reply_text("✅ Ссылка на селфи принята.")
        else:
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
        # куда положить? В VEO промпт
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

    if mode == "mj":
        # считаем это MJ промптом
        s["mj_prompt"] = text
        await update.message.reply_text(
            "🟪 *MJ — подготовка к рендеру*\n"
            "Нужны селфи и промпт.",
            parse_mode=ParseMode.MARKDOWN,
        )
        await show_card(update, ctx)
        return

    # VEO: считаем это промптом
    s["last_prompt"] = text
    await update.message.reply_text(
        "🟦 *VEO — подготовка к рендеру*\n"
        "Проверь карточку ниже и жми «Сгенерировать».",
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
        if s.get("mode") == "mj":
            s["mj_selfie_url"] = url
            await update.message.reply_text("🖼️ Селфи принято как референс.")
        else:
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
    if not (KIE_BASE_URL and KIE_GEN_PATH and KIE_STATUS_PATH and MJ_GEN_PATH and MJ_STATUS_PATH):
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
