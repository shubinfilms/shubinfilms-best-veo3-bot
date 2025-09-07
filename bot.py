# -*- coding: utf-8 -*-
# Best VEO3 Bot — PTB 20.7 + KIE Veo3 + Prompt-Master
# Версия: 2025-09-07

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

# OpenAI (старый SDK 0.28.1, опционально)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
try:
    import openai  # type: ignore
    if OPENAI_API_KEY:
        openai.api_key = OPENAI_API_KEY
except Exception:
    openai = None  # не валимся, просто отключим Prompt-Master/чат

# ---- KIE ----
KIE_API_KEY = os.getenv("KIE_API_KEY", "").strip()               # Токен
KIE_BASE_URL = os.getenv("KIE_BASE_URL", "https://api.kie.ai")   # https://api.kie.ai
KIE_GEN_PATH = os.getenv("KIE_GEN_PATH", "/api/v1/veo/generate") # POST генерация
# статус: Get Veo3 Video Details (OpenAPI выше)
KIE_STATUS_PATH = os.getenv("KIE_STATUS_PATH", "/api/v1/veo/record-info")
# 1080p video endpoint
KIE_HD_PATH = os.getenv("KIE_HD_PATH", "/api/v1/veo/get-1080p-video")

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
        return value.strip() or None
    if isinstance(value, list):
        for v in value:
            if isinstance(v, str) and v.strip():
                return v.strip()
    return None


def tg_file_direct_url(bot_token: str, file_path: str) -> str:
    # В URL присутствует токен — не логируем целиком
    return f"https://api.telegram.org/file/bot{bot_token}/{file_path.lstrip('/')}"


# ==========================
#   Состояние пользователя
# ==========================
DEFAULT_STATE = {
    "mode": None,              # 'gen_text' | 'gen_photo' | 'prompt_master' | 'chat'
    "aspect": "16:9",
    "last_prompt": None,
    "last_image_url": None,
    "generating": False,
    "generation_id": None,
    "last_ui_msg_id": None,
}

def state(ctx: ContextTypes.DEFAULT_TYPE) -> Dict[str, Any]:
    ud = ctx.user_data
    for k, v in DEFAULT_STATE.items():
        ud.setdefault(k, v)
    return ud


# ==========================
#   Кнопки / UI
# ==========================
def main_menu_kb() -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton("🎬 Сгенерировать видео по тексту", callback_data="mode:gen_text")],
        [InlineKeyboardButton("🖼️ Сгенерировать видео по фото", callback_data="mode:gen_photo")],
        [InlineKeyboardButton("🧠 Промпт-мастер (ChatGPT)", callback_data="mode:prompt_master")],
        [InlineKeyboardButton("💬 Обычный чат (ChatGPT)", callback_data="mode:chat")],
        [
            InlineKeyboardButton("❓ FAQ", callback_data="faq"),
            InlineKeyboardButton("📈 Канал с промптами", url="https://t.me/bestveo3promts"),
        ],
    ]
    rows.append([InlineKeyboardButton("16:9 ✅", callback_data="aspect:16:9"),
                 InlineKeyboardButton("9:16",   callback_data="aspect:9:16")])
    return InlineKeyboardMarkup(rows)

def aspect_row(current: str) -> List[InlineKeyboardButton]:
    if current == "9:16":
        return [InlineKeyboardButton("16:9", callback_data="aspect:16:9"),
                InlineKeyboardButton("9:16 ✅", callback_data="aspect:9:16")]
    return [InlineKeyboardButton("16:9 ✅", callback_data="aspect:16:9"),
            InlineKeyboardButton("9:16", callback_data="aspect:9:16")]

def card_keyboard(s: Dict[str, Any]) -> InlineKeyboardMarkup:
    rows: List[List[InlineKeyboardButton]] = []
    rows.append([InlineKeyboardButton("🖼️ Добавить/Удалить фото", callback_data="card:toggle_photo"),
                 InlineKeyboardButton("✍️ Изменить промпт", callback_data="card:edit_prompt")])
    rows.append(aspect_row(s["aspect"]))
    if s.get("last_prompt"):
        rows.append([InlineKeyboardButton("🚀 Сгенерировать", callback_data="card:generate")])
    rows.append([InlineKeyboardButton("🔁 Начать заново", callback_data="card:reset"),
                 InlineKeyboardButton("⬅️ Назад", callback_data="back")])
    return InlineKeyboardMarkup(rows)

def build_card_text(s: Dict[str, Any]) -> str:
    prompt_preview = (s.get("last_prompt") or "").strip()
    if len(prompt_preview) > 900:
        prompt_preview = prompt_preview[:900] + "…"
    has_prompt = "есть" if s.get("last_prompt") else "нет"
    has_ref = "есть" if s.get("last_image_url") else "нет"
    lines = [
        "🎛️ *Карточка генерации*",
        "",
        "✍️ *Промпт:*",
        f"`{prompt_preview or '—'}`",
        "",
        "*📋 Параметры:*",
        f"• Формат: *{s.get('aspect','16:9')}*",
        "• Режим: *Fast*",
        f"• Промпт: *{has_prompt}*",
        f"• Референс: *{has_ref}*",
    ]
    return "\n".join(lines)


# ==========================
#   Prompt-Master / Chat
# ==========================
async def oai_prompt_master(idea_text: str) -> Optional[str]:
    if openai is None or not OPENAI_API_KEY:
        return None
    system = (
        "You are Prompt-Master for cinematic AI video generation. "
        "Output exactly ONE prompt in English, 500–900 characters, no meta, no brand names or logos. "
        "Include lens/optics (mm/anamorphic), camera movement, lighting/palette/atmosphere, "
        "micro-details (dust, steam), subtle audio cues. Optionally one short hero line in quotes. "
        "No lists. No prefaces. Just the prompt."
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
    # По OpenAPI требуется Bearer
    token = KIE_API_KEY
    if token and not token.lower().startswith("bearer "):
        token = f"Bearer {token}"
    return {"Content-Type": "application/json", "Authorization": token or ""}

def _post_json(url: str, payload: Dict[str, Any], timeout: int = 40) -> Tuple[int, Dict[str, Any]]:
    r = requests.post(url, data=json.dumps(payload), headers=_kie_headers(), timeout=timeout)
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

def _kie_error_message(status_code: int, j: Dict[str, Any]) -> str:
    code = j.get("code", status_code)
    msg = j.get("msg") or j.get("message") or j.get("error") or ""
    if code in (401, 403):
        base = "Доступ запрещён (проверь KIE_API_KEY / Bearer)."
    elif code == 451:
        base = "Ошибка загрузки изображения (451)."
    elif code == 429:
        base = "Превышен лимит запросов (429)."
    elif code == 500:
        base = "Внутренняя ошибка сервера KIE (500)."
    else:
        base = f"KIE code {code}."
    return f"{base} {('Сообщение: ' + msg) if msg else ''}".strip()

def _build_payload_for_kie(prompt: str, aspect: str, image_url: Optional[str]) -> Dict[str, Any]:
    payload = {
        "prompt": prompt,
        "aspectRatio": aspect,    # в их доках aspectRatio
        "model": "veo3_fast",
    }
    if image_url:
        payload["imageUrl"] = image_url
    return payload

def submit_kie_generation(prompt: str, aspect: str, image_url: Optional[str]) -> Tuple[bool, Optional[str], str]:
    url = join_url(KIE_BASE_URL, KIE_GEN_PATH)
    status, j = _post_json(url, _build_payload_for_kie(prompt, aspect, image_url))
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
    # В их OpenAPI статус — GET c query taskId
    status, j = _get_json(url, {"taskId": task_id})
    code = j.get("code", status)
    if status == 200 and code == 200:
        flag, msg, data = _parse_success_flag(j)
        return True, flag, msg, _extract_result_url(data or {})
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

    # Скачиваем и отправляем файлом
    tmp_path = None
    try:
        r = requests.get(url, stream=True, timeout=120)
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
#   Поллинг KIE
# ==========================
async def poll_kie_and_send(chat_id: int, task_id: str, gen_id: str, ctx: ContextTypes.DEFAULT_TYPE):
    s = state(ctx)
    s["generating"] = True
    s["generation_id"] = gen_id

    start_ts = time.time()
    log.info("Polling start: chat=%s task=%s gen=%s", chat_id, task_id, gen_id)

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
                await ctx.bot.send_message(chat_id, "✅ Готово! (veo3_fast)" if sent else "⚠️ Не получилось отправить видео.")
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
#   Хэндлеры
# ==========================
async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = state(ctx)
    s.update({**DEFAULT_STATE})
    s["aspect"] = "16:9"
    await update.message.reply_text(
        "Привет! Я — Best VEO3 bot. Сгенерируем видео через Veo3/KIE.\nВыберите режим:",
        reply_markup=main_menu_kb(),
    )

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

    if data.startswith("aspect:"):
        _, val = data.split(":", 1)
        s["aspect"] = "9:16" if val.strip() == "9:16" else "16:9"
        await show_card(update, ctx, edit_only_markup=True)
        return

    if data.startswith("mode:"):
        _, mode = data.split(":", 1)
        s["mode"] = mode
        if mode == "gen_text":
            await query.message.reply_text("Режим: генерация по тексту. Пришлите идею или готовый промпт.")
        elif mode == "gen_photo":
            await query.message.reply_text("Режим: генерация по фото. Пришлите фото (и при желании — подпись-промпт).")
        elif mode == "prompt_master":
            await query.message.reply_text("Промпт-мастер: пришлите идею в 1–2 фразах. Верну кинопромпт (EN).")
        elif mode == "chat":
            await query.message.reply_text("Обычный чат: напишите вопрос — отвечу через ChatGPT.")
        await show_card(update, ctx)
        return

    if data == "faq":
        await query.message.reply_text(
            "FAQ:\n• Рендер Veo3 Fast.\n• Формат меняется кнопками 16:9 / 9:16.\n"
            "• Видео придёт сюда по готовности. Если ссылка не стримится — пришлю файлом.\n"
            "• Ошибки 401/403/451/500 — проверь ключи и права KIE.",
            reply_markup=main_menu_kb(),
        )
        return

    if data == "back":
        s.update({**DEFAULT_STATE})
        s["aspect"] = "16:9"
        await query.message.reply_text("Главное меню:", reply_markup=main_menu_kb())
        return

    if data == "card:toggle_photo":
        if s.get("last_image_url"):
            s["last_image_url"] = None
            await query.message.reply_text("Фото-референс удалён.")
            await show_card(update, ctx)
        else:
            await query.message.reply_text(
                "Пришлите фото. Я возьму прямую ссылку Telegram (если KIE примет).\n"
                "Или пришлите публичный URL изображения текстом."
            )
        return

    if data == "card:edit_prompt":
        await query.message.reply_text("Пришлите новый текст промпта (или идею для Prompt-Master).")
        return

    if data == "card:reset":
        keep_aspect = s.get("aspect", "16:9")
        s.update({**DEFAULT_STATE})
        s["aspect"] = keep_aspect
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
            submit_kie_generation, s["last_prompt"].strip(), s.get("aspect", "16:9"), s.get("last_image_url")
        )
        if not ok or not task_id:
            await query.message.reply_text(f"❌ Не удалось создать задачу: {msg}")
            return

        gen_id = uuid.uuid4().hex[:12]
        s["generating"] = True
        s["generation_id"] = gen_id
        log.info("Submitted: chat=%s task=%s gen=%s", update.effective_chat.id, task_id, gen_id)
        await query.message.reply_text(f"🚀 Задача отправлена (veo3_fast). taskId={task_id}")

        asyncio.create_task(poll_kie_and_send(update.effective_chat.id, task_id, gen_id, ctx))
        return

async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = state(ctx)
    text = (update.message.text or "").trim() if hasattr(str, "trim") else (update.message.text or "").strip()

    # Публичный URL изображения
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

    # По умолчанию — считаем это промптом для генерации
    s["last_prompt"] = text
    await update.message.reply_text("✍️ Промпт обновлён.")
    await show_card(update, ctx)

async def on_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = state(ctx)
    photos = update.message.photo
    if not photos:
        return
    ph = photos[-1]
    try:
        file = await ctx.bot.get_file(ph.file_id)
        file_path = file.file_path  # photos/file_123.jpg
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
    app.add_handler(CallbackQueryHandler(on_callback))
    app.add_handler(MessageHandler(filters.PHOTO, on_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_error_handler(error_handler)

    log.info(
        "Bot starting. PTB=20.7 | KIE_BASE=%s GEN=%s STATUS=%s HD=%s",
        KIE_BASE_URL, KIE_GEN_PATH, KIE_STATUS_PATH, KIE_HD_PATH
    )
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
