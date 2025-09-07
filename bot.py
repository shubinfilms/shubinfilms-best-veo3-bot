# -*- coding: utf-8 -*-
# VEO3 bot — красивый UI + Fast/Quality + Prompt-Master + счётчик генераций
# PTB 20.7 (async), требуется пакет: python-telegram-bot[rate-limiter]==20.7

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
#        ENV / CONFIG
# ==========================
load_dotenv()

TELEGRAM_TOKEN   = (os.getenv("TELEGRAM_TOKEN") or "").strip()
KIE_API_KEY      = (os.getenv("KIE_API_KEY") or "").strip()
KIE_BASE_URL     = os.getenv("KIE_BASE_URL", "https://api.kie.ai").strip()
KIE_GEN_PATH     = os.getenv("KIE_GEN_PATH", "/api/v1/veo/generate").strip()
KIE_STATUS_PATH  = os.getenv("KIE_STATUS_PATH", "/api/v1/veo/record-info").strip()
KIE_HD_PATH      = os.getenv("KIE_HD_PATH", "/api/v1/veo/get-1080p-video").strip()  # на будущее

PROMPTS_CHANNEL_URL = os.getenv("PROMPTS_CHANNEL_URL", "https://t.me/bestveo3promts").strip()
TOPUP_URL           = os.getenv("TOPUP_URL", "https://t.me/bestveo3promts/1").strip()  # заглушка-линк

# OpenAI (legacy SDK 0.28.x; можно опционально отключить)
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
try:
    import openai  # type: ignore
    if OPENAI_API_KEY:
        openai.api_key = OPENAI_API_KEY
except Exception:
    openai = None

# Параметры ожидания
POLL_INTERVAL_SECS = int(os.getenv("POLL_INTERVAL_SECS", "6"))
POLL_TIMEOUT_SECS  = int(os.getenv("POLL_TIMEOUT_SECS", str(20 * 60)))

# Баланс/кредиты
FREE_CREDITS      = int(os.getenv("FREE_CREDITS", "2"))   # стартовые
FAST_COST         = float(os.getenv("FAST_COST", "1.0"))  # стоимость «Fast»
QUALITY_COST      = float(os.getenv("QUALITY_COST", "2.0"))

# Логирование
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("veo3-bot")

USERS_FILE = "users.json"   # храним кредиты и прочее
# структура: { "<chat_id>": {"credits": 3 } }

# ==========================
#       Хранилище
# ==========================
def _load_users() -> Dict[str, Dict[str, Any]]:
    try:
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, "r", encoding="utf-8") as f:
                return json.load(f) or {}
    except Exception as e:
        log.warning("Load users.json failed: %s", e)
    return {}

def _save_users(data: Dict[str, Dict[str, Any]]) -> None:
    try:
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning("Save users.json failed: %s", e)

USERS = _load_users()

def _get_credits(chat_id: int) -> float:
    u = USERS.get(str(chat_id)) or {}
    if "credits" not in u:
        u["credits"] = float(FREE_CREDITS)
        USERS[str(chat_id)] = u
        _save_users(USERS)
    return float(u.get("credits", 0))

def _add_credits(chat_id: int, amount: float) -> None:
    u = USERS.get(str(chat_id)) or {"credits": 0}
    u["credits"] = float(u.get("credits", 0)) + float(amount)
    USERS[str(chat_id)] = u
    _save_users(USERS)

def _spend_credits(chat_id: int, amount: float) -> bool:
    cur = _get_credits(chat_id)
    if cur >= amount:
        USERS[str(chat_id)]["credits"] = round(cur - amount, 3)
        _save_users(USERS)
        return True
    return False

# ==========================
#     Вспомогательные
# ==========================
def join_url(base: str, path: str) -> str:
    u = f"{base.rstrip('/')}/{path.lstrip('/')}"
    return u.replace("://", "§§").replace("//", "/").replace("§§", "://")

def mask_secret(s: str, show: int = 8) -> str:
    s = (s or "").strip()
    return "*" * max(0, len(s) - show) + s[-show:]

def pick_first_url(value: Union[str, List[str], None]) -> Optional[str]:
    if not value:
        return None
    if isinstance(value, str):
        v = value.strip()
        return v or None
    for v in value:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None

def tg_file_direct_url(bot_token: str, file_path: str) -> str:
    return f"https://api.telegram.org/file/bot{bot_token}/{file_path.lstrip('/')}"

def _kie_headers() -> Dict[str, str]:
    tok = KIE_API_KEY
    if tok and not tok.lower().startswith("bearer "):
        tok = f"Bearer {tok}"
    return {"Content-Type": "application/json", "Authorization": tok or ""}

def _post_json(url: str, payload: Dict[str, Any], timeout: int = 50) -> Tuple[int, Dict[str, Any]]:
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

def _extract_result_url(data: Dict[str, Any]) -> Optional[str]:
    # часто приходит в data.response.resultUrls / originUrls
    resp = data.get("response") or {}
    for key in ("resultUrls", "originUrls", "resultUrl", "originUrl"):
        url = pick_first_url(resp.get(key))
        if url:
            return url
    for key in ("resultUrls", "originUrls", "resultUrl", "originUrl", "url"):
        url = pick_first_url(data.get(key))
        if url:
            return url
    return None

def _parse_success_flag(j: Dict[str, Any]) -> Tuple[Optional[int], Optional[str], Dict[str, Any]]:
    data = j.get("data") or {}
    msg  = j.get("msg") or j.get("message")
    # флаг: 0 — в работе, 1 — готово, 2/3 — ошибка
    flag = None
    for k in ("successFlag", "status", "state"):
        if k in data:
            try:
                flag = int(data[k])
                break
            except Exception:
                pass
    return flag, msg, data

def _kie_error_message(status_code: int, j: Dict[str, Any]) -> str:
    code = j.get("code", status_code)
    msg = j.get("msg") or j.get("message") or j.get("error") or ""
    if code in (401, 403):
        base = "Доступ запрещён: проверь KIE_API_KEY (Bearer)."
    elif code == 451:
        base = "Ошибка загрузки изображения (451)."
    elif code == 429:
        base = "Превышен лимит запросов (429)."
    elif code == 500:
        base = "Внутренняя ошибка KIE (500)."
    else:
        base = f"KIE code {code}."
    return f"{base} {('— ' + msg) if msg else ''}".strip()

# ==========================
#        UI / клавиатуры
# ==========================
def top_menu_kb(credits: float) -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton("🎬 Сгенерировать видео по тексту", callback_data="mode:text")],
        [InlineKeyboardButton("📷 Сгенерировать видео по фото",  callback_data="mode:photo")],
        [InlineKeyboardButton("💳 Пополнить баланс", url=TOPUP_URL)],
        [InlineKeyboardButton("🤖 Бесплатный ChatGPT", callback_data="mode:chat")],
    ]
    return InlineKeyboardMarkup(rows)

def aspect_row(cur: str) -> List[InlineKeyboardButton]:
    if cur == "9:16":
        return [InlineKeyboardButton("16:9", callback_data="aspect:16:9"),
                InlineKeyboardButton("9:16 ✅", callback_data="aspect:9:16")]
    return [InlineKeyboardButton("16:9 ✅", callback_data="aspect:16:9"),
            InlineKeyboardButton("9:16", callback_data="aspect:9:16")]

def speed_row(cur: str, quality_available: bool) -> List[InlineKeyboardButton]:
    fast   = "⚡ Fast" + (" ✅" if cur == "fast" else "")
    if not quality_available:
        return [InlineKeyboardButton(fast, callback_data="speed:fast")]
    qual   = "🎬 Quality" + (" ✅" if cur == "quality" else "")
    return [InlineKeyboardButton(fast, callback_data="speed:fast"),
            InlineKeyboardButton(qual, callback_data="speed:quality")]

def card_keyboard(s: Dict[str, Any], quality_available: bool) -> InlineKeyboardMarkup:
    rows: List[List[InlineKeyboardButton]] = []
    rows.append([InlineKeyboardButton("🖼️ Добавить/Удалить фото", callback_data="card:toggle_photo"),
                 InlineKeyboardButton("✍️ Изменить промпт", callback_data="card:edit_prompt")])
    rows.append(aspect_row(s.get("aspect", "16:9")))
    rows.append(speed_row(s.get("speed", "fast"), quality_available))
    if s.get("prompt"):
        rows.append([InlineKeyboardButton("🚀 Сгенерировать", callback_data="card:generate")])
    rows.append([InlineKeyboardButton("🔁 Начать заново", callback_data="card:reset"),
                 InlineKeyboardButton("⬅️ Назад", callback_data="back")])
    return InlineKeyboardMarkup(rows)

def build_intro_text(credits: float) -> str:
    return (
        "👋 *Привет!* Это бот *Veo 3* — генератор видео с помощью ИИ.\n\n"
        "📚 Смотри примеры роликов и правильные промпты — @veo3prompts\n"
        "📖 *Инструкция* — как пользоваться ботом\n\n"
        f"💎 *Осталось генераций:* {int(credits)}\n\n"
        "Что хочешь сгенерировать сегодня?"
    )

def build_card_text(s: Dict[str, Any], quality_available: bool) -> str:
    prompt = (s.get("prompt") or "").strip()
    if len(prompt) > 800:
        prompt = prompt[:800] + "…"
    has_ref = "есть" if s.get("image_url") else "нет"

    speed = s.get("speed", "fast")
    speed_label = "Fast ⚡" if speed == "fast" else "Quality 🎬"
    cost = FAST_COST if speed == "fast" else QUALITY_COST
    qual_line = "" if not quality_available else f"\n• Режим: *{speed_label}* (стоимость: {cost:g})"

    return (
        "✍️ *Промпт:*\n"
        f"`{prompt or '—'}`\n\n"
        "📋 *Параметры генерации:*\n"
        f"• Формат: *{s.get('aspect','16:9')}* 🎞"
        f"{qual_line}\n"
        f"• Промпт: *{'есть' if s.get('prompt') else 'нет'}* ✍️\n"
        f"• Референс: *{has_ref}*\n"
    )

# ==========================
#        Состояние
# ==========================
DEFAULT_STATE = {
    "mode": None,            # 'text' | 'photo' | 'chat'
    "aspect": "16:9",
    "speed": "fast",         # 'fast' | 'quality'
    "prompt": None,
    "image_url": None,
    "quality_available": True,  # будем динамически опускать, если придёт ошибка
    "generating": False,
    "current_task": None,
    "ui_msg_id": None,
}

def user_state(ctx: ContextTypes.DEFAULT_TYPE) -> Dict[str, Any]:
    ud = ctx.user_data
    # init defaults
    for k, v in DEFAULT_STATE.items():
        ud.setdefault(k, v)
    return ud

# ==========================
#     KIE API wrappers
# ==========================
def _payload_for_kie(prompt: str, aspect: str, speed: str, image_url: Optional[str]) -> Dict[str, Any]:
    payload = {
        "prompt": prompt,
        "aspectRatio": aspect,
        "model": "veo3_quality" if speed == "quality" else "veo3_fast",
    }
    if image_url:
        payload["imageUrl"] = image_url
    return payload

def submit_generation(prompt: str, aspect: str, speed: str, image_url: Optional[str]) -> Tuple[bool, Optional[str], str]:
    url = join_url(KIE_BASE_URL, KIE_GEN_PATH)
    status, j = _post_json(url, _payload_for_kie(prompt, aspect, speed, image_url))
    code = j.get("code", status)
    if status == 200 and code == 200:
        task_id = _extract_task_id(j)
        if task_id:
            return True, task_id, "ok"
        return False, None, "Ответ KIE без taskId."
    return False, None, _kie_error_message(status, j)

def get_status(task_id: str) -> Tuple[bool, Optional[int], Optional[str], Optional[str]]:
    url = join_url(KIE_BASE_URL, KIE_STATUS_PATH)
    status, j = _get_json(url, {"taskId": task_id})
    code = j.get("code", status)
    if status == 200 and code == 200:
        flag, msg, data = _parse_success_flag(j)
        return True, flag, msg, _extract_result_url(data or {})
    return False, None, _kie_error_message(status, j), None

# ==========================
#      Отправка видео
# ==========================
async def send_video(ctx: ContextTypes.DEFAULT_TYPE, chat_id: int, url: str) -> bool:
    # 1) пробуем отдать прямую ссылку (телеграм сам скачает)
    try:
        await ctx.bot.send_video(chat_id=chat_id, video=url, supports_streaming=True)
        return True
    except Exception as e:
        log.info("Direct URL send failed, try file. %s", e)
    # 2) скачиваем и отправляем файлом
    tmp = None
    try:
        r = requests.get(url, stream=True, timeout=180)
        r.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
            for chunk in r.iter_content(chunk_size=512 * 1024):
                if chunk:
                    f.write(chunk)
            tmp = f.name
        with open(tmp, "rb") as f:
            await ctx.bot.send_video(chat_id=chat_id, video=InputFile(f, filename="veo3.mp4"),
                                     supports_streaming=True)
        return True
    except Exception as e:
        log.exception("File send failed: %s", e)
        return False
    finally:
        if tmp:
            try: os.unlink(tmp)
            except Exception: pass

# ==========================
#     Поллинг задачи
# ==========================
async def poll_and_deliver(chat_id: int, task_id: str, ctx: ContextTypes.DEFAULT_TYPE):
    s = user_state(ctx)
    s["generating"] = True
    start = time.time()
    try:
        while True:
            ok, flag, msg, url = await asyncio.to_thread(get_status, task_id)
            if not ok:
                await ctx.bot.send_message(chat_id, f"❌ Ошибка статуса: {msg or 'неизвестно'}")
                break

            if flag == 0:  # в обработке
                if time.time() - start > POLL_TIMEOUT_SECS:
                    await ctx.bot.send_message(chat_id, "⏳ Таймаут ожидания. Попробуйте позже.")
                    break
                await asyncio.sleep(POLL_INTERVAL_SECS)
                continue

            if flag == 1:  # готово
                if not url:
                    await ctx.bot.send_message(chat_id, "⚠️ Готово, но ссылка не найдена.")
                    break
                sent = await send_video(ctx, chat_id, url)
                if sent:
                    await ctx.bot.send_message(chat_id, "✅ Готово!")
                else:
                    await ctx.bot.send_message(chat_id, "⚠️ Видео не получилось отправить. Попробуйте ещё раз.")
                break

            if flag in (2, 3):
                await ctx.bot.send_message(chat_id, f"❌ Генерация не удалась на стороне провайдера. {msg or ''}".strip())
                break

            await asyncio.sleep(POLL_INTERVAL_SECS)
    except Exception as e:
        log.exception("Poller crashed: %s", e)
        try:
            await ctx.bot.send_message(chat_id, "❌ Внутренняя ошибка при опросе статуса.")
        except Exception:
            pass
    finally:
        s["generating"] = False
        s["current_task"] = None

# ==========================
#       Prompt-Master
# ==========================
async def prompt_master(idea: str) -> Optional[str]:
    if not (openai and OPENAI_API_KEY):
        return None
    sys = (
        "You are a world-class cinematic prompt writer for Google Veo 3. "
        "Return ONE English prompt, 450–900 chars, no preface, no bullets, no brand names. "
        "Include: subject, micro-actions, lens/optics (mm/anamorphic), camera movement, "
        "lighting (time, color temp, key/fill/rim), mood/atmosphere, texture/particles, "
        "composition, color palette, environment sound cues. "
        "Optionally a short line of dialogue in quotes. No logos, no watermarks."
    )
    try:
        resp = await asyncio.to_thread(
            openai.ChatCompletion.create,
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": sys},
                      {"role": "user", "content": idea.strip()}],
            temperature=0.95,
            max_tokens=700,
        )
        out = resp["choices"][0]["message"]["content"].strip()
        return out[:1200]
    except Exception as e:
        log.exception("PromptMaster error: %s", e)
        return None

# ==========================
#        Хэндлеры
# ==========================
async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    # инициализация баланса
    _get_credits(chat_id)
    s = user_state(ctx)
    # сброс состояния
    for k, v in DEFAULT_STATE.items():
        s[k] = v

    credits = _get_credits(chat_id)
    await update.message.reply_text(
        build_intro_text(credits),
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=top_menu_kb(credits),
        disable_web_page_preview=True
    )

async def on_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    data = (q.data or "").strip()
    s = user_state(ctx)
    chat_id = update.effective_chat.id

    # режимы
    if data.startswith("mode:"):
        mode = data.split(":", 1)[1]
        s["mode"] = mode
        s["ui_msg_id"] = None  # новая карточка
        if mode == "text":
            await q.message.reply_text("✍️ Напиши идею/промпт. Хочешь — нажми «Промпт-мастер» и пришли только идею.")
        elif mode == "photo":
            await q.message.reply_text("📷 Пришли фото. Можно добавить подпись-промпт.")
        elif mode == "chat":
            await q.message.reply_text("🤖 Обычный чат: задавай вопросы. Для выхода — /start.")
        # показываем пустую карточку
        await show_card(update, ctx)
        return

    # аспект
    if data.startswith("aspect:"):
        s["aspect"] = "9:16" if data.endswith("9:16") else "16:9"
        await show_card(update, ctx, edit_only_markup=True)
        return

    # скорость
    if data.startswith("speed:"):
        val = data.split(":", 1)[1]
        if val in ("fast", "quality"):
            s["speed"] = val
        await show_card(update, ctx, edit_only_markup=True)
        return

    # карточка-кнопки
    if data == "card:toggle_photo":
        if s.get("image_url"):
            s["image_url"] = None
            await q.message.reply_text("🗑️ Референс удалён.")
        else:
            await q.message.reply_text("Пришли фото (или публичный URL картинки).")
        await show_card(update, ctx)
        return

    if data == "card:edit_prompt":
        await q.message.reply_text("Пришли новый текст промпта или идею для *Промпт-мастера*.", parse_mode=ParseMode.MARKDOWN)
        return

    if data == "card:reset":
        keep_aspect = s.get("aspect", "16:9")
        keep_quality_av = s.get("quality_available", True)
        for k, v in DEFAULT_STATE.items():
            s[k] = v
        s["aspect"] = keep_aspect
        s["quality_available"] = keep_quality_av
        await q.message.reply_text("🧹 Карточка очищена.")
        await show_card(update, ctx)
        return

    if data == "back":
        credits = _get_credits(chat_id)
        await q.message.reply_text(
            build_intro_text(credits),
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=top_menu_kb(credits),
            disable_web_page_preview=True
        )
        return

    if data == "card:generate":
        if s.get("generating"):
            await q.message.reply_text("⏳ Генерация уже идёт. Дождитесь завершения.")
            return
        if not s.get("prompt"):
            await q.message.reply_text("Сначала укажите промпт.")
            return

        # проверим кредиты
        cost = FAST_COST if s.get("speed") == "fast" else QUALITY_COST
        if not _spend_credits(chat_id, cost):
            await q.message.reply_text(
                f"💳 Недостаточно генераций (нужно {cost:g}). Нажмите «Пополнить баланс».",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Пополнить баланс", url=TOPUP_URL)]])
            )
            return

        ok, task_id, msg = await asyncio.to_thread(
            submit_generation, s["prompt"], s.get("aspect", "16:9"), s.get("speed", "fast"), s.get("image_url")
        )
        if not ok or not task_id:
            # вернём списанные кредиты, чтобы не было обидно
            _add_credits(chat_id, cost)
            # если упало из-за «quality» — выключим его до рестарта
            if "veo3_quality" in (msg or "").lower():
                s["quality_available"] = False
            await q.message.reply_text(f"❌ Не удалось создать задачу: {msg}")
            await show_card(update, ctx, edit_only_markup=True)
            return

        s["generating"] = True
        s["current_task"] = task_id
        await q.message.reply_text(f"🚀 Отправил задачу в Veo3 ({'Quality' if s['speed']=='quality' else 'Fast'}). taskId={task_id}")
        asyncio.create_task(poll_and_deliver(chat_id, task_id, ctx))
        return

async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = user_state(ctx)
    text = (update.message.text or "").strip()

    # если прислали публичный URL картинки
    low = text.lower()
    if low.startswith(("http://", "https://")) and low.split("?")[0].endswith((".jpg", ".jpeg", ".png", ".webp")):
        s["image_url"] = text.strip()
        await update.message.reply_text("🖼️ Ссылка на изображение принята.")
        await show_card(update, ctx)
        return

    # режимы
    if s.get("mode") == "chat":
        if not (openai and OPENAI_API_KEY):
            await update.message.reply_text("⚠️ ChatGPT недоступен (нет OPENAI_API_KEY).")
            return
        try:
            resp = await asyncio.to_thread(
                openai.ChatCompletion.create,
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": "You are a helpful, concise assistant."},
                          {"role": "user", "content": text}],
                temperature=0.5,
                max_tokens=600,
            )
            out = resp["choices"][0]["message"]["content"].strip()
            await update.message.reply_text(out or "…")
        except Exception as e:
            log.exception("Chat error: %s", e)
            await update.message.reply_text("⚠️ Ошибка ChatGPT.")
        return

    # кнопка «Промпт-мастер» нет как таковой — просто объясняем юзеру:
    if text.strip().lower() in ("prompt", "prompt-master", "промпт", "промпт-мастер"):
        await update.message.reply_text("Пришли коротко идею — верну готовый кинопромпт (EN).")
        return

    # если пользователь в любом режиме прислал обычный текст — считаем это идеей → прогоняем через Prompt-Master
    # (можно отключить — тогда закомментируй блок ниже)
    pm = await prompt_master(text)
    if pm:
        s["prompt"] = pm
        await update.message.reply_text("🧠 Готово! Добавил кинопромпт в карточку.")
    else:
        s["prompt"] = text
        await update.message.reply_text("✍️ Принял текст как промпт (Prompt-Master недоступен).")
    await show_card(update, ctx)

async def on_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = user_state(ctx)
    photos = update.message.photo
    if not photos:
        return
    ph = photos[-1]
    try:
        f = await ctx.bot.get_file(ph.file_id)
        if not f or not f.file_path:
            await update.message.reply_text("⚠️ Не удалось получить путь к файлу Telegram.")
            return
        url = tg_file_direct_url(TELEGRAM_TOKEN, f.file_path)
        log.info("Got TG photo path: ...%s", mask_secret(url))
        s["image_url"] = url
        await update.message.reply_text("🖼️ Фото принято как референс.")
        await show_card(update, ctx)
    except Exception as e:
        log.exception("Get photo failed: %s", e)
        await update.message.reply_text("⚠️ Не удалось обработать фото. Пришли публичный URL картинки текстом.")

async def error_handler(update: Optional[Update], ctx: ContextTypes.DEFAULT_TYPE):
    log.exception("Unhandled error: %s", ctx.error)
    try:
        if update and update.effective_chat:
            await ctx.bot.send_message(update.effective_chat.id, "⚠️ Системная ошибка. Попробуйте ещё раз.")
    except Exception:
        pass

# показать/обновить карточку
async def show_card(update: Update, ctx: ContextTypes.DEFAULT_TYPE, edit_only_markup: bool = False):
    s = user_state(ctx)
    chat_id = update.effective_chat.id
    text = build_card_text(s, s.get("quality_available", True))
    kb = card_keyboard(s, s.get("quality_available", True))

    mid = s.get("ui_msg_id")
    try:
        if mid:
            if edit_only_markup:
                await ctx.bot.edit_message_reply_markup(chat_id, mid, reply_markup=kb)
            else:
                await ctx.bot.edit_message_text(chat_id=chat_id, message_id=mid, text=text,
                                                parse_mode=ParseMode.MARKDOWN, reply_markup=kb,
                                                disable_web_page_preview=True)
        else:
            m = await (update.callback_query.message.reply_text(text, parse_mode=ParseMode.MARKDOWN,
                                                                reply_markup=kb, disable_web_page_preview=True)
                       if update.callback_query else
                       update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN,
                                                 reply_markup=kb, disable_web_page_preview=True))
            s["ui_msg_id"] = m.message_id
    except Exception as e:
        log.warning("show_card failed: %s", e)
        try:
            m = await ctx.bot.send_message(chat_id=chat_id, text=text, parse_mode=ParseMode.MARKDOWN,
                                           reply_markup=kb, disable_web_page_preview=True)
            s["ui_msg_id"] = m.message_id
        except Exception as e2:
            log.exception("show_card send failed: %s", e2)

# ==========================
#        Entry point
# ==========================
def main():
    if not TELEGRAM_TOKEN:
        raise RuntimeError("TELEGRAM_TOKEN is not set")
    if not (KIE_API_KEY and KIE_BASE_URL and KIE_GEN_PATH and KIE_STATUS_PATH):
        raise RuntimeError("KIE env vars are not properly set")

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

    log.info("Bot starting… PTB 20.7 | KIE=%s", KIE_BASE_URL)
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
