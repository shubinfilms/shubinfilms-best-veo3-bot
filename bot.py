# -*- coding: utf-8 -*-
# Best VEO3 Bot — PTB 20.7 / ApplicationBuilder only (без Updater)
# Версия: 2025-09-07

import os
import io
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

# =========================
# Инициализация и конфиг
# =========================
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
KIE_API_KEY    = os.getenv("KIE_API_KEY", "")
KIE_BASE_URL   = os.getenv("KIE_BASE_URL", "https://api.kie.ai").rstrip("/")
KIE_GEN_PATH   = os.getenv("KIE_GEN_PATH", "/api/v1/veo/generate").strip()
KIE_STATUS_PATH= os.getenv("KIE_STATUS_PATH", "/api/v1/veo/record-info").strip()
KIE_HD_PATH    = os.getenv("KIE_HD_PATH", "/api/v1/veo/get-1080p-video").strip()

KIE_ENABLE_FALLBACK = str(os.getenv("KIE_ENABLE_FALLBACK", "false")).lower() in {"1", "true", "yes"}
KIE_DEFAULT_SEED    = os.getenv("KIE_DEFAULT_SEED", "").strip()
KIE_WATERMARK_TEXT  = os.getenv("KIE_WATERMARK_TEXT", "").strip()
KIE_HD_INDEX        = int(os.getenv("KIE_HD_INDEX", "0") or 0)

POLL_INTERVAL_SECS = int(os.getenv("POLL_INTERVAL_SECS", "6") or 6)
POLL_TIMEOUT_SECS  = int(os.getenv("POLL_TIMEOUT_SECS", str(20*60)) or 20*60)

# OpenAI 0.28.1 (старый SDK)
try:
    import openai  # type: ignore
    if OPENAI_API_KEY:
        openai.api_key = OPENAI_API_KEY
except Exception:
    openai = None

# Логи
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("best-veo3-bot")

# =========================
# Утилиты
# =========================
def join_url(base: str, path: str) -> str:
    u = f"{base.rstrip('/')}/{path.lstrip('/')}"
    return u.replace("://", "§§").replace("//", "/").replace("§§", "://")

def mask_secret(s: str, show: int = 6) -> str:
    if not s: return ""
    s = s.strip()
    return ("*"*(len(s)-show))+s[-show:] if len(s)>show else "*"*len(s)

def pick_first_url(v: Union[str, List[str], None]) -> Optional[str]:
    if not v: return None
    if isinstance(v, str): return v.strip() or None
    if isinstance(v, list):
        for x in v:
            if isinstance(x, str) and x.strip(): return x.strip()
    return None

def tg_file_direct_url(bot_token: str, file_path: str) -> str:
    return f"https://api.telegram.org/file/bot{bot_token}/{file_path.lstrip('/')}"

# =========================
# Состояние пользователя
# =========================
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

# =========================
# UI
# =========================
def aspect_row(current: str) -> List[InlineKeyboardButton]:
    if current == "9:16":
        return [InlineKeyboardButton("16:9", callback_data="aspect:16:9"),
                InlineKeyboardButton("9:16 ✅", callback_data="aspect:9:16")]
    return [InlineKeyboardButton("16:9 ✅", callback_data="aspect:16:9"),
            InlineKeyboardButton("9:16", callback_data="aspect:9:16")]

def main_menu_kb() -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton("🎬 Сгенерировать видео по тексту", callback_data="mode:gen_text")],
        [InlineKeyboardButton("🖼️ Сгенерировать видео по фото",  callback_data="mode:gen_photo")],
        [InlineKeyboardButton("🧠 Промпт-мастер (ChatGPT)",       callback_data="mode:prompt_master")],
        [InlineKeyboardButton("💬 Обычный чат (ChatGPT)",         callback_data="mode:chat")],
        [InlineKeyboardButton("❓ FAQ", callback_data="faq"),
         InlineKeyboardButton("📈 Канал с промптами", url="https://t.me/bestveo3promts")],
        aspect_row("16:9"),
    ]
    return InlineKeyboardMarkup(rows)

def card_keyboard(s: Dict[str, Any]) -> InlineKeyboardMarkup:
    rows: List[List[InlineKeyboardButton]] = []
    rows.append([InlineKeyboardButton("🖼️ Добавить/Удалить фото", callback_data="card:toggle_photo"),
                 InlineKeyboardButton("✍️ Изменить промпт",       callback_data="card:edit_prompt")])
    rows.append(aspect_row(s["aspect"]))
    if s.get("last_prompt"):
        rows.append([InlineKeyboardButton("🚀 Сгенерировать", callback_data="card:generate")])
    rows.append([InlineKeyboardButton("🔁 Начать заново", callback_data="card:reset"),
                 InlineKeyboardButton("⬅️ Назад",        callback_data="back")])
    return InlineKeyboardMarkup(rows)

def build_card_text(s: Dict[str, Any]) -> str:
    p = (s.get("last_prompt") or "").strip()
    if len(p) > 900: p = p[:900] + "…"
    has_prompt = "есть" if s.get("last_prompt") else "нет"
    has_ref    = "есть" if s.get("last_image_url") else "нет"
    lines = [
        "🧾 *Промпт принят.* Выберите формат и режим или сразу генерируйте.",
        "",
        "📝 *Промпт:*",
        f"`{p or '—'}`",
        "",
        "🧱 *Параметры генерации:*",
        f"• Формат: *{s.get('aspect','16:9')}*",
        "• Режим: *Fast ⚡* (veo3_fast)",
        f"• Промпт: *{has_prompt}*",
        f"• Референс: *{has_ref}*",
    ]
    return "\n".join(lines)

# =========================
# Prompt-Master (OpenAI 0.28.1)
# =========================
async def oai_prompt_master(idea_text: str) -> Optional[str]:
    if openai is None or not OPENAI_API_KEY:
        return None
    system = (
        "You are Prompt-Master for cinematic AI video generation (Google Veo3). "
        "Return exactly ONE English prompt, 500–900 characters, no meta, no brands/logos/subtitles. "
        "Include optics (mm/anamorphic), camera moves (push-in, dolly, glide, rack focus), "
        "lighting/palette/atmosphere, micro-details (dust/steam/flares), subtle audio cues. "
        "Optionally one short hero line in quotes. Output only the prompt text."
    )
    try:
        resp = await asyncio.to_thread(
            openai.ChatCompletion.create,
            model="gpt-4o-mini",
            messages=[{"role":"system","content":system},
                      {"role":"user","content":idea_text.strip()}],
            temperature=0.8, max_tokens=800, n=1
        )
        return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        log.exception("Prompt-Master error: %s", e)
        return None

# =========================
# KIE API
# =========================
def _kie_headers() -> Dict[str, str]:
    auth = KIE_API_KEY or ""
    if auth and not auth.lower().startswith("bearer "):
        auth = f"Bearer {auth}"
    return {"Content-Type":"application/json","Authorization":auth}

def _post_json(url: str, payload: Dict[str, Any], timeout: int = 30) -> Tuple[int, Dict[str, Any]]:
    r = requests.post(url, json=payload, headers=_kie_headers(), timeout=timeout)
    try: return r.status_code, r.json()
    except Exception: return r.status_code, {"error": r.text}

def _get_json(url: str, params: Dict[str, Any], timeout: int = 30) -> Tuple[int, Dict[str, Any]]:
    r = requests.get(url, params=params, headers=_kie_headers(), timeout=timeout)
    try: return r.status_code, r.json()
    except Exception: return r.status_code, {"error": r.text}

def _extract_task_id(j: Dict[str, Any]) -> Optional[str]:
    for k in ("taskId", "taskid", "id"):
        if j.get(k): return str(j[k])
    data = j.get("data") or {}
    for k in ("taskId", "taskid", "id"):
        if data.get(k): return str(data[k])
    return None

def _extract_result_url(data: Dict[str, Any]) -> Optional[str]:
    resp = data.get("response") or {}
    for key in ("resultUrls","originUrls","resultUrl","originUrl"):
        u = pick_first_url(resp.get(key))
        if u: return u
    for key in ("resultUrls","originUrls","resultUrl","originUrl"):
        u = pick_first_url(data.get(key))
        if u: return u
    return pick_first_url(data.get("url"))

def _kie_error_to_text(status_code: int, j: Dict[str, Any]) -> str:
    code = j.get("code", status_code)
    msg  = j.get("msg") or j.get("message") or j.get("error") or ""
    if code == 402 or status_code == 402:
        base = "Недостаточно кредитов."
    elif code in (401,403) or status_code in (401,403) or "Illegal IP" in msg:
        base = "Доступ запрещён (ключ/IP)."
    else:
        base = f"KIE code {code}."
    return f"{base} {('Сообщение: '+msg) if msg else ''}".strip()

def _build_payload_for_kie(prompt: str, aspect: str, image_url: Optional[str]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": "veo3_fast",
        "prompt": prompt,
        "aspectRatio": aspect,
        "aspect_ratio": aspect,  # на всякий случай оба
    }
    if image_url:
        payload["imageUrls"] = [image_url]
        payload["image_url"] = image_url
    if KIE_DEFAULT_SEED.isdigit():
        payload["seed"] = int(KIE_DEFAULT_SEED)
    if KIE_WATERMARK_TEXT:
        payload["waterMark"] = KIE_WATERMARK_TEXT
    if KIE_ENABLE_FALLBACK and aspect == "16:9":
        payload["enableFallback"] = True
    return payload

def submit_kie_generation(prompt: str, aspect: str, image_url: Optional[str]) -> Tuple[bool, Optional[str], str]:
    url = join_url(KIE_BASE_URL, KIE_GEN_PATH)
    status, j = _post_json(url, _build_payload_for_kie(prompt, aspect, image_url))
    code = j.get("code", status)
    if status == 200 and code == 200:
        tid = _extract_task_id(j)
        return (True, tid, "Создана задача.") if tid else (False, None, "Не удалось определить taskId.")
    return False, None, _kie_error_to_text(status, j)

def get_kie_task_status(task_id: str) -> Tuple[bool, Optional[int], Optional[str], Optional[str], Optional[bool]]:
    url = join_url(KIE_BASE_URL, KIE_STATUS_PATH)
    status, j = _get_json(url, {"taskId": task_id})
    code = j.get("code", status)
    if status != 200 or code != 200:
        return False, None, _kie_error_to_text(status, j), None, None
    data = j.get("data") or {}
    flag = data.get("successFlag")
    try: flag = int(flag) if flag is not None else None
    except Exception: flag = None
    res_url = _extract_result_url(data)
    msg = j.get("msg") or j.get("message")
    fallback_flag = data.get("fallbackFlag")
    return True, flag, msg, res_url, bool(fallback_flag) if fallback_flag is not None else None

def get_kie_1080p_url(task_id: str, index: int = 0) -> Optional[str]:
    url = join_url(KIE_BASE_URL, KIE_HD_PATH)
    status, j = _get_json(url, {"taskId": task_id, "index": index})
    if status == 200 and j.get("code") == 200:
        return (j.get("data") or {}).get("resultUrl")
    return None

def _download_temp(url: str, timeout: int = 120) -> Optional[str]:
    try:
        r = requests.get(url, stream=True, timeout=timeout)
        r.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
            for chunk in r.iter_content(chunk_size=1024*256):
                if chunk: f.write(chunk)
            return f.name
    except Exception as e:
        log.exception("Download failed: %s", e)
        return None

async def send_video_with_fallback(ctx: ContextTypes.DEFAULT_TYPE, chat_id: int, url: str) -> bool:
    try:
        await ctx.bot.send_video(chat_id=chat_id, video=url, supports_streaming=True)
        return True
    except Exception as e:
        log.warning("Direct send failed → download. Err=%s", e)
    tmp = await asyncio.to_thread(_download_temp, url)
    if not tmp: return False
    try:
        with open(tmp, "rb") as f:
            await ctx.bot.send_video(chat_id=chat_id, video=InputFile(f, filename="result.mp4"), supports_streaming=True)
        return True
    except Exception as e:
        log.exception("File send failed: %s", e)
        return False

# =========================
# Поллинг задачи
# =========================
async def poll_kie_and_send(chat_id: int, task_id: str, gen_id: str, ctx: ContextTypes.DEFAULT_TYPE, aspect: str):
    s = state(ctx)
    s["generating"] = True
    s["generation_id"] = gen_id
    start_ts = time.time()
    await ctx.bot.send_message(chat_id, "⏳ Жду результат от Veo3…")

    try:
        while True:
            if s.get("generation_id") != gen_id:
                return
            ok, flag, msg, res_url, fallback_flag = await asyncio.to_thread(get_kie_task_status, task_id)
            if not ok:
                await ctx.bot.send_message(chat_id, f"❌ Ошибка статуса: {msg or 'неизвестно'}")
                break

            if flag == 0 or flag is None:
                if (time.time() - start_ts) > POLL_TIMEOUT_SECS:
                    await ctx.bot.send_message(chat_id, "⌛ Таймаут ожидания. Проверьте позже в журнале KIE.")
                    break
                await asyncio.sleep(POLL_INTERVAL_SECS)
                continue

            if flag == 1:
                if not res_url:
                    await ctx.bot.send_message(chat_id, "⚠️ Рендер завершён, но ссылка не найдена.")
                    break
                # HD для 16:9 (если не fallback)
                final_url = res_url
                if aspect == "16:9" and (fallback_flag is False or fallback_flag is None):
                    hd = await asyncio.to_thread(get_kie_1080p_url, task_id, KIE_HD_INDEX)
                    if hd: final_url = hd
                sent = await send_video_with_fallback(ctx, chat_id, final_url)
                await ctx.bot.send_message(chat_id,
                    "✅ Готово! (veo3_fast{})".format(" • fallback" if fallback_flag else "")) if sent \
                    else await ctx.bot.send_message(chat_id, "⚠️ Не удалось отправить видео.")
                break

            if flag in (2, 3):
                await ctx.bot.send_message(chat_id, f"❌ Ошибка KIE: {msg or 'без сообщения'}")
                break

            await asyncio.sleep(POLL_INTERVAL_SECS)
    finally:
        if s.get("generation_id") == gen_id:
            s["generating"] = False
            s["generation_id"] = None

# =========================
# Хэндлеры
# =========================
async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = state(ctx)
    s.update({**DEFAULT_STATE})
    s["aspect"] = "16:9"
    await update.message.reply_text(
        "Привет! Это Best VEO3 bot. Сгенерируем видео через Veo3 Fast.\nВыберите режим:",
        reply_markup=main_menu_kb(),
    )

async def show_card(update: Update, ctx: ContextTypes.DEFAULT_TYPE, edit_only_markup: bool=False):
    s = state(ctx)
    txt = build_card_text(s)
    kb  = card_keyboard(s)
    chat_id = update.effective_chat.id
    last_id = s.get("last_ui_msg_id")
    try:
        if last_id:
            if edit_only_markup:
                await ctx.bot.edit_message_reply_markup(chat_id, last_id, reply_markup=kb)
            else:
                await ctx.bot.edit_message_text(chat_id=chat_id, message_id=last_id,
                    text=txt, parse_mode=ParseMode.MARKDOWN, reply_markup=kb, disable_web_page_preview=True)
        else:
            m = await (update.callback_query.message.reply_text if update.callback_query else update.message.reply_text)(
                txt, parse_mode=ParseMode.MARKDOWN, reply_markup=kb, disable_web_page_preview=True)
            s["last_ui_msg_id"] = m.message_id
    except Exception as e:
        log.warning("show_card failed → send new. %s", e)
        m = await ctx.bot.send_message(chat_id, txt, parse_mode=ParseMode.MARKDOWN,
                                       reply_markup=kb, disable_web_page_preview=True)
        s["last_ui_msg_id"] = m.message_id

async def on_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    data = q.data or ""
    await q.answer()
    s = state(ctx)

    if data.startswith("aspect:"):
        s["aspect"] = "9:16" if data.endswith("9:16") else "16:9"
        await show_card(update, ctx, edit_only_markup=True)
        return

    if data.startswith("mode:"):
        mode = data.split(":",1)[1]
        s["mode"] = mode
        msg = {
            "gen_text":      "Режим: по тексту. Пришлите идею или готовый промпт.",
            "gen_photo":     "Режим: по фото. Пришлите фото (и подпись — по желанию).",
            "prompt_master": "Промпт-мастер: пришлите идею (1–2 фразы). Верну 1 EN-промпт.",
            "chat":          "Обычный чат: задайте вопрос.",
        }.get(mode, "Режим обновлён.")
        await q.message.reply_text(msg)
        await show_card(update, ctx)
        return

    if data == "faq":
        await q.message.reply_text(
            "FAQ:\n• Используется Fast (veo3_fast). Только форматы 16:9 и 9:16.\n"
            "• Ссылку/видео пришлю сюда. Ошибки 401/402 — ключ/IP/кредиты KIE.",
            reply_markup=main_menu_kb(),
        )
        return

    if data == "back":
        s.update({**DEFAULT_STATE})
        s["aspect"] = "16:9"
        await q.message.reply_text("Главное меню:", reply_markup=main_menu_kb())
        return

    if data == "card:toggle_photo":
        if s.get("last_image_url"):
            s["last_image_url"] = None
            await q.message.reply_text("Фото-референс удалён.")
            await show_card(update, ctx)
        else:
            await q.message.reply_text("Пришлите фото или публичный URL картинки.")
        return

    if data == "card:edit_prompt":
        await q.message.reply_text("Пришлите новый текст промпта (или идею — сделаю EN-кинопромпт).")
        return

    if data == "card:reset":
        keep = s.get("aspect","16:9")
        s.update({**DEFAULT_STATE})
        s["aspect"] = keep
        await q.message.reply_text("Карточка очищена.")
        await show_card(update, ctx)
        return

    if data == "card:generate":
        if not s.get("last_prompt"):
            await q.message.reply_text("Сначала укажите текст промпта.")
            return
        if s.get("generating"):
            await q.message.reply_text("⏳ Генерация уже идёт. Дождитесь результата.")
            return
        prompt = s["last_prompt"].strip()
        aspect = s.get("aspect","16:9")
        image  = s.get("last_image_url")
        ok, task_id, msg = await asyncio.to_thread(submit_kie_generation, prompt, aspect, image)
        if not ok or not task_id:
            await q.message.reply_text(f"❌ Не удалось создать задачу: {msg}")
            return
        gen_id = uuid.uuid4().hex[:12]
        s["generating"]  = True
        s["generation_id"]= gen_id
        await q.message.reply_text(f"🚀 Отправляю задачу в VEO3 Fast… taskId={task_id}")
        asyncio.create_task(poll_kie_and_send(update.effective_chat.id, task_id, gen_id, ctx, aspect))
        return

async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = state(ctx)
    text = (update.message.text or "").strip()

    # публичный URL изображения
    if text.lower().startswith(("http://","https://")) and any(text.lower().endswith(e) for e in (".jpg",".jpeg",".png",".webp")):
        s["last_image_url"] = text
        await update.message.reply_text("✅ Ссылка на изображение принята.")
        await show_card(update, ctx)
        return

    mode = s.get("mode")
    if mode == "prompt_master":
        pr = await oai_prompt_master(text)
        if not pr:
            await update.message.reply_text("⚠️ Не удалось получить промпт от Prompt-Master.")
            return
        s["last_prompt"] = pr
        await update.message.reply_text("🧠 Готово! Промпт добавлен в карточку.")
        await show_card(update, ctx)
        return

    if mode == "chat":
        if openai is None or not OPENAI_API_KEY:
            await update.message.reply_text("⚠️ ChatGPT недоступен (нет ключа).")
            return
        try:
            resp = await asyncio.to_thread(
                openai.ChatCompletion.create,
                model="gpt-4o-mini",
                messages=[{"role":"system","content":"You are a helpful, concise assistant."},
                          {"role":"user","content":text}],
                temperature=0.5, max_tokens=700
            )
            ans = resp["choices"][0]["message"]["content"].strip()
            await update.message.reply_text(ans)
        except Exception as e:
            log.exception("Chat error: %s", e)
            await update.message.reply_text("⚠️ Ошибка обращения к ChatGPT.")
        return

    # по умолчанию — именно текст промпта
    s["last_prompt"] = text
    await update.message.reply_text("✍️ Промпт обновлён.")
    await show_card(update, ctx)

async def on_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = state(ctx)
    phs = update.message.photo
    if not phs: return
    ph = phs[-1]
    try:
        f = await ctx.bot.get_file(ph.file_id)
        if not f.file_path:
            await update.message.reply_text("⚠️ Не удалось получить путь к файлу Telegram.")
            return
        url = tg_file_direct_url(TELEGRAM_TOKEN, f.file_path)
        log.info("Photo via Telegram path: ...%s", mask_secret(url, 10))
        s["last_image_url"] = url
        await update.message.reply_text("🖼️ Фото принято как референс.")
        await show_card(update, ctx)
    except Exception as e:
        log.exception("Get photo failed: %s", e)
        await update.message.reply_text("⚠️ Не удалось обработать фото. Пришлите публичный URL текстом.")

async def error_handler(update: Optional[Update], context: ContextTypes.DEFAULT_TYPE):
    log.exception("Unhandled error: %s", context.error)
    try:
        if update and update.effective_chat:
            await context.bot.send_message(update.effective_chat.id, "⚠️ Системная ошибка. Попробуйте ещё раз.")
    except Exception:
        pass

# =========================
# Точка входа
# =========================
def main():
    if not TELEGRAM_TOKEN:
        raise RuntimeError("TELEGRAM_TOKEN is not set")
    if not (KIE_BASE_URL and KIE_GEN_PATH and KIE_STATUS_PATH):
        raise RuntimeError("KIE_* env vars are not properly set")

    app = (ApplicationBuilder().token(TELEGRAM_TOKEN).rate_limiter(AIORateLimiter()).build())
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(on_callback))
    app.add_handler(MessageHandler(filters.PHOTO, on_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_error_handler(error_handler)

    log.info("Bot starting. KIE_BASE=%s GEN=%s STATUS=%s HD=%s PTB=20.7",
             KIE_BASE_URL, KIE_GEN_PATH, KIE_STATUS_PATH, KIE_HD_PATH)
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
