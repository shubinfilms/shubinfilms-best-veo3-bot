# -*- coding: utf-8 -*-
# SHUBIN AI VIDEO — Veo3 Fast + ChatGPT (Prompt-Master & Chat)
# Версия: 2025-09-07

import os
import json
import time
import tempfile
import traceback
from typing import Any, Iterable, Optional, Tuple

import requests
from dotenv import load_dotenv

import telebot
from telebot import types

# =========================
# ENV
# =========================
load_dotenv()

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN")
KIE_API_KEY      = os.getenv("KIE_API_KEY")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "")
PROMPTS_CHANNEL  = os.getenv("PROMPTS_CHANNEL_URL", "https://t.me/bestveo3promts")

if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN не найден (добавьте в переменные окружения Render).")
if not KIE_API_KEY:
    raise RuntimeError("KIE_API_KEY не найден (добавьте в переменные окружения Render).")

# =========================
# Telegram bot
# =========================
bot = telebot.TeleBot(TELEGRAM_TOKEN, parse_mode="Markdown")

USERS_FILE = "users.json"
try:
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r", encoding="utf-8") as fh:
            users = set(json.load(fh))
    else:
        users = set()
except Exception:
    users = set()

# Простое состояние чата: chat_id -> {phase, prompt, ratio, mode}
STATE = {}

# =========================
# OpenAI (опционально)
# =========================
try:
    from openai import OpenAI
    oai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception:
    oai_client = None

def _oai_model() -> str:
    # можешь заменить на более дешёвую/иную модель
    return "gpt-5"

def chat_completion(messages: list[dict]) -> str:
    if not oai_client:
        raise RuntimeError("OPENAI_API_KEY не задан, ChatGPT-режим недоступен.")
    model = _oai_model()
    # Пытаемся сначала chat.completions, потом Responses API
    try:
        r = oai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=1,  # у новых моделей допустимо только 1
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e1:
        try:
            r = oai_client.responses.create(model=model, input=messages)
            out = getattr(r, "output_text", "") or ""
            return out.strip()
        except Exception as e2:
            raise RuntimeError(f"OpenAI error: {e1} | fallback: {e2}")

# =========================
# KIE API
# =========================
BASE = "https://api.kie.ai"
HDRS = {
    "Authorization": f"Bearer {KIE_API_KEY}",
    "Content-Type": "application/json",
}
MODEL = "veo3_fast"

WAIT_MAX_SEC       = 30 * 60   # максимум ждём 30 минут
POLL_INTERVAL_SEC  = 7
TIMER_EDIT_STEP    = 3
URL_CHECK_INTERVAL = 8

def _with_retries(fn, tries: int = 5, delay: float = 2.0, backoff: float = 2.0):
    last = None
    d = delay
    for _ in range(tries):
        try:
            return fn()
        except Exception as e:
            last = e
            time.sleep(d)
            d *= backoff
    if last:
        raise last

def _post_json(url: str, payload: dict, timeout: int = 60) -> dict:
    def do():
        r = requests.post(url, headers=HDRS, json=payload, timeout=timeout)
        return {"status": r.status_code, "json": r.json()}
    return _with_retries(do)

def _get_json(url: str, params: dict, timeout: int = 40) -> dict:
    def do():
        r = requests.get(url, headers=HDRS, params=params, timeout=timeout)
        return {"status": r.status_code, "json": r.json()}
    return _with_retries(do)

def kie_generate(prompt: str, ratio: str, enable_fallback: bool = True) -> Tuple[Optional[str], Optional[str]]:
    try:
        res = _post_json(f"{BASE}/api/v1/veo/generate", {
            "prompt": prompt,
            "model": MODEL,
            "aspectRatio": ratio,
            "enableFallback": enable_fallback
        })
        data = res.get("json") or {}
        if res.get("status") == 200 and data.get("code") == 200:
            d = data.get("data") or {}
            task_id = (
                d.get("taskId") or d.get("id") or d.get("task_id")
                or data.get("taskId") or data.get("id")
            )
            if task_id:
                return str(task_id), None
            return None, "taskId не найден в ответе."
        return None, data.get("msg") or f"HTTP {res.get('status')}"
    except Exception as e:
        return None, str(e)

def kie_status_raw(task_id: str) -> Tuple[Optional[dict], Optional[str]]:
    try:
        res = _get_json(f"{BASE}/api/v1/veo/record-info", {"taskId": task_id})
        data = res.get("json") or {}
        if res.get("status") == 200 and data.get("code") == 200:
            return data.get("data"), None
        return None, data.get("msg") or f"HTTP {res.get('status')}"
    except Exception as e:
        return None, str(e)

# =========================
# парсинг ответов
# =========================
def _iter_values(obj: Any) -> Iterable[Any]:
    if isinstance(obj, dict):
        for v in obj.values():
            for it in _iter_values(v):
                yield it
    elif isinstance(obj, list):
        for v in obj:
            for it in _iter_values(v):
                yield it
    else:
        yield obj

def _find_success_flag(data: dict) -> Optional[int]:
    def _walk(d: Any) -> Optional[int]:
        if isinstance(d, dict):
            for k, v in d.items():
                lk = str(k).lower()
                if lk in ("successflag", "success_flag", "flag", "status"):
                    try:
                        iv = int(v)
                        if iv in (0, 1, 2, 3):
                            return iv
                    except Exception:
                        pass
                if isinstance(v, (dict, list)):
                    ans = _walk(v)
                    if ans is not None:
                        return ans
        elif isinstance(d, list):
            for it in d:
                ans = _walk(it)
                if ans is not None:
                    return ans
        return None
    return _walk(data)

def _extract_video_urls(data: dict) -> list[str]:
    urls: list[str] = []
    for v in _iter_values(data):
        if isinstance(v, str):
            s = v.strip()
            # иногда backend шлёт строку-JSON с массивом
            if s.startswith("[") and s.endswith("]"):
                try:
                    arr = json.loads(s)
                    if isinstance(arr, list):
                        for u in arr:
                            if isinstance(u, str) and "http" in u and (u.endswith(".mp4") or u.endswith(".mov") or ".m3u8" in u):
                                urls.append(u)
                except Exception:
                    pass
            if "http" in s and (s.endswith(".mp4") or s.endswith(".mov") or ".m3u8" in s):
                urls.append(s)
    # уникализация
    seen = set()
    out: list[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out

def _download_to_temp(url: str, tries: int = 5) -> str:
    last = None
    for i in range(tries):
        try:
            with requests.get(url, stream=True, timeout=180) as resp:
                resp.raise_for_status()
                ext = ".mp4" if ".m3u8" not in url else ".ts"
                f = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
                with f as fh:
                    for chunk in resp.iter_content(chunk_size=1_048_576):
                        if chunk:
                            fh.write(chunk)
                return f.name
        except Exception as e:
            last = e
            time.sleep(2 + i * 2)
    raise last if last else RuntimeError("download failed")

# =========================
# UI
# =========================
def kb_main() -> types.InlineKeyboardMarkup:
    kb = types.InlineKeyboardMarkup(row_width=1)
    kb.add(types.InlineKeyboardButton("🎬 Сгенерировать видео по тексту", callback_data="go_text"))
    kb.add(types.InlineKeyboardButton("📸 Сгенерировать видео по фото (скоро)", callback_data="photo_soon"))
    kb.add(types.InlineKeyboardButton("✍️ Промпт-мастер (ChatGPT)", callback_data="prompt_master"))
    kb.add(types.InlineKeyboardButton("💬 Обычный чат (ChatGPT)", callback_data="free_chat"))
    kb.add(types.InlineKeyboardButton("❓ FAQ", callback_data="faq"))
    kb.add(types.InlineKeyboardButton("💡 Канал с промптами", url=PROMPTS_CHANNEL))
    return kb

def kb_after_success() -> types.InlineKeyboardMarkup:
    kb = types.InlineKeyboardMarkup(row_width=2)
    kb.add(
        types.InlineKeyboardButton("🎬 Сгенерировать ещё", callback_data="go_text"),
        types.InlineKeyboardButton("💡 Вдохновиться промптами", url=PROMPTS_CHANNEL),
    )
    return kb

def main_menu(chat_id: int) -> None:
    bot.send_message(
        chat_id,
        "👋 Привет! Это *SHUBIN AI VIDEO* — генерируем видео через Veo3.\n"
        f"С нами уже *{len(users)}* пользователей.\n\n"
        f"Идеи и примеры здесь 👉 [Канал с промптами]({PROMPTS_CHANNEL})",
        reply_markup=kb_main(),
        disable_web_page_preview=True,
    )

# =========================
# Handlers
# =========================
@bot.message_handler(commands=["start", "menu"])
def on_start(m):
    try:
        users.add(m.from_user.id)
        with open(USERS_FILE, "w", encoding="utf-8") as fh:
            json.dump(list(users), fh, ensure_ascii=False)
    except Exception:
        pass
    STATE[m.chat.id] = {"phase": None, "mode": None}
    main_menu(m.chat.id)

@bot.callback_query_handler(func=lambda _: True)
def on_cb(c):
    cid = c.message.chat.id
    data = c.data or ""
    st = STATE.get(cid) or {}

    if data == "go_text":
        STATE[cid] = {"phase": "await_prompt", "mode": None}
        bot.answer_callback_query(c.id)
        bot.send_message(cid, "✍️ Напиши промпт — коротко опиши желаемое видео.")
        return

    if data == "photo_soon":
        bot.answer_callback_query(c.id)
        bot.send_message(cid, "📸 Режим по фото появится позже.")
        return

    if data == "prompt_master":
        bot.answer_callback_query(c.id)
        STATE[cid] = {"phase": "chat", "mode": "master"}
        bot.send_message(cid, "🧠 *Промпт-мастер*: опиши идею — верну идеальный промпт (EN). `/exit` — выход.")
        return

    if data == "free_chat":
        bot.answer_callback_query(c.id)
        STATE[cid] = {"phase": "chat", "mode": "chat"}
        bot.send_message(cid, "💬 Обычный чат. Пиши сообщения. `/exit` — выход.")
        return

    if data == "faq":
        bot.answer_callback_query(c.id)
        bot.send_message(
            cid,
            "❓ *FAQ*\n"
            "• Форматы: 16:9 и 9:16.\n"
            "• Видео придёт сюда готовым файлом.\n"
            "• Если ссылка появляется с задержкой — просто запусти ещё раз.\n"
            f"• Идеи и примеры: {PROMPTS_CHANNEL}",
            disable_web_page_preview=True,
        )
        return

    if data in ("ratio_16_9", "ratio_9_16"):
        if st.get("phase") != "await_ratio":
            return
        ratio = "16:9" if data == "ratio_16_9" else "9:16"
        st.update({"ratio": ratio, "phase": "ready"})
        STATE[cid] = st
        try:
            bot.edit_message_text(f"✅ Выбран формат: *{ratio}*.\nНажми «🚀 Запустить генерацию».",
                                  chat_id=cid, message_id=c.message.id)
        except Exception:
            pass
        kb = types.InlineKeyboardMarkup()
        kb.add(types.InlineKeyboardButton("🚀 Запустить генерацию", callback_data="run_generation"))
        bot.send_message(cid, "Готово!", reply_markup=kb)
        return

    if data == "run_generation":
        if st.get("phase") != "ready":
            bot.answer_callback_query(c.id, "Сначала выбери формат.")
            return
        bot.answer_callback_query(c.id)
        _run_generation(cid, st.get("prompt", ""), st.get("ratio", "16:9"))
        return

@bot.message_handler(commands=["exit"])
def on_exit(m):
    STATE[m.chat.id] = {"phase": None, "mode": None}
    bot.send_message(m.chat.id, "Вышел из текущего режима. Открываю меню.")
    main_menu(m.chat.id)

# — ввод промпта
@bot.message_handler(func=lambda m: (STATE.get(m.chat.id) or {}).get("phase") == "await_prompt", content_types=["text"])
def on_prompt(m):
    prompt = (m.text or "").strip()
    STATE[m.chat.id] = {"phase": "await_ratio", "prompt": prompt, "mode": None}
    kb = types.InlineKeyboardMarkup(row_width=2)
    kb.add(
        types.InlineKeyboardButton("🎞 16:9", callback_data="ratio_16_9"),
        types.InlineKeyboardButton("📱 9:16", callback_data="ratio_9_16"),
    )
    bot.send_message(m.chat.id, "🎚 Выбери формат и запускай генерацию.", reply_markup=kb)
    bot.send_message(m.chat.id, f"✅ Принял промпт:\n«{prompt}»")

# — ChatGPT режимы
@bot.message_handler(func=lambda m: (STATE.get(m.chat.id) or {}).get("phase") == "chat", content_types=["text"])
def on_chat_modes(m):
    st = STATE.get(m.chat.id) or {}
    mode = st.get("mode")
    txt = (m.text or "").strip()
    try:
        if mode == "master":
            sys = {
                "role": "system",
                "content": (
                    "Ты эксперт по написанию кинематографичных промптов для Google Veo 3. "
                    "Возвращай только сам промпт на английском, без комментариев. "
                    "Упоминай оптику, движение камеры, свет, атмосферу, детали. Без текста в кадре."
                )
            }
            out = chat_completion([sys, {"role": "user", "content": txt}])
            bot.send_message(m.chat.id, f"📝 Промпт для *Veo3*:\n```\n{out}\n```")
        else:
            sys = {"role": "system", "content": "Ты дружелюбный и краткий помощник."}
            out = chat_completion([sys, {"role": "user", "content": txt}])
            bot.send_message(m.chat.id, out or "…")
    except Exception as e:
        bot.send_message(m.chat.id, f"❌ Ошибка ChatGPT: {e}")

# =========================
# CORE генерация Veo3
# =========================
def _run_generation(chat_id: int, prompt: str, ratio: str):
    t0 = time.time()
    timer_msg = bot.send_message(chat_id, "⏳ Генерация идёт…")
    shown_sec = 0

    def tick():
        nonlocal shown_sec
        sec = int(time.time() - t0)
        if sec - shown_sec >= TIMER_EDIT_STEP:
            shown_sec = sec
            try:
                bot.edit_message_text(f"⏳ Генерация идёт… *{sec} сек*",
                                      chat_id=chat_id, message_id=timer_msg.id)
            except Exception:
                pass

    # 1) создать задачу
    task_id, err = kie_generate(prompt=prompt, ratio=ratio, enable_fallback=True)
    if err or not task_id:
        try:
            bot.edit_message_text(f"❌ Не удалось создать задачу: {err}",
                                  chat_id=chat_id, message_id=timer_msg.id)
        except Exception:
            bot.send_message(chat_id, f"❌ Не удалось создать задачу: {err}")
        return
    bot.send_message(chat_id, f"🧾 Задача создана. taskId={task_id}")

    # 2) ждём статус и ссылку
    deadline = time.time() + WAIT_MAX_SEC
    urls: list[str] = []

    while time.time() < deadline:
        info, serr = kie_status_raw(task_id)
        tick()
        if serr:
            time.sleep(POLL_INTERVAL_SEC)
            continue

        flag = _find_success_flag(info or {})
        if flag in (2, 3):
            try:
                bot.edit_message_text("❌ Генерация не удалась на стороне провайдера.",
                                      chat_id=chat_id, message_id=timer_msg.id)
            except Exception:
                bot.send_message(chat_id, "❌ Генерация не удалась на стороне провайдера.")
            return

        urls = _extract_video_urls(info or {})
        if urls:
            break

        time.sleep(URL_CHECK_INTERVAL)

    if not urls:
        try:
            bot.edit_message_text("⚠️ Видео, похоже, готово, но ссылка ещё не доступна. Попробуйте позже.",
                                  chat_id=chat_id, message_id=timer_msg.id)
        except Exception:
            bot.send_message(chat_id, "⚠️ Видео, похоже, готово, но ссылка ещё не доступна. Попробуйте позже.")
        return

    # 3) скачать и отправить
    video_url = urls[0]
    try:
        path = _download_to_temp(video_url)
        try:
            bot.edit_message_text("📥 Загружаю видео в Telegram…",
                                  chat_id=chat_id, message_id=timer_msg.id)
        except Exception:
            pass

        with open(path, "rb") as f:
            bot.send_video(
                chat_id, f,
                caption=f"✅ Готово! Формат: *{ratio}*",
                supports_streaming=True,
                reply_markup=kb_after_success(),
            )

        bot.send_message(
            chat_id,
            f"🔥 Нужны идеи промптов? Подписывайся: {PROMPTS_CHANNEL}",
            disable_web_page_preview=True,
        )
    except Exception:
        try:
            bot.edit_message_text("✅ Видео готово, но не удалось загрузить файл (лимиты/сеть).",
                                  chat_id=chat_id, message_id=timer_msg.id)
        except Exception:
            pass
        bot.send_message(chat_id, "Попробуй ещё раз. Если повторится — проверим размер файла и лимиты Telegram.")
    finally:
        try:
            if "path" in locals() and os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

# — общий фоллбек
@bot.message_handler(content_types=["text", "photo", "document", "video", "sticker", "audio", "voice"])
def on_fallback(m):
    st = STATE.get(m.chat.id) or {}
    if st.get("phase") == "chat" and m.content_type != "text":
        bot.send_message(m.chat.id, "Пожалуйста, напиши текст. `/exit` — выход.")
        return
    bot.send_message(m.chat.id, "Открой меню: /menu")

# =========================
# RUN
# =========================
if __name__ == "__main__":
    print("✅ Бот запущен. Ожидаю сообщения…")
    bot.polling(none_stop=True, long_polling_timeout=60)
