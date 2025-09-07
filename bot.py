# -*- coding: utf-8 -*-
# SHUBIN AI VIDEO — Veo3 Fast + ChatGPT (Prompt-Master & Chat)
# Stack: pyTelegramBotAPI (telebot) + requests + python-dotenv + openai

import os, json, time, tempfile, requests, traceback
from typing import Optional, Tuple, Any, Iterable, List
from dotenv import load_dotenv
import telebot
from telebot import types

# ===== ENV =====
load_dotenv()
TELEGRAM_TOKEN     = os.getenv("TELEGRAM_TOKEN", "")
KIE_API_KEY        = os.getenv("KIE_API_KEY", "")
KIE_BASE_URL       = os.getenv("KIE_BASE_URL", "https://api.kie.ai")
KIE_GEN_PATH       = os.getenv("KIE_GEN_PATH", "/api/v1/veo/generate")
KIE_STATUS_PATH    = os.getenv("KIE_STATUS_PATH", "/api/v1/veo/record-info")
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY", "")  # опционально
PROMPTS_CHANNEL_URL= os.getenv("PROMPTS_CHANNEL_URL", "https://t.me/bestveo3promts")

if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN отсутствует")
if not KIE_API_KEY:
    raise RuntimeError("KIE_API_KEY отсутствует")

# ===== bot / storage =====
bot = telebot.TeleBot(TELEGRAM_TOKEN, parse_mode="Markdown")
USERS_FILE = "users.json"
users = set()
if os.path.exists(USERS_FILE):
    try:
        users = set(json.load(open(USERS_FILE, "r", encoding="utf-8")))
    except Exception:
        users = set()

# chat state
STATE: dict[int, dict] = {}   # chat_id -> {...}

# ===== KIE config =====
HDRS = {"Authorization": f"Bearer {KIE_API_KEY}", "Content-Type": "application/json"}
MODEL = "veo3_fast"
WAIT_MAX_SECS     = 30 * 60
POLL_INTERVAL_SECS= 7
TIMER_EDIT_STEP   = 3

def _u(path: str) -> str:
    return f"{KIE_BASE_URL.rstrip('/')}/{path.lstrip('/')}"

# ===== OpenAI (Prompt-Master & Chat) =====
try:
    from openai import OpenAI
    oai = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception:
    oai = None

def _oai_model() -> str:
    # можно заменить на более дешёвую, если нужно
    return "gpt-5"

def _oai_chat(messages: List[dict]) -> str:
    if not oai:
        raise RuntimeError("OPENAI_API_KEY не задан — режим ChatGPT отключён")
    model = _oai_model()
    try:
        r = oai.chat.completions.create(model=model, messages=messages, temperature=1)
        return (r.choices[0].message.content or "").strip()
    except Exception as e1:
        # fallback — Responses API
        r = oai.responses.create(model=model, input=messages)
        text = getattr(r, "output_text", "") or ""
        if text:
            return text.strip()
        raise RuntimeError(f"OpenAI error: {e1}")

# ===== сетевые утилиты (ретраи) =====
def _with_retries(fn, tries=4, delay=2, backoff=2):
    last = None; d = delay
    for _ in range(tries):
        try:
            return fn()
        except Exception as e:
            last = e
            time.sleep(d); d *= backoff
    raise last or RuntimeError("retry failed")

def _post_json(url: str, payload: dict, timeout=60) -> dict:
    def run():
        r = requests.post(url, headers=HDRS, json=payload, timeout=timeout)
        j = {}
        try: j = r.json()
        except Exception: j = {"error": r.text}
        return {"status": r.status_code, "json": j}
    return _with_retries(run)

def _get_json(url: str, params: dict, timeout=40) -> dict:
    def run():
        r = requests.get(url, headers=HDRS, params=params, timeout=timeout)
        j = {}
        try: j = r.json()
        except Exception: j = {"error": r.text}
        return {"status": r.status_code, "json": j}
    return _with_retries(run)

def _download_to_temp(url: str) -> str:
    with requests.get(url, stream=True, timeout=180) as resp:
        resp.raise_for_status()
        suffix = ".mp4" if ".m3u8" not in url else ".ts"
        f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        with f as fh:
            for chunk in resp.iter_content(chunk_size=1_048_576):
                if chunk: fh.write(chunk)
        return f.name

# ===== KIE endpoints =====
def kie_generate(prompt: str, ratio: str, enable_fallback: bool = True) -> Tuple[Optional[str], Optional[str]]:
    res = _post_json(_u(KIE_GEN_PATH), {
        "prompt": prompt,
        "model": MODEL,
        "aspectRatio": ratio,
        "enableFallback": enable_fallback
    })
    data = res.get("json") or {}
    if res.get("status") == 200 and data.get("code") == 200:
        d = data.get("data") or {}
        task_id = d.get("taskId") or d.get("id") or data.get("taskId") or d.get("task_id")
        if task_id: return str(task_id), None
        return None, "taskId не найден в ответе"
    return None, data.get("msg") or f"HTTP {res.get('status')}"

def kie_status(task_id: str) -> Tuple[Optional[dict], Optional[str]]:
    res = _get_json(_u(KIE_STATUS_PATH), {"taskId": task_id})
    data = res.get("json") or {}
    if res.get("status") == 200 and data.get("code") == 200:
        return data.get("data") or {}, None
    return None, data.get("msg") or f"HTTP {res.get('status')}"

# ===== парсинг ответов KIE =====
def _iter_values(obj: Any):
    if isinstance(obj, dict):
        for v in obj.values(): yield from _iter_values(v)
    elif isinstance(obj, list):
        for v in obj: yield from _iter_values(v)
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
                        if iv in (0, 1, 2, 3): return iv
                    except Exception:
                        pass
                ans = _walk(v)
                if ans is not None: return ans
        elif isinstance(d, list):
            for it in d:
                ans = _walk(it)
                if ans is not None: return ans
        return None
    return _walk(data)

def _extract_video_urls(data: dict) -> List[str]:
    urls: List[str] = []
    for v in _iter_values(data):
        if isinstance(v, str):
            s = v.strip()
            # иногда ссылки приходят строкой-массивом: "[...]", распарсим
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
    # unique, сохраняя порядок
    seen, out = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u); out.append(u)
    return out

# ===== UI =====
def _menu_kb() -> types.InlineKeyboardMarkup:
    kb = types.InlineKeyboardMarkup(row_width=1)
    kb.add(types.InlineKeyboardButton("🎬 Сгенерировать видео по тексту", callback_data="go_text"))
    kb.add(types.InlineKeyboardButton("📸 Сгенерировать по фото (скоро)", callback_data="photo_soon"))
    kb.add(types.InlineKeyboardButton("🧠 Промпт-мастер (ChatGPT)", callback_data="prompt_master"))
    kb.add(types.InlineKeyboardButton("💬 Обычный чат (ChatGPT)", callback_data="free_chat"))
    kb.add(types.InlineKeyboardButton("❓ FAQ", callback_data="faq"))
    kb.add(types.InlineKeyboardButton("💡 Канал с промптами", url=PROMPTS_CHANNEL_URL))
    return kb

def _after_success_kb() -> types.InlineKeyboardMarkup:
    kb = types.InlineKeyboardMarkup(row_width=2)
    kb.add(
        types.InlineKeyboardButton("🎬 Сгенерировать ещё", callback_data="go_text"),
        types.InlineKeyboardButton("💡 Вдохновиться промптами", url=PROMPTS_CHANNEL_URL),
    )
    return kb

def main_menu(chat_id: int):
    bot.send_message(
        chat_id,
        "👋 Привет! Это *SHUBIN AI VIDEO* — генерируем видео через Veo3.\n"
        f"С нами уже *{len(users)}* пользователей.\n\n"
        f"Идеи и примеры здесь 👉 [Канал с промптами]({PROMPTS_CHANNEL_URL})",
        reply_markup=_menu_kb(),
        disable_web_page_preview=True,
    )

# ===== Handlers =====
@bot.message_handler(commands=["start", "menu"])
def start_cmd(m):
    users.add(m.from_user.id)
    try: json.dump(list(users), open(USERS_FILE, "w", encoding="utf-8"))
    except Exception: pass
    STATE[m.chat.id] = {"phase": None, "mode": None}
    main_menu(m.chat.id)

@bot.callback_query_handler(func=lambda c: True)
def on_cb(c):
    cid, data = c.message.chat.id, c.data
    st = STATE.get(cid) or {}

    if data == "go_text":
        STATE[cid] = {"phase": "await_prompt", "mode": None}
        bot.answer_callback_query(c.id)
        bot.send_message(cid, "✍️ Напиши промпт (1–2 фразы).")
        return

    if data == "photo_soon":
        bot.answer_callback_query(c.id)
        bot.send_message(cid, "📸 Этот режим появится позже.")
        return

    if data == "prompt_master":
        bot.answer_callback_query(c.id)
        STATE[cid] = {"phase": "chat", "mode": "master"}
        if not oai:
            bot.send_message(cid, "❌ Ошибка ChatGPT: OPENAI_API_KEY не задан, режим недоступен.")
        bot.send_message(cid, "🧠 Режим *Промпт-мастер*: опиши идею — верну английский промпт.\n`/exit` — выход.")
        return

    if data == "free_chat":
        bot.answer_callback_query(c.id)
        STATE[cid] = {"phase": "chat", "mode": "chat"}
        if not oai:
            bot.send_message(cid, "❌ Ошибка ChatGPT: OPENAI_API_KEY не задан, режим недоступен.")
        bot.send_message(cid, "💬 Обычный чат. Пиши сообщения. `/exit` — выход.")
        return

    if data == "faq":
        bot.answer_callback_query(c.id)
        bot.send_message(cid,
            "❓ *FAQ*\n"
            "• Форматы: 16:9 и 9:16.\n"
            "• Видео приходит сюда готовым файлом.\n"
            "• Если долго нет видео — запусти ещё раз (ссылки иногда появляются с задержкой).",
            disable_web_page_preview=True)
        return

    if data in ("ratio_16_9", "ratio_9_16"):
        if st.get("phase") != "await_ratio": return
        ratio = "16:9" if data == "ratio_16_9" else "9:16"
        st.update({"ratio": ratio, "phase": "ready"}); STATE[cid] = st
        try:
            bot.edit_message_text(f"✅ Выбран формат: *{ratio}*. Нажми «🚀 Запустить генерацию».",
                                  chat_id=cid, message_id=c.message.id)
        except Exception: pass
        kb = types.InlineKeyboardMarkup()
        kb.add(types.InlineKeyboardButton("🚀 Запустить генерацию", callback_data="run_generation"))
        bot.send_message(cid, "Готово!", reply_markup=kb)
        return

    if data == "run_generation":
        if st.get("phase") != "ready":
            bot.answer_callback_query(c.id, "Сначала выбери формат.")
            return
        bot.answer_callback_query(c.id)
        _run_generation(cid, st["prompt"], st["ratio"])
        return

@bot.message_handler(commands=["exit"])
def exit_mode(m):
    STATE[m.chat.id] = {"phase": None, "mode": None}
    bot.send_message(m.chat.id, "Вышел из режима. Открываю меню.")
    main_menu(m.chat.id)

# — ввод промпта
@bot.message_handler(func=lambda m: (STATE.get(m.chat.id) or {}).get("phase") == "await_prompt", content_types=["text"])
def on_prompt(m):
    prompt = (m.text or "").strip()
    STATE[m.chat.id] = {"phase": "await_ratio", "prompt": prompt, "mode": None}
    kb = types.InlineKeyboardMarkup(row_width=2)
    kb.add(types.InlineKeyboardButton("🎞 16:9", callback_data="ratio_16_9"),
           types.InlineKeyboardButton("📱 9:16", callback_data="ratio_9_16"))
    bot.send_message(m.chat.id, "🎚 Выбери формат и запускай генерацию.", reply_markup=kb)
    bot.send_message(m.chat.id, f"✅ Принял промпт:\n`{prompt}`")

# — режимы ChatGPT
@bot.message_handler(func=lambda m: (STATE.get(m.chat.id) or {}).get("phase") == "chat", content_types=["text"])
def chat_modes(m):
    mode = (STATE.get(m.chat.id) or {}).get("mode")
    user_text = (m.text or "").strip()
    try:
        if not oai:
            bot.send_message(m.chat.id, "❌ ChatGPT недоступен (нет OPENAI_API_KEY).")
            return
        if mode == "master":
            sys = {"role": "system", "content":
                   "Ты эксперт по написанию кинематографичных промптов для Google Veo 3. "
                   "Верни только ОДИН промпт на английском, без комментариев. "
                   "Укажи оптику/движение/свет/атмосферу/детали. Без текста в кадре."}
            out = _oai_chat([sys, {"role": "user", "content": user_text}])
            bot.send_message(m.chat.id, f"📝 Промпт для *Veo3*:\n```\n{out}\n```")
        else:
            sys = {"role": "system", "content": "Ты дружелюбный и краткий помощник."}
            out = _oai_chat([sys, {"role": "user", "content": user_text}])
            bot.send_message(m.chat.id, out or "…")
    except Exception as e:
        bot.send_message(m.chat.id, f"❌ Ошибка ChatGPT: {e}")

# ===== Core: генерация Veo3 =====
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

    # 1) создаём задачу
    task_id, err = kie_generate(prompt, ratio, enable_fallback=True)
    if err or not task_id:
        try:
            bot.edit_message_text(f"❌ Не удалось создать задачу: {err}", chat_id=chat_id, message_id=timer_msg.id)
        except Exception:
            bot.send_message(chat_id, f"❌ Не удалось создать задачу: {err}")
        return
    bot.send_message(chat_id, f"🧾 Задача создана. taskId={task_id}")

    # 2) опрос статуса
    deadline = time.time() + WAIT_MAX_SECS
    urls: List[str] = []

    while time.time() < deadline:
        info, serr = kie_status(task_id)
        tick()
        if serr:
            time.sleep(POLL_INTERVAL_SECS); continue

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

        time.sleep(POLL_INTERVAL_SECS)

    if not urls:
        try:
            bot.edit_message_text("⚠️ Видео готово, но ссылка ещё не появилась. Попробуйте позже.",
                                  chat_id=chat_id, message_id=timer_msg.id)
        except Exception:
            bot.send_message(chat_id, "⚠️ Видео готово, но ссылка ещё не появилась. Попробуйте позже.")
        return

    # 3) скачиваем и отправляем
    video_url = urls[0]
    try:
        path = _download_to_temp(video_url)
        try:
            bot.edit_message_text("📥 Загружаю видео в Telegram…", chat_id=chat_id, message_id=timer_msg.id)
        except Exception:
            pass

        with open(path, "rb") as f:
            bot.send_video(chat_id, f, supports_streaming=True,
                           caption=f"✅ Готово! Формат: *{ratio}*", reply_markup=_after_success_kb())
        bot.send_message(chat_id, f"🔥 Больше идей промптов: {PROMPTS_CHANNEL_URL}", disable_web_page_preview=True)
    except Exception:
        try:
            bot.edit_message_text("✅ Видео готово, но не удалось отправить файл (сеть/лимиты).",
                                  chat_id=chat_id, message_id=timer_msg.id)
        except Exception:
            pass
    finally:
        try:
            if 'path' in locals() and os.path.exists(path): os.remove(path)
        except Exception:
            pass

# ===== Fallback =====
@bot.message_handler(content_types=["text", "photo", "document", "video", "sticker", "audio", "voice"])
def fallback(m):
    st = STATE.get(m.chat.id) or {}
    if st.get("phase") == "chat" and m.content_type != "text":
        bot.send_message(m.chat.id, "Напиши текст. `/exit` — выход.")
        return
    bot.send_message(m.chat.id, "Открой меню: /menu")

# ===== RUN =====
if __name__ == "__main__":
    print("✅ Bot is running…")
    # устойчивый polling (без PTB/Updater)
    bot.infinity_polling(timeout=60, long_polling_timeout=60, skip_pending=True)
