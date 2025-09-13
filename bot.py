# -*- coding: utf-8 -*-
# BOT2 — Veo/MJ/Banana/Prompt-Master — PTB 21.x
# Версия: 2025-09-13 (cleanup + prices + promos + Redis persistence)

import os
import json
import time
import asyncio
import logging
import re
from typing import Dict, Any, List, Tuple, Optional

import requests
import redis
from dotenv import load_dotenv

from telegram import (
    Update, InlineKeyboardButton, InlineKeyboardMarkup,
    InputMediaPhoto, LabeledPrice, Bot
)
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler,
    CallbackQueryHandler, filters, AIORateLimiter, PreCheckoutQueryHandler
)

# ---- Banana API wrapper
from kie_banana import create_banana_task, get_banana_status

# ==========================
#   ENV / INIT
# ==========================
load_dotenv()

def _env(k: str, d: str = "") -> str:
    v = os.getenv(k)
    return (v if v is not None else d).strip()

TELEGRAM_TOKEN = _env("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN is not set")

REDIS_URL = _env("REDIS_URL")
if not REDIS_URL:
    raise RuntimeError("REDIS_URL is not set (нужен для постоянного баланса/промо)")

PROMPTS_CHANNEL_URL = _env("PROMPTS_CHANNEL_URL", "https://t.me/bestveo3promts")
STARS_BUY_URL       = _env("STARS_BUY_URL", "https://t.me/PremiumBot")

# ---- OpenAI (Prompt-Master) — опционально
OPENAI_API_KEY  = _env("OPENAI_API_KEY")
OPENAI_API_BASE = _env("OPENAI_API_BASE")
try:
    from openai import OpenAI
    oai = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE or None) if OPENAI_API_KEY else None
except Exception:
    oai = None

# ---- KIE (VEO/MJ)
KIE_API_KEY         = _env("KIE_API_KEY")
KIE_BASE_URL        = _env("KIE_BASE_URL", "https://api.kie.ai")
KIE_VEO_GEN_PATH    = _env("KIE_VEO_GEN_PATH",    "/api/v1/veo/generate")
KIE_VEO_STATUS_PATH = _env("KIE_VEO_STATUS_PATH", "/api/v1/veo/record-info")
KIE_VEO_1080_PATH   = _env("KIE_VEO_1080_PATH",   "/api/v1/veo/get-1080p-video")
KIE_MJ_GENERATE     = _env("KIE_MJ_GENERATE",     "/api/v1/mj/generate")
KIE_MJ_STATUS       = _env("KIE_MJ_STATUS",       "/api/v1/mj/record-info")

# ---- Redis
rdb = redis.from_url(REDIS_URL, decode_responses=True)
R_PREFIX = _env("REDIS_PREFIX", "bot2")

def rkey(*parts: str) -> str:
    return ":".join([R_PREFIX, *parts])

# ---- Promo codes (одноразово на пользователя)
# Пример: {"TEST10":10, "MEGA50":50}
PROMO_CODES: Dict[str, int] = {}
try:
    pc_json = _env("PROMO_CODES_JSON")
    if pc_json:
        PROMO_CODES = json.loads(pc_json)
except Exception:
    PROMO_CODES = {}

# ---- Logs
LOG_LEVEL = _env("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("bot2")

# ==========================
#   Prices / Tokens
# ==========================
TOKEN_COSTS = {
    "veo_fast": 50,
    "veo_quality": 150,
    "veo_photo": 50,
    "mj": 15,
    "banana": 5,
}

# Stars packs with bonuses
STARS_PACKS: Dict[int, int] = {50:50, 100:110, 200:220, 300:330, 400:440, 500:550}

# ==========================
#   Balance (Redis)
# ==========================
def get_balance(uid: int) -> int:
    try:
        return int(rdb.get(rkey("bal", str(uid))) or 0)
    except Exception:
        return 0

def set_balance(uid: int, v: int):
    rdb.set(rkey("bal", str(uid)), max(0, int(v)))

def add_tokens(uid: int, add: int):
    set_balance(uid, get_balance(uid) + int(add))

def try_charge(uid: int, need: int) -> Tuple[bool, int]:
    bal = get_balance(uid)
    if bal < need:
        return False, bal
    set_balance(uid, bal - need)
    return True, bal - need

def signup_bonus_if_needed(uid: int, amount: int = 10) -> bool:
    flag = rdb.get(rkey("bonus", str(uid)))
    if flag:
        return False
    # бонус только если у пользователя ещё было 0
    if get_balance(uid) == 0:
        add_tokens(uid, amount)
    rdb.set(rkey("bonus", str(uid)), "1")
    return True

# ==========================
#   UI
# ==========================
WELCOME = (
    "🎬 Veo 3 — съёмочная команда: опиши идею и получи готовый клип!\n"
    "🖌️ MJ — художник: нарисует изображение по твоему тексту.\n"
    "🍌 Banana — Редактор изображений из будущего\n"
    "🧠 Prompt-Master — вернёт профессиональный кинопромпт.\n"
    "💬 Обычный чат — общение с ИИ.\n"
    "💎 Ваш баланс: *{balance}*\n"
    "📈 Больше идей и примеров: {prompts_url}\n\n"
    "Выберите режим 👇"
)

def main_menu_kb() -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton("🎬 VEO Fast  💎50",      callback_data="mode:veo_fast")],
        [InlineKeyboardButton("🎬 VEO Quality  💎150",   callback_data="mode:veo_quality")],
        [InlineKeyboardButton("🖼️ MJ  💎15",            callback_data="mode:mj")],
        [InlineKeyboardButton("🍌 Banana  💎5",         callback_data="mode:banana")],
        [InlineKeyboardButton("📸 Оживить фото (VEO) 💎50", callback_data="mode:veo_photo")],
        [InlineKeyboardButton("🧠 Prompt-Master",       callback_data="mode:prompt_master")],
        [InlineKeyboardButton("💬 Обычный чат (бесплатно)", callback_data="mode:chat")],
        [
            InlineKeyboardButton("❓ FAQ", callback_data="faq"),
            InlineKeyboardButton("📈 Канал с промптами", url=PROMPTS_CHANNEL_URL),
        ],
        [InlineKeyboardButton("💳 Пополнить баланс", callback_data="topup")],
    ]
    return InlineKeyboardMarkup(rows)

def stars_topup_kb() -> InlineKeyboardMarkup:
    rows: List[List[InlineKeyboardButton]] = []
    for stars in [50, 100, 200, 300, 400, 500]:
        tokens = STARS_PACKS[stars]
        rows.append([InlineKeyboardButton(f"⭐ {stars} → 💎 {tokens}", callback_data=f"buy:{stars}")])
    rows.append([InlineKeyboardButton("🛒 Где купить Stars", url=STARS_BUY_URL)])
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="back")])
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

def build_card_text_veo(s: Dict[str, Any], bal: int) -> str:
    prompt_preview = (s.get("last_prompt") or "").strip()
    if len(prompt_preview) > 900:
        prompt_preview = prompt_preview[:900] + "…"
    has_prompt = "есть" if s.get("last_prompt") else "нет"
    has_ref = "есть" if s.get("last_image_url") else "нет"
    model = "Fast" if s.get("model") == "veo3_fast" else ("Quality" if s.get("model") == "veo3" else "—")
    price = TOKEN_COSTS['veo_quality'] if s.get('model') == 'veo3' else TOKEN_COSTS['veo_fast']
    lines = [
        "🎬 *Карточка VEO*",
        "",
        "✍️ *Промпт:*",
        f"`{prompt_preview or '—'}`",
        "",
        "🧩 *Параметры:*",
        f"• Формат: *{s.get('aspect') or '—'}*",
        f"• Модель: *{model}*",
        f"• Промпт: *{has_prompt}*",
        f"• Фото-реф: *{has_ref}*",
        "",
        f"💎 *Стоимость запуска:* {price}",
        f"💼 Баланс: *{bal}*",
    ]
    return "\n".join(lines)

def card_keyboard_veo(s: Dict[str, Any]) -> InlineKeyboardMarkup:
    rows: List[List[InlineKeyboardButton]] = []
    rows.append([InlineKeyboardButton("🖼️ Добавить/Удалить фото", callback_data="card:toggle_photo"),
                 InlineKeyboardButton("✍️ Изменить промпт",       callback_data="card:edit_prompt")])
    rows.append([InlineKeyboardButton("🧠 Prompt-Master (помочь с текстом)", callback_data="card:pm")])
    rows.append(aspect_row(s.get("aspect") or "16:9"))
    rows.append(model_row(s.get("model") or "veo3_fast"))
    rows.append([InlineKeyboardButton("🚀 Сгенерировать", callback_data="card:generate")])
    rows.append([InlineKeyboardButton("🔁 Начать заново", callback_data="card:reset"),
                 InlineKeyboardButton("⬅️ Назад",         callback_data="back")])
    rows.append([InlineKeyboardButton("💳 Пополнить баланс", callback_data="topup")])
    return InlineKeyboardMarkup(rows)

def banana_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("➕ Добавить фото", callback_data="banana:add_photo")],
        [InlineKeyboardButton("🧹 Очистить фото", callback_data="banana:reset")],
        [InlineKeyboardButton("🚀 Начать",        callback_data="banana:start")],
        [InlineKeyboardButton("⬅️ Назад",         callback_data="back")],
    ])

def mj_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("✍️ Добавить описание", callback_data="mj:add_prompt")],
        [InlineKeyboardButton("🏙️ Сгенерировать 16:9", callback_data="mj:start")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="back")]
    ])

# ==========================
#   State
# ==========================
DEFAULT_STATE = {
    "mode": None,
    "aspect": "16:9",
    "model": "veo3_fast",
    "last_prompt": None,
    "last_image_url": None,
    "last_task_id": None,
    "mj_last_wait": 0.0,
    "banana_mode": False,
    "banana_photos": [],
    "banana_prompt": None,
    "pm_return_to": None,
}

def S(ctx: ContextTypes.DEFAULT_TYPE) -> Dict[str, Any]:
    ud = ctx.user_data
    for k, v in DEFAULT_STATE.items():
        if k not in ud:
            ud[k] = (list(v) if isinstance(v, list) else v)
    # гарантируем тип списка
    if not isinstance(ud.get("banana_photos"), list):
        ud["banana_photos"] = []
    return ud

# ==========================
#   HTTP helpers (KIE)
# ==========================
def _kie_headers_json() -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    tok = (KIE_API_KEY or "").strip()
    if tok and not tok.lower().startswith("bearer "):
        tok = f"Bearer {tok}"
    if tok:
        h["Authorization"] = tok
    return h

def _post_json(url: str, payload: Dict[str, Any], timeout: int = 40):
    r = requests.post(url, json=payload, headers=_kie_headers_json(), timeout=timeout)
    try: return r.status_code, r.json()
    except Exception: return r.status_code, {"error": r.text}

def _get_json(url: str, params: Dict[str, Any], timeout: int = 40):
    r = requests.get(url, params=params, headers=_kie_headers_json(), timeout=timeout)
    try: return r.status_code, r.json()
    except Exception: return r.status_code, {"error": r.text}

def join_url(base: str, path: str) -> str:
    u = f"{base.rstrip('/')}/{path.lstrip('/')}"
    return u.replace("://", "§§").replace("//", "/").replace("§§", "://")

# ==========================
#   VEO
# ==========================
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
    status, j = _post_json(join_url(KIE_BASE_URL, KIE_VEO_GEN_PATH),
                           _build_payload_for_veo(prompt, aspect, image_url, model_key))
    code = j.get("code", status)
    if status == 200 and code == 200:
        data = j.get("data") or j
        tid = (data.get("taskId") or data.get("taskid") or data.get("id"))
        if tid: return True, str(tid), "ok"
        return False, None, "Ответ без taskId."
    return False, None, j.get("msg") or j.get("message") or j.get("error") or f"HTTP {status}"

def get_kie_veo_status(task_id: str) -> Tuple[bool, Optional[str], Optional[str]]:
    status, j = _get_json(join_url(KIE_BASE_URL, KIE_VEO_STATUS_PATH), {"taskId": task_id})
    code = j.get("code", status)
    if status == 200 and code == 200:
        data = j.get("data") or {}
        state = str(data.get("state") or "").lower() or None
        res = None
        # извлечём resultUrls из resultJson/resultInfoJson
        for key in ("resultJson", "resultInfoJson"):
            val = data.get(key)
            if not val: continue
            if isinstance(val, str):
                try: d = json.loads(val)
                except Exception: d = {}
            else:
                d = val if isinstance(val, dict) else {}
            urls = d.get("resultUrls") or d.get("videoUrls") or []
            if isinstance(urls, list) and urls:
                res = urls[0]; break
        return True, state, res
    return False, None, j.get("msg") or j.get("message") or j.get("error") or f"HTTP {status}"

async def try_get_1080_url(task_id: str) -> Optional[str]:
    try:
        status, j = _get_json(join_url(KIE_BASE_URL, KIE_VEO_1080_PATH), {"taskId": task_id}, timeout=20)
        code = j.get("code", status)
        if status == 200 and code == 200:
            data = j.get("data") or {}
            u = (data.get("url") or data.get("downloadUrl"))
            if u: return u
    except Exception:
        pass
    return None

# ==========================
#   MJ
# ==========================
def mj_generate(prompt: str) -> Tuple[bool, Optional[str], str]:
    payload = {
        "taskType": "mj_txt2img",
        "prompt": prompt,
        "speed": "turbo",
        "aspectRatio": "16:9",
        "version": "7",
        "enableTranslation": True,
    }
    status, j = _post_json(join_url(KIE_BASE_URL, KIE_MJ_GENERATE), payload)
    code = j.get("code", status)
    if status == 200 and code == 200:
        data = j.get("data") or j
        tid = (data.get("taskId") or data.get("taskid") or data.get("id"))
        if tid: return True, str(tid), "ok"
        return False, None, "Ответ без taskId."
    return False, None, j.get("msg") or j.get("message") or j.get("error") or f"HTTP {status}"

def mj_status(task_id: str) -> Tuple[bool, Optional[int], Optional[Dict[str, Any]]]:
    status, j = _get_json(join_url(KIE_BASE_URL, KIE_MJ_STATUS), {"taskId": task_id})
    code = j.get("code", status)
    if status == 200 and code == 200:
        data = j.get("data") or {}
        try: flag = int(data.get("successFlag"))
        except Exception: flag = None
        return True, flag, data
    return False, None, None

def _extract_mj_image_urls(d: Dict[str, Any]) -> List[str]:
    out = []
    rj = d.get("resultInfoJson") or {}
    arr = rj.get("resultUrls") or []
    if isinstance(arr, list):
        for u in arr:
            if isinstance(u, str) and u.startswith("http"):
                out.append(u)
    elif isinstance(arr, str):
        try:
            ll = json.loads(arr)
            for u in ll:
                if isinstance(u, str) and u.startswith("http"):
                    out.append(u)
        except Exception:
            pass
    return out

# ==========================
#   Prompt-Master
# ==========================
async def oai_prompt_master(idea_text: str) -> Optional[str]:
    txt = (idea_text or "").strip()
    if not txt:
        return None

    # вытащим реплики из кавычек и маркеры «озвучка»
    quoted = re.findall(r"“([^”]+)”|\"([^\"]+)\"", txt)
    quoted_dialogue = "; ".join([a or b for (a, b) in quoted if (a or b)]).strip()
    vo_match = re.search(r"(?:озвучка|голос|диктор|voiceover|vo)\s*:?\s*(.+)", txt, flags=re.I)
    is_ru_vo = bool(re.search(r"на\s+русском|russian\s+(?:voice|voiceover|audio)|русская\s+озвучка", txt, re.I))
    dialogue_text = (quoted_dialogue or (vo_match.group(1).strip() if vo_match else "")).strip()

    text_wo_dialogue = txt
    if quoted:
        text_wo_dialogue = re.sub(r"“[^”]+”|\"[^\"]+\"", "", text_wo_dialogue)
    if vo_match:
        text_wo_dialogue = text_wo_dialogue.replace(vo_match.group(0), "")
    text_wo_dialogue = text_wo_dialogue.strip()

    if oai:
        system = (
            "You are a Prompt-Master for cinematic AI video generation (Veo-style). "
            "Return ONE multi-line prompt in ENGLISH using EXACT labels and order:\n"
            "High-quality cinematic 4K video (16:9).\n"
            "Scene: ...\nCamera: ...\nAction: ...\nDialogue: ...\nLip-sync: ...\nAudio: ...\n"
            "Lighting: ...\nWardrobe/props: ...\nFraming: ...\n"
            "Constraints: No subtitles. No on-screen text. No logos.\n\n"
            "Rules: Keep 16:9, be specific (600–1100 chars). "
            "If the user supplies dialogue or asks for Russian voiceover, keep Dialogue EXACTLY as given in Russian "
            "and require per-syllable lip-sync for Russian."
        )
        try:
            resp = await asyncio.to_thread(
                oai.chat.completions.create,
                model=_env("OPENAI_PM_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps({
                        "idea": text_wo_dialogue[:900],
                        "dialogue_raw": dialogue_text[:300],
                        "need_russian_vo": is_ru_vo
                    }, ensure_ascii=False)}
                ],
                temperature=float(_env("OPENAI_PM_TEMPERATURE", "0.8")),
                max_tokens=int(_env("OPENAI_PM_MAXTOKENS", "900"))
            )
            out = (resp.choices[0].message.content or "").strip()
            if out:
                return out[:1400]
        except Exception as e:
            log.warning("OpenAI PM failed: %s", e)

    # fallback
    Scene = text_wo_dialogue or "Visually rich, story-driven moment in a believable world."
    Camera = "Slow dolly-in, gentle arcs, depth-of-field pulls; 24–30fps, natural motion blur."
    Action = "Grounded micro-actions, tactile props; subtle performance beats."
    Lighting = "Soft key + cool rim; practicals; slight haze for volumetrics."
    Ward = "Authentic wardrobe and era-true props; no branding."
    Framing = "Rule-of-thirds; foreground occlusion; tasteful lens breathing."
    Constraints = "No subtitles. No on-screen text. No logos."

    if is_ru_vo:
        Dialogue = dialogue_text or "—"
        Lip = "Exact per-syllable Russian lip-sync; visemes aligned to phonemes."
        Audio = "Russian voiceover, cinematic ambience and gentle foley; score supports emotion."
    else:
        Dialogue = dialogue_text or "—"
        Lip = "If dialogue present, ensure precise lip-sync; map visemes to phonemes."
        Audio = "Cinematic ambience, gentle foley, warm score."

    return (
        "High-quality cinematic 4K video (16:9).\n"
        f"Scene: {Scene}\n"
        f"Camera: {Camera}\n"
        f"Action: {Action}\n"
        f"Dialogue: {Dialogue}\n"
        f"Lip-sync: {Lip}\n"
        f"Audio: {Audio}\n"
        f"Lighting: {Lighting}\n"
        f"Wardrobe/props: {Ward}\n"
        f"Framing: {Framing}\n"
        f"Constraints: {Constraints}"
    )

# ==========================
#   Handlers
# ==========================
async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    new_bonus = signup_bonus_if_needed(uid, 10)
    if new_bonus:
        await update.message.reply_text("🎁 Добро пожаловать! На баланс зачислено 10 бесплатных токенов 💎")
    await update.message.reply_text(
        WELCOME.format(balance=get_balance(uid), prompts_url=PROMPTS_CHANNEL_URL),
        parse_mode=ParseMode.MARKDOWN, reply_markup=main_menu_kb()
    )

async def topup_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "💳 Пополнение токенов через Telegram Stars.\n"
        f"Если звёзд не хватает — купите в {STARS_BUY_URL}",
        reply_markup=stars_topup_kb()
    )

async def faq(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.callback_query.answer()
    await update.callback_query.message.reply_text(
        "📖 FAQ — Вопросы и ответы\n\n"
        "⭐ Общая инфа\n"
        "Что умеет бот?\n"
        "Генерация видео (VEO), картинок (MJ), редактирование фото (Banana), подсказки для идей (Prompt-Master) и чат (ChatGPT).\n"
        "Звёзды (💎) — внутренняя валюта бота. Баланс видно в меню. Покупаются через Telegram.\n"
        "⸻\n\n"
        "🎬 VEO (Видео)\n"
        "• Fast — быстрый ролик, 2–5 мин. Стоимость: 50💎.\n"
        "• Quality — высокое качество, 5–10 мин. Стоимость: 150💎.\n"
        "• Animate — оживление фото.\n"
        "👉 Опишите идею (локация, стиль, настроение) и ждите готовый клип.\n"
        "⸻\n\n"
        "🖼️ MJ (Картинки)\n"
        "• Стоимость: 15💎. Время: 30–90 сек.\n"
        "👉 Чем детальнее промпт (цвет, свет, стиль), тем лучше результат.\n"
        "⸻\n\n"
        "🍌 Banana (Редактор фото)\n"
        "• Стоимость: 5💎. До 4 фото + промпт.\n"
        "• После загрузки фото бот пишет «📸 Фото принято (n/4)».\n"
        "• Генерация только после нажатия «🚀 Начать».\n"
        "⸻\n\n"
        "🧠 Prompt-Master\n"
        "• Бесплатно. Опишите идею: локация, стиль, настроение, свет, камера, реплики.\n"
        "• Получите готовый кинопромпт в нужной структуре.\n"
        "⸻\n\n"
        "💬 ChatGPT\n"
        "• Бесплатно. Общение и ответы на любые вопросы.\n"
        "⸻\n\n"
        "❓ Сколько ждать?\n"
        "• Картинки / Banana — до 2 минут.\n"
        "• Видео Fast — 2–5 минут.\n"
        "• Видео Quality — 5–10 минут.\n\n"
        "Где пополнить баланс?\n"
        "— Через кнопку «Пополнить баланс» в меню.\n\n"
        "Можно ли писать на русском?\n"
        "— Да, все режимы понимают русский и английский.",
    )

async def show_card_veo_by(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = S(ctx)
    uid = update.effective_user.id if update.effective_user else update.callback_query.from_user.id
    text = build_card_text_veo(s, get_balance(uid))
    if update.callback_query:
        await update.callback_query.message.reply_text(text, parse_mode=ParseMode.MARKDOWN,
                                                      reply_markup=card_keyboard_veo(s), disable_web_page_preview=True)
    else:
        await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN,
                                        reply_markup=card_keyboard_veo(s), disable_web_page_preview=True)

# ---------- Callbacks ----------
async def on_cb(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    data = q.data or ""
    uid = q.from_user.id
    s = S(ctx)

    if data == "back":
        await q.message.reply_text("🏠 Главное меню:", reply_markup=main_menu_kb()); return

    if data == "topup":
        await q.message.reply_text(
            f"💳 Пополнение токенов через Stars.\nЕсли не хватает — купите в {STARS_BUY_URL}",
            reply_markup=stars_topup_kb()
        ); return

    if data.startswith("buy:"):
        stars = int(data.split(":")[1])
        tokens = STARS_PACKS.get(stars, 0)
        if tokens <= 0:
            await q.message.reply_text("⚠️ Такой пакет недоступен."); return
        title = f"{stars}⭐ → 💎{tokens}"
        payload = json.dumps({"kind":"stars_pack","stars":stars,"tokens":tokens})
        try:
            await ctx.bot.send_invoice(
                chat_id=uid, title=title, description="Пакет пополнения токенов (XTR/Stars)",
                payload=payload, provider_token="", currency="XTR",
                prices=[LabeledPrice(label=title, amount=stars)]
            )
        except Exception as e:
            log.warning("Invoice failed: %s", e)
            await q.message.reply_text(
                f"Если счёт не открылся — купите 1⭐ в {STARS_BUY_URL} и попробуйте снова.",
                reply_markup=stars_topup_kb()
            )
        return

    # Режимы
    if data == "mode:veo_fast":
        s.update({"mode":"veo_text","model":"veo3_fast","aspect":"16:9"})
        await q.message.reply_text("📝 Пришлите идею/промпт для видео (VEO Fast).")
        await show_card_veo_by(update, ctx); return

    if data == "mode:veo_quality":
        s.update({"mode":"veo_text","model":"veo3","aspect":"16:9"})
        await q.message.reply_text("📝 Пришлите идею/промпт для видео (VEO Quality).")
        await show_card_veo_by(update, ctx); return

    if data == "mode:veo_photo":
        s.update({"mode":"veo_photo","model":"veo3_fast","aspect":"9:16"})
        await q.message.reply_text("🖼️ Пришлите фото (подпись-промпт — по желанию).")
        await show_card_veo_by(update, ctx); return

    if data == "mode:mj":
        s.update({"mode":"mj"})
        await q.message.reply_text("🖼️ Режим MJ: добавьте описание и запустите генерацию.", reply_markup=mj_kb()); return

    if data == "mode:banana":
        s.update({"mode":"banana","banana_mode":True,"banana_photos":[],"banana_prompt":None})
        await q.message.reply_text(
            "🍌 *Banana включён*\n"
            "Сначала пришлите *до 4 фото* (вложениями). После каждого фото — «📸 Фото принято (n/4)».\n"
            "Затем *напишите*, что нужно сделать: заменить одежду/макияж, сменить локацию, добавить людей, объединить фото и т.д.\n"
            "Когда всё готово — жмите «🚀 Начать».",
            parse_mode=ParseMode.MARKDOWN, reply_markup=banana_kb()
        ); return

    if data == "banana:add_photo":
        await q.message.reply_text("Пришлите ещё фото (до 4).", reply_markup=banana_kb()); return

    if data == "banana:reset":
        s["banana_photos"] = []; s["banana_prompt"] = None
        await q.message.reply_text("🧹 Сессия Banana очищена. Пришлите новые фото.", reply_markup=banana_kb()); return

    if data == "banana:start":
        photos = s.get("banana_photos") or []
        prompt = s.get("banana_prompt")
        if not photos:
            await q.message.reply_text("⚠️ Сначала пришлите хотя бы одно фото.", reply_markup=banana_kb()); return
        if not prompt:
            await q.message.reply_text("✍️ Напишите, что изменить на фото (одним сообщением).", reply_markup=banana_kb()); return
        price = TOKEN_COSTS["banana"]
        ok, rest = try_charge(uid, price)
        if not ok:
            await q.message.reply_text(
                f"💎 Недостаточно токенов: нужно {price}, на балансе {rest}.\nПополните через Stars: {STARS_BUY_URL}",
                reply_markup=stars_topup_kb()
            ); return
        await q.message.reply_text("🍌 Запускаю Banana… ⏳", reply_markup=banana_kb())
        asyncio.create_task(wait_for_banana_result(uid, photos, prompt, ctx, price)); return

    if data == "mode:prompt_master":
        s["pm_return_to"] = s.get("mode") or "veo_text"
        s["mode"] = "prompt_master"
        await q.message.reply_text(
            "🧠 *Prompt-Master готов!* Напишите в одном сообщении:\n"
            "• Идея и локация. Стиль/настроение, свет, ключевые предметы.\n"
            "• Действие и динамика камеры. Реплики — в кавычках.\n"
            "• Если нужна озвучка на русском — так и напишите.",
            parse_mode=ParseMode.MARKDOWN
        ); return

    if data == "mode:chat":
        s["mode"]="chat"; await q.message.reply_text("💬 Обычный чат включён. Напишите вопрос."); return

    if data == "faq":
        await faq(update, ctx); return

    # VEO-параметры
    if data.startswith("aspect:"):
        s["aspect"] = "9:16" if data.endswith("9:16") else "16:9"
        await show_card_veo_by(update, ctx); return

    if data.startswith("model:"):
        s["model"] = "veo3" if data.endswith("veo3") else "veo3_fast"
        await show_card_veo_by(update, ctx); return

    if data == "card:toggle_photo":
        if s.get("last_image_url"):
            s["last_image_url"] = None
            await q.message.reply_text("🧹 Фото-референс удалён.")
        else:
            await q.message.reply_text("📎 Пришлите фото вложением или публичный URL изображения.")
        await show_card_veo_by(update, ctx); return

    if data == "card:edit_prompt":
        await q.message.reply_text("✍️ Пришлите новый текст промпта."); return

    if data == "card:pm":
        s["pm_return_to"] = s.get("mode") or "veo_text"
        s["mode"] = "prompt_master"
        await q.message.reply_text(
            "🧠 *Prompt-Master готов!* Напишите в одном сообщении:\n"
            "• Идея, локация, стиль/настроение и свет.\n"
            "• Что происходит + динамика камеры.\n"
            "• Реплики — в кавычках. Если нужна русская озвучка — укажите.",
            parse_mode=ParseMode.MARKDOWN
        ); return

    if data == "card:reset":
        keep_aspect = s.get("aspect") or "16:9"
        keep_model  = s.get("model") or "veo3_fast"
        s.clear(); s.update({**DEFAULT_STATE})
        s["aspect"] = keep_aspect; s["model"] = keep_model
        await q.message.reply_text("🗂️ Карточка очищена.")
        await show_card_veo_by(update, ctx); return

    if data == "card:generate":
        if not s.get("last_prompt"):
            await q.message.reply_text("✍️ Сначала укажите текст промпта."); return
        price = TOKEN_COSTS['veo_quality'] if s.get('model') == 'veo3' else TOKEN_COSTS['veo_fast']
        ok, rest = try_charge(uid, price)
        if not ok:
            await q.message.reply_text(
                f"💎 Недостаточно токенов: нужно {price}, на балансе {rest}.\nПополните через Stars: {STARS_BUY_URL}",
                reply_markup=stars_topup_kb()
            ); return
        await q.message.reply_text(f"🚀 Задача отправлена ({'💎 Quality' if s.get('model')=='veo3' else '⚡ Fast'}).")
        await q.message.reply_text("⏳ Генерация запущена.")
        ok2, task_id, msg = submit_kie_veo(
            s["last_prompt"].strip(), s.get("aspect","16:9"), s.get("last_image_url"), s.get("model","veo3_fast")
        )
        if not ok2 or not task_id:
            add_tokens(uid, price)
            await q.message.reply_text(f"❌ Не удалось создать VEO-задачу: {msg}\n💎 Токены возвращены."); return
        s["last_task_id"] = task_id
        asyncio.create_task(poll_veo_and_send(uid, task_id, s.get("aspect")=="9:16", s.get("model")=="veo3", ctx, price))
        return

    # MJ
    if data == "mj:add_prompt":
        s["mode"] = "mj_txt"
        await q.message.reply_text("✍️ Пришлите текстовый prompt для картинки (формат 16:9)."); return

    if data == "mj:start":
        prompt = s.get("last_prompt")
        if not prompt:
            await q.message.reply_text("⚠️ Сначала добавьте описание (кнопка выше).", reply_markup=mj_kb()); return
        price = TOKEN_COSTS["mj"]
        ok, rest = try_charge(uid, price)
        if not ok:
            await q.message.reply_text(
                f"💎 Недостаточно токенов: нужно {price}, на балансе {rest}.\nПополните через Stars: {STARS_BUY_URL}",
                reply_markup=stars_topup_kb()
            ); return
        await q.message.reply_text(f"🎨 Генерация фото запущена…\nPrompt: `{prompt}`", parse_mode=ParseMode.MARKDOWN)
        ok2, task_id, msg = mj_generate(prompt.strip())
        if not ok2 or not task_id:
            add_tokens(uid, price)
            await q.message.reply_text(f"❌ Не удалось создать MJ-задачу: {msg}\n💎 Токены возвращены."); return
        await q.message.reply_text(f"🆔 MJ taskId: `{task_id}`", parse_mode=ParseMode.MARKDOWN)
        asyncio.create_task(poll_mj_and_send(uid, task_id, ctx, price)); return

# ---------- TEXT & PHOTO ----------
async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = S(ctx); uid = update.effective_user.id
    text = (update.message.text or "").strip()

    # промокоды
    if text.lower().startswith("/promo"):
        parts = text.split(maxsplit=1)
        if len(parts) < 2:
            await update.message.reply_text("Использование: /promo КОД"); return
        code = parts[1].strip()
        amount = int(PROMO_CODES.get(code, 0))
        if amount <= 0:
            await update.message.reply_text("⚠️ Неверный или истёкший промокод."); return
        used_key = rkey("promo", code, str(uid))
        if rdb.get(used_key):
            await update.message.reply_text("⚠️ Этот промокод уже использован вашим аккаунтом."); return
        add_tokens(uid, amount); rdb.set(used_key, "1")
        await update.message.reply_text(f"🎁 Промокод активирован: +{amount}💎\nБаланс: *{get_balance(uid)}* 💎", parse_mode=ParseMode.MARKDOWN)
        return

    # Banana — текст после фото = промпт
    if s.get("banana_mode") and s.get("mode") == "banana":
        s["banana_prompt"] = text
        await update.message.reply_text(
            "✍️ Промпт сохранён. Проверьте, что добавили все фото (до 4).\nГотовы? Жмите «🚀 Начать».",
            reply_markup=banana_kb()
        ); return

    # PM
    if s.get("mode") == "prompt_master":
        pm = await oai_prompt_master(text[:1000])
        if not pm:
            await update.message.reply_text("⚠️ Prompt-Master сейчас недоступен. Попробуйте позже."); return
        s["last_prompt"] = pm
        s["mode"] = s.pop("pm_return_to", "veo_text")
        await update.message.reply_text("🧠 Готово! Промпт добавлен в карточку.")
        await show_card_veo_by(update, ctx); return

    # MJ — ввод промпта
    if s.get("mode") == "mj_txt":
        s["last_prompt"] = text
        await update.message.reply_text(
            f"✅ Prompt сохранён:\n\n`{text}`\n\nТеперь нажмите «🏙️ Сгенерировать 16:9»",
            parse_mode=ParseMode.MARKDOWN, reply_markup=mj_kb()
        ); return

    # Чат (бесплатно)
    if s.get("mode") == "chat":
        # Просто эхо-ответ (без внешнего API, чтобы не блокировать)
        await update.message.reply_text(text); return

    # По умолчанию — это VEO-промпт
    s["last_prompt"] = text
    await update.message.reply_text("🟦 VEO — подготовка к рендеру. Проверьте карточку ниже и жмите «Сгенерировать».")
    await show_card_veo_by(update, ctx)

async def on_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = S(ctx)
    photos = update.message.photo
    if not photos: return
    ph = photos[-1]
    file = await ctx.bot.get_file(ph.file_id)
    url = file.file_path if file.file_path.startswith("http") else f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{file.file_path}"

    if s.get("banana_mode") and s.get("mode") == "banana":
        if len(s["banana_photos"]) >= 4:
            await update.message.reply_text("⚠️ Уже 4/4. Очистите лишнее через «🧹 Очистить фото».", reply_markup=banana_kb()); return
        s["banana_photos"].append(url)
        await update.message.reply_text(
            f"📸 Фото принято ({len(s['banana_photos'])}/4).\n"
            "Теперь *напишите,* что нужно сделать с изображениями: объединить людей, поменять одежду/макияж, сменить локацию и т.п.",
            parse_mode=ParseMode.MARKDOWN, reply_markup=banana_kb()
        ); return

    # иначе — это референс для VEO
    s["last_image_url"] = url
    await update.message.reply_text("🖼️ Фото принято как референс."); await show_card_veo_by(update, ctx)

# ==========================
#   Pollers
# ==========================
async def poll_mj_and_send(uid: int, task_id: str, ctx: ContextTypes.DEFAULT_TYPE, price: int):
    start = time.time()
    last_wait = 0.0
    while True:
        ok, flag, data = mj_status(task_id)
        if not ok:
            add_tokens(uid, price)
            await ctx.bot.send_message(uid, "❌ MJ сейчас недоступен. 💎 Токены возвращены."); return
        if flag == 0:
            if time.time() - last_wait >= 40:
                await ctx.bot.send_message(uid, "🖼️✨ Рисую…")
                last_wait = time.time()
            await asyncio.sleep(8); continue
        if flag in (2, 3):
            add_tokens(uid, price)
            msg = (data or {}).get("errorMessage") or "MidJourney недоступен."
            await ctx.bot.send_message(uid, f"❌ MJ: {msg}\n💎 Токены возвращены."); return
        if flag == 1:
            urls = _extract_mj_image_urls(data or {})
            if not urls:
                add_tokens(uid, price)
                await ctx.bot.send_message(uid, "⚠️ Готово, но ссылки не найдены. 💎 Токены возвращены."); return
            if len(urls) == 1:
                await ctx.bot.send_photo(uid, urls[0])
            else:
                media = [InputMediaPhoto(u) for u in urls[:10]]
                await ctx.bot.send_media_group(uid, media)
            await ctx.bot.send_message(uid, "✅ *Готово!*", parse_mode=ParseMode.MARKDOWN,
                                       reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🚀 Ещё MJ", callback_data="mode:mj")]]))
            return
        if time.time() - start > 15*60:
            add_tokens(uid, price)
            await ctx.bot.send_message(uid, "⌛ MJ долго не отвечает. 💎 Токены возвращены."); return

async def poll_veo_and_send(uid: int, task_id: str, expect_vertical: bool, is_quality: bool,
                            ctx: ContextTypes.DEFAULT_TYPE, price: int):
    start = time.time()
    while True:
        ok, state, res = get_kie_veo_status(task_id)
        if not ok:
            add_tokens(uid, price)
            await ctx.bot.send_message(uid, "❌ Ошибка статуса VEO. 💎 Токены возвращены."); return
        if state == "success" and res:
            final_url = res
            if not expect_vertical and is_quality:
                u1080 = await try_get_1080_url(task_id)
                if u1080: final_url = u1080
            await ctx.bot.send_message(uid, "🎞️ Рендер завершён — отправляю файл…")
            try:
                await ctx.bot.send_video(uid, final_url, supports_streaming=True)
            except Exception:
                await ctx.bot.send_message(uid, f"🔗 Результат: {final_url}")
            await ctx.bot.send_message(uid, "✅ Готово!", reply_markup=InlineKeyboardMarkup(
                [[InlineKeyboardButton("🚀 Сгенерировать ещё видео", callback_data="mode:veo_fast")]]
            ))
            return
        if state in ("fail", "2", "3"):
            add_tokens(uid, price)
            await ctx.bot.send_message(uid, "❌ VEO: задача завершилась с ошибкой. 💎 Токены возвращены."); return
        if time.time() - start > 20*60:
            add_tokens(uid, price)
            await ctx.bot.send_message(uid, "⌛ Превышено время ожидания VEO. 💎 Токены возвращены."); return
        await asyncio.sleep(6)

async def wait_for_banana_result(uid: int, photos: List[str], prompt: str,
                                 ctx: ContextTypes.DEFAULT_TYPE, price: int):
    try:
        ok, task_id, err = create_banana_task(prompt, photos)
        if not ok or not task_id:
            add_tokens(uid, price)
            await ctx.bot.send_message(uid, f"❌ Не удалось создать Banana-задачу: {err or 'нет ответа'}\n💎 Токены возвращены.")
            return
        start = time.time()
        last_ping = 0.0
        while True:
            ok2, state, urls_or_err = get_banana_status(task_id)
            if not ok2:
                add_tokens(uid, price)
                await ctx.bot.send_message(uid, f"❌ Ошибка статуса Banana: {urls_or_err or 'нет ответа'}\n💎 Токены возвращены.")
                return
            if state == "success":
                urls = urls_or_err or []
                if not urls:
                    add_tokens(uid, price)
                    await ctx.bot.send_message(uid, "⚠️ Готово, но ссылок нет. 💎 Токены возвращены.")
                    return
                if len(urls) == 1:
                    await ctx.bot.send_photo(uid, urls[0])
                else:
                    media = [InputMediaPhoto(u) for u in urls[:10]]
                    await ctx.bot.send_media_group(uid, media)
                await ctx.bot.send_message(uid, "✅ Готово! 🍌 Banana завершил обработку.")
                s = S(ctx); s["banana_mode"]=False; s["banana_photos"]=[]; s["banana_prompt"]=None
                return
            if state == "fail":
                add_tokens(uid, price)
                await ctx.bot.send_message(uid, "❌ Banana: задача завершилась с ошибкой. 💎 Токены возвращены."); return
            if time.time() - last_ping >= 40:
                await ctx.bot.send_message(uid, "🍌 Обрабатываю…")
                last_ping = time.time()
            if time.time() - start > 15*60:
                add_tokens(uid, price)
                await ctx.bot.send_message(uid, "⌛ Превышено время ожидания Banana. 💎 Токены возвращены.")
                return
            await asyncio.sleep(6)
    except Exception as e:
        add_tokens(uid, price)
        try:
            await ctx.bot.send_message(uid, f"💥 Внутренняя ошибка Banana: {e}\n💎 Токены возвращены.")
        except Exception:
            pass

# ==========================
#   Payments (Stars/XTR)
# ==========================
async def precheckout(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        await update.pre_checkout_query.answer(ok=True)
    except Exception:
        await update.pre_checkout_query.answer(ok=False, error_message="Платёж отклонён. Попробуйте снова.")

async def paid(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    sp = update.message.successful_payment
    try:
        meta = json.loads(sp.invoice_payload)
    except Exception:
        meta = {}
    stars = int(sp.total_amount)
    tokens = int(meta.get("tokens") or STARS_PACKS.get(stars, 0))
    add_tokens(update.effective_user.id, tokens)
    await update.message.reply_text(
        f"✅ Оплата получена: +{tokens}💎\nБаланс: *{get_balance(update.effective_user.id)}* 💎",
        parse_mode=ParseMode.MARKDOWN
    )

# ==========================
#   Entry
# ==========================
def main():
    # на всякий снимаем webhook — типичный баг «бот молчит»
    try:
        Bot(TELEGRAM_TOKEN).delete_webhook(drop_pending_updates=True)
        log.info("Webhook deleted")
    except Exception as e:
        log.warning("Delete webhook failed: %s", e)

    app = (ApplicationBuilder()
           .token(TELEGRAM_TOKEN)
           .rate_limiter(AIORateLimiter())
           .build())

    # команды
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("topup", topup_cmd))
    # платежи
    app.add_handler(PreCheckoutQueryHandler(precheckout))
    app.add_handler(MessageHandler(filters.SUCCESSFUL_PAYMENT, paid))
    # кнопки
    app.add_handler(CallbackQueryHandler(on_cb))
    # сообщения
    app.add_handler(MessageHandler(filters.PHOTO, on_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    log.info("Bot starting (prices set; Redis on; promos=%s)", "on" if PROMO_CODES else "off")
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True, stop_signals=None)

if __name__ == "__main__":
    main()
