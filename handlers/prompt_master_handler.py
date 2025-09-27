"""Prompt-Master handlers, state helpers and prompt builder."""

from __future__ import annotations

import html
import json
import logging
import re
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

from telegram import InlineKeyboardMarkup, Update
from telegram.constants import ChatType, ParseMode
from telegram.ext import ContextTypes

from keyboards import (
    CB_PM_PREFIX,
    prompt_master_keyboard,
    prompt_master_mode_keyboard,
    prompt_master_result_keyboard,
)

logger = logging.getLogger(__name__)

PM_STATE_KEY = "mode"
PM_ENGINE_KEY = "pm_engine"
PM_LANG_KEY = "pm_lang"
PM_PROMPTS_KEY = "pm_prompts"

PM_ENGINES = {"veo", "mj", "banana", "animate", "suno"}

CYRILLIC_RE = re.compile(r"[а-яё]", re.IGNORECASE)
LOW_LIGHT_HINTS = ("ноч", "night", "неон", "neon", "луна", "moon")
WARM_LIGHT_HINTS = ("закат", "sunset", "golden hour", "тепл")
CAMERA_HINTS = (
    ("drone", ("drone sweep", "дрон с плавным пролётом")),
    ("крупный план", ("close-up", "крупный план")),
    ("портрет", ("portrait lens", "портретный объектив 85mm")),
    ("широкоуголь", ("wide-angle lens", "широкоугольный объектив 24mm")),
    ("дрон", ("drone sweep", "дрон с плавным пролётом")),
    ("macro", ("macro lens", "макрообъектив")),
)

PM_ROOT_TEXT = {
    "ru": "🧠 <b>Prompt-Master</b>\nВыберите движок, под который нужно подготовить промпт.",
    "en": "🧠 <b>Prompt-Master</b>\nPick the engine you want a perfect prompt for.",
}

PM_ENGINE_HINTS = {
    "veo": {
        "ru": "Опишите идею ролика: сюжет, эмоции, окружение. Я соберу структурированный JSON для VEO.",
        "en": "Describe the video idea: story, emotions, surroundings. I will craft a structured JSON for VEO.",
    },
    "mj": {
        "ru": "Расскажите, какой кадр хотите получить в Midjourney. Добавьте стиль и детали.",
        "en": "Describe the Midjourney shot you need, including style and key details.",
    },
    "banana": {
        "ru": "Что нужно поправить на фото? Я соберу чек-лист для Banana и сохраню черты лица.",
        "en": "What should be fixed on the photo? I will prepare a Banana checklist keeping the real face.",
    },
    "animate": {
        "ru": "Опишите, как оживить фото: эмоции, движения камеры и лица. Лицо останется прежним.",
        "en": "Describe how the photo should come alive: facial motion and camera drift. The face will stay true.",
    },
    "suno": {
        "ru": "Опишите настроение и сюжет трека. Я подготовлю каркас промпта для Suno.",
        "en": "Describe the mood and story of the track. I'll return a neat Suno prompt skeleton.",
    },
}

SAFETY_PHRASES = {
    "banana": {
        "ru": "Сохранить реальное лицо и черты, без подмены. Без деформаций, без лишних аксессуаров.",
        "en": "Keep the real face and traits, no swaps. No distortions, no extra accessories.",
    },
    "animate": {
        "ru": "Не менять внешность. Сохранить реальное лицо, мимику без искажений.",
        "en": "Do not alter appearance. Keep the real face and undistorted expressions.",
    },
}

ENGINE_DISPLAY = {
    "veo": {"ru": "VEO", "en": "VEO"},
    "mj": {"ru": "Midjourney", "en": "Midjourney"},
    "banana": {"ru": "Banana", "en": "Banana"},
    "animate": {"ru": "VEO Animate", "en": "VEO Animate"},
    "suno": {"ru": "Suno", "en": "Suno"},
}


@dataclass
class PromptOut:
    engine: Literal["veo", "mj", "banana", "animate", "suno"]
    body: str
    is_json: bool


_LAST_PROMPTS: Dict[Tuple[int, str], PromptOut] = {}


def detect_language(text: str) -> str:
    return "ru" if CYRILLIC_RE.search(text or "") else "en"


def _normalize_text(text: str) -> str:
    return " ".join((text or "").split())


def _choose_camera_detail(text: str, lang: str) -> str:
    lowered = text.lower()
    for needle, variants in CAMERA_HINTS:
        if needle in lowered:
            return variants[1] if lang == "ru" else variants[0]
    return "Стедикам с плавным въездом" if lang == "ru" else "Steadycam push-in"


def _choose_lighting_detail(text: str, lang: str) -> str:
    lowered = text.lower()
    if any(hint in lowered for hint in LOW_LIGHT_HINTS):
        return "Неоновый контровый свет и мягкая дымка" if lang == "ru" else "Neon rim light with gentle haze"
    if any(hint in lowered for hint in WARM_LIGHT_HINTS):
        return "Тёплый закатный свет, длинные тени" if lang == "ru" else "Warm sunset glow with long shadows"
    return "Мягкий рассеянный свет с акцентом" if lang == "ru" else "Soft diffused key light"


def _choose_palette(text: str, lang: str) -> str:
    lowered = text.lower()
    if "неон" in lowered or "neon" in lowered:
        return "Неоновая палитра: бирюзовый, фуксия, контрастный свет" if lang == "ru" else "Neon palette: teal, magenta, high contrast"
    if "пастел" in lowered or "pastel" in lowered:
        return "Пастельные тона: нежный розовый, песочный, молочный" if lang == "ru" else "Pastel hues: blush pink, sand, ivory"
    return "Кинематографичная цветокоррекция с глубокими тенями" if lang == "ru" else "Cinematic grading with deep shadows"


def _choose_style(lang: str) -> str:
    return "Гиперреалистичный стиль, премиальная детализация" if lang == "ru" else "Hyper-realistic style with premium detail"


def _animate_motion(lang: str) -> str:
    return (
        "Плавный параллакс камеры, лёгкое дыхание кадра, мимика естественная"
        if lang == "ru"
        else "Soft camera parallax, gentle breathing motion, natural facial micro-expressions"
    )


def build_prompt(
    engine: Literal["veo", "mj", "banana", "animate", "suno"],
    text: str,
    lang: str,
    user_prefs: Optional[Dict[str, str]] = None,
) -> PromptOut:
    cleaned = _normalize_text(text)
    camera = _choose_camera_detail(text, lang)
    lighting = _choose_lighting_detail(text, lang)
    palette = _choose_palette(text, lang)
    style = _choose_style(lang)

    if engine == "veo":
        payload = {
            "scene": cleaned,
            "camera": camera,
            "motion": (
                "Динамичный сторителлинг, 8 секунд, без резких рывков"
                if lang == "ru"
                else "Dynamic storytelling, 8 seconds, no abrupt cuts"
            ),
            "lighting": lighting,
            "palette": palette,
            "details": (
                "Уточнить героев, окружение, ключевой акцент. Финал оставить выразительным."
                if lang == "ru"
                else "Clarify subjects, environment, key accent. End on a striking beat."
            ),
        }
        body = json.dumps(payload, ensure_ascii=False, indent=2)
        return PromptOut(engine="veo", body=body, is_json=True)

    if engine == "mj":
        payload = {
            "prompt": (
                f"{cleaned}, {style.lower()}"
                if lang == "ru"
                else f"{cleaned}, {style.lower()}"
            ),
            "camera": camera,
            "lighting": lighting,
            "palette": palette,
            "render": "--ar 16:9 --v 6" if lang == "en" else "--ar 16:9 --v 6",
        }
        body = json.dumps(payload, ensure_ascii=False, indent=2)
        return PromptOut(engine="mj", body=body, is_json=True)

    if engine == "banana":
        safety = SAFETY_PHRASES["banana"][lang]
        lines = [
            ("📝 Задача" if lang == "ru" else "📝 Task") + f": {cleaned}",
            ("Правки" if lang == "ru" else "Adjustments") + ":",
            "• " + ("Работа со светом/кожей по описанию." if lang == "ru" else "Tweak light/skin as described."),
            "• " + ("Уточнить фон и детали одежды при необходимости." if lang == "ru" else "Refine background and outfit details."),
            "",
            ("Безопасность" if lang == "ru" else "Safety") + f": {safety}",
        ]
        body = "\n".join(lines)
        return PromptOut(engine="banana", body=body, is_json=False)

    if engine == "animate":
        safety = SAFETY_PHRASES["animate"][lang]
        lines = [
            ("🎬 Оживление" if lang == "ru" else "🎬 Animation") + f": {cleaned}",
            ("Движение" if lang == "ru" else "Motion") + f": {_animate_motion(lang)}",
            ("Камера" if lang == "ru" else "Camera") + f": {camera}",
            ("Свет" if lang == "ru" else "Lighting") + f": {lighting}",
            "",
            ("Безопасность" if lang == "ru" else "Safety") + f": {safety}",
        ]
        body = "\n".join(lines)
        return PromptOut(engine="animate", body=body, is_json=False)

    # Suno skeleton
    mood_line = "Настроение: вдохновляющее" if lang == "ru" else "Mood: inspiring"
    tempo_line = "Темп: 96-102 BPM" if lang == "ru" else "Tempo: 96-102 BPM"
    story_title = "Сюжет" if lang == "ru" else "Story"
    closing = "Сервис скоро будет доступен, но подготовьте текст заранее." if lang == "ru" else "Service launches soon, prepare the text ahead of time."
    lines = [
        "🎵 Suno Prompt", mood_line, tempo_line, f"{story_title}: {cleaned}", closing
    ]
    body = "\n".join(lines)
    return PromptOut(engine="suno", body=body, is_json=False)


def _store_prompt(chat_id: int, prompt: PromptOut) -> None:
    _LAST_PROMPTS[(chat_id, prompt.engine)] = prompt


def get_pm_prompt(chat_id: int, engine: str) -> Optional[PromptOut]:
    return _LAST_PROMPTS.get((chat_id, engine))


def clear_pm_prompts(chat_id: int) -> None:
    keys = [key for key in _LAST_PROMPTS if key[0] == chat_id]
    for key in keys:
        _LAST_PROMPTS.pop(key, None)


def _resolve_ui_lang(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    lang = context.user_data.get(PM_LANG_KEY)
    if lang in {"ru", "en"}:
        return lang
    user = update.effective_user
    if user and isinstance(user.language_code, str):
        if user.language_code.lower().startswith("ru"):
            return "ru"
    return "en"


def _header_for_engine(engine: str, lang: str) -> str:
    display = ENGINE_DISPLAY.get(engine, {}).get(lang, engine.upper())
    if lang == "ru":
        return f"<b>Готовый промпт для {html.escape(display)}</b>"
    return f"<b>Ready prompt for {html.escape(display)}</b>"


def _format_block(prompt: PromptOut) -> str:
    if prompt.is_json:
        return f"<blockquote><pre>{html.escape(prompt.body)}</pre></blockquote>"
    escaped = html.escape(prompt.body).replace("\n", "<br/>")
    return f"<blockquote>{escaped}</blockquote>"


async def prompt_master_open(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    lang = _resolve_ui_lang(update, context)
    text = PM_ROOT_TEXT.get(lang, PM_ROOT_TEXT["en"])

    if message is not None:
        await message.reply_text(
            text,
            reply_markup=prompt_master_keyboard(lang),
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )
        return

    query = update.callback_query
    if query is None:
        return
    await query.answer()
    await query.edit_message_text(
        text,
        reply_markup=prompt_master_keyboard(lang),
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True,
    )


def _set_pm_state(context: ContextTypes.DEFAULT_TYPE, engine: Optional[str]) -> None:
    if engine is None:
        context.user_data.pop(PM_STATE_KEY, None)
        context.user_data.pop(PM_ENGINE_KEY, None)
        return
    context.user_data[PM_STATE_KEY] = "pm"
    context.user_data[PM_ENGINE_KEY] = engine


async def _send_engine_hint(
    query, update: Update, context: ContextTypes.DEFAULT_TYPE, engine: str
) -> None:
    lang = _resolve_ui_lang(update, context)
    hint = PM_ENGINE_HINTS.get(engine, PM_ENGINE_HINTS["veo"]).get(lang, PM_ENGINE_HINTS["veo"]["en"])
    markup = prompt_master_mode_keyboard(lang)
    if query.message is not None:
        await query.edit_message_text(
            hint,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
            reply_markup=markup,
        )
    else:
        chat = update.effective_chat
        if chat is not None:
            await context.bot.send_message(
                chat_id=chat.id,
                text=hint,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
                reply_markup=markup,
            )


async def prompt_master_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query is None or query.data is None:
        return

    data = query.data
    await query.answer()
    lang = _resolve_ui_lang(update, context)

    if data.startswith(f"{CB_PM_PREFIX}insert:"):
        return

    if data.startswith(f"{CB_PM_PREFIX}copy:"):
        parts = data.split(":", 2)
        engine = parts[2] if len(parts) > 2 else ""
        chat = query.message.chat if query.message else update.effective_chat
        chat_id = chat.id if chat else None
        prompt = get_pm_prompt(chat_id, engine) if chat_id is not None else None
        if prompt is None:
            await query.answer("Промпт не найден" if lang == "ru" else "Prompt not found", show_alert=True)
            return
        await query.answer("Готово" if lang == "ru" else "Done")
        text = prompt.body if not prompt.is_json else f"<pre>{html.escape(prompt.body)}</pre>"
        parse_mode = ParseMode.HTML if prompt.is_json else None
        target_chat = chat_id if chat_id is not None else (update.effective_chat.id if update.effective_chat else None)
        if target_chat is not None:
            await context.bot.send_message(chat_id=target_chat, text=text, parse_mode=parse_mode)
        return

    if data in {f"{CB_PM_PREFIX}back", f"{CB_PM_PREFIX}menu"}:
        chat = query.message.chat if query.message else update.effective_chat
        if chat is not None:
            clear_pm_prompts(chat.id)
        _set_pm_state(context, None)
        await query.edit_message_text(
            PM_ROOT_TEXT.get(lang, PM_ROOT_TEXT["en"]),
            reply_markup=prompt_master_keyboard(lang),
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )
        return

    if data == f"{CB_PM_PREFIX}switch":
        _set_pm_state(context, None)
        await query.edit_message_text(
            PM_ROOT_TEXT.get(lang, PM_ROOT_TEXT["en"]),
            reply_markup=prompt_master_keyboard(lang),
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )
        return

    engine = data.removeprefix(CB_PM_PREFIX)
    if engine in PM_ENGINES:
        _set_pm_state(context, engine)
        user = update.effective_user
        user_id = user.id if user else None
        logger.info("pm.mode_set | user_id=%s engine=%s", user_id, engine)
        await _send_engine_hint(query, update, context, engine)
        return

    await query.edit_message_text(
        PM_ROOT_TEXT.get(lang, PM_ROOT_TEXT["en"]),
        reply_markup=prompt_master_keyboard(lang),
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True,
    )


async def prompt_master_handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if message is None or not isinstance(message.text, str):
        return

    if context.user_data.get(PM_STATE_KEY) != "pm":
        return

    engine = context.user_data.get(PM_ENGINE_KEY)
    if engine not in PM_ENGINES:
        return

    idea = message.text.strip()
    if not idea:
        return

    lang = detect_language(idea)
    context.user_data[PM_LANG_KEY] = lang
    user_id = update.effective_user.id if update.effective_user else None
    result = build_prompt(engine, idea, lang, context.user_data.get("pm_prefs") or {})
    logger.info(
        "pm.generate | user_id=%s engine=%s lang=%s len=%s",
        user_id,
        engine,
        lang,
        len(idea),
    )

    chat = update.effective_chat
    if chat is not None:
        _store_prompt(chat.id, result)

    prompts = context.user_data.setdefault(PM_PROMPTS_KEY, {})
    prompts[engine] = result.body
    context.user_data["pm_last_engine"] = engine

    if chat and chat.type == ChatType.PRIVATE:
        try:
            await message.delete()
        except Exception:
            logger.debug("prompt_master.delete_failed", exc_info=True)

    header = _header_for_engine(engine, lang)
    formatted = f"{header}\n{_format_block(result)}"
    markup = prompt_master_result_keyboard(engine, lang)
    try:
        await message.reply_text(
            formatted,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
            reply_markup=markup,
        )
    except Exception:
        logger.exception("prompt_master.reply_failed")


async def prompt_master_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    if chat is not None:
        clear_pm_prompts(chat.id)
    _set_pm_state(context, None)
    context.user_data.pop(PM_PROMPTS_KEY, None)
    context.user_data.pop(PM_LANG_KEY, None)

    lang = _resolve_ui_lang(update, context)
    text = "🧹 Prompt-Master сброшен." if lang == "ru" else "🧹 Prompt-Master reset."
    keyboard = prompt_master_keyboard(lang)

    message = update.effective_message
    if message is not None:
        await message.reply_text(
            f"{text}\n\n{PM_ROOT_TEXT.get(lang, PM_ROOT_TEXT['en'])}",
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
            reply_markup=keyboard,
        )
    else:
        target = chat.id if chat is not None else None
        if target is not None:
            await context.bot.send_message(
                target,
                f"{text}\n\n{PM_ROOT_TEXT.get(lang, PM_ROOT_TEXT['en'])}",
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
                reply_markup=keyboard,
            )


# Backwards compatibility alias
prompt_master_process = prompt_master_handle_text


__all__ = [
    "PromptOut",
    "build_prompt",
    "clear_pm_prompts",
    "detect_language",
    "get_pm_prompt",
    "prompt_master_callback",
    "prompt_master_handle_text",
    "prompt_master_open",
    "prompt_master_process",
    "prompt_master_reset",
]
