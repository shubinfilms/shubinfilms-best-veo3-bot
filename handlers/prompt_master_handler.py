"""Prompt-Master handlers and auto-generation helpers."""

from __future__ import annotations

import html
import json
import logging
import re
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from keyboards import CB_PM_PREFIX, prompt_master_keyboard

logger = logging.getLogger(__name__)

PM_HINT = "🧠 *Prompt-Master*\nВыберите, что хотите сделать:"

CYRILLIC_RE = re.compile(r"[а-яё]", re.IGNORECASE)

PHOTO_KEYWORDS = (
    "оживи",
    "оживление",
    "talking photo",
    "animate photo",
    "живое фото",
    "make it talk",
)
BANANA_KEYWORDS = (
    "banana",
    "ретуш",
    "ретушь",
    "замени фон",
    "edit photo",
    "photo edit",
    "редакт",
)
MJ_KEYWORDS = (
    "midjourney",
    " mj",
    "#mj",
    "изображ",
    "картин",
    "artwork",
    "poster",
    "concept art",
)
SUNO_KEYWORDS = (
    "suno",
    "песня",
    "song",
    "lyrics",
    "rap",
    "музык",
    "трек",
    "гимн",
)
VIDEO_KEYWORDS = (
    "veo",
    "video",
    "clip",
    "ролик",
    "трейлер",
    "видео",
    "cinematic",
)

CAMERA_HINTS = (
    ("drone", ("drone")),
    ("аэро", ("drone")),
    ("крупный план", ("close-up")),
    ("close-up", ("close-up")),
    ("портрет", ("portrait lens")),
    ("широкоугольн", ("wide-angle lens")),
    ("wide", ("wide-angle lens")),
)

LOW_LIGHT_HINTS = (
    "ноч",
    "night",
    "неон",
    "neon",
    "луна",
    "moon",
)

WARM_LIGHT_HINTS = (
    "закат",
    "sunset",
    "golden hour",
    "тепл",
)


@dataclass
class PromptResult:
    """Container with generated prompt data."""

    engine: str
    raw: str
    is_json: bool


ENGINE_DISPLAY: Dict[str, Dict[str, str]] = {
    "veo": {"ru": "VEO", "en": "VEO"},
    "mj": {"ru": "Midjourney", "en": "Midjourney"},
    "banana": {"ru": "Banana", "en": "Banana"},
    "photo_live": {"ru": "VEO оживление", "en": "VEO Photo Live"},
    "suno": {"ru": "Suno", "en": "Suno"},
}


async def prompt_master_open(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    from_callback: bool = False,
) -> None:
    """Send or edit the Prompt-Master root menu."""

    message = update.effective_message
    if message is not None:
        await message.reply_text(
            PM_HINT,
            reply_markup=prompt_master_keyboard(),
            parse_mode=ParseMode.MARKDOWN,
            disable_web_page_preview=True,
        )
        return

    query = update.callback_query
    if query is None:
        return

    if not from_callback:
        await query.answer()

    await query.edit_message_text(
        PM_HINT,
        reply_markup=prompt_master_keyboard(),
        parse_mode=ParseMode.MARKDOWN,
        disable_web_page_preview=True,
    )


async def prompt_master_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle Prompt-Master keyboard interactions."""

    query = update.callback_query
    if query is None or query.data is None:
        return

    user_id = update.effective_user.id if update.effective_user else None
    logger.info("prompt_master.callback | user_id=%s data=%s", user_id, query.data)

    await query.answer()

    code = query.data.removeprefix(CB_PM_PREFIX)
    if code == "back":
        await prompt_master_open(update, context, from_callback=True)
        return

    feature_name = {
        "video": "🎬 Видеопромпт (VEO)",
        "mj_gen": "🖼️ Промпт генерации фото (MJ)",
        "photo_live": "🫥 Оживление фото (VEO)",
        "banana_edit": "✂️ Редактирование фото (Banana)",
        "suno_lyrics": "🎵 Текст песни (Suno)",
    }.get(code, "Неизвестно")

    await query.edit_message_text(
        f"{feature_name}\n\n⚙️ Функция скоро будет доступна. А пока — опишите задачу текстом, я помогу сформулировать промпт.",
        reply_markup=prompt_master_keyboard(),
        parse_mode=ParseMode.MARKDOWN,
        disable_web_page_preview=True,
    )


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def detect_language(text: str) -> str:
    """Return `ru` if Cyrillic letters found, otherwise `en`."""

    return "ru" if CYRILLIC_RE.search(text or "") else "en"


def _has_keyword(text: str, keywords: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def classify_prompt_engine(text: str) -> str:
    """Classify user request into one of supported engines."""

    lowered = text.lower()
    if "#veo" in lowered:
        return "veo"
    if _has_keyword(lowered, SUNO_KEYWORDS):
        return "suno"
    if _has_keyword(lowered, BANANA_KEYWORDS):
        return "banana"
    if _has_keyword(lowered, PHOTO_KEYWORDS):
        return "photo_live"
    if "#mj" in lowered or " mj" in lowered:
        return "mj"
    if _has_keyword(lowered, MJ_KEYWORDS):
        return "mj"
    if _has_keyword(lowered, VIDEO_KEYWORDS):
        return "veo"
    return "veo"


def _choose_camera_detail(text: str, lang: str) -> str:
    lowered = text.lower()
    for needle, camera_value in CAMERA_HINTS:
        if needle in lowered:
            choice = camera_value[0]
            break
    else:
        choice = "steadycam push-in" if lang == "en" else "стедикам со плавным въездом"
    if lang == "ru":
        if choice == "drone":
            return "Дрон с плавным пролётом"
        if choice == "close-up":
            return "Крупный план с мягким фокусом"
        if choice == "portrait lens":
            return "Портретный объектив 85mm, мелкая глубина резкости"
        if choice == "wide-angle lens":
            return "Широкоугольный объектив 24mm для динамики кадра"
        return "Стедикам с плавным въездом вперёд"
    if choice == "drone":
        return "Drone sweep with smooth glide"
    if choice == "close-up":
        return "Close-up with delicate focus"
    if choice == "portrait lens":
        return "Portrait lens 85mm, shallow depth of field"
    if choice == "wide-angle lens":
        return "Wide-angle 24mm lens for energy"
    return "Steadycam push-in"


def _choose_lighting_detail(text: str, lang: str) -> str:
    lowered = text.lower()
    if _has_keyword(lowered, LOW_LIGHT_HINTS):
        return (
            "Неоновый контровый свет, мягкие рефлексы и дымка"
            if lang == "ru"
            else "Neon rim light with gentle reflections and haze"
        )
    if _has_keyword(lowered, WARM_LIGHT_HINTS):
        return (
            "Тёплый закатный свет с длинными тенями"
            if lang == "ru"
            else "Warm golden-hour glow with elongated shadows"
        )
    return (
        "Мягкий рассеянный свет с акцентом на главного героя"
        if lang == "ru"
        else "Soft diffused light with a spotlight on the subject"
    )


def _choose_style_detail(lang: str) -> str:
    return (
        "Гиперреалистичный кинематографичный стиль, сочные цвета"
        if lang == "ru"
        else "Hyper-realistic cinematic look with rich color grading"
    )


def _choose_audio_detail(text: str, lang: str) -> str:
    lowered = text.lower()
    if "ambient" in lowered or "эмбиент" in lowered:
        return (
            "Эмбиентный звуковой фон с мягким синтезатором"
            if lang == "ru"
            else "Ambient soundscape with gentle synth pads"
        )
    if "rap" in lowered or "рэп" in lowered:
        return (
            "Ритмичный бит с лёгким басом"
            if lang == "ru"
            else "Rhythmic beat with subtle bass"
        )
    return (
        "Атмосферные фоли и лёгкий эмбиент"
        if lang == "ru"
        else "Atmospheric foley with light ambient layers"
    )


def _enhance_composition(lang: str) -> str:
    return (
        "Трёхплановая композиция, выразительный передний план"
        if lang == "ru"
        else "Layered three-plane composition with a defined foreground"
    )


def _format_json_block(raw: str) -> str:
    return f"<blockquote><pre>{html.escape(raw)}</pre></blockquote>"


def _format_text_block(raw: str) -> str:
    escaped = html.escape(raw).replace("\n", "<br/>")
    return f"<blockquote>{escaped}</blockquote>"


def _header_for_engine(engine: str, lang: str) -> str:
    name = ENGINE_DISPLAY.get(engine, {}).get(lang, engine.upper())
    if lang == "ru":
        return f"<b>Готовый промпт для {name}</b>"
    return f"<b>Ready prompt for {name}</b>"


def _build_buttons(engine: str, lang: str) -> InlineKeyboardMarkup:
    display = ENGINE_DISPLAY.get(engine, {}).get(lang, engine.upper())
    if lang == "ru":
        copy_text = "📋 Скопировать"
        insert_text = f"⚡ Вставить в карточку {display}"
    else:
        copy_text = "📋 Copy"
        insert_text = f"⚡ Insert into {display} card"
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(copy_text, callback_data=f"pm:copy:{engine}"),
                InlineKeyboardButton(insert_text, callback_data=f"pm:insert:{engine}"),
            ]
        ]
    )


def _build_veo_json(text: str, lang: str) -> PromptResult:
    camera_detail = _choose_camera_detail(text, lang)
    lighting_detail = _choose_lighting_detail(text, lang)
    composition = _enhance_composition(lang)
    style_detail = _choose_style_detail(lang)
    audio_detail = _choose_audio_detail(text, lang)
    cleaned = _normalize_text(text)
    if lang == "ru":
        payload = {
            "scene": f"{cleaned}. {composition}.",
            "camera": camera_detail,
            "action": f"Герои выполняют задуманное действие, динамика держит внимание. Длительность: 8 секунд.",
            "lighting": lighting_detail,
            "style": style_detail,
            "audio": audio_detail,
        }
    else:
        payload = {
            "scene": f"{cleaned}. {composition}.",
            "camera": camera_detail,
            "action": "Carry out the described idea with purposeful motion. Duration: 8 seconds.",
            "lighting": lighting_detail,
            "style": style_detail,
            "audio": audio_detail,
        }
    return PromptResult("veo", json.dumps(payload, ensure_ascii=False, indent=2), True)


def _build_mj_json(text: str, lang: str) -> PromptResult:
    camera_detail = _choose_camera_detail(text, lang)
    lighting_detail = _choose_lighting_detail(text, lang)
    style_detail = _choose_style_detail(lang)
    cleaned = _normalize_text(text)
    if lang == "ru":
        payload = {
            "prompt": f"{cleaned}, кинематографичная детализация, четыре вариации композиции",
            "style": f"{style_detail}, натуральные фактуры",
            "camera": camera_detail,
            "lighting": lighting_detail,
        }
    else:
        payload = {
            "prompt": f"{cleaned}, cinematic detail, four distinct variations",
            "style": f"{style_detail}, natural textures",
            "camera": camera_detail,
            "lighting": lighting_detail,
        }
    return PromptResult("mj", json.dumps(payload, ensure_ascii=False, indent=2), True)


def _build_face_edit_prompt(text: str, lang: str, engine: str) -> PromptResult:
    camera_detail = _choose_camera_detail(text, lang)
    lighting_detail = _choose_lighting_detail(text, lang)
    composition = _enhance_composition(lang)
    cleaned = _normalize_text(text)
    safety = "keep the real face unchanged, no distortion, no extra limbs, natural details"
    if lang == "ru":
        raw = (
            f"Идея: {cleaned}.\n"
            f"Камера: {camera_detail}.\n"
            f"Композиция: {composition}.\n"
            f"Свет: {lighting_detail}.\n"
            f"Безопасность: {safety}."
        )
    else:
        raw = (
            f"Concept: {cleaned}.\n"
            f"Camera: {camera_detail}.\n"
            f"Composition: {composition}.\n"
            f"Lighting: {lighting_detail}.\n"
            f"Safety: {safety}."
        )
    return PromptResult(engine, raw, False)


def _detect_mood(text: str) -> Tuple[str, str]:
    lowered = text.lower()
    if any(word in lowered for word in ("sad", "грусть", "melancholy", "печаль")):
        return "ambient", "melancholic"
    if any(word in lowered for word in ("энерг", "drive", "энергия", "upbeat", "радост")):
        return "synthwave", "upbeat"
    if any(word in lowered for word in ("dark", "мрач", "тревож")):
        return "dark electronic", "tense"
    return "cinematic pop", "emotional"


def _build_suno_prompt(text: str, lang: str) -> PromptResult:
    genre, mood = _detect_mood(text)
    cleaned = text.strip()
    has_lyrics = "\n" in cleaned or len(cleaned.split()) > 30
    instruments = (
        "Синтезаторы, ударные, атмосферные гитары"
        if lang == "ru"
        else "Synths, drums, atmospheric guitars"
    )
    if lang == "ru":
        prompt_lines = [
            f"Жанр: {genre if genre != 'cinematic pop' else 'современный кинематографичный поп'}",
            f"Настроение: {mood if mood != 'emotional' else 'эмоциональное и вдохновляющее'}",
            f"Сюжет: {cleaned if not has_lyrics else 'история и образ описаны в тексте ниже'}",
            f"Инструменты: {instruments}",
        ]
        if has_lyrics:
            prompt_lines.append("Куплет/припев:")
            prompt_lines.append(cleaned)
    else:
        prompt_lines = [
            f"Genre: {genre}",
            f"Mood: {mood}",
            f"Story: {cleaned if not has_lyrics else 'use the lyrics below as verse/chorus'}",
            f"Instruments: {instruments}",
        ]
        if has_lyrics:
            prompt_lines.append("Lyrics:")
            prompt_lines.append(cleaned)
    raw = "\n".join(prompt_lines)
    return PromptResult("suno", raw, False)


def build_prompt_result(text: str) -> Tuple[PromptResult, str]:
    """Build prompt for given text returning result and language."""

    lang = detect_language(text)
    engine = classify_prompt_engine(text)
    if engine == "veo":
        result = _build_veo_json(text, lang)
    elif engine == "mj":
        result = _build_mj_json(text, lang)
    elif engine == "banana":
        result = _build_face_edit_prompt(text, lang, "banana")
    elif engine == "photo_live":
        result = _build_face_edit_prompt(text, lang, "photo_live")
    else:
        result = _build_suno_prompt(text, lang)
    return result, lang


def format_prompt_message(result: PromptResult, lang: str) -> str:
    """Return HTML-formatted message ready for Telegram."""

    header = _header_for_engine(result.engine, lang)
    block = _format_json_block(result.raw) if result.is_json else _format_text_block(result.raw)
    return f"{header}\n{block}"


async def prompt_master_process(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process user free-form text and reply with generated prompt."""

    message = update.message
    if message is None or not isinstance(message.text, str):
        return
    text = message.text.strip()
    if not text:
        return

    try:
        await message.delete()
    except Exception:
        logger.debug("prompt_master.delete_failed", exc_info=True)

    result, lang = build_prompt_result(text)
    formatted = format_prompt_message(result, lang)
    markup = _build_buttons(result.engine, lang)

    try:
        await message.reply_text(
            formatted,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
            reply_markup=markup,
        )
    except Exception:
        logger.exception("prompt_master.reply_failed")

