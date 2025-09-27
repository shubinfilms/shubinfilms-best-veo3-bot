"""Prompt-Master prompt generator for multiple engines."""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from utils.html_render import safe_lines

class Engine(str, Enum):
    """Supported Prompt-Master engines."""

    VEO_VIDEO = "veo"
    MJ = "mj"
    BANANA_EDIT = "banana"
    VEO_ANIMATE = "animate"
    SUNO = "suno"


@dataclass
class PromptPayload:
    """Structured payload returned by the prompt builder."""

    title: str
    subtitle: Optional[str]
    body_md: str
    code_block: Optional[str]
    insert_payload: Dict[str, Any]
    copy_text: str
    card_text: str
    buttons: List[str] = field(default_factory=list)

    @property
    def body(self) -> str:  # pragma: no cover - legacy compatibility
        return self.card_text


_FACE_SAFETY = {
    "ru": "Сохраняем черты лица без искажений; аккуратная ретушь, без замены лица/стиля лица.",
    "en": "Preserve the person’s facial features without distortion; gentle retouch, no face swap/style change.",
}

_FACE_SWAP_BAN = {
    "ru": "Запрещено подменять или заменять лицо.",
    "en": "Face replacement or swapping is strictly forbidden.",
}

_TITLES = {
    Engine.VEO_VIDEO: {"ru": "Готовый промпт для VEO", "en": "Ready prompt for VEO"},
    Engine.MJ: {"ru": "Готовый промпт для Midjourney", "en": "Ready prompt for Midjourney"},
    Engine.BANANA_EDIT: {"ru": "Чек-лист для Banana", "en": "Banana edit checklist"},
    Engine.VEO_ANIMATE: {"ru": "Чек-лист для “Оживить фото”", "en": "Checklist for “Animate photo”"},
    Engine.SUNO: {"ru": "Каркас запроса для Suno", "en": "Prompt scaffold for Suno"},
}

_MJ_RENDER = "--ar 16:9 --v 6"

_SUNO_DEFAULT_GENRE = {
    "ru": "Кинематографичный электропоп",
    "en": "Cinematic electro-pop",
}

_SUNO_DEFAULT_MOOD = {
    "ru": "вдохновляющее",
    "en": "uplifting",
}

_SUNO_DEFAULT_INSTRUMENTS = {
    "ru": "синтезаторы, атмосферные пад-пэды, легкие ударные",
    "en": "synths, airy pads, light percussion",
}

_SUNO_REFERENCES = {
    "ru": "По настроению — Imogen Heap, Woodkid.",
    "en": "Reference vibe: Imogen Heap, Woodkid.",
}


def _normalize_text(value: str) -> str:
    return " ".join((value or "").strip().split())


def _short_scene(text: str, *, width: int = 180) -> str:
    cleaned = _normalize_text(text)
    if not cleaned:
        return ""
    return textwrap.shorten(cleaned, width=width, placeholder="…")




def _veo_payload(user_text: str, lang: str) -> PromptPayload:
    scene = _short_scene(user_text)
    camera = "Steadicam dolly out" if lang == "en" else "Стефикам с плавным выездом"
    motion = (
        "Dynamic storytelling, 8 seconds, no abrupt motions"
        if lang == "en"
        else "Динамичный сторителлинг, 8 секунд, без резких рывков"
    )
    lighting = (
        "Soft diffused lighting with highlighted subject"
        if lang == "en"
        else "Мягкий рассеянный свет с акцентом"
    )
    palette = (
        "Cinematic color grading with deep shadows"
        if lang == "en"
        else "Кинематографичная цветокоррекция с глубокими тенями"
    )
    details = (
        "Clarify characters, setting, and the main focal moment. Close with an expressive beat."
        if lang == "en"
        else "Уточнить героев, окружение и ключевой акцент. Финал сделать выразительным."
    )
    payload = {
        "scene": scene,
        "camera": camera,
        "motion": motion,
        "lighting": lighting,
        "palette": palette,
        "details": details,
    }
    copy_text = json.dumps(payload, ensure_ascii=False, indent=2)
    duration_line = "Видеоролик длится ~8 секунд." if lang == "ru" else "Video runs for ~8 seconds."
    face_line = _FACE_SAFETY[lang]
    insert_payload = {
        "engine": Engine.VEO_VIDEO.value,
        "format": "16:9",
        "model": "Fast",
        "prompt": payload,
    }
    return PromptPayload(
        title=_TITLES[Engine.VEO_VIDEO][lang],
        subtitle=None,
        body_md=safe_lines([duration_line, face_line]),
        code_block=copy_text,
        insert_payload=insert_payload,
        copy_text=copy_text,
        card_text=copy_text,
    )


def _mj_payload(user_text: str, lang: str) -> PromptPayload:
    scene = _short_scene(user_text, width=200)
    style = "hyper-realistic, premium detail" if lang == "en" else "гиперреалистичный стиль, премиальная детализация"
    camera = "Portrait 35mm low-angle" if lang == "en" else "Портрет 35мм, низкая точка"
    lighting = "Soft diffused cinematic" if lang == "en" else "Мягкий рассеянный кинематографичный свет"
    palette = "Harmonious, film-like grading" if lang == "en" else "Гармоничная кинематографичная палитра"
    payload = {
        "prompt": f"{scene}, {style}",
        "camera": camera,
        "lighting": lighting,
        "palette": palette,
        "render": _MJ_RENDER,
    }
    copy_text = json.dumps(payload, ensure_ascii=False, indent=2)
    note = (
        "MJ создаёт 4 изображения из одного промпта."
        if lang == "ru"
        else "MJ generates 4 images from a single prompt."
    )
    face_line = _FACE_SAFETY[lang]
    insert_payload = {
        "engine": Engine.MJ.value,
        "prompt": payload,
    }
    return PromptPayload(
        title=_TITLES[Engine.MJ][lang],
        subtitle=note,
        body_md=safe_lines([face_line]),
        code_block=copy_text,
        insert_payload=insert_payload,
        copy_text=copy_text,
        card_text=copy_text,
    )


def _banana_tasks(user_text: str, lang: str) -> List[str]:
    base_tasks = [
        (
            "Сохранить черты лица, не менять форму глаз/носа/рта/пропорции"
            if lang == "ru"
            else "Preserve facial traits without altering eyes, nose, mouth or proportions"
        ),
        ("Фон: уточнить/очистить по описанию" if lang == "ru" else "Background: refine/clean as described"),
        ("Одежда и цвет: привести к описанному стилю" if lang == "ru" else "Wardrobe & color: align with described style"),
        ("Удалить лишние объекты и шум" if lang == "ru" else "Remove unwanted objects and noise"),
        (
            "Свет и тон: сделать аккуратным и естественным"
            if lang == "ru"
            else "Light & tone: keep balanced and natural"
        ),
        (
            "Лёгкая ретушь кожи, без «заломов» пластики"
            if lang == "ru"
            else "Gentle skin retouch without plastic creases"
        ),
    ]
    idea_task = (
        f"Основная задача: {user_text.strip()}" if lang == "ru" else f"Primary request: {user_text.strip()}"
    )
    return [idea_task, *base_tasks]


def _banana_payload(user_text: str, lang: str) -> PromptPayload:
    tasks = _banana_tasks(user_text, lang)
    checklist_title = "Чек-лист:" if lang == "ru" else "Checklist:"
    face_line = _FACE_SAFETY[lang]
    ban_line = _FACE_SWAP_BAN[lang]
    copy_text = "\n".join(task for task in tasks)
    insert_payload = {
        "engine": Engine.BANANA_EDIT.value,
        "banana_tasks": tasks,
    }
    list_items = [checklist_title, *[f"• {item}" for item in tasks], face_line, ban_line]
    return PromptPayload(
        title=_TITLES[Engine.BANANA_EDIT][lang],
        subtitle=None,
        body_md=safe_lines(list_items),
        code_block=None,
        insert_payload=insert_payload,
        copy_text=copy_text,
        card_text="\n".join([*tasks, face_line, ban_line]),
    )


def _animate_payload(user_text: str, lang: str) -> PromptPayload:
    hints = [
        (
            "Эмоции и микромимика: мягкое мигание, лёгкая улыбка, спокойное дыхание"
            if lang == "ru"
            else "Emotion & micro-expression: gentle blinking, subtle smile, calm breathing"
        ),
        (
            "Движение камеры: микропанорама/параллакс с плавным отклонением"
            if lang == "ru"
            else "Camera: micro-panorama/parallax with soft drift"
        ),
        (
            "Волосы и ткань: лёгкое движение от ветра"
            if lang == "ru"
            else "Hair & fabric: light movement from a breeze"
        ),
        (
            "Итог: естественно, без пластика и без смещения черт"
            if lang == "ru"
            else "Result: natural, no plastic look, no feature shifting"
        ),
    ]
    face_line = _FACE_SAFETY[lang]
    ban_line = _FACE_SWAP_BAN[lang]
    idea_line = (
        "Описание кадра: " if lang == "ru" else "Frame description: "
    ) + _normalize_text(user_text)
    list_items = [
        "Шаги:" if lang == "ru" else "Steps:",
        idea_line,
        *hints,
        face_line,
        ban_line,
    ]
    bullet_lines = [list_items[0], *[f"• {item}" for item in list_items[1:]]]
    insert_payload = {
        "engine": Engine.VEO_ANIMATE.value,
        "animate_hints": hints,
        "description": _normalize_text(user_text),
    }
    copy_text = "\n".join(hints)
    return PromptPayload(
        title=_TITLES[Engine.VEO_ANIMATE][lang],
        subtitle=None,
        body_md=safe_lines(bullet_lines),
        code_block=None,
        insert_payload=insert_payload,
        copy_text=copy_text,
        card_text="\n".join([idea_line, *hints]),
    )


def _guess_genre(text: str, lang: str) -> str:
    lowered = text.lower()
    mapping = [
        ("rock", "Альтернативный рок" if lang == "ru" else "Alternative rock"),
        ("rap", "Лиричный хип-хоп" if lang == "ru" else "Lyrical hip-hop"),
        ("hip-hop", "Лиричный хип-хоп" if lang == "ru" else "Lyrical hip-hop"),
        ("поп", "Современный поп" if lang == "ru" else "Modern pop"),
        ("synth", "Синтвейв" if lang == "ru" else "Synthwave"),
        ("джаз", "Нео-соул/джаз" if lang == "ru" else "Neo-soul/jazz"),
        ("jazz", "Нео-соул/джаз" if lang == "ru" else "Neo-soul/jazz"),
    ]
    for needle, result in mapping:
        if needle in lowered:
            return result
    return _SUNO_DEFAULT_GENRE[lang]


def _generate_story_lines(text: str, lang: str) -> List[str]:
    idea = _normalize_text(text)
    if not idea:
        return ["Сумеречный город, герой ищет надежду." if lang == "ru" else "Dusky city, hero searching for hope."]
    if len(idea.split()) > 12:
        first = textwrap.shorten(idea, width=80, placeholder="…")
        return [first]
    if lang == "ru":
        return [f"Герой переживает: {idea}.", "В припеве — катарсис и свет."]
    return [f"Protagonist faces: {idea}.", "Chorus brings catharsis and light."]


def _suno_payload(user_text: str, lang: str) -> PromptPayload:
    idea = _normalize_text(user_text)
    genre = _guess_genre(user_text, lang)
    mood = _SUNO_DEFAULT_MOOD[lang]
    instruments = _SUNO_DEFAULT_INSTRUMENTS[lang]
    has_lines = "\n" in user_text.strip()
    if has_lines:
        lyrics = user_text.strip()
    else:
        story_lines = _generate_story_lines(user_text, lang)
        lyrics = "\n".join(story_lines)
    lines = [
        ("Жанр: " if lang == "ru" else "Genre: ") + genre,
        ("Настроение: " if lang == "ru" else "Mood: ") + mood,
        ("Сюжет/картина: " if lang == "ru" else "Story: ") + lyrics,
        ("Инструменты: " if lang == "ru" else "Instruments: ") + instruments,
        (
            "Референсы: " + _SUNO_REFERENCES[lang]
            if lang == "ru"
            else "References: " + _SUNO_REFERENCES[lang]
        ),
    ]
    if has_lines:
        lines.append(
            "Текст куплета/припева включён из запроса."
            if lang == "ru"
            else "Verse/chorus text taken from user input."
        )
    copy_lines = [
        f"Genre: {genre}",
        f"Mood: {mood}",
        f"Story: {lyrics}",
        f"Instruments: {instruments}",
    ]
    if has_lines:
        copy_lines.append("Lyrics included from user input")
    insert_payload = {
        "engine": Engine.SUNO.value,
        "suno": {
            "genre": genre,
            "mood": mood,
            "story": idea or lyrics,
            "instruments": instruments,
            "lyrics": lyrics if has_lines else None,
        },
    }
    return PromptPayload(
        title=_TITLES[Engine.SUNO][lang],
        subtitle=None,
        body_md=safe_lines(lines),
        code_block=None,
        insert_payload=insert_payload,
        copy_text="\n".join(copy_lines),
        card_text="\n".join(lines),
    )


def build_veo_prompt(user_text: str, lang: str) -> PromptPayload:
    return _veo_payload(user_text, "ru" if lang == "ru" else "en")


def build_mj_prompt(user_text: str, lang: str) -> PromptPayload:
    return _mj_payload(user_text, "ru" if lang == "ru" else "en")


def build_banana_prompt(user_text: str, lang: str) -> PromptPayload:
    return _banana_payload(user_text, "ru" if lang == "ru" else "en")


def build_animate_prompt(user_text: str, lang: str) -> PromptPayload:
    return _animate_payload(user_text, "ru" if lang == "ru" else "en")


def build_suno_prompt(user_text: str, lang: str) -> PromptPayload:
    return _suno_payload(user_text, "ru" if lang == "ru" else "en")


async def build_prompt(engine: Engine, user_text: str, lang: str) -> PromptPayload:
    """Build a prompt payload for the requested engine."""

    lang = "ru" if lang == "ru" else "en"
    if engine == Engine.VEO_VIDEO:
        return build_veo_prompt(user_text, lang)
    if engine == Engine.MJ:
        return build_mj_prompt(user_text, lang)
    if engine == Engine.BANANA_EDIT:
        return build_banana_prompt(user_text, lang)
    if engine == Engine.VEO_ANIMATE:
        return build_animate_prompt(user_text, lang)
    if engine == Engine.SUNO:
        return build_suno_prompt(user_text, lang)
    raise ValueError(f"Unsupported engine: {engine}")
