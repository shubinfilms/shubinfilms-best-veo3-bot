"""Prompt-Master prompt generator for multiple engines."""

from __future__ import annotations

import json
import math
import re
import textwrap
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypedDict
from typing import Literal

from utils.html_render import render_pm_html, safe_lines

class Engine(str, Enum):
    """Supported Prompt-Master engines."""

    VEO_VIDEO = "veo"
    MJ = "mj"
    BANANA_EDIT = "banana"
    VEO_ANIMATE = "animate"
    SUNO = "suno"


class PMResultBase(TypedDict):
    """Typed payload returned by Prompt-Master generators."""

    engine: Literal["veo", "mj", "banana", "animate", "suno"]
    language: str
    body_html: str
    raw_payload: Optional[Dict[str, Any]]
    copy_text: str


class PMResult(PMResultBase, total=False):
    card_text: str


@dataclass
class _Payload:
    """Intermediate payload before converting to Telegram HTML."""

    title: str
    subtitle: Optional[str]
    body_md: str
    code_block: Optional[str]
    insert_payload: Dict[str, Any]
    copy_text: str
    card_text: str
    buttons: List[str] = field(default_factory=list)


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


_VOICEOVER_MARKER_RE = re.compile(
    r"(озвучк[аи]?|диалог(?:ов)?|реплик[аи]?|voice ?over|dialog(?:ue)?|lines|script)\s*:\s*",
    re.IGNORECASE,
)

_WRITE_DIALOGUE_RE = re.compile(
    r"(?:напиш(?:и|ите)|придум(?:ай|айте)|нужен|нужно|нужны|хочу|давай)\s+"
    r"(?:коротк[ийоеуюя]?|небольшой|маленький|.*?)(?:диалог|озвучк|реплик|текст)"
    r"|write\s+(?:a\s+)?(?:short\s+)?(?:dialogue|voice ?over|script|lines)"
    r"|create\s+(?:the\s+)?dialogue",
    re.IGNORECASE,
)

_VOICEOVER_FALLBACK_NAMES = {
    "ru": ["Голос 1", "Голос 2", "Голос 3", "Голос 4", "Голос 5"],
    "en": ["Voice 1", "Voice 2", "Voice 3", "Voice 4", "Voice 5"],
}

_ROLE_KEYWORDS_RU = [
    ("учитель", "Учитель"),
    ("учени", "Ученик"),
    ("героин", "Героиня"),
    ("герой", "Герой"),
    ("мама", "Мама"),
    ("папа", "Папа"),
    ("девуш", "Девушка"),
    ("парен", "Парень"),
    ("команд", "Команда"),
    ("капита", "Капитан"),
]

_ROLE_KEYWORDS_EN = [
    ("teacher", "Teacher"),
    ("student", "Student"),
    ("hero", "Hero"),
    ("heroine", "Heroine"),
    ("mother", "Mother"),
    ("father", "Father"),
    ("girl", "Girl"),
    ("boy", "Boy"),
    ("team", "Team lead"),
    ("captain", "Captain"),
]

_STOPWORDS_RU = {
    "и",
    "в",
    "на",
    "с",
    "к",
    "по",
    "из",
    "за",
    "для",
    "о",
    "от",
    "это",
    "как",
    "что",
    "чтобы",
    "про",
    "еще",
    "у",
    "бы",
    "но",
    "же",
    "так",
    "а",
}

_STOPWORDS_EN = {
    "and",
    "in",
    "on",
    "with",
    "for",
    "the",
    "a",
    "of",
    "to",
    "into",
    "is",
    "are",
    "be",
    "as",
    "at",
    "by",
    "that",
    "this",
    "it",
    "from",
    "an",
    "we",
    "you",
}

_FEATURE_KEYWORDS = {
    "storm": ["гроз", "шторм", "storm", "thunder", "lightning"],
    "rain": ["дожд", "rain"],
    "snow": ["снег", "snow"],
    "night": ["ноч", "night", "moon"],
    "dawn": ["рассвет", "утро", "sunrise", "morning"],
    "classroom": ["класс", "аудитор", "school", "classroom", "lecture"],
    "forest": ["лес", "роща", "forest", "woods"],
    "city": ["город", "street", "улиц", "city", "downtown"],
    "desert": ["пустын", "dune", "desert"],
}

_FEATURE_DETAILS = {
    "storm": {
        "lighting": {
            "ru": "Вспышки молний через окно дают резкий контраст и подчеркивают силуэты героев.",
            "en": "Lightning bursts from the window carve sharp contrasts across the characters.",
        },
        "palette": {
            "ru": "Холодные сине-стальные тона со всполохами белого света.",
            "en": "Cold steel-blues with crisp white lightning highlights.",
        },
        "details": {
            "ru": "Капли дождя на стекле, дрожащие отражения и туманное дыхание.",
            "en": "Rain streaks on glass, trembling reflections and misted breath.",
        },
        "ambience": {
            "ru": "Шум дождя и гул ветра за стеклом.",
            "en": "Rain battering the windows with a low wind rumble.",
        },
        "sfx": {
            "ru": "Приглушённые раскаты грома для переходов между фразами.",
            "en": "Soft thunder rumbles punctuate the transitions.",
        },
    },
    "rain": {
        "lighting": {
            "ru": "Мягкие отражения влажных поверхностей добавляют глубину.",
            "en": "Soft reflections off wet surfaces add depth.",
        },
        "palette": {
            "ru": "Глубокие зелёно-синие полутона и теплые бликовые акценты на коже.",
            "en": "Deep teal shadows with warm highlights on skin.",
        },
        "details": {
            "ru": "Водяные капли, блеск улицы и лёгкая дымка в воздухе.",
            "en": "Water droplets, slick pavement glints and a light mist in the air.",
        },
        "ambience": {
            "ru": "Фоновый шум дождя и редкие машины на удалении.",
            "en": "Steady rainfall with distant traffic hush.",
        },
    },
    "snow": {
        "lighting": {
            "ru": "Холодный отражённый свет от снега выравнивает тон кожи.",
            "en": "Snow bounce light levels the skin tones with a cold sheen.",
        },
        "palette": {
            "ru": "Чистые белые и голубые полутона с мягкими розовыми акцентами тепла.",
            "en": "Clean whites and pale blues with subtle warm rose accents.",
        },
        "details": {
            "ru": "Кристаллики снега на одежде и пар дыхания в морозном воздухе.",
            "en": "Snow crystals on fabric and plumes of breath in the cold air.",
        },
        "ambience": {
            "ru": "Приглушённая тишина и лёгкое поскрипывание снега.",
            "en": "Muffled quiet with gentle crunches of snow.",
        },
    },
    "night": {
        "lighting": {
            "ru": "Ночные источники — мягкие неоновые блики и контрастные тени.",
            "en": "Night ambience with gentle neon glows and defined shadows.",
        },
        "palette": {
            "ru": "Смешение глубоких индиго и тёплых янтарных акцентов.",
            "en": "Mix of deep indigos with warm amber accents.",
        },
        "ambience": {
            "ru": "Тихий ночной фон, редкие отдалённые звуки города.",
            "en": "Muted nighttime city hum with sparse distant sounds.",
        },
    },
    "dawn": {
        "lighting": {
            "ru": "Рассеянный свет рассвета с мягким золотым заполнением.",
            "en": "Diffused dawn glow with soft golden fill light.",
        },
        "palette": {
            "ru": "Пастельные оттенки персика и голубого, создающие надежду.",
            "en": "Pastel peach and sky blue hues evoking hope.",
        },
        "ambience": {
            "ru": "Пробуждающийся город, далёкие птицы и лёгкий ветер.",
            "en": "Waking city ambience with distant birds and a light breeze.",
        },
    },
    "classroom": {
        "details": {
            "ru": "Листы с формулами, след мела на доске и мягкий свет из окна.",
            "en": "Sheets with formulas, chalk residue on the board and soft window light.",
        },
        "ambience": {
            "ru": "Тихие шаги, шелест страниц и лёгкий гул вентиляции.",
            "en": "Soft footsteps, pages rustling and a mild ventilation hum.",
        },
        "sfx": {
            "ru": "Лёгкие акценты мела и щелчок ручки в моменты решений.",
            "en": "Subtle chalk taps and a pen click marking breakthroughs.",
        },
    },
    "forest": {
        "details": {
            "ru": "Лесная подстилка, пыльца в лучах света и влажный мох на деревьях.",
            "en": "Forest floor texture, pollen in light shafts and damp moss on trunks.",
        },
        "ambience": {
            "ru": "Пение птиц, лёгкий ветер в листве и далёкие звуки природы.",
            "en": "Birdsong, gentle wind in leaves and distant woodland calls.",
        },
    },
    "city": {
        "details": {
            "ru": "Неоновые отражения на мокром асфальте, стекло и металл вокруг героев.",
            "en": "Neon reflections on wet asphalt with glass and metal surroundings.",
        },
        "ambience": {
            "ru": "Приглушённые звуки улицы, редкие машины и далёкие сигналы.",
            "en": "Soft street ambience with distant traffic and signals.",
        },
    },
    "desert": {
        "details": {
            "ru": "Песчаная взвесь в воздухе, текстура дюн и солнечные блики.",
            "en": "Dust in the air, dune textures and blazing sun flares.",
        },
        "palette": {
            "ru": "Тёплые охристые тона с контрастными небесно-голубыми оттенками.",
            "en": "Warm ochre palette contrasted with clear sky blues.",
        },
        "ambience": {
            "ru": "Порывы сухого ветра и гул пустыни.",
            "en": "Dry wind gusts and a low desert hum.",
        },
    },
}

_CAMERA_TYPE_HINTS = [
    (["дрон", "drone", "fpv"], {"ru": "Дрон-съёмка", "en": "Drone shot"}),
    (["ручн", "handheld"], {"ru": "Ручная камера", "en": "Handheld camera"}),
    (["стедика", "gimbal", "steadicam"], {"ru": "Стедикам", "en": "Steadicam rig"}),
]

_CAMERA_MOVEMENT_HINTS = [
    (["панорам", "orbit"], {"ru": "плавный круговой облет вокруг героев", "en": "smooth orbital move around the subjects"}),
    (["долли", "проезд", "подъезд", "push", "pull"], {"ru": "медленный проезд вперёд для усиления эмоций", "en": "slow dolly-in to amplify emotion"}),
    (["крупн", "close"], {"ru": "мягкий переход к крупному плану", "en": "gentle drift into a close-up"}),
]

_CAMERA_ANGLE_HINTS = [
    (["верхн", "bird", "top"], {"ru": "верхний ракурс", "en": "top angle"}),
    (["низк", "low"], {"ru": "низкий драматичный ракурс", "en": "low dramatic angle"}),
    (["плеч", "over-the-shoulder"], {"ru": "ракурс с плеча", "en": "over-the-shoulder angle"}),
]

_CAMERA_LENS_HINTS = [
    (["широкоуг", "wide"], "24mm"),
    (["портрет", "portrait"], "50mm"),
    (["телевик", "tele"], "85mm"),
]

_VEO_BASE = {
    "ru": {
        "lighting": "Мягкий кинематографичный свет с акцентом на героях и объёме пространства.",
        "palette": "Гармоничная цветокоррекция с глубокими тенями и аккуратными световыми акцентами.",
        "details": "Фокус на мимике, фактурах окружения и ключевом действии кадра.",
        "audio_ambience": "Тонкий фон локации поддерживает эмоцию без отвлечения.",
        "audio_sfx": "Неброские звуковые переходы подчеркивают монтажные beat'ы.",
        "notes": "8s total; сохранить черты лиц/минимум искажений; без субтитров/оверлеев",
        "safety": "Запрет читаемых надписей в кадре, без бренд-логотипов, без запрещённого контента",
        "subtitle": "Видео ≈ 8 секунд",
    },
    "en": {
        "lighting": "Soft cinematic lighting emphasizing the characters and spatial depth.",
        "palette": "Harmonised film-grade palette with rich shadows and precise highlights.",
        "details": "Keep facial nuances, environment textures and the story beat in focus.",
        "audio_ambience": "Subtle location bed underscores the emotion without distraction.",
        "audio_sfx": "Delicate transitions accentuate the editorial beats.",
        "notes": "8s total; preserve facial features/no distortions; no subtitles/overlays",
        "safety": "No readable text, no brand logos, no prohibited content",
        "subtitle": "Video ≈ 8 seconds",
    },
}


@dataclass
class _VoiceoverPlan:
    lines: List[Tuple[str, str]]
    origin: Literal["provided", "generated", "none"]
    requested: bool

    @property
    def has_voiceover(self) -> bool:
        return bool(self.lines)


def _split_voiceover_block(text: str) -> Tuple[str, str]:
    match = _VOICEOVER_MARKER_RE.search(text)
    if not match:
        return text, ""
    head = text[: match.start()].rstrip()
    tail = text[match.end() :].strip()
    return head, tail


def _parse_voiceover_lines(block: str) -> List[Tuple[str, str]]:
    if not block:
        return []
    lines: List[Tuple[str, str]] = []
    for raw_line in block.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.startswith("- "):
            stripped = stripped[2:].strip()
        if not stripped:
            continue
        if ":" in stripped:
            speaker, text = stripped.split(":", 1)
            speaker = speaker.strip()
            text_value = text.strip()
        else:
            speaker = ""
            text_value = stripped
        if not text_value:
            continue
        lines.append((speaker, text_value))
    return lines


def _detect_role_names(idea_text: str, lang: str) -> List[str]:
    names: List[str] = []
    lowered = idea_text.lower()
    if lang == "ru":
        for needle, value in _ROLE_KEYWORDS_RU:
            if needle in lowered:
                names.append(value)
        candidates = re.findall(r"\b[А-ЯЁ][а-яё]{2,}\b", idea_text)
        for name in candidates:
            lowered_name = name.lower()
            if "напиш" in lowered_name or "диалог" in lowered_name:
                continue
            names.append(name)
    else:
        for needle, value in _ROLE_KEYWORDS_EN:
            if needle in lowered:
                names.append(value)
        candidates = re.findall(r"\b[A-Z][a-z]{2,}\b", idea_text)
        for name in candidates:
            lowered_name = name.lower()
            if lowered_name in {"write", "dialogue"}:
                continue
            names.append(name)
    seen: set[str] = set()
    ordered: List[str] = []
    for name in names:
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(name)
    if ordered:
        return ordered
    return []


def _extract_keywords(idea_text: str, lang: str, *, limit: int = 3) -> List[str]:
    words = re.findall(r"[A-Za-zА-Яа-яЁё-]+", idea_text.lower())
    stopwords = _STOPWORDS_RU if lang == "ru" else _STOPWORDS_EN
    result: List[str] = []
    for word in words:
        normalized = word.strip("-")
        if not normalized or len(normalized) < 3:
            continue
        if normalized in stopwords:
            continue
        if normalized in result:
            continue
        result.append(normalized)
        if len(result) >= limit:
            break
    if not result:
        result = ["идея"] if lang == "ru" else ["story"]
    while len(result) < limit:
        result.append(result[-1])
    return result[:limit]


def _generate_voiceover_lines(idea_text: str, lang: str) -> List[Tuple[str, str]]:
    roles = _detect_role_names(idea_text, lang)
    if not roles:
        roles = _VOICEOVER_FALLBACK_NAMES[lang][:2]
    elif len(roles) == 1:
        roles.append(roles[0])
    keywords = _extract_keywords(idea_text, lang)
    if lang == "ru":
        templates = [
            "{a}: Держим фокус на {k0}, не теряя спокойствия.",
            "{b}: Добавим {k1}, и решение станет ясным.",
            "{a}: Отлично, фиксируем результат и идём дальше.",
        ]
    else:
        templates = [
            "{a}: Stay with {k0}, keep the tone grounded.",
            "{b}: Add {k1}, everything clicks into place.",
            "{a}: Perfect, lock the beat and carry it home.",
        ]
    lines: List[Tuple[str, str]] = []
    for idx, template in enumerate(templates):
        text_value = template.format(a=roles[0], b=roles[1], k0=keywords[0], k1=keywords[1], k2=keywords[2])
        speaker = roles[0] if idx != 1 else roles[1]
        lines.append((speaker, text_value.split(":", 1)[1].strip()))
    return lines


def _build_voiceover(text: str, lang: str) -> Tuple[str, _VoiceoverPlan]:
    base_text, block = _split_voiceover_block(text)
    provided_lines = _parse_voiceover_lines(block)
    requested = bool(_WRITE_DIALOGUE_RE.search(text))
    if provided_lines:
        return base_text.strip(), _VoiceoverPlan(lines=provided_lines, origin="provided", requested=requested)
    if requested:
        generated = _generate_voiceover_lines(base_text or text, lang)
        return base_text.strip(), _VoiceoverPlan(lines=generated, origin="generated", requested=True)
    return base_text.strip(), _VoiceoverPlan(lines=[], origin="none", requested=requested)


def _voiceover_time_slots(count: int) -> List[str]:
    if count <= 0:
        return []
    total = min(4.0, 3.0 + 0.3 * max(count - 1, 0))
    start = 2.1
    end = min(6.5, start + total)
    duration = (end - start) / count
    slots: List[str] = []
    current = start
    for _ in range(count):
        slot_end = current + duration
        slot_end = min(slot_end, 7.5)
        slots.append(f"{_format_time(current)}–{_format_time(slot_end)}s")
        current = slot_end
    return slots


def _format_time(value: float) -> str:
    rounded = max(0.0, round(value + 1e-6, 1))
    if math.isclose(rounded, int(rounded)):
        return f"{int(rounded)}.0"
    return f"{rounded:.1f}"


def _format_voiceover_entries(plan: _VoiceoverPlan, lang: str) -> List[Dict[str, str]]:
    if not plan.has_voiceover:
        return []
    slots = _voiceover_time_slots(len(plan.lines))
    fallback = _VOICEOVER_FALLBACK_NAMES[lang]
    entries: List[Dict[str, str]] = []
    for idx, ((character, line), slot) in enumerate(zip(plan.lines, slots)):
        name = character.strip()
        if not name:
            name = fallback[idx % len(fallback)]
        entries.append({"character": name, "line": line, "time": slot})
    return entries


def _strip_trailing(text: str) -> str:
    return text.rstrip(".?! ")


def _ensure_sentence(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    if text.endswith(('.', '!', '?')):
        return text
    return f"{text}."


def _detect_features(text: str) -> List[str]:
    lowered = text.lower()
    found: List[str] = []
    for feature, needles in _FEATURE_KEYWORDS.items():
        if any(needle in lowered for needle in needles):
            found.append(feature)
    # merge rain/storm to avoid duplicates
    if "storm" in found and "rain" in found:
        found.remove("rain")
    return found


def _apply_feature_text(base: Dict[str, str], features: Sequence[str], lang: str) -> Dict[str, str]:
    lighting_parts = [base["lighting"]]
    palette_parts = [base["palette"]]
    detail_parts = [base["details"]]
    ambience_parts = [base["audio_ambience"]]
    sfx_parts = [base["audio_sfx"]]
    for feature in features:
        info = _FEATURE_DETAILS.get(feature)
        if not info:
            continue
        lighting = info.get("lighting", {}).get(lang)
        palette = info.get("palette", {}).get(lang)
        detail = info.get("details", {}).get(lang)
        ambience = info.get("ambience", {}).get(lang)
        sfx = info.get("sfx", {}).get(lang)
        if lighting:
            lighting_parts.append(lighting)
        if palette:
            palette_parts.append(palette)
        if detail:
            detail_parts.append(detail)
        if ambience:
            ambience_parts.append(ambience)
        if sfx:
            sfx_parts.append(sfx)
    return {
        "lighting": " ".join(lighting_parts),
        "palette": " ".join(palette_parts),
        "details": " ".join(detail_parts),
        "audio_ambience": " ".join(ambience_parts),
        "audio_sfx": " ".join(sfx_parts),
    }


def _apply_camera_hints(text: str, lang: str) -> Dict[str, str]:
    lowered = text.lower()
    camera_type = "Стедикам" if lang == "ru" else "Steadicam"
    movement = "плавный дуговой проход вокруг ключевой точки" if lang == "ru" else "smooth arc move around the key moment"
    angle = "уровень глаз" if lang == "ru" else "eye-level angle"
    lens = "35mm"
    for needles, values in _CAMERA_TYPE_HINTS:
        if any(needle in lowered for needle in needles):
            camera_type = values.get(lang, camera_type)
            break
    for needles, values in _CAMERA_MOVEMENT_HINTS:
        if any(needle in lowered for needle in needles):
            movement = values.get(lang, movement)
            break
    for needles, values in _CAMERA_ANGLE_HINTS:
        if any(needle in lowered for needle in needles):
            angle = values.get(lang, angle)
            break
    lens_match = re.search(r"(\d{2})\s*(?:mm|мм)", lowered)
    if lens_match:
        lens = f"{lens_match.group(1)}mm"
    else:
        for needles, value in _CAMERA_LENS_HINTS:
            if any(needle in lowered for needle in needles):
                lens = value
                break
    if lang == "ru":
        movement = _ensure_sentence(movement).rstrip('.')
    else:
        movement = _ensure_sentence(movement).rstrip('.')
    return {
        "type": camera_type,
        "movement": movement,
        "angle": angle,
        "lens": lens,
    }


def _timeline_actions(idea_sentence: str, features: Sequence[str], lang: str) -> List[Dict[str, str]]:
    clean_idea = _strip_trailing(idea_sentence)
    if lang == "ru":
        base = [
            "Камера мягко устанавливает локацию и героев.",
            f"Ключевой момент: {clean_idea}.",
            "Темп замедляется, камера ловит реакцию и детали.",
            "Финальный beat: короткая пауза и удержание взгляда.",
        ]
    else:
        base = [
            "Camera eases into the setting and characters.",
            f"Key beat: {clean_idea}.",
            "Tempo softens while we catch reactions and texture.",
            "Final beat: brief hold for a memorable freeze.",
        ]
    extras = []
    for feature in features:
        info = _FEATURE_DETAILS.get(feature)
        if not info:
            continue
        detail = info.get("details", {}).get(lang)
        ambience = info.get("ambience", {}).get(lang)
        if detail and lang == "ru":
            extras.append(detail)
        elif detail and lang == "en":
            extras.append(detail)
        if ambience and lang == "ru":
            extras.append(ambience)
        elif ambience and lang == "en":
            extras.append(ambience)
    if extras:
        addition = extras[0]
        if lang == "ru":
            base[1] = _ensure_sentence(base[1])[:-1] + f" {addition}"
        else:
            base[1] = _ensure_sentence(base[1])[:-1] + f" {addition}"
    times = ["0–2s", "2–5s", "5–7s", "7–8s"]
    return [{"t": t, "action": _ensure_sentence(text)} for t, text in zip(times, base)]


def _scene_description(sentences: List[str], idea_sentence: str, lang: str) -> Dict[str, str]:
    first = sentences[0] if sentences else idea_sentence
    last = sentences[-1] if len(sentences) > 1 else idea_sentence
    if lang == "ru":
        setting = _ensure_sentence(first)
        initial = _ensure_sentence(f"Камера открывает сцену: {_strip_trailing(first)}")
        final = _ensure_sentence(f"Финал подчёркивает {_strip_trailing(last)}")
    else:
        setting = _ensure_sentence(first)
        initial = _ensure_sentence(f"Camera opens on {_strip_trailing(first)}")
        final = _ensure_sentence(f"Closing frame highlights {_strip_trailing(last)}")
    return {
        "setting": setting,
        "initial_state": initial,
        "final_state": final,
    }


def _prepare_voiceover_payload(plan: _VoiceoverPlan, lang: str) -> Tuple[Dict[str, Any], bool]:
    entries = _format_voiceover_entries(plan, lang)
    audio: Dict[str, Any] = {}
    if entries:
        audio["voiceover"] = entries
    return audio, bool(entries)


def _veo_payload(user_text: str, lang: str) -> _Payload:
    base_text, voice_plan = _build_voiceover(user_text, lang)
    idea_sentence = _ensure_sentence(_short_scene(base_text or user_text, width=200)) or (
        "Кинематографичная сцена." if lang == "ru" else "Cinematic scene."
    )
    sentences = [sent.strip() for sent in re.split(r"[.!?]\s+", base_text) if sent.strip()]
    features = _detect_features(user_text)
    base_locale = _VEO_BASE[lang]
    feature_text = _apply_feature_text(base_locale, features, lang)
    camera = _apply_camera_hints(user_text, lang)
    timeline = _timeline_actions(idea_sentence, features, lang)
    scene_description = _scene_description(sentences, idea_sentence, lang)
    audio, has_voiceover = _prepare_voiceover_payload(voice_plan, lang)
    audio["ambience"] = feature_text["audio_ambience"]
    audio["sfx"] = feature_text["audio_sfx"]
    veo_json: Dict[str, Any] = {
        "idea": idea_sentence,
        "scene_description": scene_description,
        "timeline": timeline,
        "camera": camera,
        "lighting": feature_text["lighting"],
        "palette": feature_text["palette"],
        "details": feature_text["details"],
        "audio": audio,
        "notes": base_locale["notes"] + ("; lip-sync required" if has_voiceover else ""),
        "safety": base_locale["safety"],
    }
    copy_text = json.dumps(veo_json, ensure_ascii=False, indent=2)
    subtitle = base_locale["subtitle"]
    insert_payload = {
        "engine": Engine.VEO_VIDEO.value,
        "prompt": veo_json,
        "duration_hint": "≈8 секунд" if lang == "ru" else "≈8 seconds",
        "lip_sync_required": has_voiceover,
        "voiceover_origin": voice_plan.origin,
        "voiceover_requested": voice_plan.requested,
    }
    return _Payload(
        title=_TITLES[Engine.VEO_VIDEO][lang],
        subtitle=subtitle,
        body_md="",
        code_block=copy_text,
        insert_payload=insert_payload,
        copy_text=copy_text,
        card_text=copy_text,
    )


def _mj_payload(user_text: str, lang: str) -> _Payload:
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
    return _Payload(
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


def _banana_payload(user_text: str, lang: str) -> _Payload:
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
    return _Payload(
        title=_TITLES[Engine.BANANA_EDIT][lang],
        subtitle=None,
        body_md=safe_lines(list_items),
        code_block=None,
        insert_payload=insert_payload,
        copy_text=copy_text,
        card_text="\n".join([*tasks, face_line, ban_line]),
    )


def _animate_payload(user_text: str, lang: str) -> _Payload:
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
    return _Payload(
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


def _suno_payload(user_text: str, lang: str) -> _Payload:
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
    return _Payload(
        title=_TITLES[Engine.SUNO][lang],
        subtitle=None,
        body_md=safe_lines(lines),
        code_block=None,
        insert_payload=insert_payload,
        copy_text="\n".join(copy_lines),
        card_text="\n".join(lines),
    )


def _payload_to_html(payload: _Payload) -> str:
    parts: List[str] = [f"**{payload.title}**"]
    if payload.subtitle:
        parts.append(f"_{payload.subtitle}_")
    if payload.body_md:
        parts.append(payload.body_md)
    if payload.code_block:
        code_value = payload.code_block.strip()
        language = "json" if code_value.startswith(("{", "[")) else ""
        parts.append(f"```{language}\n{code_value}\n```")
    if payload.buttons:
        parts.append(safe_lines(payload.buttons))
    markdown = "\n\n".join(part for part in parts if part)
    return render_pm_html(markdown)


def _to_result(engine: Engine, lang: str, payload: _Payload) -> PMResult:
    return PMResult(
        engine=engine.value,
        language=lang,
        body_html=_payload_to_html(payload),
        raw_payload=payload.insert_payload or None,
        copy_text=payload.copy_text,
        card_text=payload.card_text,
    )


def build_veo_prompt(user_text: str, lang: str) -> PMResult:
    lang = "ru" if lang == "ru" else "en"
    return _to_result(Engine.VEO_VIDEO, lang, _veo_payload(user_text, lang))


def build_mj_prompt(user_text: str, lang: str) -> PMResult:
    lang = "ru" if lang == "ru" else "en"
    return _to_result(Engine.MJ, lang, _mj_payload(user_text, lang))


def build_banana_prompt(user_text: str, lang: str) -> PMResult:
    lang = "ru" if lang == "ru" else "en"
    return _to_result(Engine.BANANA_EDIT, lang, _banana_payload(user_text, lang))


def build_animate_prompt(user_text: str, lang: str) -> PMResult:
    lang = "ru" if lang == "ru" else "en"
    return _to_result(Engine.VEO_ANIMATE, lang, _animate_payload(user_text, lang))


def build_suno_prompt(user_text: str, lang: str) -> PMResult:
    lang = "ru" if lang == "ru" else "en"
    return _to_result(Engine.SUNO, lang, _suno_payload(user_text, lang))


async def build_prompt(engine: Engine, user_text: str, lang: str) -> PMResult:
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
