# -*- coding: utf-8 -*-
"""Prompt-Master core: generate structured cinematic prompts."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, Optional, Tuple

LOGGER = logging.getLogger(__name__)

_USE_NEW_CLIENT = False
_client: Any = None
try:  # pragma: no cover - optional dependency
    from openai import OpenAI  # type: ignore

    _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
    _USE_NEW_CLIENT = True
except Exception:  # pragma: no cover - fallback to legacy SDK
    try:
        import openai  # type: ignore

        openai.api_key = os.getenv("OPENAI_API_KEY", "")
        _client = openai
        _USE_NEW_CLIENT = False
    except Exception:  # pragma: no cover - no client available
        _client = None
        _USE_NEW_CLIENT = False


SYSTEM_PROMPT = """You are Prompt-Master 2.0 — a creative cinematic prompt writer.
TASK:
- Keep the user's core idea intact.
- Deliver cinematic, premium, hyper-realistic descriptions.
- No TV/news vibe. Be modern, stylish, emotional, yet plausible.
- Respect any explicit technical hints (camera, format, speed).
- No subtitles, on-screen text or logos unless explicitly requested.

LANGUAGE:
- Detect whether the user idea is Russian or English.
- Respond in that language only (section titles + content).
- Never mix languages. No emojis.

OUTPUT STRICTLY AS JSON with keys:
{
  "scene": "...",
  "camera": "...",
  "action": "...",
  "dialogue": "...",
  "atmosphere": "...",
  "lighting": "...",
  "wardrobe_props": "...",
  "framing": "..."
}

Each value must be a concise paragraph (<= 260 chars) with rich cinematic detail. Mention sound/voice only if user clearly implies dialogue or voiceover. Keep authenticity and realism.
"""


_CYRILLIC_RE = re.compile(r"[а-яА-ЯёЁ]")
VOICE_HINT_RE = re.compile(r"\b(voice|озвучк|диктор|голос|озвучить|озвучка|narration)\b", re.IGNORECASE)
MUSIC_HINT_RE = re.compile(r"\b(music|музык|бит|саунд|soundtrack|score|beat)\b", re.IGNORECASE)

CAMERA_HINTS = [
    r"\b(\d{2,3}mm)\b",
    r"\bprime lens\b",
    r"\bshallow depth of field\b",
    r"\bDOF\b",
    r"\banamorphic\b",
    r"\bhandheld\b",
    r"\bsteadycam\b",
    r"\bgimbal\b",
    r"\bdrone\b",
    r"\bFPV\b",
    r"\bclose[- ]?up\b",
    r"\bmacro\b",
    r"\bwide[- ]?shot\b",
    r"\bmedium\b",
    r"\bslow[- ]?motion\b",
    r"\bslow[- ]?mo\b",
    r"\breal[- ]?time\b",
    r"\btime[- ]?remap\b",
    r"\btilt[- ]?shift\b",
    r"\bpan\b",
    r"\btilt\b",
    r"\bzoom\b",
    r"\bdolly\b",
    r"\bpush[- ]?in\b",
    r"\bpull[- ]?back\b",
]


def _lang(text: str) -> str:
    return "ru" if _CYRILLIC_RE.search(text or "") else "en"


def _voice_req(text: str) -> bool:
    return bool(VOICE_HINT_RE.search(text or ""))


def _music_req(text: str) -> bool:
    return bool(MUSIC_HINT_RE.search(text or ""))


def _cam_tokens(text: str) -> str:
    found = []
    for pat in CAMERA_HINTS:
        matches = re.findall(pat, text, flags=re.IGNORECASE)
        if not matches:
            continue
        if isinstance(matches[0], tuple):
            for tup in matches:
                for token in tup:
                    if token:
                        found.append(token)
        else:
            found.extend(matches)
    tokens, seen = [], set()
    for token in (item.strip().lower() for item in found if item):
        if token and token not in seen:
            seen.add(token)
            tokens.append(token)
    return ", ".join(tokens)


def _build_user_instruction(
    text: str,
    lang: str,
    v_req: bool,
    m_req: bool,
    camera_tokens: Optional[str] = None,
) -> str:
    lines = [
        "USER IDEA:",
        text.strip(),
        "",
        "CONSTRAINTS:",
        "- Keep realism, clean motion, plausible physics.",
        "- No TV/news tone. Be modern, stylish, emotional.",
    ]
    if m_req:
        lines.append("- Provide modern music suggestion that fits mood (hip-hop/ambient/cinematic/electronic).")
    if v_req:
        lines.append(
            "- Add Russian voiceover only, natural and expressive (no radio anchor)."
            if lang == "ru"
            else "- Add English voiceover only, natural and expressive (no radio anchor)."
        )
    lines.append("- If user forbids text/logos/subtitles — keep none.")
    cams = camera_tokens if camera_tokens is not None else _cam_tokens(text)
    if cams:
        lines.append(f"- Camera technical hints to respect: {cams}")
    return "\n".join(lines)

SECTION_ORDER = [
    "scene",
    "camera",
    "action",
    "dialogue",
    "atmosphere",
    "lighting",
    "wardrobe_props",
    "framing",
]


SECTION_LABELS = {
    "ru": {
        "scene": "Сцена",
        "camera": "Камера",
        "action": "Действие",
        "dialogue": "Диалог",
        "atmosphere": "Атмосфера",
        "lighting": "Свет",
        "wardrobe_props": "Костюмы и реквизит",
        "framing": "Кадрирование",
    },
    "en": {
        "scene": "Scene",
        "camera": "Camera",
        "action": "Action",
        "dialogue": "Dialogue",
        "atmosphere": "Atmosphere",
        "lighting": "Lighting",
        "wardrobe_props": "Wardrobe/props",
        "framing": "Framing",
    },
}


def _clean_value(value: str) -> str:
    return value.replace("```", "``\`").strip()


def _fallback_sections(raw_text: str, lang: str) -> Dict[str, str]:
    snippet = raw_text.strip()
    if len(snippet) > 180:
        snippet = snippet[:180].rstrip() + "…"

    if lang == "ru":
        return {
            "scene": snippet or "Коротко опишите ключевые персонажи и место действия.",
            "camera": "85 мм, плавные панорамы, лёгкий хэндхелд для живости.",
            "action": "Реалистичное, пластичное движение без артефактов.",
            "dialogue": "Естественные реплики в тон сцены; при отсутствии диалогов — сделать акцент на эмоциях героев.",
            "atmosphere": "Современное кинематографичное настроение с вниманием к деталям.",
            "lighting": "Мягкий объёмный свет, акцентирующий лица и фактуру.",
            "wardrobe_props": "Стильные костюмы и реквизит, поддерживающие идею.",
            "framing": "Комбинация средних и крупных планов, аккуратные движения камеры.",
        }

    return {
        "scene": snippet or "Outline the key characters and location succinctly.",
        "camera": "85mm lens feel, smooth pans, subtle handheld for realism.",
        "action": "Grounded, fluid motion with natural pacing and clean choreography.",
        "dialogue": "Conversational lines that match the tone; if none, highlight emotional beats.",
        "atmosphere": "Modern cinematic mood with polished yet believable detail.",
        "lighting": "Soft volumetric lighting that sculpts faces and textures.",
        "wardrobe_props": "Stylish wardrobe and props reinforcing the concept.",
        "framing": "Mix of medium and close shots with deliberate camera moves.",
    }


def _format_sections(sections: Dict[str, str], lang: str) -> str:
    labels = SECTION_LABELS["ru" if lang == "ru" else "en"]
    defaults = _fallback_sections("", lang)
    lines = []
    for key in SECTION_ORDER:
        value = sections.get(key, "").strip()
        if not value:
            value = defaults[key]
        lines.append(f"{labels[key]}: {_clean_value(value)}")
    return "\n".join(lines)


def _parse_sections(text: str) -> Optional[Dict[str, str]]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    sections: Dict[str, str] = {}
    for key in SECTION_ORDER:
        value = data.get(key)
        if value is None:
            continue
        if not isinstance(value, str):
            value = str(value)
        sections[key] = value.strip()
    return sections if sections else None


async def _ask_openai(system_prompt: str, user_prompt: str, lang: str, raw_text: str) -> Dict[str, str]:
    if _client is None:
        return _fallback_sections(raw_text, lang)

    def _call_sync() -> str:
        if _USE_NEW_CLIENT:
            response = _client.chat.completions.create(  # type: ignore[union-attr]
                model="gpt-4o-mini",
                temperature=0.6,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return (response.choices[0].message.content or "").strip()

        response = _client.ChatCompletion.create(  # type: ignore[union-attr]
            model="gpt-4o-mini",
            temperature=0.6,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return (response["choices"][0]["message"]["content"] or "").strip()

    try:
        raw = await asyncio.to_thread(_call_sync)
    except Exception:  # pragma: no cover - network issues fallback
        LOGGER.exception("Prompt-Master LLM call failed")
        return _fallback_sections(raw_text, lang)

    sections = _parse_sections(raw)
    if not sections:
        LOGGER.warning("Prompt-Master response is not valid JSON")
        return _fallback_sections(raw_text, lang)
    return sections


async def call_llm_to_make_kino_prompt(
    user_text: str,
    user_lang: str = "ru",
    *,
    voice_requested: Optional[bool] = None,
    music_requested: Optional[bool] = None,
    camera_hints: Optional[str] = None,
) -> str:
    text = (user_text or "").strip()
    lang = user_lang or _lang(text)
    v_req = voice_requested if voice_requested is not None else _voice_req(text)
    m_req = music_requested if music_requested is not None else _music_req(text)
    cams = camera_hints if camera_hints is not None else _cam_tokens(text)
    user_prompt = _build_user_instruction(text, lang, v_req, m_req, cams)
    sections = await _ask_openai(SYSTEM_PROMPT, user_prompt, lang, text)
    return _format_sections(sections, lang)


async def build_cinema_prompt(user_text: str, user_lang: str = "ru") -> Tuple[str, Dict[str, Any]]:
    """
    Возвращает (kino_prompt_text, meta).
    Никаких отправок в Telegram здесь нет.
    """

    text = (user_text or "").strip()
    lang = user_lang or _lang(text)
    voice_requested = _voice_req(text)
    music_requested = _music_req(text)
    camera_tokens = _cam_tokens(text)

    prompt_text = await call_llm_to_make_kino_prompt(
        text,
        user_lang=lang,
        voice_requested=voice_requested,
        music_requested=music_requested,
        camera_hints=camera_tokens,
    )

    meta: Dict[str, Any] = {
        "lang": lang,
        "voice_requested": voice_requested,
        "music_requested": music_requested,
        "camera_hints": camera_tokens,
    }
    return prompt_text, meta


__all__ = ["build_cinema_prompt", "call_llm_to_make_kino_prompt"]
