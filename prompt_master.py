# -*- coding: utf-8 -*-
"""
Prompt-Master 2.0 — структурный генератор кинематографичных промптов.
ENV:
- OPENAI_API_KEY
Зависимости: openai (любой из SDK: новый или старый).
"""

import os
import re
import json
from typing import Any, Dict

# ---------- OpenAI client (new/old) ----------
_USE_NEW = False
_client = None
try:
    from openai import OpenAI  # new SDK
    _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
    _USE_NEW = True
except Exception:
    try:
        import openai  # old SDK
        openai.api_key = os.getenv("OPENAI_API_KEY", "")
        _client = openai
        _USE_NEW = False
    except Exception:
        _client = None

_DEFAULT_QUOTE_MODE = "generator"


def _normalize_quote_mode(value: str | None) -> str:
    if value is None:
        return _DEFAULT_QUOTE_MODE
    text = value.strip().lower()
    if not text or text in {"auto", "default"}:
        return _DEFAULT_QUOTE_MODE
    if text in {"generator", "gen", "g", "on", "1", "true", "yes"}:
        return "generator"
    if text in {"bot"}:
        return "bot"
    if text in {"off", "0", "false", "no"}:
        return "off"
    return _DEFAULT_QUOTE_MODE


PM_QUOTE_MODE = _normalize_quote_mode(os.getenv("PM_QUOTE_MODE"))

_PM_HEADER_RE = re.compile(r"^\s*>?\s*Карточка Prompt-Master\b", re.IGNORECASE)

SYSTEM_PROMPT = """You are Prompt-Master 2.0 — a creative cinematic prompt writer.
GOALS:
- Keep the user's core idea intact.
- Transform it into a premium, cinematic, hyper-realistic breakdown.
- Absolutely avoid "TV/news" tone. Be modern, stylish, emotional, yet plausible.
- Shots must be clean, realistic, smooth; avoid artifacts or broken logic.
- No subtitles, no on-screen text, no logos unless explicitly requested.

LANGUAGE & VOICE:
- If the user writes in Russian or asks for Russian voice-over — keep Russian voiceover with natural emotionality (not a radio anchor).
- If language is English — keep English voiceover.
- If voice/voiceover is mentioned, include a concise 🎙 Озвучка/Voice block with language, character, emotion.
- If not mentioned, omit the voice block.

MUSIC:
- If music is requested or implied, propose modern genres fitting the mood (hip-hop, ambient, cinematic score, electronic, atmospheric) — never "TV bed".

TECH:
- Detect technical camera hints in the user text (e.g., 85mm prime, shallow DOF, drone shot, handheld, FPV, timelapse/real-time, anamorphic).
- Normalize them into the 🎥 Camera block. Respect real-time vs slow-mo if explicitly requested.

OUTPUT FORMAT (strict):
- 🎬 Сцена: ...
- 🎭 Действие: ...
- 🌌 Атмосфера: ...
- 🎥 Камера: ...   (lens, movement, framing, speed)
- 💡 Свет: ...
- 🌍 Окружение: ...
- 🔊 Звук/Музыка: ...   (modern styles only, if relevant)
- 🎙 Озвучка: ...       (only if requested or clearly implied)
- 🎨 Стиль: ...
- 📝 Текст/субтитры: ... (usually "нет"/"none")

Keep it under ~2200 characters unless the user explicitly asks for long form.
"""

# ---------- Helpers: lang/intent/camera ----------
_CYRILLIC_RE = re.compile(r"[а-яА-ЯёЁ]")
VOICE_HINT_RE = re.compile(r"\b(voice|озвучк|диктор|голос|озвучить|озвучка|narration)\b", re.IGNORECASE)
MUSIC_HINT_RE = re.compile(r"\b(music|музык|бит|саунд|soundtrack|score|beat)\b", re.IGNORECASE)

CAMERA_HINTS = [
    r"\b(\d{2,3}mm)\b", r"\bprime lens\b", r"\bshallow depth of field\b", r"\bDOF\b",
    r"\banamorphic\b", r"\bhandheld\b", r"\bsteadycam\b", r"\bgimbal\b", r"\bdrone\b", r"\bFPV\b",
    r"\bclose[- ]?up\b", r"\bmacro\b", r"\bwide[- ]?shot\b", r"\bmedium\b",
    r"\bslow[- ]?motion\b", r"\bslow[- ]?mo\b", r"\breal[- ]?time\b", r"\btime[- ]?remap\b",
    r"\btilt[- ]?shift\b", r"\bpan\b", r"\btilt\b", r"\bzoom\b", r"\bdolly\b", r"\bpush[- ]?in\b", r"\bpull[- ]?back\b",
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
        m = re.findall(pat, text, flags=re.IGNORECASE)
        if m:
            if isinstance(m[0], tuple):
                for tup in m:
                    for x in tup:
                        if x:
                            found.append(x)
            else:
                found.extend(m)
    tokens, seen = [], set()
    for t in (x.strip().lower() for x in found if x):
        if t and t not in seen:
            seen.add(t); tokens.append(t)
    return ", ".join(tokens)


def _strip_prompt_master_header(text: str) -> str:
    lines = (text or "").splitlines()
    idx = 0
    total = len(lines)

    while idx < total and not lines[idx].strip():
        idx += 1

    if idx < total and _PM_HEADER_RE.match(lines[idx]):
        idx += 1
        while idx < total and not lines[idx].strip():
            idx += 1

    remainder = "\n".join(lines[idx:])
    return remainder.strip()


def ensure_quote_block(text: str) -> str:
    content = (text or "").strip()
    if not content:
        return ""

    result = []
    for raw_line in content.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            result.append(">")
            continue
        if stripped.startswith(">"):
            inner = stripped.lstrip(">").lstrip()
            result.append(f"> {inner}" if inner else ">")
        else:
            result.append(f"> {stripped}")
    return "\n".join(result)


def _postprocess_output(text: str) -> tuple[str, str]:
    cleaned = _strip_prompt_master_header(text)

    if PM_QUOTE_MODE == "bot":
        formatted = cleaned
    elif PM_QUOTE_MODE == "off":
        formatted = cleaned
    else:  # default behaviour — quote in generator
        formatted = ensure_quote_block(cleaned)

    return formatted, cleaned

def _build_user_instruction(text: str, lang: str, v_req: bool, m_req: bool) -> str:
    lines = ["USER IDEA:", text.strip(), "", "CONSTRAINTS:",
             "- Keep realism, clean motion, plausible physics.",
             "- No TV/news tone. Be modern, stylish, emotional."]
    if m_req:
        lines.append("- Provide modern music suggestion that fits mood (hip-hop/ambient/cinematic/electronic).")
    if v_req:
        lines.append("- Add Russian voiceover only, natural and expressive (no radio anchor)." if lang=="ru"
                     else "- Add English voiceover only, natural and expressive (no radio anchor).")
    lines.append("- If user forbids text/logos/subtitles — keep none.")
    cams = _cam_tokens(text)
    if cams:
        lines.append(f"- Camera technical hints to respect: {cams}")
    return "\n".join(lines)

def _ask_openai(system_prompt: str, user_prompt: str, lang: str) -> str:
    if _client is None:
        # Fallback без API — минимальный шаблон, чтобы UX не ломался
        voice = "🎙 Озвучка: русский, тёплый, живой" if lang=="ru" else "🎙 Voice: English, warm, natural"
        return (
            "🎬 Сцена: " + (user_prompt[:180]) + "...\n"
            "🎭 Действие: Плавные, реалистичные, без артефактов.\n"
            "🌌 Атмосфера: Современная, эмоциональная, кинематографичная.\n"
            "🎥 Камера: 85mm prime, shallow DOF, плавные панорамы.\n"
            "💡 Свет: Мягкий, объёмный, с аккуратными бликами.\n"
            "🌍 Окружение: Детально, но без перегруза, фокус на главном.\n"
            "🔊 Звук/Музыка: Современная подача (ambient/hip-hop/cinematic), без TV-подложки.\n"
            f"{voice}\n"
            "🎨 Стиль: Премиальный, гиперреалистичный, рекламный.\n"
            "📝 Текст/субтитры: нет\n"
        )
    if _USE_NEW:
        resp = _client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.6,
            messages=[{"role":"system","content":system_prompt},
                      {"role":"user","content":user_prompt}],
        )
        return (resp.choices[0].message.content or "").strip()
    else:
        resp = _client.ChatCompletion.create(
            model="gpt-4o-mini",
            temperature=0.6,
            messages=[{"role":"system","content":system_prompt},
                      {"role":"user","content":user_prompt}],
        )
        return (resp["choices"][0]["message"]["content"] or "").strip()

def generate_prompt(user_text: str) -> Dict[str, Any]:
    """
    Вход: текст пользователя.
    Выход:
      - text_markdown: Markdown промпт
      - meta: {lang, voice_requested, music_requested, camera_hints, quote_mode, raw_text_markdown}
    """
    txt = (user_text or "").strip()
    lang = _lang(txt)
    vreq = _voice_req(txt)
    mreq = _music_req(txt)

    user_instr = _build_user_instruction(txt, lang, vreq, mreq)
    out = _ask_openai(SYSTEM_PROMPT, user_instr, lang)
    formatted, cleaned = _postprocess_output(out)
    return {
        "text_markdown": formatted,
        "meta": {
            "lang": lang,
            "voice_requested": vreq,
            "music_requested": mreq,
            "camera_hints": _cam_tokens(txt),
            "quote_mode": PM_QUOTE_MODE,
            "raw_text_markdown": cleaned,
        },
    }


def generate_prompt_master(user_text: str) -> str:
    """Return only the Prompt-Master text block."""

    result = generate_prompt(user_text)
    if isinstance(result, dict):
        return (result.get("text_markdown") or "").strip()
    if isinstance(result, str):
        return result.strip()
    return ""

if __name__ == "__main__":
    demo = "High-quality cinematic 4K. 85mm prime lens for shallow DOF. Озвучка по-русски, современная музыка."
    print(json.dumps(generate_prompt(demo), ensure_ascii=False, indent=2))
