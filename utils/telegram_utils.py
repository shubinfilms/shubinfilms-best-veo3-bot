from __future__ import annotations

import re
import unicodedata
from time import time
from typing import Any, Optional, Set

from telegram.error import BadRequest


COMMAND_PREFIX = "/"

_SPACE_RE = re.compile(r"\s+")
_VARIATION_SELECTOR_RE = re.compile(r"[\u200d\ufe00-\ufe0f]")


def _normalize_button_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = _VARIATION_SELECTOR_RE.sub("", normalized)
    normalized = _SPACE_RE.sub(" ", normalized).strip()
    return normalized


def _casefold(text: str) -> str:
    return _normalize_button_text(text).casefold()


LABEL_ALIASES: dict[str, str] = {}
_EXACT_LABELS: Set[str] = set()
_PREFIX_LABELS: Set[str] = set()
_MENU_LABELS: Set[str] = set()


def _register_label(canonical: str, *aliases: str, prefix: bool = False) -> None:
    if not canonical:
        return
    _MENU_LABELS.add(_normalize_button_text(canonical))
    all_variants = (canonical, *aliases)
    for variant in all_variants:
        if not variant:
            continue
        folded = _casefold(variant)
        if not folded:
            continue
        LABEL_ALIASES[folded] = canonical
        if prefix:
            _PREFIX_LABELS.add(folded)
        else:
            _EXACT_LABELS.add(folded)


_register_label("🎬 ГЕНЕРАЦИЯ ВИДЕО", "🎬 Генерация видео", "🎬 Видео", "🎬 Видео (VEO)", "🎬 Video", "🎬 Video prompt (VEO)", prefix=True)
_register_label("🎨 ГЕНЕРАЦИЯ ИЗОБРАЖЕНИЙ", "🎨 Генерация изображений", "🎨 Изображения", "🎨 Изображения (MJ)", "🎨 Midjourney", "🎨 Image prompt (MJ)", prefix=True)
_register_label("🎵 Генерация музыки", "🎵 Музыка", "🎵 Музыка (Suno)", "🎵 Track (Suno)", "🎵 Suno", prefix=True)
_register_label("🧠 Prompt-Master", "🧠", prefix=True)
_register_label("💎 Баланс", "💎", prefix=True)
_register_label("💬 Обычный чат", "💬", prefix=True)
_register_label("🏠 В меню", "⬅️ В меню")
_register_label("ℹ️ FAQ", "ℹ️ Общие вопросы", "⚡ Токены и возвраты")
_register_label("🍌 Banana JSON", "🎨 Midjourney JSON", prefix=True)
_register_label("📋 Скопировать", "📋 Copy")
_register_label("⬇️ Вставить в карточку", "⬇️ Insert into", prefix=True)
_register_label("🔁 Сменить движок", "🔁 Switch engine", prefix=True)
_register_label("Подтвердить")
_register_label("Отменить")
_register_label("Назад")
_register_label("Назад в меню")
_register_label("⬅️ Назад", "⬅️ Назад (в главное)", "⬅️ Назад к разделам", "⬅️ Back", "◀️ Назад")
_register_label("Повторить", "🔁 Повторить")
_register_label("🔁 Изменить ввод")
_register_label("Показать ещё")
_register_label("Сменить формат")
_register_label("Горизонтальный (16:9)")
_register_label("Вертикальный (9:16)")
_register_label("16:9")
_register_label("9:16")
_register_label("⚡ Fast", "⚡ Fast ✅", prefix=True)
_register_label("💎 Quality", "💎 Quality ✅", prefix=True)
_register_label("🖼 Добавить/Удалить референс", prefix=True)
_register_label("🚀 Сгенерировать", "🚀 Сгенерировать ещё видео", prefix=True)
_register_label("🚀 Начать генерацию Banana", prefix=True)
_register_label("➕ Добавить ещё фото")
_register_label("🧹 Очистить фото")
_register_label("✍️ Изменить промпт")
_register_label("📝 Текст песни")
_register_label("✏️ Название")
_register_label("🎨 Стиль")
_register_label("🎼 Режим", prefix=True)
_register_label("⏳ Генерация", prefix=True)
_register_label("💳 Пополнить баланс")
_register_label("🧾 История операций")
_register_label("👥 Пригласить друга")
_register_label("🎁 Активировать промокод")
_register_label("📤 Поделиться")
_register_label("🛒 Где купить Stars")
_register_label("🎬", prefix=True)
_register_label("🎨", prefix=True)
_register_label("🎵", prefix=True)
_register_label("💎", prefix=True)
_register_label("💬", prefix=True)
_register_label("🧠", prefix=True)
_register_label("🧩 Banana")
_register_label("🎵 Suno (текст песни)")

MENU_LABELS = tuple(sorted(_MENU_LABELS))

_LABEL_PATTERN_PARTS: list[str] = []
for value in sorted({_casefold(label) for label in LABEL_ALIASES}, key=len, reverse=True):
    escaped = re.escape(value)
    if value in _PREFIX_LABELS:
        pattern = rf"{escaped}(?:\s|$)"
    else:
        pattern = rf"^(?:{escaped})$"
    _LABEL_PATTERN_PARTS.append(pattern)

_COMMAND_PATTERN = r"/[a-z0-9_]+(?:@[a-z0-9_]+)?"
_LABEL_PATTERN_PARTS.append(_COMMAND_PATTERN)

COMMAND_OR_BUTTON_REGEX = re.compile("|".join(_LABEL_PATTERN_PARTS))

_PLACEHOLDER_PROMPTS = {
    "⏳ sending request…",
    "⚠️ generation failed, please try later.",
    "⚠️ generation failed, please try later. 💎 токены возвращены.",
}


BUTTON_LABELS = set(MENU_LABELS)


def is_button_label(text: Optional[str]) -> bool:
    if not text:
        return False
    normalized = _casefold(text)
    if not normalized:
        return False
    if normalized in _EXACT_LABELS:
        return True
    for prefix in _PREFIX_LABELS:
        if normalized.startswith(prefix):
            return True
    return False


def is_command_text(text: Optional[str]) -> bool:
    if not text:
        return False
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.startswith(COMMAND_PREFIX):
        return True
    normalized = _casefold(stripped)
    if normalized in _EXACT_LABELS:
        return True
    for prefix in _PREFIX_LABELS:
        if normalized.startswith(prefix):
            return True
    if COMMAND_OR_BUTTON_REGEX.search(normalized):
        return True
    return False


def should_capture_to_prompt(text: Optional[str]) -> bool:
    """Разрешаем только обычный пользовательский текст (не команды и не лейблы)."""

    if not text:
        return False
    stripped = text.strip()
    if not stripped:
        return False
    if is_command_text(stripped):
        return False
    if is_button_label(stripped):
        return False
    if stripped.casefold() in _PLACEHOLDER_PROMPTS:
        return False
    return True


# простая защита от частых одинаковых edit
_last_edit_cache: dict[tuple[int, int], dict[str, Any]] = {}


def safe_edit(bot: Any, chat_id: int, message_id: int, text: str, **kwargs: Any):
    key = (int(chat_id), int(message_id))
    entry = _last_edit_cache.get(key)
    now = time()
    if entry and entry.get("text") == text and now - entry.get("ts", 0.0) < 30:
        return entry.get("result")

    try:
        result = bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=text, **kwargs)
    except BadRequest as exc:
        if "message is not modified" in str(exc).lower():
            _last_edit_cache[key] = {"text": text, "ts": now, "result": None}
            return None
        raise
    else:
        _last_edit_cache[key] = {"text": text, "ts": now, "result": result}
        return result


__all__ = [
    "COMMAND_PREFIX",
    "MENU_LABELS",
    "LABEL_ALIASES",
    "COMMAND_OR_BUTTON_REGEX",
    "BUTTON_LABELS",
    "is_command_text",
    "is_button_label",
    "should_capture_to_prompt",
    "safe_edit",
]

