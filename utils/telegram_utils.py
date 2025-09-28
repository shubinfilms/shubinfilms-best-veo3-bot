from __future__ import annotations

from time import time
from typing import Any, Optional

from telegram.error import BadRequest


COMMAND_PREFIX = "/"

# Короткий словарь лейблов кнопок (то, что не должно попадать в промпт)
BUTTON_LABELS = {
    "🎬 ГЕНЕРАЦИЯ ВИДЕО",
    "🎨 ГЕНЕРАЦИЯ ИЗОБРАЖЕНИЙ",
    "🎵 Генерация музыки",
    "🧠 Prompt-Master",
    "💎 Баланс",
    "💬 Обычный чат",
    "Подтвердить",
    "Назад",
    "Отменить",
    "Сменить формат",
    "Добавить/Удалить референс",
    "Fast ✅",
    "Quality 💎",
    "16:9 ✅",
    "9:16",
    "Изображение (Midjourney)",
    "Редактирование фото (Banana)",
    "Оживление фото",
    "Трек (Suno)",
    "Видеопромпт (VEO)",
    "Сменить движок",
    "Горизонтальный (16:9)",
    "Вертикальный (9:16)",
}


def is_command_text(text: Optional[str]) -> bool:
    return bool(text) and text.strip().startswith(COMMAND_PREFIX)


def is_button_label(text: Optional[str]) -> bool:
    if not text:
        return False
    t = text.strip()
    return bool(t) and t in BUTTON_LABELS


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
    "BUTTON_LABELS",
    "is_command_text",
    "is_button_label",
    "should_capture_to_prompt",
    "safe_edit",
]

