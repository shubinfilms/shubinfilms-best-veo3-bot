from __future__ import annotations

import json
from typing import Any, Optional

from telegram.error import BadRequest


MENU_LABELS = {
    "/menu",
    "🎬 Генерация видео",
    "🎨 Генерация изображений",
    "🎵 Генерация музыки",
    "🧠 Prompt-Master",
    "💬 Обычный чат",
    "💎 Баланс",
    "Генерация изображений (MJ) — 💎 10",
    "Редактор изображений (Banana) — 💎 5",
    "Генерация видео (Veo Fast) — 💎 50",
    "Генерация видео (Veo Quality) — 💎 150",
    "Оживить изображение (Veo) — 💎 50",
}

# Дополнительные лейблы, которые не должны попадать в промпт
_LEGACY_BUTTON_LABELS = {
    "🎬 ГЕНЕРАЦИЯ ВИДЕО",
    "🎨 ГЕНЕРАЦИЯ ИЗОБРАЖЕНИЙ",
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


def _normalize_text(value: Optional[str]) -> str:
    if not value:
        return ""
    return value.strip()


def is_command_text(text: Optional[str]) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    if normalized.startswith("/"):
        return True
    return normalized in MENU_LABELS or normalized in _LEGACY_BUTTON_LABELS


def should_capture_to_prompt(text: Optional[str]) -> bool:
    """Return ``True`` only for free-form user text that should go into prompts."""

    normalized = _normalize_text(text)
    if not normalized:
        return False
    return not is_command_text(normalized)


def is_ack_needed_for_text(text: Optional[str]) -> bool:
    """ACK показываем только на пользовательский текст."""

    normalized = _normalize_text(text)
    if not normalized:
        return False
    return not is_command_text(normalized)


_last_payload: dict[tuple[int, int], tuple[str, str]] = {}


def _serialize_markup(markup: Any) -> str:
    if markup is None:
        return ""
    if hasattr(markup, "to_dict"):
        try:
            payload = markup.to_dict()
        except Exception:
            payload = markup
    else:
        payload = markup
    try:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(payload)


async def safe_edit(
    bot,
    chat_id: int,
    msg_id: int,
    text: str,
    reply_markup,
    *,
    disable_web_page_preview: bool = True,
    parse_mode: Any = "HTML",
) -> bool:
    key = (int(chat_id), int(msg_id))
    payload = (text, _serialize_markup(reply_markup))
    if _last_payload.get(key) == payload:
        return False
    try:
        await bot.edit_message_text(
            chat_id=chat_id,
            message_id=msg_id,
            text=text,
            reply_markup=reply_markup,
            parse_mode=parse_mode,
            disable_web_page_preview=disable_web_page_preview,
        )
    except BadRequest as exc:
        if "message is not modified" in str(exc).lower():
            _last_payload[key] = payload
            return False
        raise
    else:
        _last_payload[key] = payload
        return True


async def show_card(
    bot,
    chat_id: int,
    text: str,
    reply_markup,
    prev_msg_id: Optional[int],
    *,
    force_new: bool = False,
    disable_web_page_preview: bool = True,
    parse_mode: Any = "HTML",
):
    if prev_msg_id and not force_new:
        try:
            changed = await safe_edit(
                bot,
                chat_id,
                prev_msg_id,
                text,
                reply_markup,
                disable_web_page_preview=disable_web_page_preview,
                parse_mode=parse_mode,
            )
        except BadRequest:
            changed = False
        if changed:
            return prev_msg_id
    msg = await bot.send_message(
        chat_id,
        text,
        reply_markup=reply_markup,
        parse_mode=parse_mode,
        disable_web_page_preview=disable_web_page_preview,
    )
    new_id = msg.message_id
    _last_payload[(int(chat_id), int(new_id))] = (text, _serialize_markup(reply_markup))
    return new_id


__all__ = [
    "MENU_LABELS",
    "is_command_text",
    "should_capture_to_prompt",
    "is_ack_needed_for_text",
    "safe_edit",
    "show_card",
]

