from __future__ import annotations

import logging
from typing import Optional

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from texts import TXT_KB_AI_DIALOG
from ui.card import build_card
from utils.input_state import is_waiting as is_waiting_input

try:  # pragma: no cover - optional dependency during tests
    from redis_utils import rds as redis_client
except Exception:  # pragma: no cover - fallback when redis is unavailable
    redis_client = None

log = logging.getLogger(__name__)

_STATE_KEY_TMPL = "chat:free:{chat_id}"
_MEMORY_STATE: dict[int, str] = {}


def _state_key(chat_id: int) -> str:
    return _STATE_KEY_TMPL.format(chat_id=int(chat_id))


def _memory_get(chat_id: int) -> Optional[str]:
    return _MEMORY_STATE.get(int(chat_id))


def _memory_set(chat_id: int, value: Optional[str]) -> None:
    if value is None:
        _MEMORY_STATE.pop(int(chat_id), None)
        return
    _MEMORY_STATE[int(chat_id)] = str(value)


async def is_enabled(chat_id: int | None) -> bool:
    try:
        chat_key = int(chat_id) if chat_id is not None else None
    except (TypeError, ValueError):
        return False
    if chat_key is None:
        return False

    key = _state_key(chat_key)
    if redis_client is not None:
        try:
            raw = redis_client.get(key)
        except Exception:  # pragma: no cover - diagnostics only
            log.debug("chat.free.state_load_failed", exc_info=True, extra={"chat_id": chat_key})
        else:
            if raw is not None:
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", "ignore")
                text = str(raw).strip().lower()
                _memory_set(chat_key, text)
                return text == "on"
    cached = _memory_get(chat_key)
    return (cached or "off").strip().lower() == "on"


async def set_enabled(chat_id: int | None, enabled: bool) -> None:
    try:
        chat_key = int(chat_id) if chat_id is not None else None
    except (TypeError, ValueError):
        return
    if chat_key is None:
        return

    value = "on" if enabled else "off"
    _memory_set(chat_key, value)
    if redis_client is None:
        return
    try:
        redis_client.set(_state_key(chat_key), value)
    except Exception:  # pragma: no cover - diagnostics only
        log.debug("chat.free.state_store_failed", exc_info=True, extra={"chat_id": chat_key})


def render_status_card(*, enabled: bool) -> dict:
    if enabled:
        subtitle = "Свободный чат активирован."
        body_lines = (
            "Пишите сообщения — бот ответит сразу, без карточек.",
            "Команда /reset очищает историю переписки.",
        )
        toggle_label = "✖️ Выкл диалог"
    else:
        subtitle = "Свободный чат выключен."
        body_lines = (
            "Нажмите «Вкл диалог», чтобы перейти в обычное общение.",
            "Карточки режимов продолжат работать как раньше.",
        )
        toggle_label = "🧠 Вкл диалог"

    rows = [
        [InlineKeyboardButton(toggle_label, callback_data="dialog:toggle")],
        [InlineKeyboardButton("⬅️ В меню", callback_data="home:open")],
    ]
    return build_card(TXT_KB_AI_DIALOG, subtitle, rows, body_lines=body_lines)


def is_waiting_card_input(user_id: Optional[int]) -> bool:
    if user_id is None:
        return False
    return bool(is_waiting_input(int(user_id)))


__all__ = ["is_enabled", "set_enabled", "render_status_card", "is_waiting_card_input"]
