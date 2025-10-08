from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import TelegramError

logger = logging.getLogger("user-errors")

_TEMPLATES: dict[str, str] = {
    "content_policy": (
        "Не получилось выполнить запрос. Похоже, в тексте есть запрещённый или "
        "негативный контент. Попробуйте переформулировать короче и нейтральнее."
    ),
    "timeout": (
        "Сервис сейчас отвечает дольше обычного. Нажмите «Повторить» или попробуйте позже."
    ),
    "backend_fail": "Не удалось получить результат. Проверьте текст запроса и попробуйте снова.",
    "invalid_input": "Нужна фотография или корректный промт. Пришлите изображение и повторите.",
}


async def send_user_error(
    ctx: Any,
    kind: str,
    *,
    details: Optional[Mapping[str, Any]] = None,
    retry_cb: Optional[str] = None,
) -> None:
    """Send a friendly error message to the user and log the event."""

    template = _TEMPLATES.get(kind, _TEMPLATES["backend_fail"])
    chat_id = None
    user_id = None
    mode = None
    reason = None
    req_id = None

    if details:
        chat_id = details.get("chat_id")
        user_id = details.get("user_id")
        mode = details.get("mode")
        reason = details.get("reason")
        req_id = details.get("req_id")

    logger.info(
        "ERR_USER_SENT",
        extra={
            "mode": mode,
            "kind": kind,
            "reason": reason,
            "user_id": user_id,
            "chat_id": chat_id,
            "req_id": req_id,
        },
    )

    bot = getattr(ctx, "bot", None)
    if bot is None or chat_id is None:
        return

    reply_markup = None
    if retry_cb:
        reply_markup = InlineKeyboardMarkup(
            [[InlineKeyboardButton("🔁 Повторить", callback_data=retry_cb)]]
        )

    try:
        await bot.send_message(chat_id=chat_id, text=template, reply_markup=reply_markup)
    except TelegramError:
        logger.exception(
            "user_error.send_fail", extra={"chat_id": chat_id, "kind": kind, "mode": mode}
        )
    except Exception:  # pragma: no cover - defensive guard
        logger.exception(
            "user_error.send_fail", extra={"chat_id": chat_id, "kind": kind, "mode": mode}
        )
