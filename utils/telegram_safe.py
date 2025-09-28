from __future__ import annotations

import logging
from typing import Any, Optional

from telegram.constants import ParseMode
from telegram.error import BadRequest


_logger = logging.getLogger("telegram-safe")


async def safe_edit_message(
    ctx: Any,
    chat_id: int,
    message_id: int,
    new_text: Optional[str] = None,
    reply_markup: Any = None,
    *,
    parse_mode: ParseMode = ParseMode.HTML,
    disable_web_page_preview: bool = True,
) -> bool:
    """Safely edit a Telegram message without crashing on no-op updates.

    Returns ``True`` if the message text or markup was edited. ``False`` means the
    content was already up to date. Other exceptions are propagated to let the
    caller decide how to recover (e.g. resend the card).
    """

    bot = getattr(ctx, "bot", None)
    if bot is None:
        raise RuntimeError("Context has no bot instance")

    if new_text is None:
        text_payload = ""
    else:
        text_payload = str(new_text)

    try:
        await bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text=text_payload,
            reply_markup=reply_markup,
            parse_mode=parse_mode,
            disable_web_page_preview=disable_web_page_preview,
        )
        return True
    except BadRequest as exc:
        lowered = str(exc).lower()
        if "message is not modified" not in lowered:
            raise
        _logger.debug(
            "safe_edit_message.noop",
            extra={"chat_id": chat_id, "message_id": message_id},
        )
        if reply_markup is None:
            return False
        try:
            await bot.edit_message_reply_markup(
                chat_id=chat_id,
                message_id=message_id,
                reply_markup=reply_markup,
            )
        except BadRequest as markup_exc:
            markup_lowered = str(markup_exc).lower()
            if "message is not modified" not in markup_lowered:
                raise
            _logger.debug(
                "safe_edit_message.reply_markup_noop",
                extra={"chat_id": chat_id, "message_id": message_id},
            )
            return False
        return True
