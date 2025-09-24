from __future__ import annotations

import logging
from typing import Any, Optional

from telegram.constants import ParseMode


async def upsert_card(
    ctx: Any,
    chat_id: int,
    state_dict: dict[str, Any],
    state_key: str,
    text: str,
    reply_markup: Optional[Any] = None,
    parse_mode: ParseMode = ParseMode.HTML,
    disable_web_page_preview: bool = True,
) -> Optional[int]:
    """Update an existing UI card or send a new one.

    Returns the message id of the card on success, otherwise ``None``.
    """

    mid = state_dict.get(state_key)
    if mid:
        try:
            await ctx.bot.edit_message_text(
                chat_id=chat_id,
                message_id=mid,
                text=text,
                reply_markup=reply_markup,
                parse_mode=parse_mode,
                disable_web_page_preview=disable_web_page_preview,
            )
            return mid
        except Exception as exc:  # pragma: no cover - network issues
            logger = getattr(getattr(ctx, "application", None), "logger", logging.getLogger(__name__))
            logger.warning("edit %s failed: %s", state_key, exc)
            state_dict[state_key] = None

    try:
        msg = await ctx.bot.send_message(
            chat_id=chat_id,
            text=text,
            reply_markup=reply_markup,
            parse_mode=parse_mode,
            disable_web_page_preview=disable_web_page_preview,
        )
        state_dict[state_key] = msg.message_id
        return msg.message_id
    except Exception as exc:  # pragma: no cover - network issues
        logger = getattr(getattr(ctx, "application", None), "logger", logging.getLogger(__name__))
        logger.error("send %s failed: %s", state_key, exc)
        return None
