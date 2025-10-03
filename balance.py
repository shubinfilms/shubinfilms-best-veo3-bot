"""Helpers for balance checks and top-up prompts."""

from __future__ import annotations

import logging
from typing import Optional

from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from redis_utils import get_balance
from texts import common_text

log = logging.getLogger(__name__)


def insufficient_balance_keyboard() -> InlineKeyboardMarkup:
    """Return inline keyboard with shortcuts to the top-up menu."""

    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton(common_text("topup.inline.open"), callback_data="topup:open")],
            [InlineKeyboardButton(common_text("topup.inline.back"), callback_data="back")],
        ]
    )


async def ensure_tokens(
    ctx: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    user_id: int,
    need: int,
    *,
    reply_to: Optional[int] = None,
) -> bool:
    """Ensure the user has at least ``need`` tokens; prompt to top up otherwise."""

    try:
        have = int(get_balance(user_id))
    except Exception as exc:  # pragma: no cover - network/redis failure
        log.exception("ensure_tokens.get_balance_failed", extra={"user_id": user_id, "need": need})
        have = 0

    if have >= need:
        return True

    text = common_text("balance.insufficient", need=need, have=have)
    keyboard = insufficient_balance_keyboard()

    try:
        await ctx.bot.send_message(
            chat_id=chat_id,
            text=text,
            reply_markup=keyboard,
            reply_to_message_id=reply_to,
            parse_mode=ParseMode.MARKDOWN,
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        log.warning(
            "ensure_tokens.notify_failed",
            extra={"user_id": user_id, "chat_id": chat_id, "need": need, "err": str(exc)},
        )
    return False


__all__ = ["ensure_tokens", "insufficient_balance_keyboard"]
