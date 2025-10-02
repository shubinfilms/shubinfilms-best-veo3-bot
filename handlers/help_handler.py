"""Handlers for the /help and /support commands."""
from __future__ import annotations

import logging
from typing import Optional

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from settings import SUPPORT_USER_ID, SUPPORT_USERNAME
from texts import help_text
from utils.logging_extra import build_log_extra
from utils.telegram_safe import safe_send_text

logger = logging.getLogger("bot.commands.help")


def _support_url(username: str) -> str:
    clean = username.lstrip("@") or "BestAi_Support"
    return f"https://t.me/{clean}"


def _resolve_language(update: Update) -> Optional[str]:
    user = getattr(update, "effective_user", None)
    return getattr(user, "language_code", None)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send localized support information to the user."""

    logger.info("update.received", extra=build_log_extra(update))
    logger.info("command.dispatch", extra=build_log_extra(update, ctx_command="/help"))

    chat = update.effective_chat
    if chat is None:
        logger.warning(
            "send.fail",
            extra=build_log_extra(
                update,
                ctx_command="/help",
                message_type="text",
                reason="no_chat",
            ),
        )
        return

    language_code = _resolve_language(update)
    text, button_label = help_text(language_code, SUPPORT_USERNAME)
    markup = InlineKeyboardMarkup(
        [[InlineKeyboardButton(button_label, url=_support_url(SUPPORT_USERNAME))]]
    )

    result = await safe_send_text(
        context,
        chat_id=chat.id,
        text=text,
        reply_markup=markup,
        parse_mode=None,
        disable_web_page_preview=True,
    )

    base_kwargs = dict(
        ctx_command="/help",
        message_type="text",
        text_length=len(text),
        support_user_id=SUPPORT_USER_ID,
        support_username=SUPPORT_USERNAME,
    )

    if result.ok:
        logger.info(
            "send.ok",
            extra=build_log_extra(
                update,
                message_id=result.message_id,
                **base_kwargs,
            ),
        )
        return

    error = result.error
    logger.warning(
        "send.fail",
        extra=build_log_extra(
            update,
            error=str(error) if error else "unknown",
            error_type=error.__class__.__name__ if error else None,
            **base_kwargs,
        ),
    )


support_command = help_command


__all__ = ["help_command", "support_command"]
