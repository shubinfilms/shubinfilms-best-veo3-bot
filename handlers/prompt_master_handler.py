"""Prompt-Master conversation handler."""

from __future__ import annotations

import logging
from contextlib import suppress

from telegram import Update
from telegram.ext import (
    CallbackQueryHandler,
    CommandHandler,
    ConversationHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

LOGGER = logging.getLogger(__name__)

ASK_PROMPT = 1
START_MESSAGE = "Пришлите текст промпта. /cancel — выход."
ACCEPT_TEMPLATE = "Принято: {text}"
ERROR_MESSAGE = "⚠️ Системная ошибка. Попробуйте ещё раз."
CANCEL_MESSAGE = "Диалог завершён."


def _get_reply_target(update: Update):
    if update.message:
        return update.message
    if update.callback_query and update.callback_query.message:
        return update.callback_query.message
    return None


async def _send_error(update: Update) -> None:
    message = _get_reply_target(update)
    if message is None:
        return
    with suppress(Exception):
        await message.reply_text(ERROR_MESSAGE)


def _log_exception(message: str) -> None:
    LOGGER.exception(message)


async def _start_dialog(update: Update) -> int:
    message = _get_reply_target(update)
    if message is None:
        return ConversationHandler.END
    await message.reply_text(START_MESSAGE)
    return ASK_PROMPT


async def prompt_master_start_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    try:
        return await _start_dialog(update)
    except Exception:
        _log_exception("Failed to start Prompt-Master via command")
        await _send_error(update)
        return ConversationHandler.END


async def prompt_master_start_callback(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    query = update.callback_query
    try:
        if query:
            await query.answer()
        return await _start_dialog(update)
    except Exception:
        _log_exception("Failed to start Prompt-Master via callback")
        await _send_error(update)
        return ConversationHandler.END


async def prompt_master_start_message(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    try:
        return await _start_dialog(update)
    except Exception:
        _log_exception("Failed to start Prompt-Master via reply button")
        await _send_error(update)
        return ConversationHandler.END


async def prompt_master_receive(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    try:
        text = (update.message.text or "").strip()
        message = update.message
        if message:
            await message.reply_text(ACCEPT_TEMPLATE.format(text=text))
        return ASK_PROMPT
    except Exception:
        _log_exception("Failed to process Prompt-Master input")
        await _send_error(update)
        return ASK_PROMPT


async def prompt_master_cancel(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    try:
        message = _get_reply_target(update)
        if message is not None:
            await message.reply_text(CANCEL_MESSAGE)
    except Exception:
        _log_exception("Failed to cancel Prompt-Master dialog")
        await _send_error(update)
    return ConversationHandler.END


prompt_master_conv = ConversationHandler(
    entry_points=[
        CommandHandler("promptmaster", prompt_master_start_command),
        CallbackQueryHandler(
            prompt_master_start_callback,
            pattern=r"^(pm:start|prompt_master)$",
        ),
        MessageHandler(filters.Regex(r"^🧠 Prompt-Master$"), prompt_master_start_message),
    ],
    states={
        ASK_PROMPT: [
            MessageHandler(filters.TEXT & ~filters.COMMAND, prompt_master_receive),
        ]
    },
    fallbacks=[CommandHandler("cancel", prompt_master_cancel)],
    name="prompt_master",
)
