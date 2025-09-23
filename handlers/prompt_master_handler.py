# handlers/prompt_master_handler.py
from __future__ import annotations
from telegram import Update
from telegram.ext import (
    ConversationHandler,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

ASK_PROMPT = 1

async def pm_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.effective_chat.send_message("Пришлите текст промпта. /cancel — выход.")
    return ASK_PROMPT

async def pm_recv(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = (update.message.text or "").strip()
    if not text:
        await update.effective_chat.send_message("Пусто. Пришлите текст или /cancel.")
        return ASK_PROMPT
    # Здесь в дальнейшем можно вставить PromptMaster/OpenAI-логику
    await update.effective_chat.send_message(f"Принято:\n\n{text}")
    return ConversationHandler.END

async def pm_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.effective_chat.send_message("Отменено.")
    return ConversationHandler.END

prompt_master_conv = ConversationHandler(
    entry_points=[CommandHandler("promptmaster", pm_start)],
    states={ASK_PROMPT: [MessageHandler(filters.TEXT & ~filters.COMMAND, pm_recv)]},
    fallbacks=[CommandHandler("cancel", pm_cancel)],
    name="prompt_master_conv",
)
