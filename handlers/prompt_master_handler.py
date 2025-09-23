# handlers/prompt_master_handler.py
from __future__ import annotations
from typing import Final
from telegram import Update
from telegram.ext import (
    ConversationHandler,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# Состояния диалога
ASK_PROMPT: Final[int] = 1

# Вспомогательная подсказка (если бот где-то её показывает)
PROMPT_MASTER_HINT: Final[str] = (
    "Пришлите текст запроса (prompt). Напишите /cancel чтобы выйти."
)

async def pm_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.effective_chat.send_message(PROMPT_MASTER_HINT)
    return ASK_PROMPT

async def pm_receive(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = (update.message.text or "").strip()
    if not text:
        await update.effective_chat.send_message("Пусто. Пришлите текст или /cancel.")
        return ASK_PROMPT

    # Здесь можно вставить логику PromptMaster / OpenAI / и т.п.
    await update.effective_chat.send_message(f"Ваш промпт принят:\n\n{text}")

    # Завершаем диалог (или верните ASK_PROMPT, если хотите продолжать)
    return ConversationHandler.END

async def pm_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.effective_chat.send_message("Отменено.")
    return ConversationHandler.END

# Готовый ConversationHandler, который ожидает команду /promptmaster
prompt_master_conv = ConversationHandler(
    entry_points=[CommandHandler("promptmaster", pm_start)],
    states={
        ASK_PROMPT: [MessageHandler(filters.TEXT & ~filters.COMMAND, pm_receive)],
    },
    fallbacks=[CommandHandler("cancel", pm_cancel)],
    name="prompt_master_conv",
)
