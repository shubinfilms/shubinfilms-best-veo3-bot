# -*- coding: utf-8 -*-
import logging
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (ContextTypes, ConversationHandler,
                          CommandHandler, MessageHandler, filters)

from prompt_master import generate_prompt

PROMPT_MASTER_HEADER = "🧠 Prompt-Master"
PROMPT_MASTER_BODY = (
    "Опиши идею (1–3 предложения). Если нужна озвучка — укажи язык и характер голоса. "
    "Можешь добавить тех.детали (например: “85mm prime, shallow DOF, real-time”)."
)
PROMPT_MASTER_INVITE_TEXT = f"{PROMPT_MASTER_HEADER}\n\n{PROMPT_MASTER_BODY}"

ASK_IDEA = 1
log = logging.getLogger("prompt_master")

async def prompt_master_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(PROMPT_MASTER_INVITE_TEXT)
    return ASK_IDEA

async def prompt_master_generate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id if update.effective_user else None
    text = update.message.text or ""
    result = generate_prompt(text)
    md = result["text_markdown"]
    meta = result["meta"]

    log.info("PromptMaster: uid=%s len=%s lang=%s voice=%s music=%s cam=%s",
             user_id, len(text), meta["lang"], meta["voice_requested"],
             meta["music_requested"], meta["camera_hints"])

    await update.message.reply_text(md, parse_mode=ParseMode.MARKDOWN)
    return ConversationHandler.END

prompt_master_conv = ConversationHandler(
    entry_points=[CommandHandler("promptmaster", prompt_master_start)],
    states={ASK_IDEA: [MessageHandler(filters.TEXT & ~filters.COMMAND, prompt_master_generate)]},
    fallbacks=[],
    name="prompt_master_conv",
    persistent=False,
)
