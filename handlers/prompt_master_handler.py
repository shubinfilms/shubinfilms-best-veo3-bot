"""Prompt-Master conversation handler that generates cinematic prompts."""

from __future__ import annotations

import asyncio
import html
import logging
import os
from contextlib import suppress
from typing import Awaitable, Callable, Optional

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Message, Update
from telegram.constants import ChatAction
from telegram.ext import (
    CallbackQueryHandler,
    CommandHandler,
    ConversationHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

LOGGER = logging.getLogger(__name__)

PROMPT_MASTER_OPEN = "pm:open"
PROMPT_MASTER_CANCEL = "pm:cancel"

# Conversation states
PM_WAIT = 1
PM_WAITING = (PM_WAIT,)

REQUEST_MESSAGE = "Пришлите тему/идею. /cancel — выход."
READY_MESSAGE_PREFIX = "🧠 Готово! Вот ваш кинопромпт:"
ERROR_MESSAGE = "⚠️ Не удалось сгенерировать промпт, попробуйте ещё раз."
CANCEL_MESSAGE = "Отмена"

SYS_PROMPT = (
    "Ты профессиональный сценарист. Сгенерируй лаконичный, структурированный кинопромпт на английском, "
    "но диалоги и lip-sync на языке пользователя. Структура: Scene, Camera, Action, Dialogue, Lip-sync, Audio, "
    "Lighting, Wardrobe/props, Framing. Без лишних пояснений, только разделы с короткими абзацами."
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
try:
    import openai  # type: ignore

    if OPENAI_API_KEY:
        openai.api_key = OPENAI_API_KEY
except Exception:  # pragma: no cover - optional dependency
    openai = None  # type: ignore

def configure_prompt_master(*, update_veo_card: Optional[Callable[[int, ContextTypes.DEFAULT_TYPE], Awaitable[None]]]) -> None:
    """Compatibility shim: Prompt-Master no longer updates VEO cards automatically."""

    _ = update_veo_card  # preserved for API compatibility


def _effective_message(update: Update) -> Optional[Message]:
    if update.message:
        return update.message
    if update.callback_query and update.callback_query.message:
        return update.callback_query.message
    return None


async def _send_request_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = _effective_message(update)
    if message is not None:
        await message.reply_text(REQUEST_MESSAGE)
        return
    chat = update.effective_chat
    if chat is not None:
        await context.bot.send_message(chat.id, REQUEST_MESSAGE)


def _remember_prompt(context: ContextTypes.DEFAULT_TYPE, prompt_text: str) -> None:
    context.chat_data.setdefault("prompt_master", {})["last_prompt"] = prompt_text
async def _call_openai(user_lang: str, topic: str) -> str:
    if openai is None or not OPENAI_API_KEY:
        raise RuntimeError("OpenAI client is not configured")

    def _sync_call() -> str:
        response = openai.ChatCompletion.create(  # type: ignore[union-attr]
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": f"Language: {user_lang}\nTopic: {topic}"},
            ],
            temperature=0.7,
            max_tokens=700,
        )
        choice = response.choices[0].message["content"].strip()
        return choice

    return await asyncio.to_thread(_sync_call)


async def _generate_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE, topic: str) -> Optional[str]:
    chat = update.effective_chat
    if chat is not None:
        with suppress(Exception):
            await context.bot.send_chat_action(chat_id=chat.id, action=ChatAction.TYPING)

    user_lang = "en"
    if update.effective_user and update.effective_user.language_code:
        user_lang = update.effective_user.language_code

    try:
        prompt_text = await _call_openai(user_lang, topic)
    except Exception:
        LOGGER.exception("Prompt-Master generation failed")
        return None
    return prompt_text.strip()


async def _pm_start(update: Update, context: ContextTypes.DEFAULT_TYPE, *, via: str) -> int:
    chat_id = update.effective_chat.id if update.effective_chat else None
    user_id = update.effective_user.id if update.effective_user else None
    LOGGER.info("PROMPT_MASTER_START | via=%s chat_id=%s user_id=%s", via, chat_id, user_id)
    context.user_data.pop("pm_text", None)
    await _send_request_message(update, context)
    return PM_WAIT


async def pm_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return await _pm_start(update, context, via="command")


async def pm_start_cb(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if update.callback_query:
        await update.callback_query.answer()
    return await _pm_start(update, context, via="button")


async def prompt_master_generate(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    message = update.message
    if message is None:
        return PM_WAIT

    topic = (message.text or "").strip()
    if not topic:
        LOGGER.info(
            "PROMPT_MASTER_EMPTY_TOPIC | chat_id=%s user_id=%s",
            update.effective_chat.id if update.effective_chat else None,
            update.effective_user.id if update.effective_user else None,
        )
        await message.reply_text(REQUEST_MESSAGE)
        return PM_WAIT

    prompt_text = await _generate_prompt(update, context, topic)
    if not prompt_text:
        LOGGER.warning(
            "PROMPT_MASTER_ERROR | chat_id=%s user_id=%s",
            update.effective_chat.id if update.effective_chat else None,
            update.effective_user.id if update.effective_user else None,
        )
        await message.reply_text(ERROR_MESSAGE)
        return PM_WAIT

    context.user_data["pm_text"] = prompt_text
    _remember_prompt(context, prompt_text)

    await message.reply_html(
        f"{READY_MESSAGE_PREFIX}\n<pre>{html.escape(prompt_text)}</pre>",
        reply_markup=InlineKeyboardMarkup(
            [[InlineKeyboardButton("↩️ Назад", callback_data=PROMPT_MASTER_CANCEL)]]
        ),
    )
    LOGGER.info(
        "PROMPT_MASTER_GENERATED | chat_id=%s user_id=%s length=%d",
        update.effective_chat.id if update.effective_chat else None,
        update.effective_user.id if update.effective_user else None,
        len(prompt_text),
    )
    return PM_WAIT


async def prompt_master_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if update.callback_query:
        await update.callback_query.answer()
    message = _effective_message(update)
    if message is not None:
        await message.reply_text(CANCEL_MESSAGE)
    LOGGER.info(
        "PROMPT_MASTER_CANCELLED | chat_id=%s user_id=%s",
        update.effective_chat.id if update.effective_chat else None,
        update.effective_user.id if update.effective_user else None,
    )
    context.user_data.pop("pm_text", None)
    return ConversationHandler.END


prompt_master_conv = ConversationHandler(
    entry_points=[
        CommandHandler("pm", pm_start),
        CallbackQueryHandler(pm_start_cb, pattern=fr"^{PROMPT_MASTER_OPEN}$"),
    ],
    states={
        PM_WAIT: [
            MessageHandler(filters.TEXT & ~filters.COMMAND, prompt_master_generate),
            CallbackQueryHandler(prompt_master_cancel, pattern=fr"^{PROMPT_MASTER_CANCEL}$"),
            CallbackQueryHandler(pm_start_cb, pattern=fr"^{PROMPT_MASTER_OPEN}$"),
        ]
    },
    fallbacks=[
        CommandHandler("cancel", prompt_master_cancel),
        CallbackQueryHandler(prompt_master_cancel, pattern=fr"^{PROMPT_MASTER_CANCEL}$"),
    ],
    name="prompt_master",
)


__all__ = [
    "PROMPT_MASTER_OPEN",
    "PROMPT_MASTER_CANCEL",
    "PM_WAITING",
    "prompt_master_conv",
    "configure_prompt_master",
]
