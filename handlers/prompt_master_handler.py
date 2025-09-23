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

PROMPT_MASTER_OPEN = "PROMPT_MASTER_OPEN"
PROMPT_MASTER_CANCEL = "PROMPT_MASTER_CANCEL"

# Conversation states
PM_WAITING = range(1)
_PM_STATE = PM_WAITING[0]

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

_veo_card_updater: Optional[
    Callable[[int, ContextTypes.DEFAULT_TYPE], Awaitable[None]]
] = None


def configure_prompt_master(*, update_veo_card: Optional[Callable[[int, ContextTypes.DEFAULT_TYPE], Awaitable[None]]]) -> None:
    """Register dependencies required by Prompt-Master handlers."""

    global _veo_card_updater
    _veo_card_updater = update_veo_card


def _result_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("📎 Вставить в VEO ещё раз", callback_data=PROMPT_MASTER_OPEN)],
            [InlineKeyboardButton("↩️ Назад", callback_data=PROMPT_MASTER_CANCEL)],
        ]
    )


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


def _store_prompt(context: ContextTypes.DEFAULT_TYPE, prompt_text: str) -> None:
    context.chat_data.setdefault("veo_card", {})["prompt"] = prompt_text
    context.chat_data.setdefault("prompt_master", {})["last_prompt"] = prompt_text
    context.user_data["last_prompt"] = prompt_text


def _stored_prompt(context: ContextTypes.DEFAULT_TYPE) -> str:
    veo_card = context.chat_data.get("veo_card") or {}
    prompt = veo_card.get("prompt")
    if isinstance(prompt, str):
        return prompt
    last_prompt = context.chat_data.get("prompt_master", {}).get("last_prompt")
    return last_prompt if isinstance(last_prompt, str) else ""


async def _update_veo_card_if_visible(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    if chat is None or not callable(_veo_card_updater):
        return
    last_ui_msg_id = context.user_data.get("last_ui_msg_id")
    if not last_ui_msg_id:
        return
    try:
        await _veo_card_updater(chat.id, context)  # type: ignore[arg-type]
    except Exception:  # pragma: no cover - UI update should not crash flow
        LOGGER.exception("Failed to update VEO card from Prompt-Master")


def _is_result_message(message: Optional[Message]) -> bool:
    if not message or not message.text:
        return False
    return message.text.startswith(READY_MESSAGE_PREFIX)


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


async def prompt_master_open(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    reapply = False
    answer_text: Optional[str] = None
    if query:
        if _is_result_message(query.message):
            reapply = True

    if reapply:
        prompt = _stored_prompt(context)
        if prompt:
            _store_prompt(context, prompt)
            await _update_veo_card_if_visible(update, context)
            answer_text = "Промпт вставлен в карточку VEO."

    if query:
        await query.answer(answer_text)

    await _send_request_message(update, context)
    return _PM_STATE


async def prompt_master_reapply(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    if query:
        prompt = _stored_prompt(context)
        if prompt:
            _store_prompt(context, prompt)
            await _update_veo_card_if_visible(update, context)
            await query.answer("Промпт вставлен в карточку VEO.")
        else:
            await query.answer()
            await _send_request_message(update, context)
    return _PM_STATE


async def prompt_master_generate(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    message = update.message
    if message is None:
        return _PM_STATE

    topic = (message.text or "").strip()
    if not topic:
        await message.reply_text(REQUEST_MESSAGE)
        return _PM_STATE

    prompt_text = await _generate_prompt(update, context, topic)
    if not prompt_text:
        await message.reply_text(ERROR_MESSAGE)
        return _PM_STATE

    _store_prompt(context, prompt_text)
    await _update_veo_card_if_visible(update, context)

    await message.reply_html(
        f"{READY_MESSAGE_PREFIX}\n<pre>{html.escape(prompt_text)}</pre>",
        reply_markup=_result_keyboard(),
    )
    return _PM_STATE


async def prompt_master_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if update.callback_query:
        await update.callback_query.answer()
    message = _effective_message(update)
    if message is not None:
        await message.reply_text(CANCEL_MESSAGE)
    return ConversationHandler.END


prompt_master_conv = ConversationHandler(
    entry_points=[
        CallbackQueryHandler(prompt_master_open, pattern=fr"^{PROMPT_MASTER_OPEN}$"),
    ],
    states={
        _PM_STATE: [
            MessageHandler(filters.TEXT & ~filters.COMMAND, prompt_master_generate),
            CallbackQueryHandler(prompt_master_reapply, pattern=fr"^{PROMPT_MASTER_OPEN}$"),
            CallbackQueryHandler(prompt_master_cancel, pattern=fr"^{PROMPT_MASTER_CANCEL}$"),
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
