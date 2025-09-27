"""Prompt-Master MVP handlers."""

from __future__ import annotations

import logging

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from keyboards import CB_PM_PREFIX, prompt_master_keyboard

logger = logging.getLogger(__name__)

PM_HINT = "🧠 *Prompt-Master*\nВыберите, что хотите сделать:"


async def prompt_master_open(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    from_callback: bool = False,
) -> None:
    """Send or edit the Prompt-Master root menu."""

    message = update.effective_message
    if message is not None:
        await message.reply_text(
            PM_HINT,
            reply_markup=prompt_master_keyboard(),
            parse_mode=ParseMode.MARKDOWN,
            disable_web_page_preview=True,
        )
        return

    query = update.callback_query
    if query is None:
        return

    if not from_callback:
        await query.answer()

    await query.edit_message_text(
        PM_HINT,
        reply_markup=prompt_master_keyboard(),
        parse_mode=ParseMode.MARKDOWN,
        disable_web_page_preview=True,
    )


async def prompt_master_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle Prompt-Master keyboard interactions."""

    query = update.callback_query
    if query is None or query.data is None:
        return

    user_id = update.effective_user.id if update.effective_user else None
    logger.info("prompt_master.callback | user_id=%s data=%s", user_id, query.data)

    await query.answer()

    code = query.data.removeprefix(CB_PM_PREFIX)
    if code == "back":
        await prompt_master_open(update, context, from_callback=True)
        return

    feature_name = {
        "video": "🎬 Видеопромпт (VEO)",
        "mj_gen": "🖼️ Промпт генерации фото (MJ)",
        "photo_live": "🫥 Оживление фото (VEO)",
        "banana_edit": "✂️ Редактирование фото (Banana)",
        "suno_lyrics": "🎵 Текст песни (Suno)",
    }.get(code, "Неизвестно")

    await query.edit_message_text(
        f"{feature_name}\n\n⚙️ Функция скоро будет доступна. А пока — опишите задачу текстом, я помогу сформулировать промпт.",
        reply_markup=prompt_master_keyboard(),
        parse_mode=ParseMode.MARKDOWN,
        disable_web_page_preview=True,
    )
