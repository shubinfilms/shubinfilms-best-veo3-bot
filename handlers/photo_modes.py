"""Helpers to show photo engine selection menus."""

from __future__ import annotations

import logging

from telegram import Update
from telegram.ext import ContextTypes

from keyboards import photo_engines_kb

log = logging.getLogger("handlers.photo_modes")

_PHOTO_ENGINES_TITLE = "📸 Выберите нейросеть для фотографий:"


async def open_menu(update: Update, ctx: ContextTypes.DEFAULT_TYPE, payload=None) -> None:
    query = update.callback_query
    if query is not None:
        try:
            await query.answer()
        except Exception:
            pass
        message = query.message
    else:
        message = update.effective_message
    chat = update.effective_chat
    markup = photo_engines_kb()
    if message is not None and getattr(message, "text", None):
        try:
            await message.edit_text(_PHOTO_ENGINES_TITLE, reply_markup=markup)
            log.debug("photo_modes.menu.edit", extra={"chat_id": getattr(chat, "id", None)})
            return
        except Exception:
            log.debug("photo_modes.menu.edit_failed", exc_info=True)
    if chat is not None:
        await ctx.bot.send_message(chat.id, _PHOTO_ENGINES_TITLE, reply_markup=markup)
        log.debug("photo_modes.menu.sent", extra={"chat_id": chat.id})


__all__ = ["open_menu"]
