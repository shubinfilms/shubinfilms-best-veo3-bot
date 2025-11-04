from __future__ import annotations

from typing import Sequence

from telegram import InlineKeyboardButton, Update
from telegram.ext import ContextTypes

from keyboards import CB, FEATURE_PM_ENABLED, kb_main, main_menu_kb, photo_engines_kb
from texts import (
    TXT_AI_DIALOG_CHOOSE,
    TXT_AI_DIALOG_NORMAL,
    TXT_AI_DIALOG_PM,
    TXT_KB_AI_DIALOG,
    TXT_KB_MUSIC,
    TXT_KB_PHOTO,
    TXT_KB_PROFILE,
    TXT_KB_VIDEO,
    TXT_MENU_TITLE,
)
from ui.card import build_card

from logging_utils import get_logger

log = get_logger("handlers.menu")

_MAIN_MENU_SUBTITLE = "Выберите раздел:"


async def open_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = "📋 *Главное меню*\n_Выберите раздел:_"
    message = update.effective_message
    if message is None:
        return
    await message.reply_text(text, reply_markup=main_menu_kb(), parse_mode="Markdown")


async def show_profile(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    target = query.message if query else update.effective_message
    if target is None:
        return
    await target.reply_text("👤 Профиль\nБаланс: …\nТариф: …")


async def show_kb(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    target = query.message if query else update.effective_message
    if target is None:
        return
    await target.reply_text("📚 База знаний (в разработке)")


async def open_photo_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    target = query.message if query else update.effective_message
    if target is None:
        return
    await target.reply_text(
        "📸 Выберите нейросеть для фотографий:",
        reply_markup=photo_engines_kb(),
    )


async def open_music_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    target = query.message if query else update.effective_message
    if target is None:
        return
    await target.reply_text("🎧 Музыка: Suno / загрузка каппеллы / инструкции (скоро).")


async def open_video_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    target = query.message if query else update.effective_message
    if target is None:
        return
    await target.reply_text("📹 Видео: Kling / VEO (скоро выбор).")


async def open_dialog_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    target = query.message if query else update.effective_message
    if target is None:
        return
    context.user_data["dialog_mode"] = "plain"
    await target.reply_text(
        "🧠 Обычный диалог включён. Пишите текст — отвечу без дополнительных меню."
    )


async def on_menu_dialog(update, context) -> None:
    """Switch the user into chat mode immediately."""

    chat = getattr(update, "effective_chat", None)
    if chat is None:
        return
    chat_id = chat.id
    await context.bot.send_message(chat_id, "Напиши запрос — это обычный чат с ИИ.")
    try:
        context.user_data["mode"] = "chat"
    except Exception:
        log.debug("menu.dialog.set_mode.failed", exc_info=True)
    else:
        log.debug("menu.dialog.mode_set", extra={"chat_id": chat_id})


def build_main_menu_card() -> dict:
    """Return the unified card used for the /menu screen."""

    markup = kb_main()
    return build_card(TXT_MENU_TITLE, _MAIN_MENU_SUBTITLE, markup.inline_keyboard)


def build_profile_card(balance: str, warning: str | None = None) -> dict:
    rows = [
        [InlineKeyboardButton("💎 Пополнить баланс", callback_data="btn:profile|view=topup")],
        [InlineKeyboardButton("🧾 История операций", callback_data="btn:profile|view=history")],
        [InlineKeyboardButton("👥 Пригласить друга", callback_data="btn:profile|view=invite")],
        [InlineKeyboardButton("🎁 Активировать промокод", callback_data="btn:profile|view=promo")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="btn:profile|view=back")],
    ]
    body: Sequence[str] | None = (warning,) if warning else None
    return build_card(TXT_KB_PROFILE, f"Ваш баланс: {balance} 💎", rows, body_lines=body)


def build_photo_card() -> dict:
    markup = photo_engines_kb()
    return build_card(TXT_KB_PHOTO, "Выбери движок для фото:", markup.inline_keyboard)


def build_music_card() -> dict:
    rows = [
        [InlineKeyboardButton("🎼 Инструментал", callback_data="music:inst")],
        [InlineKeyboardButton("🎙 Вокал", callback_data="music:vocal")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="back")],
    ]
    return build_card(TXT_KB_MUSIC, "Выберите режим:", rows)


def build_video_card(*, veo_fast_cost: int, veo_photo_cost: int, sora2_cost: int) -> dict:
    rows = [
        [
            InlineKeyboardButton(
                f"Генерация видео (Veo Fast) — 💎 {veo_fast_cost}",
                callback_data="mode:veo_text_fast",
            )
        ],
        [
            InlineKeyboardButton(
                f"Оживить изображение (Veo) — 💎 {veo_photo_cost}",
                callback_data=CB.VIDEO_VEO_ANIMATE,
            )
        ],
        [
            InlineKeyboardButton(
                f"🎬 Sora2 — генерация по тексту — 💎 {sora2_cost}",
                callback_data="video:type:sora2",
            )
        ],
        [InlineKeyboardButton("⬅️ Назад", callback_data="back")],
    ]
    return build_card(TXT_KB_VIDEO, "Выберите действие:", rows)


def build_dialog_card() -> dict:
    subtitle = TXT_AI_DIALOG_CHOOSE if FEATURE_PM_ENABLED else "Напиши запрос — это обычный чат с ИИ."
    if FEATURE_PM_ENABLED:
        rows = [
            [
                InlineKeyboardButton(TXT_AI_DIALOG_NORMAL, callback_data="mode:chat"),
                InlineKeyboardButton(TXT_AI_DIALOG_PM, callback_data="mode:prompt_master"),
            ],
            [InlineKeyboardButton("⬅️ Назад", callback_data="back")],
        ]
    else:
        rows = [[InlineKeyboardButton("⬅️ Назад", callback_data="back")]]
    return build_card(TXT_KB_AI_DIALOG, subtitle, rows)


__all__ = [
    "open_main_menu",
    "show_profile",
    "show_kb",
    "open_photo_mode",
    "open_music_mode",
    "open_video_mode",
    "open_dialog_mode",
    "on_menu_dialog",
    "build_dialog_card",
    "build_main_menu_card",
    "build_music_card",
    "build_photo_card",
    "build_profile_card",
    "build_video_card",
]
