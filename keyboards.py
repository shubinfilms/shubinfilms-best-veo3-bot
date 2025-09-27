from telegram import InlineKeyboardButton, InlineKeyboardMarkup

CB_FAQ_PREFIX = "faq:"
CB_PM_PREFIX = "pm:"


def faq_keyboard() -> InlineKeyboardMarkup:
    rows = [
        [
            InlineKeyboardButton("🎬 Видео (VEO)", callback_data=f"{CB_FAQ_PREFIX}veo"),
            InlineKeyboardButton("🎨 Изображения (MJ)", callback_data=f"{CB_FAQ_PREFIX}mj"),
        ],
        [
            InlineKeyboardButton("🧩 Banana", callback_data=f"{CB_FAQ_PREFIX}banana"),
            InlineKeyboardButton("🎵 Музыка (Suno)", callback_data=f"{CB_FAQ_PREFIX}suno"),
        ],
        [
            InlineKeyboardButton("💎 Баланс и оплата", callback_data=f"{CB_FAQ_PREFIX}billing"),
            InlineKeyboardButton("⚡ Токены и возвраты", callback_data=f"{CB_FAQ_PREFIX}tokens"),
        ],
        [
            InlineKeyboardButton("💬 Обычный чат", callback_data=f"{CB_FAQ_PREFIX}chat"),
            InlineKeyboardButton("🧠 Prompt-Master", callback_data=f"{CB_FAQ_PREFIX}pm"),
        ],
        [
            InlineKeyboardButton("ℹ️ Общие вопросы", callback_data=f"{CB_FAQ_PREFIX}common"),
            InlineKeyboardButton("⬅️ Назад (в главное)", callback_data=f"{CB_FAQ_PREFIX}back"),
        ],
    ]
    return InlineKeyboardMarkup(rows)


def prompt_master_keyboard() -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton("🎬 Видеопромпт (VEO)", callback_data=f"{CB_PM_PREFIX}video")],
        [InlineKeyboardButton("🖼️ Промпт генерации фото (MJ)", callback_data=f"{CB_PM_PREFIX}mj_gen")],
        [InlineKeyboardButton("🫥 Оживление фото (VEO)", callback_data=f"{CB_PM_PREFIX}photo_live")],
        [InlineKeyboardButton("✂️ Редактирование фото (Banana)", callback_data=f"{CB_PM_PREFIX}banana_edit")],
        [InlineKeyboardButton("🎵 Текст песни (Suno)", callback_data=f"{CB_PM_PREFIX}suno_lyrics")],
        [InlineKeyboardButton("↩️ Назад", callback_data=f"{CB_PM_PREFIX}back")],
    ]
    return InlineKeyboardMarkup(rows)
