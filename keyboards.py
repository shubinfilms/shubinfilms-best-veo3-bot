from telegram import InlineKeyboardButton, InlineKeyboardMarkup

CB_FAQ_PREFIX = "faq:"
CB_PM_PREFIX = "pm:"


_PM_LABELS = {
    "veo": {"ru": "🎬 Видеопромпт (VEO)", "en": "🎬 Video prompt (VEO)"},
    "mj": {"ru": "🖼️ Изображение (Midjourney)", "en": "🖼️ Image prompt (MJ)"},
    "animate": {"ru": "🫥 Оживление фото", "en": "🫥 Photo animate"},
    "banana": {"ru": "✂️ Редактирование фото (Banana)", "en": "✂️ Photo edit (Banana)"},
    "suno": {"ru": "🎵 Трек (Suno)", "en": "🎵 Track (Suno)"},
    "back": {"ru": "⬅️ Назад", "en": "⬅️ Back"},
}

_ENGINE_DISPLAY = {
    "veo": {"ru": "VEO", "en": "VEO"},
    "mj": {"ru": "Midjourney", "en": "Midjourney"},
    "banana": {"ru": "Banana", "en": "Banana"},
    "animate": {"ru": "VEO Animate", "en": "VEO Animate"},
    "suno": {"ru": "Suno", "en": "Suno"},
}


def _label(key: str, lang: str) -> str:
    data = _PM_LABELS.get(key, {})
    return data.get(lang, data.get("en", ""))


def prompt_master_keyboard(lang: str = "ru") -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton(_label("veo", lang), callback_data=f"{CB_PM_PREFIX}veo")],
        [InlineKeyboardButton(_label("mj", lang), callback_data=f"{CB_PM_PREFIX}mj")],
        [InlineKeyboardButton(_label("animate", lang), callback_data=f"{CB_PM_PREFIX}animate")],
        [InlineKeyboardButton(_label("banana", lang), callback_data=f"{CB_PM_PREFIX}banana")],
        [InlineKeyboardButton(_label("suno", lang), callback_data=f"{CB_PM_PREFIX}suno")],
        [InlineKeyboardButton(_label("back", lang), callback_data=f"{CB_PM_PREFIX}back")],
    ]
    return InlineKeyboardMarkup(rows)


def prompt_master_mode_keyboard(lang: str = "ru") -> InlineKeyboardMarkup:
    back = _label("back", lang)
    switch = "🔁 Сменить движок" if lang == "ru" else "🔁 Switch engine"
    rows = [
        [InlineKeyboardButton(back, callback_data=f"{CB_PM_PREFIX}back")],
        [InlineKeyboardButton(switch, callback_data=f"{CB_PM_PREFIX}switch")],
    ]
    return InlineKeyboardMarkup(rows)


def prompt_master_result_keyboard(engine: str, lang: str = "ru") -> InlineKeyboardMarkup:
    display = _ENGINE_DISPLAY.get(engine, {}).get(lang, engine.upper())
    copy_text = "📋 Скопировать" if lang == "ru" else "📋 Copy"
    insert_text = (
        f"⬇️ Вставить в карточку ({display})"
        if lang == "ru"
        else f"⬇️ Insert into {display} card"
    )
    back = _label("back", lang)
    switch = "🔁 Сменить движок" if lang == "ru" else "🔁 Switch engine"
    rows = [
        [
            InlineKeyboardButton(copy_text, callback_data=f"{CB_PM_PREFIX}copy:{engine}"),
            InlineKeyboardButton(insert_text, callback_data=f"{CB_PM_PREFIX}insert:{engine}"),
        ],
        [
            InlineKeyboardButton(back, callback_data=f"{CB_PM_PREFIX}back"),
            InlineKeyboardButton(switch, callback_data=f"{CB_PM_PREFIX}switch"),
        ],
    ]
    return InlineKeyboardMarkup(rows)


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
