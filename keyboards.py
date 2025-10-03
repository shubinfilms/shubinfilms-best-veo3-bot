from telegram import InlineKeyboardButton, InlineKeyboardMarkup

CB_FAQ_PREFIX = "faq:"
CB_PM_PREFIX = "pm:"

CB_PM_BACK = f"{CB_PM_PREFIX}back"
CB_PM_MENU = f"{CB_PM_PREFIX}menu"
CB_PM_SWITCH = f"{CB_PM_PREFIX}switch"
CB_PM_COPY_PREFIX = f"{CB_PM_PREFIX}copy:"
CB_PM_INSERT_PREFIX = f"{CB_PM_PREFIX}insert:"


class CB:
    VIDEO_MENU = "cb:menu:video"
    VIDEO_MENU_BACK = "cb:menu:video:back"
    VIDEO_PICK_VEO = "cb:video:veo"
    VIDEO_PICK_SORA2 = "cb:video:sora2"
    VIDEO_PICK_SORA2_DISABLED = "cb:video:sora2:disabled"
    VIDEO_MODE_VEO_FAST = "cb:video:veo:mode:fast"
    VIDEO_MODE_VEO_QUALITY = "cb:video:veo:mode:quality"
    VIDEO_MODE_VEO_PHOTO = "cb:video:veo:mode:photo"
    VIDEO_MODE_SORA_TEXT = "cb:video:sora2:mode:ttv"
    VIDEO_MODE_SORA_IMAGE = "cb:video:sora2:mode:itv"


# Backwards compatibility constants (to be removed once call sites migrate).
CB_VIDEO_MENU = CB.VIDEO_MENU
CB_VIDEO_ENGINE_VEO = CB.VIDEO_PICK_VEO
CB_VIDEO_ENGINE_SORA2 = CB.VIDEO_PICK_SORA2
CB_VIDEO_ENGINE_SORA2_DISABLED = CB.VIDEO_PICK_SORA2_DISABLED
CB_VIDEO_MODE_FAST = CB.VIDEO_MODE_VEO_FAST
CB_VIDEO_MODE_QUALITY = CB.VIDEO_MODE_VEO_QUALITY
CB_VIDEO_MODE_PHOTO = CB.VIDEO_MODE_VEO_PHOTO
CB_VIDEO_MODE_SORA_TEXT = CB.VIDEO_MODE_SORA_TEXT
CB_VIDEO_MODE_SORA_IMAGE = CB.VIDEO_MODE_SORA_IMAGE
CB_VIDEO_BACK = CB.VIDEO_MENU_BACK


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
        [InlineKeyboardButton(_label("back", lang), callback_data=CB_PM_BACK)],
    ]
    return InlineKeyboardMarkup(rows)


def prompt_master_mode_keyboard(lang: str = "ru") -> InlineKeyboardMarkup:
    back = _label("back", lang)
    switch = "🔁 Сменить движок" if lang == "ru" else "🔁 Switch engine"
    rows = [
        [InlineKeyboardButton(back, callback_data=CB_PM_BACK)],
        [InlineKeyboardButton(switch, callback_data=CB_PM_SWITCH)],
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
            InlineKeyboardButton(copy_text, callback_data=f"{CB_PM_COPY_PREFIX}{engine}"),
            InlineKeyboardButton(insert_text, callback_data=f"{CB_PM_INSERT_PREFIX}{engine}"),
        ],
        [
            InlineKeyboardButton(back, callback_data=CB_PM_BACK),
            InlineKeyboardButton(switch, callback_data=CB_PM_SWITCH),
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


def suno_modes_keyboard() -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton("🎼 Инструментал", callback_data="suno:mode:instrumental")],
        [InlineKeyboardButton("🎤 Вокал", callback_data="suno:mode:lyrics")],
        [InlineKeyboardButton("🎚️ Ковер", callback_data="suno:mode:cover")],
    ]
    return InlineKeyboardMarkup(rows)


def suno_start_keyboard() -> InlineKeyboardMarkup:
    rows = [[InlineKeyboardButton("▶️ Начать генерацию", callback_data="suno:start")]]
    return InlineKeyboardMarkup(rows)


def suno_start_disabled_keyboard() -> InlineKeyboardMarkup:
    rows = [[InlineKeyboardButton("⏳ Идёт генерация…")]]
    return InlineKeyboardMarkup(rows)


def mj_upscale_root_keyboard(grid_id: str) -> InlineKeyboardMarkup:
    buttons = [
        [
            InlineKeyboardButton(
                "✨ Улучшить качество",
                callback_data=f"mj.upscale.menu:{grid_id}",
            )
        ],
        [
            InlineKeyboardButton(
                "🔁 Сгенерировать ещё",
                callback_data=f"mj.gallery.again:{grid_id}",
            )
        ],
        [InlineKeyboardButton("🏠 Назад в меню", callback_data="mj.gallery.back")],
    ]
    return InlineKeyboardMarkup(buttons)


def mj_upscale_select_keyboard(grid_id: str, *, count: int) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    safe_count = max(int(count), 0)
    for idx in range(1, safe_count + 1):
        if idx == 1:
            title = "Первая фотография"
        elif idx == 2:
            title = "Вторая фотография"
        elif idx == 3:
            title = "Третья фотография"
        elif idx == 4:
            title = "Четвёртая фотография"
        else:
            title = f"{idx}-я фотография"
        rows.append(
            [
                InlineKeyboardButton(
                    title,
                    callback_data=f"mj.upscale:{grid_id}:{idx}",
                )
            ]
        )
    rows.append(
        [InlineKeyboardButton("⬅️ Назад", callback_data=f"mj.upscale.menu:{grid_id}")]
    )
    return InlineKeyboardMarkup(rows)
