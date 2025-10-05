from typing import Optional

from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
)


EMOJI = {
    "video": "🎬",
    "image": "🎨",
    "music": "🎵",
    "chat": "💬",
    "prompt": "🧠",
    "profile": "👥",
    "back": "⬅️",
    "pay": "💎",
}

AI_MENU_CB = "ai:menu"
AI_TO_SIMPLE_CB = "dialog_default"
AI_TO_PROMPTMASTER_CB = "prompt_master"

VIDEO_MENU_CB = "video:menu"
IMAGE_MENU_CB = "image:menu"
MUSIC_MENU_CB = "music:menu"
PROFILE_MENU_CB = "profile:menu"
KNOWLEDGE_MENU_CB = "kb:menu"

# Backward compatible aliases (deprecated)
CB_PROFILE = PROFILE_MENU_CB
CB_KB = KNOWLEDGE_MENU_CB
CB_PHOTO = IMAGE_MENU_CB
CB_MUSIC = MUSIC_MENU_CB
CB_VIDEO = VIDEO_MENU_CB
CB_CHAT = AI_MENU_CB


def main_menu_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton(text="👥 Профиль", callback_data=PROFILE_MENU_CB)],
            [InlineKeyboardButton(text="📚 База знаний", callback_data=KNOWLEDGE_MENU_CB)],
            [
                InlineKeyboardButton(text="📸 Режим фото", callback_data=IMAGE_MENU_CB),
                InlineKeyboardButton(text="🎧 Режим музыки", callback_data=MUSIC_MENU_CB),
            ],
            [
                InlineKeyboardButton(text="📹 Режим видео", callback_data=VIDEO_MENU_CB),
                InlineKeyboardButton(text="🧠 Диалог с ИИ", callback_data=AI_MENU_CB),
            ],
        ]
    )


def kb_home_menu() -> InlineKeyboardMarkup:
    return main_menu_kb()


def reply_kb_home() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="👥 Профиль")],
            [KeyboardButton(text="📚 База знаний")],
            [
                KeyboardButton(text="📸 Режим фото"),
                KeyboardButton(text="🎧 Режим музыки"),
            ],
            [
                KeyboardButton(text="📹 Режим видео"),
                KeyboardButton(text="🧠 Диалог с ИИ"),
            ],
        ],
        resize_keyboard=True,
        is_persistent=True,
    )


def _row(*buttons: InlineKeyboardButton) -> list[list[InlineKeyboardButton]]:
    return [list(buttons)]


def kb_btn(text: str, callback: str) -> InlineKeyboardButton:
    """Единая фабрика кнопок для инлайн-клавиатуры."""

    return InlineKeyboardButton(text=text, callback_data=callback)


def build_menu(rows: list[list[tuple[str, str]]]) -> InlineKeyboardMarkup:
    """Построить клавиатуру из строк ``(text, callback)``."""

    markup_rows: list[list[InlineKeyboardButton]] = []
    for row in rows:
        markup_rows.append([kb_btn(text, cb) for text, cb in row])
    return InlineKeyboardMarkup(markup_rows)

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


def menu_main_like() -> InlineKeyboardMarkup:
    """Инлайн-меню, повторяющее расположение главного экрана."""

    return build_menu(
        [
            [
                (f"{EMOJI['profile']} Профиль", "profile"),
                ("📚 База знаний", "kb_docs"),
            ],
            [
                ("📸 Режим фото", "mode_photo"),
                ("🎧 Режим музыки", "mode_music"),
            ],
            [
                (f"{EMOJI['video']} Режим видео", "mode_video"),
                (f"{EMOJI['prompt']} Диалог с ИИ", "mode_chat"),
            ],
        ]
    )


def menu_bottom_unified() -> InlineKeyboardMarkup:
    """Единое меню для перехода между режимами внутри карточек."""

    return build_menu(
        [
            [(f"{EMOJI['video']} Генерация видео", "nav_video")],
            [(f"{EMOJI['image']} Генерация изображений", "nav_image")],
            [(f"{EMOJI['music']} Генерация музыки", "nav_music")],
            [(f"{EMOJI['prompt']} Prompt-Master", "nav_prompt")],
            [(f"{EMOJI['chat']} Обычный чат", "nav_chat")],
            [(f"{EMOJI['profile']} Профиль", "profile")],
        ]
    )


def menu_pay_unified() -> InlineKeyboardMarkup:
    """Инлайн-меню для способов оплаты."""

    return build_menu(
        [
            [("⭐️ Телеграм Stars", "pay_stars")],
            [("💳 Оплата картой", "pay_card")],
            [("🔐 Crypto", "pay_crypto")],
            [(f"{EMOJI['back']} Назад", "back_main")],
        ]
    )


def suno_start_keyboard() -> InlineKeyboardMarkup:
    rows = [[InlineKeyboardButton("▶️ Начать генерацию", callback_data="music:suno:start")]]
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
CB_MAIN_PROFILE = PROFILE_MENU_CB
CB_MAIN_KNOWLEDGE = KNOWLEDGE_MENU_CB
CB_MAIN_PHOTO = IMAGE_MENU_CB
CB_MAIN_MUSIC = MUSIC_MENU_CB
CB_MAIN_VIDEO = VIDEO_MENU_CB
CB_MAIN_AI_DIALOG = AI_MENU_CB
CB_MAIN_BACK = "main_back"
CB_PROFILE_TOPUP = "profile_topup"
CB_PROFILE_BACK = "profile_back"
CB_AI_MODES = AI_MENU_CB
CB_CHAT_NORMAL = AI_TO_SIMPLE_CB
CB_CHAT_PROMPTMASTER = AI_TO_PROMPTMASTER_CB
CB_PAY_STARS = "pay_stars"
CB_PAY_CARD = "pay_card"
CB_PAY_CRYPTO = "pay_crypto"


def kb_main_menu_profile_first() -> InlineKeyboardMarkup:
    from texts import (
        TXT_KB_AI_DIALOG,
        TXT_KB_KNOWLEDGE,
        TXT_KB_MUSIC,
        TXT_KB_PHOTO,
        TXT_KB_PROFILE,
        TXT_KB_VIDEO,
    )

    rows = [
        [InlineKeyboardButton(TXT_KB_PROFILE, callback_data=CB_MAIN_PROFILE)],
        [InlineKeyboardButton(TXT_KB_KNOWLEDGE, callback_data=CB_MAIN_KNOWLEDGE)],
        [
            InlineKeyboardButton(TXT_KB_PHOTO, callback_data=CB_MAIN_PHOTO),
            InlineKeyboardButton(TXT_KB_MUSIC, callback_data=CB_MAIN_MUSIC),
        ],
        [
            InlineKeyboardButton(TXT_KB_VIDEO, callback_data=CB_MAIN_VIDEO),
            InlineKeyboardButton(TXT_KB_AI_DIALOG, callback_data=CB_MAIN_AI_DIALOG),
        ],
    ]
    return InlineKeyboardMarkup(rows)


def kb_profile_topup_entry() -> InlineKeyboardMarkup:
    from texts import TXT_TOPUP_ENTRY

    return InlineKeyboardMarkup(
        [[InlineKeyboardButton(TXT_TOPUP_ENTRY, callback_data=CB_PROFILE_TOPUP)]]
    )


def kb_topup_methods(*, crypto_url: Optional[str] = None) -> InlineKeyboardMarkup:
    from texts import TXT_PAY_CRYPTO_OPEN_LINK, common_text

    markup = menu_pay_unified()
    rows = list(markup.inline_keyboard[:-1])
    if crypto_url:
        rows.insert(
            3,
            [InlineKeyboardButton(TXT_PAY_CRYPTO_OPEN_LINK, url=crypto_url)],
        )
    rows.append([InlineKeyboardButton(common_text("topup.menu.back"), callback_data=CB_PROFILE_BACK)])
    return InlineKeyboardMarkup(rows)


def kb_ai_dialog_modes() -> InlineKeyboardMarkup:
    from texts import TXT_AI_DIALOG_NORMAL, TXT_AI_DIALOG_PM
    from texts import common_text

    rows = [
        [
            InlineKeyboardButton(TXT_AI_DIALOG_NORMAL, callback_data=AI_TO_SIMPLE_CB),
            InlineKeyboardButton(TXT_AI_DIALOG_PM, callback_data=AI_TO_PROMPTMASTER_CB),
        ]
    ]
    return InlineKeyboardMarkup(rows)

