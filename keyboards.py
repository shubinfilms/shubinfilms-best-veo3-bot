from functools import lru_cache
import re
from typing import Dict, Iterable, List, Optional, Tuple

from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
)

from utils.text_normalizer import normalize_btn_text



EMOJI = {
    "video": "🎬",
    "image": "🎨",
    "music": "🎵",
    "chat": "💬",
    "prompt": "🧠",
    "profile": "👤",
    "back": "⬅️",
    "pay": "💎",
}

NAV_PROFILE = "menu:profile"
NAV_KB = "menu:kb"
NAV_PHOTO = "menu:photo"
NAV_MUSIC = "menu:music"
NAV_VIDEO = "menu:video"
NAV_DIALOG = "menu:dialog"

HOME_CB_PROFILE = NAV_PROFILE
HOME_CB_KB = NAV_KB
HOME_CB_PHOTO = NAV_PHOTO
HOME_CB_MUSIC = NAV_MUSIC
HOME_CB_VIDEO = NAV_VIDEO
HOME_CB_DIALOG = NAV_DIALOG

HUB_INLINE_CB_PROFILE = "hub:open:profile"
HUB_INLINE_CB_KB = "hub:open:kb"
HUB_INLINE_CB_PHOTO = "hub:open:photo"
HUB_INLINE_CB_MUSIC = "hub:open:music"
HUB_INLINE_CB_VIDEO = "hub:open:video"
HUB_INLINE_CB_DIALOG = "hub:open:dialog"


_PLAIN_PREFIX_RE = re.compile(r"^[\W_]+", re.UNICODE)


def _strip_prefix_symbols(label: str) -> str:
    return _PLAIN_PREFIX_RE.sub("", label or "").strip()


def iter_home_menu_buttons() -> Iterable[Tuple[str, str]]:
    """Yield flattened pairs of ``(text, callback_data)`` for the home layout."""

    for row in _get_home_menu_layout():
        for label, callback in row:
            yield label, callback


@lru_cache(maxsize=1)
def _build_text_action_variants() -> Dict[str, str]:
    variants: Dict[str, str] = {}
    for label, callback in iter_home_menu_buttons():
        variants[label] = callback
        plain = _strip_prefix_symbols(label)
        if plain and plain != label and callback != HOME_CB_PROFILE:
            variants.setdefault(plain, callback)
    return variants


class _LazyTextActionVariants(dict[str, str]):
    """Lazily materialized map of reply labels to callback payloads."""

    def __init__(self) -> None:
        super().__init__()
        self._loaded = False

    def _ensure(self) -> None:
        if not self._loaded:
            super().update(_build_text_action_variants())
            self._loaded = True

    def __getitem__(self, key: str) -> str:  # type: ignore[override]
        self._ensure()
        return super().__getitem__(key)

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:  # type: ignore[override]
        self._ensure()
        return super().get(key, default)

    def items(self):  # type: ignore[override]
        self._ensure()
        return super().items()

    def keys(self):  # type: ignore[override]
        self._ensure()
        return super().keys()

    def values(self):  # type: ignore[override]
        self._ensure()
        return super().values()

    def __iter__(self):  # type: ignore[override]
        self._ensure()
        return super().__iter__()

    def __len__(self) -> int:  # type: ignore[override]
        self._ensure()
        return super().__len__()

    def __contains__(self, key: object) -> bool:  # type: ignore[override]
        self._ensure()
        return super().__contains__(key)


class _LazyTextToAction(dict[str, str]):
    """Lazily normalized map for reply button dispatch."""

    def __init__(self, source: _LazyTextActionVariants) -> None:
        super().__init__()
        self._source = source
        self._loaded = False

    def _ensure(self) -> None:
        if not self._loaded:
            for label, callback in self._source.items():
                normalized = normalize_btn_text(label)
                if normalized:
                    self.setdefault(normalized, callback)
            self._loaded = True

    def __getitem__(self, key: str) -> str:  # type: ignore[override]
        self._ensure()
        return super().__getitem__(key)

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:  # type: ignore[override]
        self._ensure()
        return super().get(key, default)

    def items(self):  # type: ignore[override]
        self._ensure()
        return super().items()

    def keys(self):  # type: ignore[override]
        self._ensure()
        return super().keys()

    def values(self):  # type: ignore[override]
        self._ensure()
        return super().values()

    def __iter__(self):  # type: ignore[override]
        self._ensure()
        return super().__iter__()

    def __len__(self) -> int:  # type: ignore[override]
        self._ensure()
        return super().__len__()

    def __contains__(self, key: object) -> bool:  # type: ignore[override]
        self._ensure()
        return super().__contains__(key)




AI_MENU_CB = HOME_CB_DIALOG
AI_TO_SIMPLE_CB = "chat_mode_normal"
AI_TO_PROMPTMASTER_CB = "chat_mode_pm"

DIALOG_PICK_REGULAR = "dialog:choose_regular"
DIALOG_PICK_PM = "dialog:choose_promptmaster"

VIDEO_MENU_CB = HOME_CB_VIDEO
IMAGE_MENU_CB = HOME_CB_PHOTO
MUSIC_MENU_CB = HOME_CB_MUSIC
PROFILE_MENU_CB = HOME_CB_PROFILE
KNOWLEDGE_MENU_CB = HOME_CB_KB

# Backward compatible aliases (deprecated)
CB_PROFILE = PROFILE_MENU_CB
CB_KB = KNOWLEDGE_MENU_CB
CB_PHOTO = IMAGE_MENU_CB
CB_MUSIC = MUSIC_MENU_CB
CB_VIDEO = VIDEO_MENU_CB
CB_CHAT = AI_MENU_CB


def kb_main() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(_build_inline_home_rows())


def main_menu_kb() -> InlineKeyboardMarkup:
    return kb_main()


def kb_home_menu() -> InlineKeyboardMarkup:
    return kb_main()


def reply_kb_home() -> ReplyKeyboardMarkup:
    return build_main_reply_kb()


def build_main_reply_kb() -> ReplyKeyboardMarkup:
    layout = _get_home_menu_layout()
    rows: List[List[KeyboardButton]] = []
    for layout_row in layout:
        rows.append([KeyboardButton(text=label) for label, _ in layout_row])
    return ReplyKeyboardMarkup(keyboard=rows, resize_keyboard=True, is_persistent=True)


def dialog_picker_inline() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("💬 Обычный чат", callback_data=DIALOG_PICK_REGULAR),
                InlineKeyboardButton("📝 Prompt-Master", callback_data=DIALOG_PICK_PM),
            ]
        ]
    )


def build_empty_reply_kb() -> ReplyKeyboardRemove:
    return ReplyKeyboardRemove()


def _build_inline_home_rows() -> List[List[InlineKeyboardButton]]:
    layout = _get_home_menu_layout()
    rows: List[List[InlineKeyboardButton]] = []
    for row in layout:
        buttons: List[InlineKeyboardButton] = []
        for label, callback in row:
            buttons.append(InlineKeyboardButton(text=label, callback_data=callback))
        rows.append(buttons)
    return rows


@lru_cache(maxsize=1)
def _get_home_menu_layout() -> Tuple[Tuple[Tuple[str, str], Tuple[str, str]], ...]:
    from texts import (
        TXT_KB_AI_DIALOG,
        TXT_KB_KNOWLEDGE,
        TXT_KB_MUSIC,
        TXT_KB_PHOTO,
        TXT_KB_PROFILE,
        TXT_KB_VIDEO,
    )

    return (
        (
            (TXT_KB_PROFILE, HOME_CB_PROFILE),
            (TXT_KB_KNOWLEDGE, HOME_CB_KB),
        ),
        (
            (TXT_KB_PHOTO, HOME_CB_PHOTO),
            (TXT_KB_MUSIC, HOME_CB_MUSIC),
        ),
        (
            (TXT_KB_VIDEO, HOME_CB_VIDEO),
            (TXT_KB_AI_DIALOG, HOME_CB_DIALOG),
        ),
    )


TEXT_ACTION_VARIANTS: Dict[str, str] = _LazyTextActionVariants()
TEXT_TO_ACTION: Dict[str, str] = _LazyTextToAction(TEXT_ACTION_VARIANTS)  # type: ignore[arg-type]


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


def kb_banana_templates() -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton("🧼 Удалить фон", callback_data="banana:tpl:bg_remove")],
        [InlineKeyboardButton("🎨 Сменить фон на студию", callback_data="banana:tpl:bg_studio")],
        [InlineKeyboardButton("👕 Сменить одежду на чёрный пиджак", callback_data="banana:tpl:outfit_black")],
        [InlineKeyboardButton("💄 Лёгкий макияж, подчеркнуть глаза", callback_data="banana:tpl:makeup_soft")],
        [InlineKeyboardButton("🧼 Очистить стол от лишнего", callback_data="banana:tpl:desk_clean")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="banana:back")],
    ]
    return InlineKeyboardMarkup(rows)

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
    rows = [[InlineKeyboardButton("▶️ Начать генерацию", callback_data="music:start")]]
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
CB_PROFILE_TOPUP = "profile:topup"
CB_PROFILE_BACK = "profile:back"
CB_AI_MODES = AI_MENU_CB
CB_CHAT_NORMAL = AI_TO_SIMPLE_CB
CB_CHAT_PROMPTMASTER = AI_TO_PROMPTMASTER_CB
CB_PAY_STARS = "pay_stars"
CB_PAY_CARD = "pay_card"
CB_PAY_CRYPTO = "pay_crypto"


def kb_main_menu_profile_first() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(_build_inline_home_rows())


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


def kb_kb_root() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("🪄 Примеры генераций", callback_data="kb_examples")],
            [InlineKeyboardButton("✨ Готовые шаблоны", callback_data="kb_templates")],
            [InlineKeyboardButton("💡 Мини видео уроки", callback_data="kb_lessons")],
            [InlineKeyboardButton("❓ Частые вопросы", callback_data="kb_faq")],
            [InlineKeyboardButton("⬅️ Назад (в главное)", callback_data="menu_main")],
        ]
    )


def kb_kb_templates() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("🎬 Генерация видео", callback_data="tpl_video")],
            [InlineKeyboardButton("🖼️ Генерация фото", callback_data="tpl_image")],
            [InlineKeyboardButton("🎵 Генерация музыки", callback_data="tpl_music")],
            [InlineKeyboardButton("🍌 Редактор фото", callback_data="tpl_banana")],
            [InlineKeyboardButton("🤖 ИИ-фотограф", callback_data="tpl_ai_photo")],
            [InlineKeyboardButton("⬅️ Назад", callback_data="kb_entry")],
        ]
    )

