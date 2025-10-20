from __future__ import annotations

import logging
from collections.abc import MutableMapping
from typing import Awaitable, Callable, Optional

from telegram import InlineKeyboardButton, Update
from telegram.ext import ContextTypes

from ui.card import build_card
from telegram_utils import safe_answer

log = logging.getLogger(__name__)

KB_PREFIX = "KB_"
KB_ROOT = f"{KB_PREFIX}ROOT"
KB_EXAMPLES = f"{KB_PREFIX}EXAMPLES"
KB_TEMPLATES = f"{KB_PREFIX}TEMPLATES"
KB_MINI_LESSONS = f"{KB_PREFIX}MINI_LESSONS"
KB_FAQ = f"{KB_PREFIX}FAQ"
KB_TEMPLATE_PREFIX = f"{KB_PREFIX}TEMPLATE_"
KB_TEMPLATE_VIDEO = f"{KB_TEMPLATE_PREFIX}VIDEO"
KB_TEMPLATE_PHOTO = f"{KB_TEMPLATE_PREFIX}PHOTO"
KB_TEMPLATE_MUSIC = f"{KB_TEMPLATE_PREFIX}MUSIC"
KB_TEMPLATE_BANANA = f"{KB_TEMPLATE_PREFIX}BANANA"
KB_TEMPLATE_AI_PHOTO = f"{KB_TEMPLATE_PREFIX}AI_PHOTO"

_CARD_STATE_KEY = "knowledge_base_card"
_CARD_MSG_KEY = "kb"
_STATE_ACTIVE_CARD_KEY = "active_card"

_send_menu: Optional[Callable[..., Awaitable[Optional[int]]]] = None
_faq_handler: Optional[Callable[[Update, ContextTypes.DEFAULT_TYPE], Awaitable[None]]] = None
_state_getter: Optional[Callable[[ContextTypes.DEFAULT_TYPE], dict]] = None

_TEMPLATE_MESSAGES = {
    KB_TEMPLATE_VIDEO: (
        "🎬 Шаблон генерации видео:\n"
        "1. Опишите ключевую сцену и героев.\n"
        "2. Добавьте настроение, стиль съёмки и ракурсы.\n"
        "3. Укажите длительность и желаемый формат кадра."
    ),
    KB_TEMPLATE_PHOTO: (
        "📸 Шаблон генерации фото:\n"
        "1. Опишите внешний вид персонажа и окружение.\n"
        "2. Уточните стиль (реализм, кино, fashion и т. д.).\n"
        "3. Добавьте детали света, настроения и качества."
    ),
    KB_TEMPLATE_MUSIC: (
        "🎧 Шаблон генерации музыки:\n"
        "1. Название или тема трека.\n"
        "2. Жанр, настроение и ключевые инструменты.\n"
        "3. Желательные референсы или ссылки на похожие треки."
    ),
    KB_TEMPLATE_BANANA: (
        "🍌 Шаблон для редактора фото:\n"
        "1. Опишите, что изменить (фон, одежда, предметы).\n"
        "2. Уточните желаемый результат и стиль.\n"
        "3. При необходимости добавьте цветовые или световые акценты."
    ),
    KB_TEMPLATE_AI_PHOTO: (
        "🤖 Шаблон для ИИ-фотографа:\n"
        "1. Опишите персонажа: возраст, внешность, позу.\n"
        "2. Укажите место съёмки и атмосферу кадра.\n"
        "3. Добавьте технические детали: тип объектива, свет, качество."
    ),
}


def configure(
    *,
    send_menu: Callable[..., Awaitable[Optional[int]]],
    faq_handler: Optional[Callable[[Update, ContextTypes.DEFAULT_TYPE], Awaitable[None]]] = None,
    state_getter: Optional[Callable[[ContextTypes.DEFAULT_TYPE], dict]] = None,
) -> None:
    global _send_menu
    global _faq_handler
    global _state_getter
    _send_menu = send_menu
    _faq_handler = faq_handler
    _state_getter = state_getter


def _chat_data(ctx: ContextTypes.DEFAULT_TYPE) -> MutableMapping[str, object] | None:
    obj = getattr(ctx, "chat_data", None)
    return obj if isinstance(obj, MutableMapping) else None


def _get_msg_id(chat_data: MutableMapping[str, object] | None) -> Optional[int]:
    if not isinstance(chat_data, MutableMapping):
        return None
    raw = chat_data.get("kb_msg_id")
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _store_msg_id(
    chat_data: MutableMapping[str, object] | None, message_id: Optional[int]
) -> None:
    if not isinstance(chat_data, MutableMapping):
        return
    if isinstance(message_id, int):
        chat_data["kb_msg_id"] = int(message_id)
    else:
        chat_data.pop("kb_msg_id", None)


def _root_rows() -> list[list[InlineKeyboardButton]]:
    return [
        [InlineKeyboardButton("🪄 Примеры генераций", callback_data=KB_EXAMPLES)],
        [InlineKeyboardButton("✨ Готовые шаблоны", callback_data=KB_TEMPLATES)],
        [InlineKeyboardButton("💡 Мини видео уроки", callback_data=KB_MINI_LESSONS)],
        [InlineKeyboardButton("❓ Частые вопросы", callback_data=KB_FAQ)],
        [InlineKeyboardButton("⬅️ Назад в меню", callback_data="back")],
    ]


def _placeholder_rows() -> list[list[InlineKeyboardButton]]:
    return [[InlineKeyboardButton("⬅️ Назад", callback_data=KB_ROOT)]]


def _templates_rows() -> list[list[InlineKeyboardButton]]:
    return [
        [InlineKeyboardButton("🎬 Генерация видео", callback_data=KB_TEMPLATE_VIDEO)],
        [InlineKeyboardButton("📸 Генерация фото", callback_data=KB_TEMPLATE_PHOTO)],
        [InlineKeyboardButton("🎧 Генерация музыки", callback_data=KB_TEMPLATE_MUSIC)],
        [InlineKeyboardButton("🍌 Редактор фото", callback_data=KB_TEMPLATE_BANANA)],
        [InlineKeyboardButton("🤖 ИИ фотограф", callback_data=KB_TEMPLATE_AI_PHOTO)],
        [InlineKeyboardButton("⬅️ Назад", callback_data=KB_ROOT)],
    ]


def _root_card() -> dict:
    return build_card("📚 База знаний", "Выберите раздел:", _root_rows())


def _examples_card() -> dict:
    return build_card(
        "🪄 Примеры генераций",
        "Скоро добавим лучшие примеры по каждому режиму.",
        _placeholder_rows(),
    )


def _lessons_card() -> dict:
    return build_card(
        "💡 Мини видео уроки",
        "Короткие инструкции скоро будут тут.",
        _placeholder_rows(),
    )


def _templates_card() -> dict:
    return build_card("✨ Готовые шаблоны", "Выберите шаблон для генерации:", _templates_rows())


async def _send_card(
    ctx: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    card: dict,
    *,
    log_label: str,
    active_card: Optional[str] = None,
    fallback_message_id: Optional[int] = None,
) -> Optional[int]:
    if _send_menu is None:
        raise RuntimeError("knowledge_base.send_menu is not configured")

    if _state_getter is None:
        raise RuntimeError("knowledge_base.state_getter is not configured")

    state_dict = _state_getter(ctx)
    chat_data = _chat_data(ctx)
    effective_fallback = fallback_message_id
    if effective_fallback is None:
        effective_fallback = _get_msg_id(chat_data)

    payload = {
        "ctx": ctx,
        "chat_id": chat_id,
        "text": card["text"],
        "reply_markup": card["reply_markup"],
        "state_key": _CARD_STATE_KEY,
        "msg_ids_key": _CARD_MSG_KEY,
        "state_dict": state_dict,
        "fallback_message_id": effective_fallback,
        "parse_mode": card.get("parse_mode"),
        "disable_web_page_preview": card.get("disable_web_page_preview", True),
        "log_label": log_label,
    }
    result = await _send_menu(**payload)
    if isinstance(result, int):
        _store_msg_id(chat_data, result)
    elif isinstance(effective_fallback, int):
        _store_msg_id(chat_data, effective_fallback)
    if active_card is not None:
        try:
            state_dict[_STATE_ACTIVE_CARD_KEY] = active_card
        except Exception:
            pass
    return result


async def open_root(
    ctx: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    *,
    suppress_nav: bool = False,
    fallback_message_id: Optional[int] = None,
) -> Optional[int]:
    log.info(
        "[KB] open",
        extra={"chat_id": chat_id, "suppress_nav": suppress_nav},
    )
    chat_data = _chat_data(ctx)
    previous_mid = _get_msg_id(chat_data)
    effective_fallback = fallback_message_id
    if effective_fallback is None:
        effective_fallback = previous_mid
    message_id = await _send_card(
        ctx,
        chat_id,
        _root_card(),
        log_label="ui.kb.root",
        active_card="kb:root",
        fallback_message_id=fallback_message_id,
    )
    reused = bool(
        isinstance(message_id, int)
        and effective_fallback is not None
        and message_id == effective_fallback
    )
    log.info("kb.opened reused=%s msg_id=%s", reused, message_id)
    return message_id


async def show_examples(ctx: ContextTypes.DEFAULT_TYPE, chat_id: int) -> Optional[int]:
    return await _send_card(
        ctx,
        chat_id,
        _examples_card(),
        log_label="ui.kb.examples",
        active_card="kb:examples",
    )


async def show_lessons(ctx: ContextTypes.DEFAULT_TYPE, chat_id: int) -> Optional[int]:
    return await _send_card(
        ctx,
        chat_id,
        _lessons_card(),
        log_label="ui.kb.lessons",
        active_card="kb:lessons",
    )


async def show_templates(ctx: ContextTypes.DEFAULT_TYPE, chat_id: int) -> Optional[int]:
    log.info("[KB] open templates", extra={"chat_id": chat_id})
    return await _send_card(
        ctx,
        chat_id,
        _templates_card(),
        log_label="ui.kb.templates",
        active_card="kb:templates",
    )


async def handle_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query is None or query.data is None:
        return

    data = query.data
    chat = update.effective_chat or (query.message.chat if query.message else None)
    chat_id = chat.id if chat else None
    if chat_id is None:
        await safe_answer(query)
        return

    if data == KB_ROOT:
        await safe_answer(query)
        await open_root(ctx, chat_id)
        return
    if data == KB_EXAMPLES:
        await safe_answer(query)
        await show_examples(ctx, chat_id)
        return
    if data == KB_TEMPLATES:
        await safe_answer(query)
        await show_templates(ctx, chat_id)
        return
    if data == KB_MINI_LESSONS:
        await safe_answer(query)
        await show_lessons(ctx, chat_id)
        return
    if data == KB_FAQ:
        await safe_answer(query)
        await open_root(ctx, chat_id)
        if callable(_faq_handler):
            await _faq_handler(update, ctx)
        if _state_getter is not None:
            try:
                _state_getter(ctx)[_STATE_ACTIVE_CARD_KEY] = "kb:faq"
            except Exception:
                pass
        return
    if data in _TEMPLATE_MESSAGES:
        await safe_answer(query)
        await ctx.bot.send_message(chat_id=chat_id, text=_TEMPLATE_MESSAGES[data])
        return

    if data.startswith(KB_TEMPLATE_PREFIX):
        log.warning("[KB] unknown template", extra={"chat_id": chat_id, "data": data})
        await safe_answer(query, text="Шаблон недоступен", show_alert=True)
        return

    if data.startswith(KB_PREFIX):
        log.warning("[KB] unknown callback", extra={"chat_id": chat_id, "data": data})
        await safe_answer(query)


async def kb_open_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Entry point for inline callbacks opening the knowledge base."""

    query = update.callback_query
    if query is None:
        return

    await safe_answer(query)
    await _kb_render_or_send(update, context, origin="callback")


async def kb_open_entrypoint(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Entry point for text or command based knowledge base openings."""

    await _kb_render_or_send(update, context, origin="text")


async def _kb_render_or_send(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    origin: str,
) -> None:
    chat = getattr(update, "effective_chat", None)
    message = getattr(update, "effective_message", None)
    chat_id = getattr(chat, "id", None)
    if chat_id is None and message is not None:
        chat_id = getattr(message, "chat_id", None)
    if chat_id is None:
        return

    chat_data = _chat_data(context)
    fallback_message_id: Optional[int] = None
    if origin == "callback":
        query = update.callback_query
        if query and query.message:
            fallback_message_id = getattr(query.message, "message_id", None)

    message_id = await open_root(
        context,
        chat_id,
        suppress_nav=True,
        fallback_message_id=fallback_message_id,
    )

    if isinstance(chat_data, MutableMapping) and isinstance(message_id, int):
        chat_data["last_card"] = {
            "kind": "kb",
            "chat_id": chat_id,
            "message_id": message_id,
        }


__all__ = [
    "KB_EXAMPLES",
    "KB_FAQ",
    "KB_MINI_LESSONS",
    "KB_PREFIX",
    "KB_ROOT",
    "KB_TEMPLATE_PREFIX",
    "KB_TEMPLATES",
    "kb_open_entrypoint",
    "kb_open_handler",
    "configure",
    "handle_callback",
    "open_root",
    "show_examples",
    "show_lessons",
    "show_templates",
]
