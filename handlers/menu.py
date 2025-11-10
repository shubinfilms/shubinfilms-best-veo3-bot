from __future__ import annotations

from typing import MutableMapping, Optional, Sequence

from telegram import InlineKeyboardButton, Update
from telegram.ext import ContextTypes
from telegram.error import BadRequest, Forbidden, TelegramError

from keyboards import CB, kb_main, photo_engines_kb
from texts import (
    TXT_KB_AI_DIALOG,
    TXT_KB_MUSIC,
    TXT_KB_PHOTO,
    TXT_KB_PROFILE,
    TXT_KB_VIDEO,
    TXT_MENU_TITLE,
)
from ui.card import build_card
from ui.card_store import load_card_message_id, store_card_message_id

from logging_utils import get_logger
from settings import FEATURE_BANANA, FEATURE_SUNO, FEATURE_VIDEO

log = get_logger("handlers.menu")

_MAIN_MENU_SUBTITLE = "Выберите раздел:"

_MENU_NAMESPACE = "menu"
_VIDEO_NAMESPACE = "video"
_SUNO_NAMESPACE = "suno"
_DIALOG_NAMESPACE = "dialog"

_CHATDATA_KEY = "ui_card_message_ids"


def _chat_card_mapping(
    ctx: ContextTypes.DEFAULT_TYPE,
) -> MutableMapping[str, int] | None:
    mapping = getattr(ctx, "chat_data", None)
    if not isinstance(mapping, MutableMapping):
        return None
    store = mapping.get(_CHATDATA_KEY)
    if isinstance(store, MutableMapping):
        return store
    store_dict: dict[str, int] = {}
    mapping[_CHATDATA_KEY] = store_dict
    return store_dict


def _to_int(value: object) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


async def _load_card_id(
    ctx: ContextTypes.DEFAULT_TYPE,
    *,
    namespace: str,
    user_id: int,
) -> Optional[int]:
    local_store = _chat_card_mapping(ctx)
    if local_store is not None:
        cached = _to_int(local_store.get(namespace))
        if cached:
            return cached

    redis = getattr(ctx, "redis", None)
    if redis is not None:
        try:
            stored = await load_card_message_id(redis, namespace, user_id)
        except Exception as exc:  # pragma: no cover - diagnostics
            log.debug(
                "menu.card.load_failed",
                extra={"namespace": namespace, "user_id": user_id, "error": str(exc)},
            )
        else:
            if stored:
                if local_store is not None:
                    local_store[namespace] = int(stored)
                return stored
    return None


async def _store_card_id(
    ctx: ContextTypes.DEFAULT_TYPE,
    *,
    namespace: str,
    user_id: int,
    message_id: int,
) -> None:
    local_store = _chat_card_mapping(ctx)
    if local_store is not None:
        local_store[namespace] = int(message_id)

    redis = getattr(ctx, "redis", None)
    if redis is not None:
        try:
            await store_card_message_id(redis, namespace, user_id, int(message_id))
        except Exception as exc:  # pragma: no cover - diagnostics
            log.debug(
                "menu.card.store_failed",
                extra={"namespace": namespace, "user_id": user_id, "error": str(exc)},
            )


async def _answer_callback(query) -> None:
    if query is None:
        return
    try:
        await query.answer()
    except Exception:  # pragma: no cover - best effort ack
        pass


async def _ensure_card(
    update: Update,
    ctx: ContextTypes.DEFAULT_TYPE,
    *,
    namespace: str,
    card: dict,
) -> Optional[int]:
    chat = getattr(update, "effective_chat", None)
    user = getattr(update, "effective_user", None)
    if chat is None or user is None:
        return None

    chat_id = getattr(chat, "id", None)
    user_id = getattr(user, "id", None)
    if chat_id is None or user_id is None:
        return None

    message_id = await _load_card_id(ctx, namespace=namespace, user_id=int(user_id))
    text = card.get("text") or ""
    reply_markup = card.get("reply_markup")
    parse_mode = card.get("parse_mode")
    disable_web_page_preview = bool(card.get("disable_web_page_preview", True))

    if message_id is not None:
        try:
            await ctx.bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=text,
                reply_markup=reply_markup,
                parse_mode=parse_mode,
                disable_web_page_preview=disable_web_page_preview,
            )
        except BadRequest as exc:
            message = str(exc)
            lowered = message.lower()
            if "message is not modified" in lowered:
                try:
                    await ctx.bot.edit_message_reply_markup(
                        chat_id=chat_id,
                        message_id=message_id,
                        reply_markup=reply_markup,
                    )
                except BadRequest as markup_exc:
                    if "message is not modified" not in str(markup_exc).lower():
                        log.debug(
                            "menu.card.markup_failed",
                            extra={
                                "namespace": namespace,
                                "chat_id": chat_id,
                                "message_id": message_id,
                                "error": str(markup_exc),
                            },
                        )
                except TelegramError as markup_exc:  # pragma: no cover - defensive
                    log.debug(
                        "menu.card.markup_error",
                        extra={
                            "namespace": namespace,
                            "chat_id": chat_id,
                            "message_id": message_id,
                            "error": str(markup_exc),
                        },
                    )
                await _store_card_id(ctx, namespace=namespace, user_id=int(user_id), message_id=message_id)
                return message_id

            log.debug(
                "menu.card.edit_failed",
                extra={
                    "namespace": namespace,
                    "chat_id": chat_id,
                    "message_id": message_id,
                    "error": message,
                },
            )
            message_id = None
        except (Forbidden, TelegramError) as exc:
            log.debug(
                "menu.card.edit_error",
                extra={
                    "namespace": namespace,
                    "chat_id": chat_id,
                    "message_id": message_id,
                    "error": str(exc),
                },
            )
            message_id = None
        else:
            await _store_card_id(ctx, namespace=namespace, user_id=int(user_id), message_id=message_id)
            return message_id

    try:
        sent = await ctx.bot.send_message(
            chat_id=chat_id,
            text=text,
            reply_markup=reply_markup,
            parse_mode=parse_mode,
            disable_web_page_preview=disable_web_page_preview,
        )
    except TelegramError as exc:
        log.warning(
            "menu.card.send_failed",
            extra={"namespace": namespace, "chat_id": chat_id, "error": str(exc)},
        )
        return None

    await _store_card_id(
        ctx,
        namespace=namespace,
        user_id=int(user_id),
        message_id=sent.message_id,
    )
    return sent.message_id


def _filter_menu_rows(rows: Sequence[Sequence[InlineKeyboardButton]]) -> list[list[InlineKeyboardButton]]:
    filtered: list[list[InlineKeyboardButton]] = []
    for row in rows:
        new_row: list[InlineKeyboardButton] = []
        for button in row:
            data = (button.callback_data or "").strip().lower()
            if not FEATURE_VIDEO and data == "menu:video":
                continue
            if not FEATURE_SUNO and data == "menu:music":
                continue
            if not FEATURE_BANANA and data in {"menu:photo", "img_engine:banana"}:
                continue
            new_row.append(button)
        if new_row:
            filtered.append(new_row)
    return filtered


async def open_main_menu(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    skip_ack: bool = False,
) -> None:
    query = update.callback_query if not skip_ack else None
    await _answer_callback(query)

    card = build_main_menu_card()
    message_id = await _ensure_card(update, context, namespace=_MENU_NAMESPACE, card=card)

    chat = getattr(update, "effective_chat", None)
    chat_id = getattr(chat, "id", None)
    if message_id is not None and chat_id is not None:
        log.debug(
            "menu.card.rendered",
            extra={
                "namespace": _MENU_NAMESPACE,
                "chat_id": chat_id,
                "message_id": message_id,
            },
        )


async def show_profile(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await _answer_callback(query)
    target = query.message if query else update.effective_message
    if target is None:
        return
    await target.reply_text("👤 Профиль\nБаланс: …\nТариф: …")


async def show_kb(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await _answer_callback(query)
    target = query.message if query else update.effective_message
    if target is None:
        return
    await target.reply_text("📚 База знаний (в разработке)")


async def open_photo_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await _answer_callback(query)
    target = query.message if query else update.effective_message
    if target is None:
        return
    await target.reply_text(
        "📸 Выберите нейросеть для фотографий:",
        reply_markup=photo_engines_kb(),
    )


async def open_music_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not FEATURE_SUNO:
        chat = getattr(update, "effective_chat", None)
        chat_id = getattr(chat, "id", None)
        if chat_id is not None:
            try:
                await context.bot.send_message(
                    chat_id,
                    "Музыкальный режим временно недоступен.",
                )
            except TelegramError:
                log.debug("menu.music.disabled.notify_failed", exc_info=True)
        await _answer_callback(query)
        return

    await _answer_callback(query)

    card = build_music_card()
    message_id = await _ensure_card(update, context, namespace=_SUNO_NAMESPACE, card=card)

    chat = getattr(update, "effective_chat", None)
    chat_id = getattr(chat, "id", None)
    if message_id is not None and chat_id is not None:
        log.debug(
            "menu.card.rendered",
            extra={
                "namespace": _SUNO_NAMESPACE,
                "chat_id": chat_id,
                "message_id": message_id,
            },
        )


async def open_video_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not FEATURE_VIDEO:
        chat = getattr(update, "effective_chat", None)
        chat_id = getattr(chat, "id", None)
        if chat_id is not None:
            try:
                await context.bot.send_message(
                    chat_id,
                    "Видео сейчас недоступно. Загляните позже!",
                )
            except TelegramError:
                log.debug("menu.video.disabled.notify_failed", exc_info=True)
        await _answer_callback(query)
        return

    await _answer_callback(query)

    card = build_video_card()
    message_id = await _ensure_card(update, context, namespace=_VIDEO_NAMESPACE, card=card)

    chat = getattr(update, "effective_chat", None)
    chat_id = getattr(chat, "id", None)
    if message_id is not None and chat_id is not None:
        log.debug(
            "menu.card.rendered",
            extra={
                "namespace": _VIDEO_NAMESPACE,
                "chat_id": chat_id,
                "message_id": message_id,
            },
        )


async def open_dialog_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await _answer_callback(query)

    chat = getattr(update, "effective_chat", None)
    chat_id = getattr(chat, "id", None)
    user = getattr(update, "effective_user", None)
    user_id = getattr(user, "id", None)

    try:
        from bot import enable_chat_mode  # type: ignore
    except Exception as exc:  # pragma: no cover - fallback when bot not initialized
        log.debug(
            "menu.dialog.enable_import_failed",
            extra={"chat_id": chat_id, "user_id": user_id, "error": str(exc)},
        )
    else:
        try:
            await enable_chat_mode(update, context, "normal")
        except Exception as exc:  # pragma: no cover - diagnostics
            log.warning(
                "menu.dialog.enable_failed",
                extra={"chat_id": chat_id, "user_id": user_id, "error": str(exc)},
            )

    card = build_dialog_card()
    message_id = await _ensure_card(update, context, namespace=_DIALOG_NAMESPACE, card=card)

    if message_id is not None and chat_id is not None:
        log.debug(
            "menu.card.rendered",
            extra={
                "namespace": _DIALOG_NAMESPACE,
                "chat_id": chat_id,
                "message_id": message_id,
            },
        )


async def close_dialog_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await _answer_callback(query)

    chat = getattr(update, "effective_chat", None)
    chat_id = getattr(chat, "id", None)
    user = getattr(update, "effective_user", None)
    user_id = getattr(user, "id", None)

    try:
        from bot import disable_chat_mode  # type: ignore
    except Exception as exc:  # pragma: no cover - diagnostics only
        log.debug(
            "menu.dialog.disable_import_failed",
            extra={"chat_id": chat_id, "user_id": user_id, "error": str(exc)},
        )
    else:
        try:
            await disable_chat_mode(
                context,
                chat_id=chat_id,
                user_id=user_id,
                notify=False,
            )
        except Exception as exc:  # pragma: no cover - diagnostics only
            log.warning(
                "menu.dialog.disable_failed",
                extra={"chat_id": chat_id, "user_id": user_id, "error": str(exc)},
            )

    await open_main_menu(update, context, skip_ack=True)


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
    rows = _filter_menu_rows(markup.inline_keyboard)
    return build_card(TXT_MENU_TITLE, _MAIN_MENU_SUBTITLE, rows)


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
        [InlineKeyboardButton("🚀 Начать генерацию", callback_data="suno:start")],
        [InlineKeyboardButton("🎙 Прикрепить аудио", callback_data="suno:attach")],
        [InlineKeyboardButton("⬅️ В меню", callback_data="back")],
    ]
    body = [
        "1. Опишите трек: жанр, настроение, длительность.",
        "2. Прикрепите референс (voice/audio до 15 МБ) или нажмите «Прикрепить аудио».",
    ]
    return build_card(
        TXT_KB_MUSIC,
        "Suno создаёт музыку по вашему описанию и каппелле.",
        rows,
        body_lines=body,
    )


def build_video_card(*, veo_fast_cost: int, veo_photo_cost: int, sora2_cost: int) -> dict:
    rows = [
        [InlineKeyboardButton("🎞 Kling — скоро", callback_data="video:kling")],
        [
            InlineKeyboardButton(
                f"VEO Fast — текст → видео • 💎 {veo_fast_cost}",
                callback_data="mode:veo_text_fast",
            )
        ],
        [
            InlineKeyboardButton(
                f"VEO Animate — фото → видео • 💎 {veo_photo_cost}",
                callback_data=CB.VIDEO_VEO_ANIMATE,
            )
        ],
        [
            InlineKeyboardButton(
                f"🎬 Sora2 — текст → видео • 💎 {sora2_cost}",
                callback_data="video:type:sora2",
            )
        ],
        [InlineKeyboardButton("⬅️ В меню", callback_data="back")],
    ]
    body = [
        "Kling в разработке — скоро откроем доступ.",
        "Для VEO пришлите текст или 1–4 фото, затем нажмите «Начать генерацию».",
    ]
    return build_card(
        TXT_KB_VIDEO,
        "Выберите движок для генерации видео.",
        rows,
        body_lines=body,
    )


def build_dialog_card() -> dict:
    rows = [
        [InlineKeyboardButton("✖️ Выкл диалог", callback_data="dialog:off")],
        [InlineKeyboardButton("⬅️ В меню", callback_data="back")],
    ]
    body = [
        "Диалог включён. Пишите сообщения — я отвечу сразу, без карточек.",
        "Команда /reset очищает историю переписки.",
    ]
    return build_card(
        TXT_KB_AI_DIALOG,
        "Свободный чат активирован.",
        rows,
        body_lines=body,
    )


__all__ = [
    "open_main_menu",
    "show_profile",
    "show_kb",
    "open_photo_mode",
    "open_music_mode",
    "open_video_mode",
    "open_dialog_mode",
    "close_dialog_mode",
    "on_menu_dialog",
    "build_dialog_card",
    "build_main_menu_card",
    "build_music_card",
    "build_photo_card",
    "build_profile_card",
    "build_video_card",
]
