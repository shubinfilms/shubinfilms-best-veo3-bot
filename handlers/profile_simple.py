"""Simplified profile handlers focused on stability."""

from __future__ import annotations

import inspect
import logging
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Iterable, MutableMapping, Optional

import redis
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import BadRequest, TelegramError
from telegram.ext import ContextTypes

from billing import get_history
from core.balance_provider import get_balance_snapshot
from redis_utils import rds
import settings as app_settings
from .stars import open_stars_menu

log = logging.getLogger(__name__)


_LAST_PROFILE_KEY_TMPL = f"{app_settings.REDIS_PREFIX}:profile_simple:last_msg:{{chat_id}}"
_LAST_PROFILE_TTL = 6 * 60 * 60
_memory_last_ids: dict[int, int] = {}


@dataclass
class _ProfileContext:
    chat_id: Optional[int]
    user_id: Optional[int]


def _profile_key(chat_id: int) -> str:
    return _LAST_PROFILE_KEY_TMPL.format(chat_id=int(chat_id))


def _load_last_message_id(chat_id: int) -> Optional[int]:
    if chat_id is None:
        return None
    if rds is not None:
        try:
            value = rds.get(_profile_key(chat_id))
        except redis.RedisError as exc:  # pragma: no cover - defensive
            log.debug("profile_simple.redis_get_failed", extra={"chat_id": chat_id, "error": str(exc)})
        else:
            if value is None:
                return None
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    return _memory_last_ids.get(int(chat_id))


def _store_last_message_id(chat_id: int, message_id: Optional[int]) -> None:
    if chat_id is None:
        return
    if message_id is None:
        if rds is not None:
            with suppress(redis.RedisError):
                rds.delete(_profile_key(chat_id))
        _memory_last_ids.pop(int(chat_id), None)
        return

    payload = int(message_id)
    if rds is not None:
        try:
            rds.setex(_profile_key(chat_id), _LAST_PROFILE_TTL, payload)
        except redis.RedisError as exc:  # pragma: no cover - defensive
            log.debug("profile_simple.redis_set_failed", extra={"chat_id": chat_id, "error": str(exc)})
        else:
            _memory_last_ids[int(chat_id)] = payload
            return
    _memory_last_ids[int(chat_id)] = payload


async def _delete_previous_profile_message(ctx: ContextTypes.DEFAULT_TYPE, chat_id: Optional[int]) -> None:
    if chat_id is None:
        return
    last_message_id = _load_last_message_id(chat_id)
    if not last_message_id:
        return
    bot_obj = getattr(ctx, "bot", None)
    chat_data = getattr(ctx, "chat_data", None)
    if not hasattr(bot_obj, "delete_message"):
        _store_last_message_id(chat_id, None)
        if isinstance(chat_data, MutableMapping):
            chat_data.pop("profile_msg_id", None)
        return
    try:
        await bot_obj.delete_message(chat_id=chat_id, message_id=last_message_id)
    except (BadRequest, TelegramError):
        pass
    finally:
        _store_last_message_id(chat_id, None)
        if isinstance(chat_data, MutableMapping):
            chat_data.pop("profile_msg_id", None)


async def _answer_callback(update: Update) -> None:
    query = getattr(update, "callback_query", None)
    if query is None:
        return
    with suppress(BadRequest):
        await query.answer()


def _extract_context(update: Update) -> _ProfileContext:
    chat = getattr(update, "effective_chat", None)
    message = getattr(update, "effective_message", None)
    user = getattr(update, "effective_user", None)

    chat_id: Optional[int] = getattr(chat, "id", None)
    if chat_id is None and message is not None:
        chat_id = getattr(message, "chat_id", None)

    user_id: Optional[int] = getattr(user, "id", None)
    if user_id is None and chat_id is not None:
        user_id = chat_id

    return _ProfileContext(chat_id=chat_id, user_id=user_id)


def _profile_keyboard() -> InlineKeyboardMarkup:
    buttons = [
        [InlineKeyboardButton("💎 Пополнить баланс", callback_data="profile:topup")],
        [InlineKeyboardButton("🧾 История операций", callback_data="profile:history")],
        [InlineKeyboardButton("👥 Пригласить друга", callback_data="profile:invite")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="profile:back")],
    ]
    return InlineKeyboardMarkup(buttons)


def _back_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад в профиль", callback_data="profile:open")]])


async def _send_profile_message(
    update: Update,
    ctx: ContextTypes.DEFAULT_TYPE,
    *,
    text: str,
    reply_markup: InlineKeyboardMarkup,
) -> None:
    context = _extract_context(update)
    if context.chat_id is None:
        log.debug("profile_simple.missing_chat_id")
        return
    await _delete_previous_profile_message(ctx, context.chat_id)
    bot_obj = getattr(ctx, "bot", None)
    if not hasattr(bot_obj, "send_message"):
        log.debug("profile_simple.missing_bot")
        return
    message = await bot_obj.send_message(
        chat_id=context.chat_id,
        text=text,
        reply_markup=reply_markup,
        parse_mode=None,
        disable_web_page_preview=True,
    )
    if hasattr(message, "message_id") and context.chat_id is not None:
        msg_id = getattr(message, "message_id")
        _store_last_message_id(context.chat_id, msg_id)
        chat_data = getattr(ctx, "chat_data", None)
        if isinstance(chat_data, MutableMapping):
            chat_data["profile_msg_id"] = msg_id


async def profile_open(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await _answer_callback(update)
    context = _extract_context(update)
    balance_display = "0"
    user_id = context.user_id
    if user_id is not None:
        try:
            snapshot = get_balance_snapshot(int(user_id))
        except Exception:
            log.exception("profile_simple.balance_failed", extra={"user_id": user_id})
        else:
            if snapshot.display:
                balance_display = snapshot.display
            elif snapshot.value is not None:
                balance_display = str(snapshot.value)

    lines = [
        "👤 Профиль",
        f"Баланс: {balance_display} 💎",
    ]
    if user_id is not None:
        lines.append(f"ID: {user_id}")
    lines.extend([
        "",
        "Выберите действие:",
    ])

    await _send_profile_message(
        update,
        ctx,
        text="\n".join(lines),
        reply_markup=_profile_keyboard(),
    )


async def profile_topup(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await _answer_callback(update)
    context = _extract_context(update)
    chat_id = context.chat_id
    message = getattr(update, "effective_message", None)
    query = getattr(update, "callback_query", None)
    if query is not None and getattr(query, "message", None) is not None:
        message = query.message

    message_id = getattr(message, "message_id", None)

    try:
        await open_stars_menu(
            ctx,
            chat_id=chat_id,
            message_id=message_id,
            edit_message=True,
            source="profile",
        )
    except Exception:
        log.exception("profile_simple.topup_stars_failed", extra={"chat_id": chat_id})

    text = "\n".join(
        [
            "💎 Пополнение — скоро.",
            "Мы работаем над запуском оплаты прямо в боте.",
        ]
    )
    await _send_profile_message(update, ctx, text=text, reply_markup=_back_keyboard())


def _format_history_entry(entry: Any) -> str:
    if isinstance(entry, dict):
        title = entry.get("title") or entry.get("description") or entry.get("type")
        amount = entry.get("amount")
        pieces: list[str] = []
        if title:
            pieces.append(str(title))
        if amount not in (None, ""):
            pieces.append(str(amount))
        extra = entry.get("status") or entry.get("note")
        if extra:
            pieces.append(str(extra))
        if pieces:
            return " — ".join(pieces[:2]) if len(pieces) == 2 else " — ".join(pieces)
    if isinstance(entry, str):
        return entry
    return str(entry)


async def profile_history(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await _answer_callback(update)
    context = _extract_context(update)
    entries: Iterable[Any] = []
    if context.user_id is not None:
        try:
            result = get_history(int(context.user_id))
        except Exception:
            log.exception("profile_simple.history_failed", extra={"user_id": context.user_id})
        else:
            if inspect.isawaitable(result):
                result = await result
            try:
                entries = list(result)[:10]
            except TypeError:
                entries = []

    if entries:
        lines = ["🧾 История операций:"]
        for idx, item in enumerate(entries, start=1):
            text = _format_history_entry(item)
            if text:
                lines.append(f"{idx}. {text}")
    else:
        lines = ["🧾 История операций", "История операций пока пуста."]

    await _send_profile_message(update, ctx, text="\n".join(lines), reply_markup=_back_keyboard())


async def profile_invite(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await _answer_callback(update)
    context = _extract_context(update)
    bot_name = (app_settings.BOT_NAME or "").strip()

    if bot_name and context.user_id is not None:
        invite_link = f"https://t.me/{bot_name}?start={context.user_id}"
        text = "\n".join(
            [
                "👥 Пригласить друга",
                "Поделитесь ссылкой:",
                invite_link,
            ]
        )
        keyboard = InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("Скопировать ссылку", url=invite_link)],
                [InlineKeyboardButton("⬅️ Назад в профиль", callback_data="profile:open")],
            ]
        )
    else:
        text = "\n".join(
            [
                "👥 Пригласить друга",
                "Скоро включим приглашения.",
            ]
        )
        keyboard = _back_keyboard()

    await _send_profile_message(update, ctx, text=text, reply_markup=keyboard)


async def profile_back(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await _answer_callback(update)
    context = _extract_context(update)
    await _delete_previous_profile_message(ctx, context.chat_id)
    _store_last_message_id(context.chat_id, None)

    from bot import handle_menu  # Avoid circular import at module load

    await handle_menu(update, ctx, notify_chat_off=False)


__all__ = [
    "profile_back",
    "profile_history",
    "profile_invite",
    "profile_open",
    "profile_topup",
]
