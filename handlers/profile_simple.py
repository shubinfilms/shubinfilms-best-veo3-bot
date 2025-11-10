"""Lightweight profile card handlers with single-message storage."""

from __future__ import annotations

import inspect
import logging
from typing import Iterable, MutableMapping, Optional, Sequence

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import BadRequest
from telegram.ext import ContextTypes

import settings as app_settings
from services.db_async import get_user_balance_async
from ui.card_store import clear_card_message_id, load_card_message_id, store_card_message_id

try:  # pragma: no cover - optional billing module
    from billing import get_history as _billing_history
except Exception:  # pragma: no cover - fallback in tests
    _billing_history = None  # type: ignore[assignment]

from .stars import open_stars_menu

log = logging.getLogger(__name__)

_NAMESPACE = "profile"
_CHATDATA_KEY = "profile_card_message_id"


async def get_history(user_id: int) -> Sequence[dict[str, object]]:
    """Return recent billing entries for ``user_id``.

    The default implementation proxies to :mod:`billing`. Tests can monkeypatch
    this coroutine to provide custom data.
    """

    if _billing_history is None:
        return []
    result = _billing_history(int(user_id))
    if inspect.isawaitable(result):
        return await result  # type: ignore[return-value]
    return list(result or [])  # type: ignore[arg-type]


async def _load_history_entries(user_id: int) -> Sequence[dict[str, object]]:
    result = get_history(int(user_id))
    if inspect.isawaitable(result):
        result = await result  # type: ignore[awaited-non-awaitable]
    if not result:
        return []
    return list(result)


def _profile_keyboard() -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton("💎 Пополнить баланс", callback_data="btn:profile|view=topup")],
        [InlineKeyboardButton("🧾 История операций", callback_data="btn:profile|view=history")],
        [InlineKeyboardButton("👥 Пригласить друга", callback_data="btn:profile|view=invite")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="btn:profile|view=back")],
    ]
    return InlineKeyboardMarkup(rows)


def _back_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data="btn:profile")]])


async def _answer_callback(update: Update) -> None:
    query = getattr(update, "callback_query", None)
    if query is None:
        return
    try:
        await query.answer()
    except Exception:  # pragma: no cover - best effort acknowledgement
        pass


def _chat_storage(ctx: ContextTypes.DEFAULT_TYPE) -> MutableMapping[str, object] | None:
    mapping = getattr(ctx, "chat_data", None)
    return mapping if isinstance(mapping, MutableMapping) else None


async def _load_stored_message_id(ctx: ContextTypes.DEFAULT_TYPE, user_id: int) -> Optional[int]:
    redis = getattr(ctx, "redis", None)
    if redis is not None:
        try:
            stored = await load_card_message_id(redis, _NAMESPACE, user_id)
        except Exception as exc:  # pragma: no cover - diagnostics
            log.debug(
                "profile.card.load_failed",
                extra={"user_id": user_id, "error": str(exc)},
            )
        else:
            if stored is not None:
                return stored
    mapping = _chat_storage(ctx)
    if mapping is not None:
        raw = mapping.get(_CHATDATA_KEY)
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None
    return None


async def _store_message_id(
    ctx: ContextTypes.DEFAULT_TYPE, user_id: int, message_id: Optional[int]
) -> None:
    if message_id is None:
        return
    redis = getattr(ctx, "redis", None)
    if redis is not None:
        try:
            await store_card_message_id(redis, _NAMESPACE, user_id, int(message_id))
        except Exception as exc:  # pragma: no cover - diagnostics
            log.debug(
                "profile.card.store_failed",
                extra={"user_id": user_id, "error": str(exc)},
            )
    mapping = _chat_storage(ctx)
    if mapping is not None:
        mapping[_CHATDATA_KEY] = int(message_id)


async def _clear_message_id(ctx: ContextTypes.DEFAULT_TYPE, user_id: int) -> None:
    redis = getattr(ctx, "redis", None)
    if redis is not None:
        try:
            await clear_card_message_id(redis, _NAMESPACE, user_id)
        except Exception as exc:  # pragma: no cover - diagnostics
            log.debug(
                "profile.card.clear_failed",
                extra={"user_id": user_id, "error": str(exc)},
            )
    mapping = _chat_storage(ctx)
    if mapping is not None:
        mapping.pop(_CHATDATA_KEY, None)


def _format_balance(value: Optional[int]) -> str:
    if value is None:
        return "0"
    return f"{int(value):,}".replace(",", " ")


async def _balance_display(user_id: int) -> str:
    try:
        raw = await get_user_balance_async(int(user_id))
    except Exception as exc:
        log.warning(
            "profile.balance_unavailable",
            extra={"user_id": user_id, "error": str(exc)},
        )
        return "—"
    return _format_balance(raw)


async def _delete_stored_message(
    ctx: ContextTypes.DEFAULT_TYPE, chat_id: Optional[int], user_id: Optional[int]
) -> None:
    if chat_id is None or user_id is None:
        return
    message_id = await _load_stored_message_id(ctx, user_id)
    if message_id is None:
        return
    bot = getattr(ctx, "bot", None)
    if bot is None:
        await _clear_message_id(ctx, user_id)
        return
    try:
        await bot.delete_message(chat_id=chat_id, message_id=message_id)
    except BadRequest:
        pass
    except Exception as exc:  # pragma: no cover - diagnostics
        log.debug(
            "profile.card.delete_failed",
            extra={"chat_id": chat_id, "message_id": message_id, "error": str(exc)},
        )
    await _clear_message_id(ctx, user_id)


def _invite_text() -> str:
    bot_name = (app_settings.BOT_NAME or "").strip()
    username = (app_settings.BOT_USERNAME or "").strip()
    if not bot_name and not username:
        return "Скоро включим приглашения."
    handle = bot_name or username
    return (
        "Поделитесь ссылкой с друзьями и получайте бонусы!\n"
        f"Ваш бот: @{handle}"
    )


def _render_history_lines(entries: Iterable[dict[str, object]]) -> str:
    rendered: list[str] = []
    for entry in entries:
        if isinstance(entry, dict):
            label = str(entry.get("label") or entry.get("title") or "Операция")
            amount = entry.get("amount")
            balance = entry.get("balance")
            pieces = [label]
            if amount is not None:
                pieces.append(f"Δ {amount}")
            if balance is not None:
                pieces.append(f"Баланс: {balance}")
            rendered.append(" • ".join(pieces))
        else:
            rendered.append(str(entry))
    return "\n".join(rendered)


async def profile_open(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await _answer_callback(update)
    chat = getattr(update, "effective_chat", None)
    message = getattr(update, "effective_message", None)
    chat_id = getattr(chat, "id", None) or getattr(message, "chat_id", None)
    user = getattr(update, "effective_user", None)
    user_id = getattr(user, "id", None)
    if chat_id is None or user_id is None:
        return

    balance_display = await _balance_display(user_id)
    text = "\n".join([
        "👤 Профиль",
        f"Баланс: {balance_display} 💎",
    ])
    markup = _profile_keyboard()

    bot = getattr(ctx, "bot", None)
    if bot is None:
        log.debug("profile.open.no_bot", extra={"chat_id": chat_id})
        return

    stored_message_id = await _load_stored_message_id(ctx, user_id)
    sent = None
    if stored_message_id is not None:
        try:
            sent = await bot.edit_message_text(
                chat_id=chat_id,
                message_id=stored_message_id,
                text=text,
                reply_markup=markup,
                disable_web_page_preview=True,
            )
        except BadRequest as exc:
            log.debug(
                "profile.card.edit_failed",
                extra={"chat_id": chat_id, "message_id": stored_message_id, "error": str(exc)},
            )
            sent = None
            await _clear_message_id(ctx, user_id)
        except Exception as exc:  # pragma: no cover - diagnostics
            log.debug(
                "profile.card.edit_error",
                exc_info=True,
                extra={"chat_id": chat_id, "message_id": stored_message_id, "error": str(exc)},
            )
            sent = None
    if sent is None:
        sent = await bot.send_message(
            chat_id=chat_id,
            text=text,
            reply_markup=markup,
            disable_web_page_preview=True,
        )
    await _store_message_id(ctx, user_id, getattr(sent, "message_id", None))


async def profile_history(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await _answer_callback(update)
    chat = getattr(update, "effective_chat", None)
    user = getattr(update, "effective_user", None)
    chat_id = getattr(chat, "id", None)
    user_id = getattr(user, "id", None)
    await _delete_stored_message(ctx, chat_id, user_id)

    entries = await _load_history_entries(int(user_id)) if user_id is not None else []
    lines = list(entries or [])
    if not lines:
        text = "📜 История операций пока пуста."
    else:
        text = "📜 История операций:\n" + _render_history_lines(lines[:5])

    bot = getattr(ctx, "bot", None)
    if bot is None or chat_id is None:
        return
    await bot.send_message(
        chat_id=chat_id,
        text=text,
        reply_markup=_back_keyboard(),
        disable_web_page_preview=True,
    )


async def profile_invite(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await _answer_callback(update)
    chat = getattr(update, "effective_chat", None)
    user = getattr(update, "effective_user", None)
    chat_id = getattr(chat, "id", None)
    user_id = getattr(user, "id", None)
    await _delete_stored_message(ctx, chat_id, user_id)

    bot = getattr(ctx, "bot", None)
    if bot is None or chat_id is None:
        return
    await bot.send_message(
        chat_id=chat_id,
        text=_invite_text(),
        reply_markup=_back_keyboard(),
        disable_web_page_preview=True,
    )


async def profile_topup(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await _answer_callback(update)
    await open_stars_menu(update, ctx, edit_message=True, source="profile")


async def profile_back(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await _answer_callback(update)
    chat = getattr(update, "effective_chat", None)
    user = getattr(update, "effective_user", None)
    chat_id = getattr(chat, "id", None)
    user_id = getattr(user, "id", None)
    await _delete_stored_message(ctx, chat_id, user_id)

    from bot import handle_menu  # avoid circular import at module load

    await handle_menu(update, ctx, notify_chat_off=False)


__all__ = [
    "profile_back",
    "profile_history",
    "profile_invite",
    "profile_open",
    "profile_topup",
]
