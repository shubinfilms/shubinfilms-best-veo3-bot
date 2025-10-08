from __future__ import annotations

import inspect
import logging
import time
from collections.abc import MutableMapping, Sequence
from contextlib import suppress
from datetime import datetime
from typing import Any, Optional

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import ContextTypes

from ui.card import build_card
from utils.telegram_safe import safe_edit_message

log = logging.getLogger(__name__)

PROMO_WAIT_KIND = "promo"
_PROMO_WAIT_KEY = "profile_wait_state"
_PROMO_WAIT_UNTIL_KEY = "profile_wait_until"
PROMO_WAIT_TTL = 180

_HISTORY_TYPE_LABELS = {
    "credit": "Пополнение",
    "debit": "Списание",
    "refund": "Возврат",
}


def _chat_data(ctx: ContextTypes.DEFAULT_TYPE) -> MutableMapping[str, Any] | None:
    obj = getattr(ctx, "chat_data", None)
    return obj if isinstance(obj, MutableMapping) else None


def _set_nav_event(ctx: ContextTypes.DEFAULT_TYPE) -> tuple[MutableMapping[str, Any] | None, bool]:
    chat_data = _chat_data(ctx)
    if chat_data is not None:
        chat_data["nav_event"] = True
        log.info("nav.event (source=callback)")
        return chat_data, True
    return None, False


def _clear_nav_event(chat_data: MutableMapping[str, Any] | None, flag: bool) -> None:
    if flag and isinstance(chat_data, MutableMapping):
        chat_data.pop("nav_event", None)


def _callback_target(update: Update) -> tuple[Optional[int], Optional[int]]:
    query = update.callback_query
    if query and query.message:
        return query.message.chat_id, query.message.message_id
    message = update.effective_message
    if message is not None:
        return message.chat_id, message.message_id
    chat = update.effective_chat
    chat_id = getattr(chat, "id", None)
    return chat_id, None


async def _edit_card(
    ctx: ContextTypes.DEFAULT_TYPE,
    chat_id: Optional[int],
    message_id: Optional[int],
    payload: dict[str, Any],
) -> bool:
    if chat_id is None or message_id is None:
        return False
    try:
        return await safe_edit_message(
            ctx,
            chat_id,
            message_id,
            payload.get("text", ""),
            payload.get("reply_markup"),
            parse_mode=payload.get("parse_mode", ParseMode.HTML),
            disable_web_page_preview=payload.get("disable_web_page_preview", True),
        )
    except BadRequest as exc:
        log.warning(
            "profile.card.edit_failed | chat=%s mid=%s err=%s",
            chat_id,
            message_id,
            exc,
        )
    except Exception:
        log.exception("profile.card.edit_failed | chat=%s mid=%s", chat_id, message_id)
    return False


async def on_profile_menu(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    chat_data, flag = _set_nav_event(ctx)
    try:
        from bot import open_profile_card  # lazy import to avoid circular deps

        await open_profile_card(update, ctx, suppress_nav=False, edit=True)
        log.info("profile.action", extra={"action": "menu", "result": "opened"})
    finally:
        _clear_nav_event(chat_data, flag)


def _topup_url() -> Optional[str]:
    try:
        import settings as app_settings
    except Exception:  # pragma: no cover - defensive
        return None
    return getattr(app_settings, "TOPUP_URL", None)


async def on_profile_topup(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    chat_data, flag = _set_nav_event(ctx)
    try:
        query = update.callback_query
        if query is not None:
            with suppress(BadRequest):
                await query.answer()

        chat_id, message_id = _callback_target(update)

        url = _topup_url()
        if url:
            subtitle = "Пополнить баланс можно по кнопке ниже."
            rows = [
                [InlineKeyboardButton("Перейти к оплате", url=url)],
                [InlineKeyboardButton("⬅️ Назад", callback_data="profile:menu")],
            ]
        else:
            subtitle = "Пополнить баланс — в разработке."
            rows = [[InlineKeyboardButton("⬅️ Назад", callback_data="profile:menu")]]
        payload = build_card("💎 Пополнение", subtitle, rows)

        success = await _edit_card(ctx, chat_id, message_id, payload)
        log.info(
            "profile.action",
            extra={"action": "topup", "result": "ok" if success else "skipped"},
        )
    finally:
        _clear_nav_event(chat_data, flag)


async def _billing_history(user_id: int) -> list[dict[str, Any]]:
    try:
        import billing
    except ImportError:
        return []

    get_history = getattr(billing, "get_history", None)
    if get_history is None:
        return []

    try:
        result = get_history(user_id)
    except Exception:
        log.exception("profile.history.error | user=%s", user_id)
        return []

    if inspect.isawaitable(result):
        result = await result
    try:
        return list(result)
    except TypeError:
        return []


def _format_timestamp(value: Any) -> str:
    ts: Optional[datetime] = None
    if isinstance(value, (int, float)):
        try:
            ts = datetime.fromtimestamp(float(value))
        except Exception:
            ts = None
    elif isinstance(value, str):
        text = value.strip()
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                ts = datetime.strptime(text.split(".")[0], fmt)
                break
            except Exception:
                continue
        else:
            try:
                num = float(text)
            except Exception:
                num = None
            if num is not None:
                try:
                    ts = datetime.fromtimestamp(num)
                except Exception:
                    ts = None
    if ts is None:
        return "—"
    return ts.strftime("%d.%m.%Y")


def _format_history_entry(entry: dict[str, Any]) -> Optional[str]:
    try:
        raw_type = str(entry.get("type", "")).lower()
    except Exception:
        raw_type = ""
    label = _HISTORY_TYPE_LABELS.get(raw_type, raw_type.capitalize() or "Операция")

    amount_raw = entry.get("amount")
    try:
        amount = int(amount_raw)
    except Exception:
        amount = 0
    sign = "−" if amount < 0 else "+"
    amount_text = f"{sign}{abs(amount)}💎"

    ts_value = entry.get("created_at")
    if ts_value is None:
        ts_value = entry.get("timestamp")
    if ts_value is None:
        ts_value = entry.get("ts")
    date_text = _format_timestamp(ts_value)

    return f"{date_text} • {label} • {amount_text}"


async def on_profile_history(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    chat_data, flag = _set_nav_event(ctx)
    try:
        user = update.effective_user
        uid = user.id if user else None
        if uid is None:
            return

        history = await _billing_history(uid)
        entries = [_format_history_entry(item) for item in history[-5:]]
        lines = [item for item in entries if item]

        if lines:
            log.info("profile.history.count=%s", len(lines), extra={"user": uid})
            body: Sequence[str] | None = lines
        else:
            log.info("profile.history.empty", extra={"user": uid})
            body = ("История операций пока пуста.",)

        rows = [[InlineKeyboardButton("⬅️ Назад", callback_data="profile:menu")]]
        payload = build_card("🧾 История операций", "Последние операции", rows, body_lines=body)

        chat_id, message_id = _callback_target(update)
        success = await _edit_card(ctx, chat_id, message_id, payload)
        log.info(
            "profile.action",
            extra={"action": "history", "result": "ok" if success else "skipped"},
        )
    finally:
        _clear_nav_event(chat_data, flag)


def _bot_name() -> Optional[str]:
    try:
        import settings as app_settings
    except Exception:  # pragma: no cover - defensive
        return None
    name = getattr(app_settings, "BOT_NAME", None)
    if isinstance(name, str) and name:
        return name
    username = getattr(app_settings, "BOT_USERNAME", None)
    if isinstance(username, str) and username:
        return username
    return None


async def on_profile_invite(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    chat_data, flag = _set_nav_event(ctx)
    try:
        query = update.callback_query
        if query is not None:
            with suppress(BadRequest):
                await query.answer()

        user = update.effective_user
        chat_id = getattr(update.effective_chat, "id", None)
        if user is None or chat_id is None:
            return

        bot_name = _bot_name()
        if not bot_name:
            log.warning("profile.invite.no_bot_name | user=%s", user.id)
            return

        link = f"https://t.me/{bot_name}?start=ref_{user.id}"
        text = (
            "Пригласите друга — получите бонус после его первой оплаты.\n\n"
            f"Ваша ссылка: {link}"
        )
        markup = InlineKeyboardMarkup(
            [[InlineKeyboardButton("Скопировать ссылку", url=link)]]
        )
        await ctx.bot.send_message(
            chat_id=chat_id,
            text=text,
            reply_markup=markup,
            disable_web_page_preview=True,
        )
        log.info("profile.invite.sent", extra={"user": user.id})
        log.info("profile.action", extra={"action": "invite", "result": "sent"})
    finally:
        _clear_nav_event(chat_data, flag)


def _activate_promo_wait(ctx: ContextTypes.DEFAULT_TYPE) -> None:
    chat_data = _chat_data(ctx)
    if chat_data is None:
        return
    chat_data[_PROMO_WAIT_KEY] = PROMO_WAIT_KIND
    chat_data[_PROMO_WAIT_UNTIL_KEY] = time.time() + PROMO_WAIT_TTL


def clear_promo_wait(ctx: ContextTypes.DEFAULT_TYPE) -> None:
    chat_data = _chat_data(ctx)
    if chat_data is None:
        return
    chat_data.pop(_PROMO_WAIT_KEY, None)
    chat_data.pop(_PROMO_WAIT_UNTIL_KEY, None)


def is_waiting_for_promo(ctx: ContextTypes.DEFAULT_TYPE) -> bool:
    chat_data = _chat_data(ctx)
    if chat_data is None:
        return False
    if chat_data.get(_PROMO_WAIT_KEY) != PROMO_WAIT_KIND:
        return False
    deadline = chat_data.get(_PROMO_WAIT_UNTIL_KEY)
    try:
        return float(deadline) > time.time()
    except (TypeError, ValueError):
        return False


async def on_profile_promo_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    chat_data, flag = _set_nav_event(ctx)
    try:
        query = update.callback_query
        if query is not None:
            with suppress(BadRequest):
                await query.answer()

        rows = [[InlineKeyboardButton("⬅️ Назад", callback_data="profile:menu")]]
        payload = build_card(
            "🎁 Промокод",
            "Введите код одним сообщением.",
            rows,
            body_lines=("Отправьте промокод в чат.",),
        )

        chat_id, message_id = _callback_target(update)
        success = await _edit_card(ctx, chat_id, message_id, payload)
        if success:
            _activate_promo_wait(ctx)
            user = update.effective_user
            log.info("profile.promo.wait", extra={"user": getattr(user, "id", None)})
        log.info(
            "profile.action",
            extra={"action": "promo", "result": "waiting" if success else "skipped"},
        )
    finally:
        _clear_nav_event(chat_data, flag)


async def _apply_promo(user_id: int, code: str) -> bool:
    try:
        import promo
    except ImportError:
        return code.strip().upper().startswith("VE")

    apply_func = getattr(promo, "apply", None)
    if apply_func is None:
        return code.strip().upper().startswith("VE")

    try:
        result = apply_func(user_id, code)
    except Exception:
        log.exception("profile.promo.apply_failed | user=%s", user_id)
        return False

    if inspect.isawaitable(result):
        result = await result
    return bool(result)


async def on_profile_promo_apply(
    update: Update,
    ctx: ContextTypes.DEFAULT_TYPE,
    code: str,
) -> None:
    clear_promo_wait(ctx)

    user = update.effective_user
    chat = update.effective_chat
    message = update.effective_message
    if user is None or chat is None or message is None:
        return

    clean_code = code.strip()
    if not clean_code:
        await message.reply_text("❌ Неверный промокод")
        log.info("profile.promo.bad", extra={"user": user.id, "code": clean_code})
        return

    ok = await _apply_promo(user.id, clean_code)
    if ok:
        await message.reply_text("✅ Промокод активирован")
        log.info("profile.promo.ok", extra={"user": user.id, "code": clean_code})
        log.info("profile.action", extra={"action": "promo_apply", "result": "ok"})
    else:
        await message.reply_text("❌ Неверный промокод")
        log.info("profile.promo.bad", extra={"user": user.id, "code": clean_code})
        log.info("profile.action", extra={"action": "promo_apply", "result": "bad"})

    await on_profile_menu(update, ctx)


async def handle_promo_timeout(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    clear_promo_wait(ctx)
    message = update.effective_message
    if message is not None:
        await message.reply_text("⏳ Ожидание промокода истекло.")
    log.info("profile.action", extra={"action": "promo_timeout", "result": "expired"})
    await on_profile_menu(update, ctx)


__all__ = [
    "PROMO_WAIT_KIND",
    "PROMO_WAIT_TTL",
    "clear_promo_wait",
    "handle_promo_timeout",
    "is_waiting_for_promo",
    "on_profile_history",
    "on_profile_invite",
    "on_profile_menu",
    "on_profile_promo_apply",
    "on_profile_promo_start",
    "on_profile_topup",
]
