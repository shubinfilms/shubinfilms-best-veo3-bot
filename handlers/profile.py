from __future__ import annotations

import logging
import time
from collections.abc import MutableMapping, Sequence
from datetime import datetime
from typing import Any, Optional

from telegram import InlineKeyboardButton, Update
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import ContextTypes

from core.balance_provider import BalanceSnapshot
from keyboards import menu_pay_unified
from redis_utils import get_ledger_entries
from state import state
from texts import TXT_TOPUP_CHOOSE
from ui.card import build_card
from utils.telegram_safe import safe_edit_message

log = logging.getLogger(__name__)

PROMO_WAIT_KIND = "promo_code"
PROMO_WAIT_TTL = 180  # seconds

_CHATDATA_PROFILE_KEY = "profile_card"
_CHATDATA_HISTORY_HASH = "history_hash"
_CHATDATA_WAIT_KIND = "wait_kind"
_CHATDATA_WAIT_UNTIL = "wait_until"


def _with_nav_event(ctx: ContextTypes.DEFAULT_TYPE) -> tuple[MutableMapping[str, Any] | None, bool]:
    chat_data_obj = getattr(ctx, "chat_data", None)
    if isinstance(chat_data_obj, MutableMapping):
        chat_data_obj["nav_event"] = True
        return chat_data_obj, True
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


def _history_hash(chat_data: MutableMapping[str, Any] | None) -> Optional[tuple[str, int]]:
    if not isinstance(chat_data, MutableMapping):
        return None
    profile_state = chat_data.setdefault(_CHATDATA_PROFILE_KEY, {})
    if not isinstance(profile_state, MutableMapping):
        return None
    cached = profile_state.get(_CHATDATA_HISTORY_HASH)
    if isinstance(cached, tuple) and len(cached) == 2:
        return cached  # type: ignore[return-value]
    return None


def _store_history_hash(chat_data: MutableMapping[str, Any] | None, value: tuple[str, int]) -> None:
    if not isinstance(chat_data, MutableMapping):
        return
    profile_state = chat_data.setdefault(_CHATDATA_PROFILE_KEY, {})
    if not isinstance(profile_state, MutableMapping):
        profile_state = {}
        chat_data[_CHATDATA_PROFILE_KEY] = profile_state
    profile_state[_CHATDATA_HISTORY_HASH] = value


def _format_history_entry(entry: dict[str, Any]) -> Optional[str]:
    try:
        entry_type = str(entry.get("type", ""))
    except Exception:
        entry_type = ""

    amount_raw = entry.get("amount")
    try:
        amount = abs(int(amount_raw))
    except (TypeError, ValueError):
        amount = 0

    if entry_type == "debit":
        icon, amount_text = "➖", f"−{amount}"
    elif entry_type == "refund":
        icon, amount_text = "↩️", f"+{amount}"
    else:
        icon, amount_text = "➕", f"+{amount}"

    reason = entry.get("reason")
    if isinstance(reason, str):
        reason = " ".join(reason.split())
    else:
        meta = entry.get("meta")
        if isinstance(meta, dict):
            meta_reason = meta.get("model")
            reason = str(meta_reason).strip() if isinstance(meta_reason, str) else ""
        else:
            reason = ""
    reason_text = reason or "—"

    ts_value = entry.get("ts")
    try:
        ts = datetime.fromtimestamp(float(ts_value))
        ts_text = ts.strftime("%d.%m %H:%M")
    except (TypeError, ValueError):
        ts_text = "—"

    balance_after = entry.get("balance_after")
    try:
        balance_text = f"{int(balance_after)}"
    except (TypeError, ValueError):
        balance_text = "—"

    return f"{icon} {amount_text}💎 • {reason_text} • {ts_text} • Баланс: {balance_text}💎"


def _build_history_card(entries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    lines: list[str] = []
    for entry in entries:
        if isinstance(entry, dict):
            row = _format_history_entry(entry)
            if row:
                lines.append(row)

    if lines:
        body: Sequence[str] | None = lines
    else:
        body = ("История операций пока пуста.",)

    markup_rows = [[InlineKeyboardButton("⬅️ Назад", callback_data="profile:menu")]]
    return build_card(
        "🧾 История операций",
        "Последние операции",
        markup_rows,
        body_lines=body,
    )


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
            log_on_noop="profile.card.noop",
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
    chat_data, flag = _with_nav_event(ctx)
    try:
        state(ctx)["mode"] = "profile"
        from bot import open_profile_card  # lazy import to avoid cycles

        await open_profile_card(update, ctx, edit=True)
    finally:
        _clear_nav_event(chat_data, flag)


async def _show_payment_error(
    ctx: ContextTypes.DEFAULT_TYPE,
    chat_id: Optional[int],
    message_id: Optional[int],
) -> None:
    payload = build_card(
        "💎 Пополнение",
        "Пополнение временно недоступно, попробуйте позже.",
        [[InlineKeyboardButton("⬅️ Назад", callback_data="profile:menu")]],
    )
    await _edit_card(ctx, chat_id, message_id, payload)


async def on_profile_topup(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    chat_data, flag = _with_nav_event(ctx)
    try:
        chat_id, message_id = _callback_target(update)
        payload = {
            "text": TXT_TOPUP_CHOOSE,
            "reply_markup": menu_pay_unified(),
            "parse_mode": ParseMode.HTML,
            "disable_web_page_preview": True,
        }
        edited = await _edit_card(ctx, chat_id, message_id, payload)
        if not edited:
            await _show_payment_error(ctx, chat_id, message_id)
        else:
            state(ctx)["last_panel"] = "profile_topup"
    finally:
        _clear_nav_event(chat_data, flag)


async def on_profile_history(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    chat_data, flag = _with_nav_event(ctx)
    try:
        user = update.effective_user
        uid = user.id if user else None
        if uid is None:
            return

        entries = get_ledger_entries(uid, offset=0, limit=10)
        items_hash = hash(tuple(_format_history_entry(item) for item in entries))
        cached = _history_hash(chat_data)
        if cached == ("profile:history", items_hash):
            log.debug("profile.history.noop | user=%s", uid)
            return

        payload = _build_history_card(entries)
        chat_id, message_id = _callback_target(update)
        if await _edit_card(ctx, chat_id, message_id, payload):
            _store_history_hash(chat_data, ("profile:history", items_hash))
            state(ctx)["last_panel"] = "profile_history"
    finally:
        _clear_nav_event(chat_data, flag)


async def on_profile_invite(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    chat_data, flag = _with_nav_event(ctx)
    try:
        query = update.callback_query
        user = update.effective_user
        uid = user.id if user else None
        referral_url: Optional[str] = None
        if uid is not None:
            try:
                from bot import _build_referral_link  # type: ignore[attr-defined]

                referral_url = await _build_referral_link(uid, ctx)
            except Exception as exc:  # pragma: no cover - defensive
                log.warning("profile.invite.link_failed | user=%s err=%s", uid, exc)
                referral_url = None

        if query is None:
            return

        if referral_url:
            try:
                await query.answer(url=referral_url)
            except BadRequest as exc:
                log.debug("profile.invite.answer_warn | user=%s err=%s", uid, exc)
                try:
                    await query.answer("Ссылка скопирована ниже", show_alert=True)
                except Exception:
                    pass
                chat_id, _ = _callback_target(update)
                if chat_id is not None:
                    await ctx.bot.send_message(
                        chat_id,
                        f"🔗 Пригласительная ссылка:\n{referral_url}",
                        disable_web_page_preview=True,
                    )
            except Exception:
                log.exception("profile.invite.answer_failed | user=%s", uid)
            return

        try:
            await query.answer("Не удалось получить ссылку, попробуйте позже.", show_alert=True)
        except Exception:
            pass
    finally:
        _clear_nav_event(chat_data, flag)


async def on_profile_promo_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    chat_data, flag = _with_nav_event(ctx)
    try:
        chat_id, message_id = _callback_target(update)
        payload = build_card(
            "🎁 Промокод",
            "Введите код одним сообщением…",
            [[InlineKeyboardButton("⬅️ Назад", callback_data="profile:menu")]],
            body_lines=("Отправьте промокод в чат.",),
        )
        if await _edit_card(ctx, chat_id, message_id, payload):
            state(ctx)["last_panel"] = "profile_promo"
            _activate_promo_wait(ctx, chat_id, message_id)
    finally:
        _clear_nav_event(chat_data, flag)


def _activate_promo_wait(
    ctx: ContextTypes.DEFAULT_TYPE,
    chat_id: Optional[int],
    card_message_id: Optional[int],
) -> None:
    chat_data_obj = getattr(ctx, "chat_data", None)
    if not isinstance(chat_data_obj, MutableMapping):
        return
    chat_data_obj[_CHATDATA_WAIT_KIND] = PROMO_WAIT_KIND
    chat_data_obj[_CHATDATA_WAIT_UNTIL] = time.time() + PROMO_WAIT_TTL
    wait_meta = {
        "chat_id": chat_id,
        "card_msg_id": card_message_id,
    }
    chat_data_obj.setdefault(_CHATDATA_PROFILE_KEY, {})["promo_wait"] = wait_meta


def clear_promo_wait(ctx: ContextTypes.DEFAULT_TYPE) -> None:
    chat_data_obj = getattr(ctx, "chat_data", None)
    if not isinstance(chat_data_obj, MutableMapping):
        return
    chat_data_obj.pop(_CHATDATA_WAIT_KIND, None)
    chat_data_obj.pop(_CHATDATA_WAIT_UNTIL, None)
    profile_state = chat_data_obj.get(_CHATDATA_PROFILE_KEY)
    if isinstance(profile_state, MutableMapping):
        profile_state.pop("promo_wait", None)


def is_waiting_for_promo(ctx: ContextTypes.DEFAULT_TYPE) -> bool:
    chat_data_obj = getattr(ctx, "chat_data", None)
    if not isinstance(chat_data_obj, MutableMapping):
        return False
    if chat_data_obj.get(_CHATDATA_WAIT_KIND) != PROMO_WAIT_KIND:
        return False
    deadline = chat_data_obj.get(_CHATDATA_WAIT_UNTIL)
    try:
        return float(deadline) > time.time()
    except (TypeError, ValueError):
        return False


async def on_profile_promo_apply(
    update: Update,
    ctx: ContextTypes.DEFAULT_TYPE,
    code: str,
) -> None:
    chat_data, flag = _with_nav_event(ctx)
    try:
        clear_promo_wait(ctx)
        state(ctx)["mode"] = None

        user = update.effective_user
        chat = update.effective_chat
        message = update.effective_message
        if user is None or chat is None or message is None:
            return

        code_input = code.strip()
        if not code_input:
            await message.reply_text("⚠️ Отправьте промокод текстом.")
            return

        try:
            from bot import _cache_balance_snapshot, _set_cached_balance, activate_fixed_promo
        except ImportError:  # pragma: no cover - defensive
            log.exception("profile.promo.import_failed")
            await message.reply_text("⚠️ Временная ошибка, попробуйте позже.")
            return

        status, balance_after = activate_fixed_promo(user.id, code_input)
        if status == "invalid":
            await message.reply_text("Такого промокода нет.")
            await on_profile_menu(update, ctx)
            return
        if status == "already_used":
            await message.reply_text("⚠️ Этот промокод уже был использован.")
            await on_profile_menu(update, ctx)
            return
        if status != "ok" or balance_after is None:
            await message.reply_text("⚠️ Временная ошибка, попробуйте позже.")
            await on_profile_menu(update, ctx)
            return

        snapshot = BalanceSnapshot(value=balance_after, display=str(int(balance_after)))
        _cache_balance_snapshot(ctx, user.id, snapshot)
        _set_cached_balance(ctx, balance_after)

        await message.reply_text(f"✅ Промокод активирован! Баланс: {balance_after}💎")

        await on_profile_menu(update, ctx)
    finally:
        _clear_nav_event(chat_data, flag)


async def handle_promo_timeout(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    clear_promo_wait(ctx)
    state(ctx)["mode"] = None
    message = update.effective_message
    if message is not None:
        await message.reply_text("⏳ Ожидание промокода истекло.")
    await on_profile_menu(update, ctx)

