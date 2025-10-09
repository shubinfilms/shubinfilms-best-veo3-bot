from __future__ import annotations

import inspect
import logging
import time
from collections.abc import MutableMapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from html import escape
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Message, Update
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import ContextTypes

from ui.card import build_card
from utils.input_state import (
    WaitInputState,
    WaitKind,
    clear_wait_state,
    get_wait_state,
    input_state,
    set_wait_state,
)
log = logging.getLogger(__name__)

PROMO_WAIT_KIND = WaitKind.PROMO_CODE.value
_PROMO_WAIT_KEY = "profile_wait_state"
_PROMO_WAIT_UNTIL_KEY = "profile_wait_until"
_PROMO_WAIT_USER_KEY = "profile_wait_user_id"
PROMO_WAIT_TTL = 180
_PROFILE_LOCK_KEY = "profile_open_in_progress"
PROFILE_MSG_ID = "profile_msg_id"
PROFILE_OPEN_AT = "profile_open_at"
NAV_UNTIL = "nav_active_until"


_HISTORY_TYPE_LABELS = {
    "credit": "Пополнение",
    "debit": "Списание",
    "refund": "Возврат",
}


def _as_html(text: str) -> str:
    if not text:
        return ""
    lines = [escape(line) for line in str(text).splitlines()]
    return "<br/>".join(lines)


def render_profile_root(ctx: ContextTypes.DEFAULT_TYPE, data: dict | None = None) -> tuple[str, InlineKeyboardMarkup]:
    return render_profile_view(ctx, "root", data=data)


def render_profile_view(
    ctx: ContextTypes.DEFAULT_TYPE,
    view: str,
    data: dict | None = None,
) -> tuple[str, InlineKeyboardMarkup]:
    normalized = (view or "root").strip().lower()
    payload: dict = data or {}

    if normalized == "root":
        return _render_root_view(ctx, payload)
    if normalized == "topup":
        return _render_topup_view(payload)
    if normalized == "history":
        return _render_history_view(payload)
    if normalized == "invite":
        return _render_invite_view(payload)
    if normalized == "promo":
        return _render_promo_view()

    return _render_unknown_view(normalized)


def _render_root_view(
    ctx: ContextTypes.DEFAULT_TYPE,
    payload: dict,
) -> tuple[str, InlineKeyboardMarkup]:
    snapshot = payload.get("snapshot")
    referral_url = payload.get("referral_url")

    chat_state = _chat_data(ctx)
    state_payload: dict[str, Any] | None
    if isinstance(chat_state, MutableMapping):
        raw_state = chat_state.get("profile_render_state")
        state_payload = raw_state if isinstance(raw_state, dict) else None
    else:
        state_payload = None

    if snapshot is None:
        from bot import _resolve_balance_snapshot, get_user_id

        target = payload.get("snapshot_target")
        if target is None and state_payload is not None:
            target = state_payload.get("snapshot_target")
        if target is None:
            target = get_user_id(ctx)
        chat_id = payload.get("chat_id")
        if chat_id is None and state_payload is not None:
            chat_id = state_payload.get("chat_id")
        if target is None and chat_id is not None:
            target = chat_id
        if target is not None:
            snapshot = _resolve_balance_snapshot(ctx, int(target), prefer_cached=True)

    if snapshot is not None:
        from bot import _profile_balance_text

        text = _profile_balance_text(snapshot)
    else:
        text = "👤 Профиль\n💎 Баланс недоступен"

    if referral_url is None:
        if state_payload is not None:
            referral_url = state_payload.get("referral_url")
        if referral_url is None:
            referral_url = payload.get("referral_url_cached")

    from bot import balance_menu_kb

    markup = balance_menu_kb(referral_url=referral_url)
    return _as_html(text), markup


def _render_topup_view(payload: dict) -> tuple[str, InlineKeyboardMarkup]:
    topup_url = payload.get("topup_url") or ""
    rows: list[list[InlineKeyboardButton]] = []
    body: Sequence[str] | None
    subtitle: str

    if topup_url:
        rows.append([InlineKeyboardButton("Пополнить баланс", url=str(topup_url))])
        subtitle = "Пополнение через сайт"
        body = ("Откроется безопасная страница оплаты.",)
    else:
        subtitle = "Скоро"
        rows.append([InlineKeyboardButton("Пополнить баланс", callback_data="noop")])
        body = ("Пополнение будет доступно позже.",)

    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="profile:back")])
    card = build_card("💎 Пополнение", subtitle, rows, body_lines=body)
    return card["text"], card["reply_markup"]


def _render_history_view(payload: dict) -> tuple[str, InlineKeyboardMarkup]:
    entries = payload.get("entries") or []
    formatted: list[str] = []
    for entry in entries[-5:]:
        line = _format_history_entry(entry)
        if line:
            formatted.append(line)

    if not formatted:
        formatted = ["История операций пока пуста."]

    rows = [[InlineKeyboardButton("⬅️ Назад", callback_data="profile:back")]]
    card = build_card("🧾 История операций", "Последние операции", rows, body_lines=formatted)
    return card["text"], card["reply_markup"]


def _render_invite_view(payload: dict) -> tuple[str, InlineKeyboardMarkup]:
    link = payload.get("invite_link")
    rows: list[list[InlineKeyboardButton]] = []
    if link:
        rows.append([InlineKeyboardButton("Скопировать ссылку", url=str(link))])
        body = (f"Ваша ссылка: {link}",)
    else:
        rows.append([InlineKeyboardButton("Скопировать ссылку", callback_data="noop")])
        body = ("Ссылка станет доступна позже — мы работаем над запуском.",)

    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="profile:back")])
    card = build_card(
        "👥 Пригласить друга",
        "Поделитесь ссылкой и получите бонус после первой оплаты друга.",
        rows,
        body_lines=body,
    )
    return card["text"], card["reply_markup"]


def _render_promo_view() -> tuple[str, InlineKeyboardMarkup]:
    rows = [[InlineKeyboardButton("⬅️ Назад", callback_data="profile:back")]]
    body = (
        "Отправьте промокод в чат одним сообщением.",
        "Мы автоматически проверим и начислим бонусы.",
    )
    card = build_card("🎁 Промокод", "Как активировать код", rows, body_lines=body)
    return card["text"], card["reply_markup"]


def _render_unknown_view(view: str) -> tuple[str, InlineKeyboardMarkup]:
    rows = [[InlineKeyboardButton("⬅️ Назад", callback_data="profile:back")]]
    body = ("Этот раздел пока недоступен. Попробуйте позже.",)
    card = build_card(
        "ℹ️ Профиль",
        f"Раздел {view or '—'} временно недоступен",
        rows,
        body_lines=body,
    )
    return card["text"], card["reply_markup"]



async def profile_update_or_send(
    update: Update,
    ctx: ContextTypes.DEFAULT_TYPE,
    text: str,
    markup: InlineKeyboardMarkup,
    *,
    parse_mode: ParseMode = ParseMode.HTML,
) -> Message | None:
    chat = update.effective_chat
    message = update.effective_message
    chat_id = getattr(chat, "id", None)
    if chat_id is None and message is not None:
        chat_id = getattr(message, "chat_id", None)

    if chat_id is None:
        log.warning("profile.edit.fallback", extra={"reason": "missing_chat"})
        return None

    chat_data = _chat_data(ctx)
    msg_id = _get_profile_msg_id(chat_data)

    try:
        if msg_id:
            result = await ctx.bot.edit_message_text(
                chat_id=chat_id,
                message_id=msg_id,
                text=text,
                reply_markup=markup,
                parse_mode=parse_mode,
                disable_web_page_preview=True,
            )
            if isinstance(chat_data, MutableMapping):
                chat_data[PROFILE_MSG_ID] = getattr(result, "message_id", msg_id)
            log.info(
                "profile.edit.ok",
                extra={"chat_id": chat_id, "msg_id": msg_id},
            )
            return result
        raise BadRequest("no previous profile message")
    except BadRequest as exc:
        log.warning(
            "profile.edit.fallback",
            extra={"chat_id": chat_id, "msg_id": msg_id, "error": str(exc)},
        )
        result = await ctx.bot.send_message(
            chat_id,
            text,
            reply_markup=markup,
            parse_mode=parse_mode,
            disable_web_page_preview=True,
        )
        if isinstance(chat_data, MutableMapping):
            chat_data[PROFILE_MSG_ID] = getattr(result, "message_id", None)
        return result

def _chat_data(ctx: ContextTypes.DEFAULT_TYPE) -> MutableMapping[str, Any] | None:
    obj = getattr(ctx, "chat_data", None)
    return obj if isinstance(obj, MutableMapping) else None


def _set_nav_event(
    ctx: ContextTypes.DEFAULT_TYPE,
    *,
    source: str | None = None,
) -> tuple[MutableMapping[str, Any] | None, bool, bool]:
    chat_data = _chat_data(ctx)
    previous_flag = getattr(ctx, "nav_event", False)
    setattr(ctx, "nav_event", True)
    if isinstance(chat_data, MutableMapping):
        chat_data["nav_in_progress"] = True
        chat_data["nav_event"] = True
        chat_data[NAV_UNTIL] = time.monotonic() + 2.0
    if source:
        log.info("nav.event (source=%s)", source)
    return (
        chat_data if isinstance(chat_data, MutableMapping) else None,
        isinstance(chat_data, MutableMapping),
        previous_flag,
    )


def _clear_nav_event(
    chat_data: MutableMapping[str, Any] | None,
    flag: bool,
    ctx: ContextTypes.DEFAULT_TYPE,
    previous_nav: bool,
) -> None:
    if flag and isinstance(chat_data, MutableMapping):
        chat_data["nav_in_progress"] = False
        chat_data.pop("nav_event", None)
    setattr(ctx, "nav_event", previous_nav)


async def handle_profile_view(
    update: Update,
    ctx: ContextTypes.DEFAULT_TYPE,
    view: str,
) -> Message | None:
    normalized = (view or "root").strip().lower()
    if normalized == "back":
        normalized = "root"

    source = f"profile_{normalized}"
    chat_data, flag, previous_nav = _set_nav_event(ctx, source=source)
    try:
        query = update.callback_query
        if query is not None:
            with suppress(BadRequest):
                await query.answer()

        log.info("profile.cb", extra={"view": normalized})

        if normalized != "promo":
            clear_promo_wait(ctx)

        data: dict[str, Any]
        if normalized == "root":
            data = await _prepare_root_payload(update, ctx)
        elif normalized == "topup":
            topup_url = _topup_url() or ""
            if not topup_url:
                log.warning("profile.view.no_topup_url")
            data = {"topup_url": topup_url}
        elif normalized == "history":
            user_id = _resolve_profile_user_id(update, ctx)
            history = await _billing_history(user_id) if user_id is not None else []
            if not history:
                log.warning("profile.view.empty_history", extra={"user": user_id})
            data = {"entries": history}
        elif normalized == "invite":
            user_id = _resolve_profile_user_id(update, ctx)
            bot_name = _bot_name()
            invite_link: Optional[str]
            if bot_name and user_id:
                invite_link = f"https://t.me/{bot_name}?start=ref_{user_id}"
            else:
                invite_link = None
                if not bot_name:
                    log.warning("profile.view.no_bot_name")
            data = {"invite_link": invite_link}
        elif normalized == "promo":
            data = {}
        else:
            data = {"view": normalized}

        text, markup = render_profile_view(ctx, normalized, data=data)
        result = await profile_update_or_send(update, ctx, text, markup)
        if result is None:
            return None

        chat_state = _chat_data(ctx)
        if isinstance(chat_state, MutableMapping):
            chat_state["profile_last_view"] = normalized

        if normalized == "root":
            _store_root_context(ctx, data, result, update)
        elif normalized == "promo":
            user_id = _resolve_profile_user_id(update, ctx)
            chat = update.effective_chat
            chat_id = getattr(chat, "id", None)
            msg_id = getattr(result, "message_id", None)
            if user_id is not None and chat_id is not None and msg_id is not None:
                _activate_promo_wait(
                    ctx,
                    user_id=int(user_id),
                    chat_id=int(chat_id),
                    message_id=int(msg_id),
                )

        return result
    finally:
        _clear_nav_event(chat_data, flag, ctx, previous_nav)


async def on_profile_cbq(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    data = getattr(query, "data", "") if query else ""
    if not isinstance(data, str) or not data.startswith("profile:"):
        return

    _, _, raw_view = data.partition(":")
    await handle_profile_view(update, ctx, raw_view or "root")


async def _prepare_root_payload(
    update: Update,
    ctx: ContextTypes.DEFAULT_TYPE,
) -> dict[str, Any]:
    chat = update.effective_chat
    message = update.effective_message
    chat_id = getattr(chat, "id", None)
    if chat_id is None and message is not None:
        chat_id = getattr(message, "chat_id", None)

    user = update.effective_user
    user_id = getattr(user, "id", None)

    from bot import get_user_id

    resolved_user = user_id or get_user_id(ctx) or chat_id

    referral_url: Optional[str] = None
    if resolved_user is not None:
        from bot import _build_referral_link

        try:
            referral_url = await _build_referral_link(int(resolved_user), ctx)
        except Exception as exc:
            log.warning(
                "profile.view.referral_failed",
                extra={"user": resolved_user, "error": str(exc)},
            )
            referral_url = None

    if resolved_user is not None:
        snapshot_target: Optional[int] = int(resolved_user)
    elif chat_id is not None:
        snapshot_target = int(chat_id)
    else:
        snapshot_target = None

    snapshot = None
    if snapshot_target is not None:
        from bot import _resolve_balance_snapshot

        snapshot = _resolve_balance_snapshot(ctx, snapshot_target, prefer_cached=True)

    return {
        "snapshot": snapshot,
        "snapshot_target": snapshot_target,
        "referral_url": referral_url,
        "chat_id": chat_id,
    }


def _store_root_context(
    ctx: ContextTypes.DEFAULT_TYPE,
    payload: dict[str, Any],
    message: Message,
    update: Update,
) -> None:
    chat_state = _chat_data(ctx)
    if not isinstance(chat_state, MutableMapping):
        return

    chat_id = payload.get("chat_id")
    if chat_id is None:
        chat = update.effective_chat
        if chat is not None:
            chat_id = getattr(chat, "id", None)
        if chat_id is None:
            chat_id = getattr(message, "chat_id", None)

    chat_state["profile_render_state"] = {
        "snapshot_target": payload.get("snapshot_target"),
        "chat_id": chat_id,
        "referral_url": payload.get("referral_url"),
    }


def _resolve_profile_user_id(
    update: Update, ctx: ContextTypes.DEFAULT_TYPE
) -> Optional[int]:
    user = update.effective_user
    if user is not None:
        uid = getattr(user, "id", None)
        if uid is not None:
            return int(uid)

    from bot import get_user_id

    ctx_user = get_user_id(ctx)
    if ctx_user is not None:
        return int(ctx_user)

    chat = update.effective_chat
    if chat is not None:
        chat_id = getattr(chat, "id", None)
        if chat_id is not None:
            return int(chat_id)

    message = update.effective_message
    if message is not None:
        mid = getattr(message, "chat_id", None)
        if mid is not None:
            return int(mid)

    return None

async def profile_reset_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    chat_data = _chat_data(ctx)
    cleared_keys: list[str] = []
    if isinstance(chat_data, MutableMapping):
        for key in (PROFILE_MSG_ID, "profile_rendered_hash", PROFILE_OPEN_AT):
            if key in chat_data:
                value = chat_data.pop(key, None)
                if value is not None:
                    cleared_keys.append(key)
        chat_data.pop(_PROFILE_LOCK_KEY, None)

    chat = getattr(update, "effective_chat", None)
    chat_id = getattr(chat, "id", None)
    log.info(
        "profile.cache.cleared",
        extra={"chat_id": chat_id, "keys": cleared_keys},
    )

    message = update.effective_message
    if message is not None:
        try:
            await message.reply_text("♻️ Кэш профиля очищен.")
        except Exception:
            pass


@dataclass(slots=True, frozen=True)
class OpenedProfile:
    msg_id: Optional[int]
    reused: bool


def _get_profile_msg_id(chat_data: MutableMapping[str, Any] | None) -> Optional[int]:
    if not isinstance(chat_data, MutableMapping):
        return None
    raw_mid = chat_data.get(PROFILE_MSG_ID)
    try:
        return int(raw_mid) if raw_mid is not None else None
    except (TypeError, ValueError):
        return None


def _store_profile_msg_id(
    chat_data: MutableMapping[str, Any] | None, message_id: Optional[int]
) -> None:
    if not isinstance(chat_data, MutableMapping):
        return
    if isinstance(message_id, int):
        chat_data[PROFILE_MSG_ID] = int(message_id)
    else:
        chat_data.pop(PROFILE_MSG_ID, None)


async def open_profile(
    update: Update,
    ctx: ContextTypes.DEFAULT_TYPE,
    *,
    source: str,
    suppress_nav: bool = True,
) -> None:
    chat_data, flag, previous_nav = _set_nav_event(ctx, source=source)
    try:
        now = time.monotonic()
        if isinstance(chat_data, MutableMapping):
            raw_open_at = chat_data.get(PROFILE_OPEN_AT)
            try:
                last_open = float(raw_open_at)
            except (TypeError, ValueError):
                last_open = 0.0
            if now - last_open < 0.5:
                log.info(
                    "profile.open.debounced",
                    extra={"source": source, "delta": now - last_open},
                )
                return
            chat_data[PROFILE_OPEN_AT] = now
            chat_data.pop("wait_kind", None)
            try:
                previous_deadline = float(chat_data.get(NAV_UNTIL, 0.0))
            except (TypeError, ValueError):
                previous_deadline = 0.0
            chat_data[NAV_UNTIL] = max(previous_deadline, now + 2.0)
            chat_data["suppress_dialog_notice"] = True

        result = await open_profile_card(
            update,
            ctx,
            source=source,
            suppress_nav=suppress_nav,
        )

        reused = bool(result.reused) if isinstance(result, OpenedProfile) else False
        log.info(
            "profile.open",
            extra={"source": source, "reused_msg": reused},
        )
    finally:
        _clear_nav_event(chat_data, flag, ctx, previous_nav)


async def _open_profile_card_impl(
    chat_id: Optional[int],
    user_id: Optional[int],
    *,
    update: Update,
    ctx: ContextTypes.DEFAULT_TYPE,
    suppress_nav: bool = True,
    source: str = "menu",
) -> OpenedProfile:
    log.info(
        "profile.open(chat_id=%s, user_id=%s, source=%s, suppress_nav=%s)",
        chat_id,
        user_id,
        source,
        suppress_nav,
    )

    if user_id is not None:
        try:
            input_state.clear(int(user_id), reason="profile_nav")
        except Exception:
            log.debug(
                "profile.input_state_clear_failed",
                exc_info=True,
                extra={"user_id": user_id},
            )

    resolved_chat_id = chat_id
    if resolved_chat_id is None:
        chat = getattr(update, "effective_chat", None)
        if chat is None:
            message = getattr(update, "effective_message", None)
            chat = getattr(message, "chat", None)
        resolved_chat_id = getattr(chat, "id", None)

    chat_data = _chat_data(ctx)
    previous_mid = _get_profile_msg_id(chat_data)
    reuse_existing = previous_mid is not None

    if isinstance(chat_data, MutableMapping):
        chat_data["nav_event"] = True
        chat_data["nav_in_progress"] = True
        if chat_data.get(_PROFILE_LOCK_KEY):
            log.info(
                "profile.open.skip_duplicate",
                extra={"chat_id": resolved_chat_id, "user_id": user_id},
            )
            return OpenedProfile(msg_id=previous_mid, reused=True)
        chat_data[_PROFILE_LOCK_KEY] = True

    try:
        from bot import open_profile_card as _core_open_profile_card  # lazy import

        message_id = await _core_open_profile_card(
            update,
            ctx,
            suppress_nav=suppress_nav,
            edit=reuse_existing,
            force_new=not reuse_existing,
        )
    finally:
        if isinstance(chat_data, MutableMapping):
            chat_data.pop("nav_in_progress", None)
            chat_data.pop(_PROFILE_LOCK_KEY, None)

    if isinstance(message_id, int):
        _store_profile_msg_id(chat_data, message_id)
    reused_actual = bool(
        isinstance(message_id, int)
        and previous_mid is not None
        and message_id == previous_mid
    )

    log.info(
        "profile.opened reused=%s msg_id=%s",
        reused_actual,
        message_id,
    )
    return OpenedProfile(msg_id=message_id if isinstance(message_id, int) else None, reused=reused_actual)


async def open_profile_card(
    update: Update,
    ctx: ContextTypes.DEFAULT_TYPE,
    *,
    source: str = "menu",
    suppress_nav: bool = True,
) -> OpenedProfile:
    chat = getattr(update, "effective_chat", None)
    message = getattr(update, "effective_message", None)
    if chat is None and message is not None:
        chat = getattr(message, "chat", None)
    chat_id = getattr(chat, "id", None)
    if chat_id is None and message is not None:
        chat_id = getattr(message, "chat_id", None)

    user = getattr(update, "effective_user", None)
    user_id = getattr(user, "id", None)

    return await _open_profile_card_impl(
        chat_id,
        user_id,
        update=update,
        ctx=ctx,
        suppress_nav=suppress_nav,
        source=source,
    )


async def on_profile_menu(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    chat_data = _chat_data(ctx)
    if query is not None:
        if isinstance(chat_data, MutableMapping) and chat_data.get(_PROFILE_LOCK_KEY):
            with suppress(BadRequest):
                await query.answer("Открываю профиль…")
            return
        with suppress(BadRequest):
            await query.answer()

    await open_profile(update, ctx, source="menu", suppress_nav=True)


def _topup_url() -> Optional[str]:
    try:
        import settings as app_settings
    except Exception:  # pragma: no cover - defensive
        return None
    return getattr(app_settings, "TOPUP_URL", None)


async def on_profile_topup(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    log.info("profile.click", extra={"action": "topup"})
    await handle_profile_view(update, ctx, "topup")


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
    log.info("profile.click", extra={"action": "history"})
    await handle_profile_view(update, ctx, "history")


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
    log.info("profile.click", extra={"action": "invite"})
    await handle_profile_view(update, ctx, "invite")


def _activate_promo_wait(
    ctx: ContextTypes.DEFAULT_TYPE,
    *,
    user_id: int,
    chat_id: int,
    message_id: int,
) -> None:
    chat_data = _chat_data(ctx)
    expires_at = time.time() + PROMO_WAIT_TTL
    if chat_data is not None:
        chat_data[_PROMO_WAIT_KEY] = PROMO_WAIT_KIND
        chat_data[_PROMO_WAIT_UNTIL_KEY] = expires_at
        chat_data[_PROMO_WAIT_USER_KEY] = int(user_id)

    wait_state = WaitInputState(
        kind=WaitKind.PROMO_CODE,
        card_msg_id=int(message_id),
        chat_id=int(chat_id),
        meta={"source": "profile"},
        expires_at=expires_at,
    )
    set_wait_state(int(user_id), wait_state, ttl_seconds=PROMO_WAIT_TTL)


def clear_promo_wait(ctx: ContextTypes.DEFAULT_TYPE) -> None:
    chat_data = _chat_data(ctx)
    user_id: Optional[int] = None
    if isinstance(chat_data, MutableMapping):
        chat_data.pop(_PROMO_WAIT_KEY, None)
        chat_data.pop(_PROMO_WAIT_UNTIL_KEY, None)
        raw_user = chat_data.pop(_PROMO_WAIT_USER_KEY, None)
        try:
            user_id = int(raw_user)
        except (TypeError, ValueError):
            user_id = None
    if user_id is not None:
        clear_wait_state(int(user_id), reason="profile_nav")


def is_waiting_for_promo(ctx: ContextTypes.DEFAULT_TYPE) -> bool:
    chat_data = _chat_data(ctx)
    if chat_data is None:
        return False
    if chat_data.get(_PROMO_WAIT_KEY) != PROMO_WAIT_KIND:
        return False
    deadline = chat_data.get(_PROMO_WAIT_UNTIL_KEY)
    now = time.time()
    expired: bool
    try:
        expired = float(deadline) <= now
    except (TypeError, ValueError):
        expired = True
    if expired:
        clear_promo_wait(ctx)
        return False

    raw_user = chat_data.get(_PROMO_WAIT_USER_KEY)
    user_id: Optional[int]
    try:
        user_id = int(raw_user) if raw_user is not None else None
    except (TypeError, ValueError):
        user_id = None

    if user_id is not None:
        state = get_wait_state(int(user_id))
        if state and state.kind == WaitKind.PROMO_CODE and not state.is_expired(now=now):
            return True
    return True


async def on_profile_promo_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    log.info("profile.click", extra={"action": "promo"})
    await handle_profile_view(update, ctx, "promo")


async def on_profile_back(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    log.info("profile.click", extra={"action": "back"})
    await handle_profile_view(update, ctx, "back")


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
    "PROFILE_MSG_ID",
    "PROFILE_OPEN_AT",
    "NAV_UNTIL",
    "PROMO_WAIT_KIND",
    "PROMO_WAIT_TTL",
    "clear_promo_wait",
    "handle_promo_timeout",
    "is_waiting_for_promo",
    "OpenedProfile",
    "open_profile",
    "open_profile_card",
    "on_profile_history",
    "on_profile_invite",
    "on_profile_menu",
    "on_profile_back",
    "on_profile_promo_apply",
    "on_profile_promo_start",
    "on_profile_topup",
    "profile_reset_command",
    "render_profile_root",
    "render_profile_view",
    "profile_update_or_send",
    "handle_profile_view",
    "on_profile_cbq",
]
