from __future__ import annotations

import logging
import secrets
import time
from contextlib import suppress
from typing import Optional, Tuple

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes
from telegram.error import BadRequest

from keyboards import kb_main
from texts import TXT_MENU_TITLE
from ui.card import build_card
from utils.input_state import clear_wait_states
from utils.telegram_safe import safe_edit_message
from telegram_utils import safe_send

try:  # pragma: no cover - optional dependency in tests
    from redis_utils import rds as redis_client
except Exception:  # pragma: no cover - fallback when redis is unavailable
    redis_client = None

log = logging.getLogger(__name__)

_HOME_MSG_KEY_TMPL = "ui:home:{chat_id}"
_HOME_LOCK_KEY_TMPL = "lock:home:{chat_id}"
_HOME_LOCK_TTL_SEC = 2.0
_HOME_MSG_TTL_SEC = 7 * 24 * 60 * 60

_memory_msg: dict[int, Tuple[float, int]] = {}
_memory_locks: dict[int, Tuple[float, str]] = {}


def _msg_key(chat_id: int) -> str:
    return _HOME_MSG_KEY_TMPL.format(chat_id=int(chat_id))


def _lock_key(chat_id: int) -> str:
    return _HOME_LOCK_KEY_TMPL.format(chat_id=int(chat_id))


def render_home() -> Tuple[str, object, ParseMode, bool]:
    """Render the home menu card text and markup."""

    markup = kb_main()
    rows = markup.inline_keyboard
    card = build_card(TXT_MENU_TITLE, "Выберите раздел:", rows)
    text = card.get("text", "")
    reply_markup = card.get("reply_markup") or markup
    parse_mode = card.get("parse_mode", ParseMode.HTML)
    disable_preview = bool(card.get("disable_web_page_preview", True))
    return text, reply_markup, parse_mode, disable_preview


def _memory_get_msg(chat_id: int) -> Optional[int]:
    entry = _memory_msg.get(int(chat_id))
    if not entry:
        return None
    expires_at, message_id = entry
    if expires_at <= time.monotonic():
        _memory_msg.pop(int(chat_id), None)
        return None
    return message_id


def _memory_set_msg(chat_id: int, message_id: Optional[int]) -> None:
    if message_id is None:
        _memory_msg.pop(int(chat_id), None)
        return
    _memory_msg[int(chat_id)] = (
        time.monotonic() + _HOME_MSG_TTL_SEC,
        int(message_id),
    )


def _load_message_id(chat_id: int) -> Optional[int]:
    if redis_client is not None:
        try:
            raw = redis_client.get(_msg_key(chat_id))
        except Exception:  # pragma: no cover - diagnostics only
            log.debug("home.msg.load_failed", exc_info=True, extra={"chat_id": chat_id})
        else:
            if raw is not None:
                try:
                    msg_id = int(raw)
                except (TypeError, ValueError):
                    msg_id = None
                else:
                    _memory_set_msg(chat_id, msg_id)
                    return msg_id
    return _memory_get_msg(chat_id)


def _store_message_id(chat_id: int, message_id: Optional[int]) -> None:
    _memory_set_msg(chat_id, message_id)
    if redis_client is None:
        return
    try:
        if message_id is None:
            redis_client.delete(_msg_key(chat_id))
        else:
            redis_client.setex(_msg_key(chat_id), _HOME_MSG_TTL_SEC, int(message_id))
    except Exception:  # pragma: no cover - diagnostics only
        log.debug("home.msg.store_failed", exc_info=True, extra={"chat_id": chat_id})


def _acquire_lock(chat_id: int) -> Optional[str]:
    token = secrets.token_hex(8)
    if redis_client is not None:
        try:
            stored = redis_client.set(
                _lock_key(chat_id),
                token,
                nx=True,
                ex=max(1, int(_HOME_LOCK_TTL_SEC)),
            )
        except Exception:  # pragma: no cover - diagnostics only
            log.debug("home.lock.redis_failed", exc_info=True, extra={"chat_id": chat_id})
        else:
            if stored:
                return token
    now = time.monotonic()
    entry = _memory_locks.get(int(chat_id))
    if entry and entry[0] > now:
        return None
    _memory_locks[int(chat_id)] = (now + _HOME_LOCK_TTL_SEC, token)
    return token


def _release_lock(chat_id: int, token: Optional[str]) -> None:
    if not token:
        return
    if redis_client is not None:
        try:
            stored = redis_client.get(_lock_key(chat_id))
        except Exception:  # pragma: no cover - diagnostics only
            log.debug("home.lock.redis_get_failed", exc_info=True, extra={"chat_id": chat_id})
        else:
            if stored == token:
                with suppress(Exception):  # pragma: no cover - best effort
                    redis_client.delete(_lock_key(chat_id))
    entry = _memory_locks.get(int(chat_id))
    if entry and entry[1] == token:
        _memory_locks.pop(int(chat_id), None)


async def open_home(
    ctx: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    *,
    force_send: bool = False,
    fallback_message_id: Optional[int] = None,
) -> Optional[int]:
    """Render the home menu for ``chat_id`` in an idempotent manner."""

    lock_token = _acquire_lock(chat_id)
    if lock_token is None:
        log.warning("error.menu.duplicate", extra={"chat_id": chat_id})
        return _load_message_id(chat_id)

    try:
        stored_msg_id = _load_message_id(chat_id)
        if force_send:
            stored_msg_id = None
        if stored_msg_id is None and fallback_message_id is not None:
            stored_msg_id = int(fallback_message_id)

        text, reply_markup, parse_mode, disable_preview = render_home()

        reused = False
        current_msg_id: Optional[int] = stored_msg_id

        if current_msg_id is not None:
            try:
                edited = await safe_edit_message(
                    ctx,
                    chat_id,
                    current_msg_id,
                    text,
                    reply_markup=reply_markup,
                    parse_mode=parse_mode,
                    disable_web_page_preview=disable_preview,
                )
            except BadRequest as exc:
                lowered = str(exc).lower()
                if "message to edit not found" not in lowered:
                    log.debug(
                        "home.edit.failed",
                        extra={"chat_id": chat_id, "msg_id": current_msg_id, "error": str(exc)},
                    )
                current_msg_id = None
            except Exception as exc:  # pragma: no cover - unexpected runtime errors
                log.warning(
                    "home.edit.error",
                    extra={"chat_id": chat_id, "msg_id": stored_msg_id, "error": str(exc)},
                )
                current_msg_id = None
            else:
                reused = True
                _store_message_id(chat_id, current_msg_id)
                log.info(
                    "ui.home.open",
                    extra={"chat_id": chat_id, "reused": True, "msg_id": current_msg_id},
                )
                return current_msg_id

        result_msg_id: Optional[int] = current_msg_id
        if current_msg_id is None:
            try:
                result = await safe_send(
                    ctx.bot.send_message,
                    method_name="send_message",
                    kind="home_send",
                    chat_id=chat_id,
                    text=text,
                    reply_markup=reply_markup,
                    parse_mode=parse_mode,
                    disable_web_page_preview=disable_preview,
                )
            except Exception as exc:  # pragma: no cover - network or API errors
                log.warning(
                    "home.send.failed",
                    extra={"chat_id": chat_id, "error": str(exc)},
                )
                return None
            else:
                result_msg_id = getattr(result, "message_id", None)
                if isinstance(result_msg_id, int):
                    _store_message_id(chat_id, int(result_msg_id))
                else:
                    result_msg_id = None
                reused = False

        log.info(
            "ui.home.open",
            extra={"chat_id": chat_id, "reused": reused, "msg_id": result_msg_id},
        )
        return result_msg_id
    finally:
        _release_lock(chat_id, lock_token)


async def _resolve_chat_user(update: Update) -> Tuple[Optional[int], Optional[int]]:
    chat = getattr(update, "effective_chat", None)
    user = getattr(update, "effective_user", None)
    chat_id = getattr(chat, "id", None)
    user_id = getattr(user, "id", None)
    if chat_id is None and user_id is not None:
        chat_id = user_id
    return chat_id, user_id


async def _prepare_navigation(
    update: Update,
    ctx: ContextTypes.DEFAULT_TYPE,
    *,
    clear_wait: bool = True,
) -> Tuple[Optional[int], Optional[int]]:
    chat_id, user_id = await _resolve_chat_user(update)
    if clear_wait and user_id is not None:
        try:
            cleared = clear_wait_states(user_id, reason="home_nav")
            if cleared:
                log.debug(
                    "home.wait.cleared",
                    extra={"chat_id": chat_id, "user_id": user_id},
                )
        except Exception:  # pragma: no cover - defensive logging
            log.debug(
                "home.wait.clear_failed",
                exc_info=True,
                extra={"chat_id": chat_id, "user_id": user_id},
            )
    return chat_id, user_id


async def open_from_update(
    update: Update,
    ctx: ContextTypes.DEFAULT_TYPE,
    *,
    skip_ack: bool = False,
) -> Optional[int]:
    query = getattr(update, "callback_query", None)
    if query is not None and not skip_ack:
        with suppress(Exception):  # pragma: no cover - defensive
            await query.answer()

    chat_id, _ = await _prepare_navigation(update, ctx)
    if chat_id is None:
        log.debug("home.open.no_chat")
        return None

    fallback_mid = None
    if query is not None and getattr(query, "message", None) is not None:
        fallback_mid = getattr(query.message, "message_id", None)

    return await open_home(ctx, int(chat_id), fallback_message_id=fallback_mid)


async def open_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> Optional[int]:
    return await open_from_update(update, ctx)


async def cb_open(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> Optional[int]:
    return await open_from_update(update, ctx)


__all__ = [
    "cb_open",
    "open_from_update",
    "open_handler",
    "open_home",
    "render_home",
]
