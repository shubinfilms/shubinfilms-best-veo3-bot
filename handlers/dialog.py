from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, MutableMapping, Optional

from telegram.ext import ContextTypes

from handlers.menu import build_dialog_card

log = logging.getLogger(__name__)

_CARD_STATE_KEY = "dialog_menu_card"
_MSG_IDS_KEY = "dialog"

_send_menu: Optional[Callable[..., Awaitable[Optional[int]]]] = None
_state_getter: Optional[Callable[[ContextTypes.DEFAULT_TYPE], MutableMapping[str, Any]]] = None


def configure(
    *,
    send_menu: Callable[..., Awaitable[Optional[int]]],
    state_getter: Callable[[ContextTypes.DEFAULT_TYPE], MutableMapping[str, Any]],
) -> None:
    global _send_menu
    global _state_getter
    _send_menu = send_menu
    _state_getter = state_getter


async def open_menu(
    ctx: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    *,
    suppress_nav: bool = False,
    fallback_message_id: Optional[int] = None,
) -> Optional[int]:
    if _send_menu is None or _state_getter is None:
        raise RuntimeError("dialog menu handler is not configured")

    state_dict = _state_getter(ctx)
    card = build_dialog_card()

    message_id = await _send_menu(
        ctx,
        chat_id=chat_id,
        text=card["text"],
        reply_markup=card["reply_markup"],
        state_key=_CARD_STATE_KEY,
        msg_ids_key=_MSG_IDS_KEY,
        state_dict=state_dict,
        fallback_message_id=fallback_message_id,
        parse_mode=card.get("parse_mode"),
        disable_web_page_preview=card.get("disable_web_page_preview", True),
        log_label="ui.dialog.menu",
    )

    log.debug(
        "dialog.menu.opened",
        extra={
            "chat_id": chat_id,
            "message_id": message_id,
            "suppress_nav": suppress_nav,
            "fallback_mid": fallback_message_id,
        },
    )
    return message_id


__all__ = ["configure", "open_menu"]
