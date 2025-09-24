from __future__ import annotations

import logging
from typing import Any, Optional

from telegram.constants import ParseMode

from redis_utils import get_balance


async def upsert_card(
    ctx: Any,
    chat_id: int,
    state_dict: dict[str, Any],
    state_key: str,
    text: str,
    reply_markup: Optional[Any] = None,
    parse_mode: ParseMode = ParseMode.HTML,
    disable_web_page_preview: bool = True,
) -> Optional[int]:
    """Update an existing UI card or send a new one.

    Returns the message id of the card on success, otherwise ``None``.
    """

    mid = state_dict.get(state_key)
    if mid:
        try:
            await ctx.bot.edit_message_text(
                chat_id=chat_id,
                message_id=mid,
                text=text,
                reply_markup=reply_markup,
                parse_mode=parse_mode,
                disable_web_page_preview=disable_web_page_preview,
            )
            return mid
        except Exception as exc:  # pragma: no cover - network issues
            logger = getattr(getattr(ctx, "application", None), "logger", logging.getLogger(__name__))
            logger.warning("edit %s failed: %s", state_key, exc)
            state_dict[state_key] = None

    try:
        msg = await ctx.bot.send_message(
            chat_id=chat_id,
            text=text,
            reply_markup=reply_markup,
            parse_mode=parse_mode,
            disable_web_page_preview=disable_web_page_preview,
        )
        state_dict[state_key] = msg.message_id
        return msg.message_id
    except Exception as exc:  # pragma: no cover - network issues
        logger = getattr(getattr(ctx, "application", None), "logger", logging.getLogger(__name__))
        logger.error("send %s failed: %s", state_key, exc)
        return None


async def refresh_balance_card_if_open(
    user_id: int,
    chat_id: int,
    *,
    ctx: Any,
    state_dict: Optional[dict[str, Any]] = None,
    reply_markup: Optional[Any] = None,
    state_key: str = "last_ui_msg_id_balance",
    parse_mode: ParseMode = ParseMode.MARKDOWN,
    disable_web_page_preview: bool = True,
) -> Optional[int]:
    """Re-render the balance card if it is currently shown."""

    if state_dict is None:
        state_dict = getattr(ctx, "user_data", {})
    if not isinstance(state_dict, dict):
        return None

    msg_ids_raw = state_dict.get("msg_ids")
    msg_ids = msg_ids_raw if isinstance(msg_ids_raw, dict) else None
    balance_mid = msg_ids.get("balance") if msg_ids else None
    last_panel = state_dict.get("last_panel")
    if last_panel != "balance" and not balance_mid:
        return None

    try:
        balance = get_balance(user_id)
    except Exception as exc:  # pragma: no cover - network/redis issues
        logger = getattr(getattr(ctx, "application", None), "logger", logging.getLogger(__name__))
        logger.warning("refresh_balance_card_if_open failed for %s: %s", user_id, exc)
        return None

    text = f"💎 Ваш баланс: {balance}"
    mid = await upsert_card(
        ctx,
        chat_id,
        state_dict,
        state_key,
        text,
        reply_markup=reply_markup,
        parse_mode=parse_mode,
        disable_web_page_preview=disable_web_page_preview,
    )
    if mid:
        if msg_ids is None:
            msg_ids = {}
            state_dict["msg_ids"] = msg_ids
        msg_ids["balance"] = mid
        state_dict["last_panel"] = "balance"
        try:
            if hasattr(ctx, "user_data"):
                ctx.user_data["balance"] = balance
        except Exception:
            pass
    return mid
