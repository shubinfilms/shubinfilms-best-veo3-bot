from __future__ import annotations

import json
import logging
import re
from contextlib import suppress
from typing import Any, Optional

from telegram import LabeledPrice, Update
from telegram.error import BadRequest
from telegram.ext import ContextTypes

from handlers.stars import STARS_TIERS, open_stars_menu
from metrics import stars_buy_total
from telegram_utils import safe_answer

import settings as app_settings

log = logging.getLogger(__name__)

_STAR_CALLBACK_RE = re.compile(r"stars:buy:(?P<amount>\d+)")
_ALLOWED_AMOUNTS = {tier[0] for tier in STARS_TIERS}


def _coerce_amount(value: Any) -> Optional[int]:
    try:
        amount = int(value)
    except (TypeError, ValueError):
        return None
    return amount if amount > 0 else None


def _resolve_amount(update: Update, payload: Optional[dict[str, object]]) -> Optional[int]:
    if isinstance(payload, dict):
        raw = payload.get("amount")
        coerced = _coerce_amount(raw)
        if coerced is not None:
            return coerced
    query = getattr(update, "callback_query", None)
    data = getattr(query, "data", "") if query else ""
    match = _STAR_CALLBACK_RE.search(data or "")
    if match:
        return _coerce_amount(match.group("amount"))
    return None


async def _send_invoice(
    *,
    ctx: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    amount: int,
    diamonds: int,
) -> None:
    title = f"{amount}⭐ → {diamonds}💎"
    payload = json.dumps(
        {
            "type": "stars_pack",
            "stars": amount,
            "diamonds": diamonds,
            "bonus": max(diamonds - amount, 0),
        }
    )
    await ctx.bot.send_invoice(
        chat_id=chat_id,
        title=title,
        description="Пакет пополнения токенов",
        payload=payload,
        provider_token="",
        currency="XTR",
        prices=[LabeledPrice(label=title, amount=amount)],
    )


async def stars_buy(
    update: Update,
    ctx: ContextTypes.DEFAULT_TYPE,
    payload: Optional[dict[str, object]] = None,
) -> None:
    if not getattr(app_settings, "PAYMENTS_STARS_ENABLED", True):
        log.info("payments.stars.disabled")
        return

    amount = _resolve_amount(update, payload)
    query = getattr(update, "callback_query", None)
    message = getattr(query, "message", None)
    chat_id = None
    if message is not None:
        chat = getattr(message, "chat", None)
        chat_id = getattr(chat, "id", None)
        if chat_id is None:
            chat_id = getattr(message, "chat_id", None)
    if chat_id is None:
        chat = getattr(update, "effective_chat", None)
        chat_id = getattr(chat, "id", None)

    if amount is None or amount not in _ALLOWED_AMOUNTS:
        if query is not None:
            with suppress(BadRequest):
                await safe_answer(query, text="Пакет недоступен", show_alert=True)
        label_amount = str(amount) if amount is not None else "unknown"
        stars_buy_total.labels(amount=label_amount, result="invalid").inc()
        return

    diamonds = next((tier[1] for tier in STARS_TIERS if tier[0] == amount), None)
    if diamonds is None:
        diamonds = amount

    result = "ok"
    if chat_id is None:
        log.warning("payments.stars.missing_chat", extra={"amount": amount})
        result = "error"
    else:
        try:
            await _send_invoice(ctx=ctx, chat_id=int(chat_id), amount=amount, diamonds=diamonds)
        except Exception as exc:  # pragma: no cover - network issues
            log.exception("payments.stars.invoice_failed", extra={"amount": amount, "chat_id": chat_id})
            result = "error"
            if query is not None:
                with suppress(BadRequest):
                    await safe_answer(query, text="Не удалось открыть счёт", show_alert=True)
        else:
            if query is not None:
                with suppress(BadRequest):
                    await safe_answer(query)

    stars_buy_total.labels(amount=str(amount), result=result).inc()

    if chat_id is not None and message is not None:
        try:
            await open_stars_menu(
                ctx,
                chat_id=chat_id,
                message_id=getattr(message, "message_id", None),
                edit_message=True,
                source="payments",
            )
        except Exception:
            log.debug("payments.stars.refresh_failed", exc_info=True)
