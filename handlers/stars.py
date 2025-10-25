import json
import logging
from contextlib import suppress
from typing import List, Optional, Sequence

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, LabeledPrice, Message, Update
from telegram.error import BadRequest
from telegram.ext import ContextTypes

from telegram_utils import safe_answer

import settings as app_settings

log = logging.getLogger(__name__)

STARS_TIERS: Sequence[tuple[int, int]] = (
    (50, 50),
    (100, 110),
    (200, 220),
    (300, 330),
    (400, 440),
    (500, 550),
)

_DEFAULT_DISPLAY_TIERS: Sequence[int] = (50, 100, 300)
_STARS_PACK_MAP = {stars: gems for stars, gems in STARS_TIERS}


def render_stars_text() -> str:
    """Render the Stars top-up description."""

    return (
        "💎 <b>Пополнение через Telegram Stars</b>\n"
        "Если звёзд не хватает — купите в официальном боте @PremiumBot."
    )


def build_stars_kb(*, tiers: Sequence[int] | None = None) -> InlineKeyboardMarkup:
    rows: List[List[InlineKeyboardButton]] = []
    selected = list(tiers) if tiers is not None else [stars for stars, _ in STARS_TIERS]
    for stars in selected:
        gems = _STARS_PACK_MAP.get(stars, stars)
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"⭐️ {stars} → 💎 {gems}",
                    callback_data=f"stars:buy:{stars}",
                )
            ]
        )

    rows.append([InlineKeyboardButton("🛒 Где купить Stars", url="https://t.me/PremiumBot")])
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="nav:back")])
    return InlineKeyboardMarkup(rows)


def _resolve_chat_id(chat_id: Optional[int], ctx) -> Optional[int]:
    if chat_id is not None:
        return chat_id

    chat = getattr(ctx, "chat", None)
    if chat is not None:
        resolved = getattr(chat, "id", None)
        if isinstance(resolved, int):
            return resolved

    chat_data = getattr(ctx, "chat_data", None)
    if isinstance(chat_data, dict):
        candidate = chat_data.get("chat_id")
        if isinstance(candidate, int):
            return candidate

    return None


def _resolve_message_id(message_id: Optional[int], ctx) -> Optional[int]:
    if message_id is not None:
        return message_id

    message = getattr(ctx, "message", None)
    if message is not None:
        candidate = getattr(message, "message_id", None)
        if isinstance(candidate, int):
            return candidate

    return None


async def open_stars_menu(
    ctx,
    *,
    chat_id: Optional[int] = None,
    message_id: Optional[int] = None,
    edit_message: bool = True,
    source: Optional[str] = None,
    tiers: Optional[Sequence[int]] = None,
) -> Optional[Message]:
    """Show the Stars purchase screen."""

    bot = getattr(ctx, "bot", None)
    if bot is None:
        log.warning("stars.open missing_bot", extra={"source": source})
        return None

    target_chat_id = _resolve_chat_id(chat_id, ctx)
    target_message_id = _resolve_message_id(message_id, ctx)

    log.info("stars.open", extra={"source": source or "unknown", "chat_id": target_chat_id})

    text = render_stars_text()
    keyboard = build_stars_kb(tiers=tiers or _DEFAULT_DISPLAY_TIERS)

    if edit_message and target_chat_id is not None and target_message_id is not None:
        try:
            return await bot.edit_message_text(
                chat_id=target_chat_id,
                message_id=target_message_id,
                text=text,
                reply_markup=keyboard,
                parse_mode="HTML",
                disable_web_page_preview=True,
            )
        except Exception as exc:  # pragma: no cover - defensive
            log.info(
                "stars.render fallback=send",
                extra={
                    "source": source or "unknown",
                    "chat_id": target_chat_id,
                    "error": str(exc),
                },
            )

    if target_chat_id is None:
        log.warning("stars.render missing_chat", extra={"source": source})
        return None

    return await bot.send_message(
        chat_id=target_chat_id,
        text=text,
        reply_markup=keyboard,
        parse_mode="HTML",
        disable_web_page_preview=True,
    )


async def open(
    ctx: ContextTypes.DEFAULT_TYPE,
    *,
    update: Optional[Update] = None,
    chat_id: Optional[int] = None,
    message_id: Optional[int] = None,
    source: str = "profile",
    tiers: Optional[Sequence[int]] = None,
) -> Optional[Message]:
    """Render the Stars menu with the configured tiers."""

    resolved_chat_id = chat_id
    resolved_message_id = message_id
    if update is not None:
        query = getattr(update, "callback_query", None)
        message = getattr(query, "message", None)
        if message is None:
            message = getattr(update, "effective_message", None)
        if resolved_chat_id is None:
            chat_obj = getattr(update, "effective_chat", None)
            resolved_chat_id = getattr(chat_obj, "id", None)
            if resolved_chat_id is None and message is not None:
                resolved_chat_id = getattr(message, "chat_id", None)
        if resolved_message_id is None and message is not None:
            resolved_message_id = getattr(message, "message_id", None)

    return await open_stars_menu(
        ctx,
        chat_id=resolved_chat_id,
        message_id=resolved_message_id,
        edit_message=True,
        source=source,
        tiers=tiers,
    )


async def buy(
    ctx: ContextTypes.DEFAULT_TYPE,
    *,
    amount: int,
    update: Optional[Update] = None,
    source: str = "stars",
) -> None:
    """Initiate a Telegram Stars invoice for the desired ``amount``."""

    try:
        normalized_amount = int(amount)
    except (TypeError, ValueError):
        normalized_amount = 0

    if normalized_amount <= 0:
        log.warning("stars.buy.invalid_amount", extra={"amount": amount})
        return

    if not getattr(app_settings, "PAYMENTS_STARS_ENABLED", True):
        log.info("stars.buy.disabled", extra={"amount": amount})
        return

    query = getattr(update, "callback_query", None) if update is not None else None
    message = getattr(query, "message", None)
    if message is None and update is not None:
        message = getattr(update, "effective_message", None)

    chat_id = None
    if message is not None:
        chat_obj = getattr(message, "chat", None)
        chat_id = getattr(chat_obj, "id", None)
        if chat_id is None:
            chat_id = getattr(message, "chat_id", None)
    if chat_id is None and update is not None:
        chat_obj = getattr(update, "effective_chat", None)
        chat_id = getattr(chat_obj, "id", None)

    if normalized_amount not in _STARS_PACK_MAP:
        if query is not None:
            with suppress(BadRequest):
                await safe_answer(query, text="Пакет недоступен", show_alert=True)
        log.warning("stars.buy.invalid", extra={"amount": normalized_amount, "source": source})
        return

    target_chat_id = chat_id
    if target_chat_id is None:
        log.warning("stars.buy.missing_chat", extra={"amount": normalized_amount, "source": source})
        return

    diamonds = _STARS_PACK_MAP.get(normalized_amount, normalized_amount)
    title = f"{normalized_amount}⭐ → {diamonds}💎"
    invoice_payload = json.dumps(
        {
            "type": "stars_pack",
            "stars": normalized_amount,
            "diamonds": diamonds,
            "bonus": max(diamonds - normalized_amount, 0),
        }
    )

    try:
        await ctx.bot.send_invoice(
            chat_id=int(target_chat_id),
            title=title,
            description="Пакет пополнения токенов",
            payload=invoice_payload,
            provider_token="",
            currency="XTR",
            prices=[LabeledPrice(label=title, amount=normalized_amount)],
        )
    except Exception as exc:  # pragma: no cover - network issues
        log.exception(
            "stars.buy.invoice_failed",
            extra={"amount": normalized_amount, "chat_id": target_chat_id, "error": str(exc)},
        )
        if query is not None:
            with suppress(BadRequest):
                await safe_answer(query, text="Не удалось открыть счёт", show_alert=True)
        return

    if query is not None:
        with suppress(BadRequest):
            await safe_answer(query)

    log.info(
        "stars.buy.invoice_sent",
        extra={"amount": normalized_amount, "diamonds": diamonds, "chat_id": target_chat_id, "source": source},
    )
