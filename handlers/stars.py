import logging
from typing import List, Optional, Sequence

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Message

log = logging.getLogger(__name__)

STARS_TIERS: Sequence[tuple[int, int]] = (
    (50, 50),
    (100, 110),
    (200, 220),
    (300, 330),
    (400, 440),
    (500, 550),
)


def render_stars_text() -> str:
    """Render the Stars top-up description."""

    return (
        "💎 <b>Пополнение через Telegram Stars</b>\n"
        "Если звёзд не хватает — купите в официальном боте @PremiumBot."
    )


def build_stars_kb() -> InlineKeyboardMarkup:
    rows: List[List[InlineKeyboardButton]] = []
    for stars, gems in STARS_TIERS:
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
    keyboard = build_stars_kb()

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
