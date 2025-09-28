from __future__ import annotations

import inspect
import logging
import os
from typing import Any, Optional, Tuple

from urllib.parse import quote_plus

from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode

from redis_utils import get_balance

import html

from utils.suno_state import (
    SunoState,
    lyrics_preview as suno_lyrics_preview,
    load as load_suno_state,
    style_preview as suno_style_preview,
)

_SUNO_MODEL_RAW = (os.getenv("SUNO_MODEL") or "v5").strip()
_SUNO_MODEL_LABEL = _SUNO_MODEL_RAW.upper() if _SUNO_MODEL_RAW else "V5"

_COPY_TEXT_SUPPORTED = "copy_text" in inspect.signature(InlineKeyboardButton.__init__).parameters

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


def _suno_keyboard(
    suno_state: SunoState, *, price: int, generating: bool
) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []

    rows.append([InlineKeyboardButton("✏️ Название", callback_data="suno:edit:title")])
    rows.append([InlineKeyboardButton("🎨 Стиль", callback_data="suno:edit:style")])

    mode_label = "Со словами" if suno_state.has_lyrics else "Инструментал"
    rows.append([
        InlineKeyboardButton(
            f"🎼 Режим: {mode_label}",
            callback_data="suno:toggle:instrumental",
        )
    ])

    if suno_state.has_lyrics:
        rows.append([
            InlineKeyboardButton("📝 Текст песни", callback_data="suno:edit:lyrics")
        ])

    generate_caption = "⏳ Генерация…" if generating else f"🎵 Генерация музыки — {price}💎"
    rows.append([
        InlineKeyboardButton(generate_caption, callback_data="suno:start")
    ])
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="back")])
    return InlineKeyboardMarkup(rows)


def render_suno_card(
    suno_state: SunoState,
    *,
    price: int,
    balance: Optional[int] = None,
    generating: bool = False,
) -> Tuple[str, InlineKeyboardMarkup]:
    safe_title = html.escape(suno_state.title) if suno_state.title else "—"
    style_display = suno_style_preview(suno_state.style, limit=120)
    safe_style = html.escape(style_display) if style_display else "—"
    mode_label = "Со словами" if suno_state.has_lyrics else "Инструментал"
    lyrics_preview = suno_lyrics_preview(suno_state.lyrics if suno_state.has_lyrics else None)
    safe_lyrics = html.escape(lyrics_preview) if lyrics_preview else "—"

    lines = ["🎵 Генерация музыки"]
    if balance is not None:
        lines.append(f"Баланс: {int(balance)}")
    lines.append(f"Модель: {html.escape(_SUNO_MODEL_LABEL)}")
    lines.append(f"Режим: {mode_label}")
    lines.append(f"• Название: {safe_title}")
    lines.append(f"• Стиль: {safe_style}")
    if suno_state.has_lyrics:
        lines.append(f"• Текст: <code>{safe_lyrics}</code>")
    lines.append("")
    lines.append(f"💎 Цена: {price}💎 за попытку")
    if generating:
        lines.append("⏳ Генерация запущена — ожидайте результат.")

    text = "\n".join(lines)
    keyboard = _suno_keyboard(suno_state, price=price, generating=generating)
    return text, keyboard


async def refresh_suno_card(
    ctx: Any,
    chat_id: int,
    state_dict: dict[str, Any],
    *,
    price: int,
    state_key: str = "last_ui_msg_id_suno",
) -> Optional[int]:
    suno_state_obj = load_suno_state(ctx)
    state_dict["suno_state"] = suno_state_obj.to_dict()
    generating = bool(state_dict.get("suno_generating"))
    balance_val = state_dict.get("suno_balance")
    try:
        balance_num = int(balance_val) if balance_val is not None else None
    except Exception:
        balance_num = None
    text, markup = render_suno_card(
        suno_state_obj,
        price=price,
        balance=balance_num,
        generating=generating,
    )
    last_key = "_last_text_suno"
    if state_dict.get(last_key) == text and state_dict.get(state_key):
        return state_dict.get(state_key)

    mid = await upsert_card(
        ctx,
        chat_id,
        state_dict,
        state_key,
        text,
        reply_markup=markup,
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True,
    )
    if mid:
        state_dict[last_key] = text
    else:
        state_dict[last_key] = None
    return mid


def referral_card_text(link: str, referrals: int, earned: int) -> str:
    safe_link = link.strip()
    return (
        "👥 <b>Рефералов:</b> {referrals}\n"
        "💎 <b>Заработано:</b> {earned}\n"
        "🔗 <b>Ваша ссылка:</b> <code>{link}</code>\n\n"
        "Приглашайте друзей — получайте <b>10%</b> в 💎 от их пополнений Stars."
    ).format(referrals=int(referrals), earned=int(earned), link=safe_link)


def referral_card_keyboard(link: str, *, share_text: Optional[str] = None) -> InlineKeyboardMarkup:
    share_caption = share_text or "Присоединяйся к Best VEO3 Bot!"
    url_encoded = quote_plus(link)
    text_encoded = quote_plus(share_caption)
    share_url = f"https://t.me/share/url?url={url_encoded}&text={text_encoded}"

    rows: list[list[InlineKeyboardButton]] = []
    if _COPY_TEXT_SUPPORTED:
        rows.append([InlineKeyboardButton("📋 Скопировать ссылку", copy_text=link)])
    else:
        rows.append([
            InlineKeyboardButton(
                "📋 Скопировать ссылку",
                switch_inline_query_current_chat=link,
            )
        ])
    rows.append([InlineKeyboardButton("📤 Поделиться", url=share_url)])
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="ref:back")])
    return InlineKeyboardMarkup(rows)


async def show_referral_card(
    ctx: Any,
    chat_id: int,
    state_dict: dict[str, Any],
    *,
    link: str,
    referrals: int,
    earned: int,
    share_text: Optional[str] = None,
) -> Optional[int]:
    text = referral_card_text(link, referrals, earned)
    markup = referral_card_keyboard(link, share_text=share_text)
    mid = await upsert_card(
        ctx,
        chat_id,
        state_dict,
        "last_ui_msg_id_balance",
        text,
        reply_markup=markup,
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True,
    )
    if mid:
        msg_ids_raw = state_dict.get("msg_ids")
        msg_ids = msg_ids_raw if isinstance(msg_ids_raw, dict) else None
        if msg_ids is None:
            msg_ids = {}
            state_dict["msg_ids"] = msg_ids
        msg_ids["balance"] = mid
        state_dict["last_panel"] = "referral"
    return mid


def pm_main_kb() -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton("🎬 Промпт для видео", callback_data="pm:video")],
        [InlineKeyboardButton("🖼️ Оживление фото", callback_data="pm:animate")],
        [InlineKeyboardButton("🍌 Banana JSON", callback_data="pm:banana")],
        [InlineKeyboardButton("🎨 Midjourney JSON", callback_data="pm:mj")],
        [InlineKeyboardButton("🎵 Suno (текст песни)", callback_data="pm:suno")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="pm:home")],
    ]
    return InlineKeyboardMarkup(rows)


_PM_USE_LABELS = {
    "video": "Veo",
    "animate": "Veo Animate",
    "banana": "Banana",
    "mj": "Midjourney",
    "suno": "Suno",
}


def pm_result_kb(kind: str) -> InlineKeyboardMarkup:
    use_target = _PM_USE_LABELS.get(kind, "")
    use_caption = "⚡ Использовать сейчас" if not use_target else f"⚡ Использовать в {use_target}"
    rows = [
        [InlineKeyboardButton("🔁 Изменить ввод", callback_data="pm:back")],
        [InlineKeyboardButton(use_caption, callback_data=f"pm:reuse:{kind}")],
        [InlineKeyboardButton("📋 Скопировать", callback_data="pm:copy")],
        [
            InlineKeyboardButton("⬅️ Назад к разделам", callback_data="pm:menu"),
            InlineKeyboardButton("🏠 В меню", callback_data="pm:home"),
        ],
    ]
    return InlineKeyboardMarkup(rows)
