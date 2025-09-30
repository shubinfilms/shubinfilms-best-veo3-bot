from __future__ import annotations

import inspect
import json
import hashlib
import logging
import os
from typing import Any, Optional, Tuple, MutableMapping

from urllib.parse import quote_plus

from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.error import BadRequest

from redis_utils import get_balance

import html

from utils.suno_state import (
    LyricsSource,
    SunoState,
    lyrics_preview as suno_lyrics_preview,
    load as load_suno_state,
    save as save_suno_state,
    style_preview as suno_style_preview,
    suno_is_ready_to_start,
)
from utils.suno_modes import (
    default_style_text as suno_default_style_text,
    get_mode_config as get_suno_mode_config,
)
from keyboards import suno_start_keyboard
from texts import SUNO_START_READY_MESSAGE, t
from telegram_utils import safe_edit, SafeEditResult
from utils.telegram_safe import safe_edit_message

logger = logging.getLogger(__name__)

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
    *,
    force_new: bool = False,
) -> Optional[int]:
    """Update an existing UI card or send a new one.

    Returns the message id of the card on success, otherwise ``None``.
    """

    mid = state_dict.get(state_key)
    if force_new and isinstance(mid, int):
        try:
            await ctx.bot.delete_message(chat_id, mid)
        except BadRequest as exc:
            app_logger = getattr(getattr(ctx, "application", None), "logger", logging.getLogger(__name__))
            app_logger.debug("delete %s bad request: %s", state_key, exc)
        except Exception as exc:  # pragma: no cover - network issues
            app_logger = getattr(getattr(ctx, "application", None), "logger", logging.getLogger(__name__))
            app_logger.warning("delete %s failed: %s", state_key, exc)
        state_dict[state_key] = None
        msg_ids = state_dict.get("msg_ids")
        if isinstance(msg_ids, dict):
            for key, value in list(msg_ids.items()):
                if value == mid:
                    msg_ids[key] = None
    mid = state_dict.get(state_key)
    if mid:
        try:
            await safe_edit_message(
                ctx,
                chat_id,
                mid,
                text,
                reply_markup,
                parse_mode=parse_mode,
                disable_web_page_preview=disable_web_page_preview,
            )
            return mid
        except BadRequest as exc:
            app_logger = getattr(getattr(ctx, "application", None), "logger", logging.getLogger(__name__))
            app_logger.debug("edit %s bad request: %s", state_key, exc)
            state_dict[state_key] = None
        except Exception as exc:  # pragma: no cover - network issues
            app_logger = getattr(getattr(ctx, "application", None), "logger", logging.getLogger(__name__))
            app_logger.warning("edit %s failed: %s", state_key, exc)
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
        app_logger = getattr(getattr(ctx, "application", None), "logger", logging.getLogger(__name__))
        app_logger.error("send %s failed: %s", state_key, exc)
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
        app_logger = getattr(getattr(ctx, "application", None), "logger", logging.getLogger(__name__))
        app_logger.warning("refresh_balance_card_if_open failed for %s: %s", user_id, exc)
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
    suno_state: SunoState,
    *,
    price: int,
    generating: bool,
    flow: Optional[str],
) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []

    mode = flow or suno_state.mode
    if mode == "instrumental":
        rows.append([InlineKeyboardButton("✏️ Название", callback_data="suno:edit:title")])
        rows.append([InlineKeyboardButton("🎛️ Стиль", callback_data="suno:edit:style")])
    elif mode == "lyrics":
        rows.append([InlineKeyboardButton("✏️ Название", callback_data="suno:edit:title")])
        rows.append([InlineKeyboardButton("🎛️ Стиль", callback_data="suno:edit:style")])
        rows.append([
            InlineKeyboardButton(
                "🧾/✨ Источник текста",
                callback_data="suno:card:lyrics_source:toggle",
            )
        ])
        if suno_state.lyrics_source == LyricsSource.USER:
            rows.append([InlineKeyboardButton("📝 Текст", callback_data="suno:edit:lyrics")])
    elif mode == "cover":
        rows.append([InlineKeyboardButton("✏️ Название", callback_data="suno:edit:title")])
        rows.append([InlineKeyboardButton("🎧 Источник", callback_data="suno:edit:cover")])
        rows.append([InlineKeyboardButton("🎛️ Стиль", callback_data="suno:edit:style")])

    bottom_row = [InlineKeyboardButton("🔄 Сменить режим", callback_data="suno:menu")]
    bottom_row.append(InlineKeyboardButton("✖️ Отмена", callback_data="suno:cancel"))
    rows.append(bottom_row)
    return InlineKeyboardMarkup(rows)


def render_suno_card(
    suno_state: SunoState,
    *,
    price: int,
    balance: Optional[int] = None,
    generating: bool = False,
    waiting_enqueue: bool = False,
) -> Tuple[str, InlineKeyboardMarkup, bool]:
    mode = suno_state.mode
    config = get_suno_mode_config(mode)

    def _safe_style_preview(value: Optional[str], limit: int = 160) -> str:
        func = globals().get("suno_style_preview")
        if callable(func):
            try:
                preview = func(value, limit=limit)
            except Exception:
                preview = None
            if preview:
                return str(preview)
        text = (value or "").strip()
        if not text:
            return ""
        if len(text) <= limit:
            return text
        return text[: max(1, limit - 1)].rstrip() + "…"

    title_value = html.escape(suno_state.title.strip()) if suno_state.title else "—"
    style_preview = _safe_style_preview(suno_state.style, limit=160)
    style_value = html.escape(style_preview) if style_preview else "—"

    lyrics_preview_value = suno_lyrics_preview(suno_state.lyrics)
    lyrics_value = html.escape(lyrics_preview_value) if lyrics_preview_value else "—"

    if suno_state.kie_file_id:
        source_value = f"загружено ✅ (id: {html.escape(suno_state.kie_file_id)})"
    elif suno_state.cover_source_label:
        source_value = html.escape(suno_state.cover_source_label)
    elif suno_state.cover_source_url:
        source_value = html.escape(suno_state.cover_source_url)
    else:
        source_value = "—"

    lines: list[str] = [f"{config.emoji} <b>{html.escape(config.title)}</b>"]
    lines.append(f"🧠 Модель: {html.escape(_SUNO_MODEL_LABEL)}")
    if balance is not None:
        lines.append(f"💎 Баланс: {int(balance)}💎")

    if mode == "instrumental":
        lines.append(f"🏷️ {t('suno.field.title')}: <i>{title_value}</i>")
        lines.append(f"🎹 {t('suno.field.style')}: <i>{style_value}</i>")
    elif mode == "lyrics":
        lines.append(f"🏷️ {t('suno.field.title')}: <i>{title_value}</i>")
        lines.append(f"🎹 {t('suno.field.style')}: <i>{style_value}</i>")
        if suno_state.lyrics_source == LyricsSource.USER:
            lines.append(f"📥 {t('suno.field.lyrics_source')}: <i>{html.escape(t('suno.lyrics_source.user'))}</i>")
            lines.append(f"📜 {t('suno.field.lyrics')}: <i>{lyrics_value}</i>")
        else:
            lines.append(f"📥 {t('suno.field.lyrics_source')}: <i>{html.escape(t('suno.lyrics_source.ai'))}</i>")
    else:
        lines.append(f"🏷️ {t('suno.field.title')}: <i>{title_value}</i>")
        lines.append(f"🎧 {t('suno.field.source')}: <i>{source_value}</i>")
        lines.append(f"🎹 {t('suno.field.style')}: <i>{style_value}</i>")

    lines.append(f"💎 {t('suno.field.cost')}: {price}💎")

    if waiting_enqueue:
        lines.append("⏳ Отправляем запрос…")
    elif generating:
        lines.append("⏳ Генерация запущена, ждём трек…")

    text = "\n".join(lines)

    ready = suno_is_ready_to_start(suno_state)
    keyboard = _suno_keyboard(
        suno_state,
        price=price,
        generating=generating or waiting_enqueue,
        flow=mode,
    )
    return text, keyboard, ready


async def sync_suno_start_message(
    ctx: Any,
    chat_id: int,
    state_dict: MutableMapping[str, Any],
    *,
    suno_state: SunoState,
    ready: bool,
    generating: bool,
    waiting_enqueue: bool,
) -> Optional[int]:
    should_show = ready and not generating and not waiting_enqueue

    raw_id = state_dict.get("suno_start_msg_id")
    start_msg_id = raw_id if isinstance(raw_id, int) else None
    if start_msg_id is None and isinstance(suno_state.start_msg_id, int):
        start_msg_id = suno_state.start_msg_id

    if not should_show:
        if isinstance(start_msg_id, int):
            try:
                await ctx.bot.delete_message(chat_id, start_msg_id)
            except BadRequest as exc:
                logger.debug("delete suno start failed: %s", exc)
            except Exception as exc:  # pragma: no cover - network issues
                logger.warning("delete suno start error: %s", exc)
            start_msg_id = None
    else:
        text = SUNO_START_READY_MESSAGE
        markup = suno_start_keyboard()
        suno_state.start_clicked = False
        suno_state.start_emoji_msg_id = None
        if isinstance(start_msg_id, int):
            try:
                await safe_edit_message(
                    ctx,
                    chat_id,
                    start_msg_id,
                    text,
                    markup,
                    parse_mode=ParseMode.HTML,
                    disable_web_page_preview=True,
                )
            except BadRequest as exc:
                logger.debug("edit suno start failed: %s", exc)
                start_msg_id = None
            except Exception as exc:  # pragma: no cover - network issues
                logger.warning("edit suno start error: %s", exc)
                start_msg_id = None
        if start_msg_id is None:
            try:
                msg = await ctx.bot.send_message(
                    chat_id=chat_id,
                    text=text,
                    reply_markup=markup,
                    parse_mode=ParseMode.HTML,
                    disable_web_page_preview=True,
                )
                start_msg_id = msg.message_id
            except Exception as exc:  # pragma: no cover - network issues
                logger.error("send suno start failed: %s", exc)
                start_msg_id = None

    if isinstance(start_msg_id, int):
        state_dict["suno_start_msg_id"] = start_msg_id
        msg_ids = state_dict.get("msg_ids")
        if isinstance(msg_ids, MutableMapping):
            msg_ids["suno_start"] = start_msg_id
    else:
        state_dict.pop("suno_start_msg_id", None)
        msg_ids = state_dict.get("msg_ids")
        if isinstance(msg_ids, MutableMapping):
            msg_ids.pop("suno_start", None)

    suno_state.start_msg_id = start_msg_id
    state_dict["suno_state"] = suno_state.to_dict()
    return start_msg_id


async def refresh_suno_card(
    ctx: Any,
    chat_id: int,
    state_dict: dict[str, Any],
    *,
    price: int,
    state_key: str = "last_ui_msg_id_suno",
    force_new: bool = False,
) -> Optional[int]:
    suno_state_obj = load_suno_state(ctx)
    flow = state_dict.get("suno_flow")
    if isinstance(flow, str) and flow in {"instrumental", "lyrics", "cover"}:
        suno_state_obj.mode = flow  # type: ignore[assignment]
    state_dict["suno_state"] = suno_state_obj.to_dict()
    generating = bool(state_dict.get("suno_generating"))
    waiting_enqueue = bool(state_dict.get("suno_waiting_enqueue"))
    balance_val = state_dict.get("suno_balance")
    try:
        balance_num = int(balance_val) if balance_val is not None else None
    except Exception:
        balance_num = None
    text, markup, ready = render_suno_card(
        suno_state_obj,
        price=price,
        balance=balance_num,
        generating=generating,
        waiting_enqueue=waiting_enqueue,
    )
    card_state_raw = state_dict.get("suno_card")
    card_state: MutableMapping[str, Any]
    if isinstance(card_state_raw, MutableMapping):
        card_state = card_state_raw
    else:
        card_state = {"msg_id": None, "last_text_hash": None, "last_markup_hash": None}
        state_dict["suno_card"] = card_state

    msg_id = card_state.get("msg_id")
    if not isinstance(msg_id, int):
        msg_id = state_dict.get(state_key)
        if not isinstance(msg_id, int):
            msg_id = None

    if force_new and isinstance(msg_id, int):
        try:
            await ctx.bot.delete_message(chat_id, msg_id)
        except BadRequest as exc:
            app_logger = getattr(getattr(ctx, "application", None), "logger", logging.getLogger(__name__))
            app_logger.debug("delete suno card failed: %s", exc)
        except Exception as exc:  # pragma: no cover - network issues
            app_logger = getattr(getattr(ctx, "application", None), "logger", logging.getLogger(__name__))
            app_logger.warning("delete suno card failed: %s", exc)
        msg_id = None
        state_dict[state_key] = None
        card_state["msg_id"] = None
        card_state["last_text_hash"] = None
        card_state["last_markup_hash"] = None

    if markup is None:
        markup_payload: Any = None
    else:
        try:
            markup_payload = markup.to_dict()
        except AttributeError:
            markup_payload = markup
    markup_json = json.dumps(markup_payload, ensure_ascii=False, sort_keys=True)
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    markup_hash = hashlib.sha256(markup_json.encode("utf-8")).hexdigest()

    last_text_hash = card_state.get("last_text_hash") if isinstance(card_state, MutableMapping) else None
    last_markup_hash = card_state.get("last_markup_hash") if isinstance(card_state, MutableMapping) else None

    payload_changed = not (
        isinstance(last_text_hash, str)
        and isinstance(last_markup_hash, str)
        and last_text_hash == text_hash
        and last_markup_hash == markup_hash
        and isinstance(msg_id, int)
    )

    def _log_card_event(method: str, reason: str, old_id: Optional[int], new_id: Optional[int]) -> None:
        try:
            payload = json.dumps(
                {
                    "card_msg_id_old": old_id,
                    "card_msg_id_new": new_id,
                    "method": method,
                    "reason": reason,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        except Exception:
            payload = str(
                {
                    "card_msg_id_old": old_id,
                    "card_msg_id_new": new_id,
                    "method": method,
                    "reason": reason,
                }
            )
        logger.info("EVT_CARD_EDIT | %s", payload)

    if not payload_changed and isinstance(msg_id, int):
        card_state["msg_id"] = msg_id
        card_state["last_text_hash"] = text_hash
        card_state["last_markup_hash"] = markup_hash
        card_state["chat_id"] = chat_id
        suno_state_obj.card_message_id = msg_id
        suno_state_obj.card_text_hash = text_hash
        suno_state_obj.card_markup_hash = markup_hash
        suno_state_obj.card_chat_id = chat_id
        suno_state_obj.last_card_hash = text_hash
        await sync_suno_start_message(
            ctx,
            chat_id,
            state_dict,
            suno_state=suno_state_obj,
            ready=ready,
            generating=generating,
            waiting_enqueue=waiting_enqueue,
        )
        save_suno_state(ctx, suno_state_obj)
        state_dict["suno_state"] = suno_state_obj.to_dict()
        state_dict[state_key] = msg_id
        msg_ids = state_dict.get("msg_ids")
        if isinstance(msg_ids, dict):
            msg_ids["suno"] = msg_id
        state_dict["_last_text_suno"] = text
        _log_card_event("skip", "not_changed", msg_id, msg_id)
        return msg_id

    result: SafeEditResult = await safe_edit(
        ctx.bot,
        chat_id,
        msg_id,
        text,
        markup,
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True,
        state=card_state,
        resend_on_not_modified=True,
    )

    new_msg_id = result.message_id or card_state.get("msg_id")
    if isinstance(new_msg_id, int):
        state_dict[state_key] = new_msg_id
        card_state["msg_id"] = new_msg_id
        msg_ids = state_dict.get("msg_ids")
        if isinstance(msg_ids, dict):
            msg_ids["suno"] = new_msg_id
    card_state["chat_id"] = chat_id

    suno_state_obj.card_message_id = card_state.get("msg_id") if isinstance(card_state.get("msg_id"), int) else None
    card_text_hash = card_state.get("last_text_hash")
    card_markup_hash = card_state.get("last_markup_hash")
    suno_state_obj.card_text_hash = card_text_hash if isinstance(card_text_hash, str) else None
    suno_state_obj.card_markup_hash = card_markup_hash if isinstance(card_markup_hash, str) else None
    suno_state_obj.card_chat_id = chat_id
    suno_state_obj.last_card_hash = suno_state_obj.card_text_hash

    await sync_suno_start_message(
        ctx,
        chat_id,
        state_dict,
        suno_state=suno_state_obj,
        ready=ready,
        generating=generating,
        waiting_enqueue=waiting_enqueue,
    )

    save_suno_state(ctx, suno_state_obj)
    state_dict["suno_state"] = suno_state_obj.to_dict()

    reason = "changed"
    method = "edit"
    if result.status == "sent":
        method = "send"
        reason = "missing_msg"
    elif result.status == "resent":
        method = "send"
        reason = result.reason or "missing_msg"
    elif result.status == "skipped":
        method = "skip"
        reason = result.reason or "not_modified"

    _log_card_event(method, reason, msg_id, new_msg_id if isinstance(new_msg_id, int) else None)

    state_dict["_last_text_suno"] = text
    return new_msg_id if isinstance(new_msg_id, int) else msg_id


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
