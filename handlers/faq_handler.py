"""FAQ handlers rendering HTML-safe content."""

import html
import logging
import re
from typing import Awaitable, Callable, Optional

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from logging_utils import build_log_extra, get_context_logger

from keyboards import CB_FAQ_PREFIX, faq_keyboard
from texts import FAQ_INTRO, FAQ_SECTIONS
from telegram_utils import safe_edit, safe_send

logger = logging.getLogger(__name__)


_ShowMainMenu = Optional[Callable[[Update, ContextTypes.DEFAULT_TYPE], Awaitable[None]]]
_MetricCallback = Optional[Callable[[str], None]]

_show_main_menu: _ShowMainMenu = None
_on_root_view: Optional[Callable[[], None]] = None
_on_section_view: _MetricCallback = None


def configure_faq(
    *,
    show_main_menu: Optional[Callable[[Update, ContextTypes.DEFAULT_TYPE], Awaitable[None]]] = None,
    on_root_view: Optional[Callable[[], None]] = None,
    on_section_view: _MetricCallback = None,
) -> None:
    """Configure dependencies used by FAQ handlers."""

    global _show_main_menu
    global _on_root_view
    global _on_section_view
    _show_main_menu = show_main_menu
    _on_root_view = on_root_view
    _on_section_view = on_section_view


async def faq_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send the FAQ root menu."""

    log = get_context_logger(context)
    user = update.effective_user
    chat = update.effective_chat
    log.info(
        "update.received",
        **build_log_extra(
            {},
            update=update,
            command="/faq",
        ),
    )

    message = update.effective_message
    if message is None or message.chat is None:
        return
    if callable(_on_root_view):
        try:
            _on_root_view()
        except Exception:  # pragma: no cover - metrics should not break flow
            logger.exception("faq.root_metric_failed")
    try:
        await safe_send(
            context.bot.send_message,
            method_name="send_message",
            kind="faq",
            chat_id=message.chat_id,
            text=_markdown_to_html(FAQ_INTRO),
            reply_markup=faq_keyboard(),
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )
    except Exception as exc:
        log.warning(
            "faq.command.failed",
            **build_log_extra(
                {"error": repr(exc)},
                user_id=user.id if user else None,
                chat_id=chat.id if chat else None,
                command="/faq",
                update_type=type(update).__name__,
            ),
        )


async def faq_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle FAQ inline button clicks."""

    query = update.callback_query
    if query is None or query.data is None:
        return

    user_id = update.effective_user.id if update.effective_user else None
    log = get_context_logger(context)
    try:
        await query.answer()
    except Exception as exc:
        log.warning("cbq.answer.fail", **build_log_extra({"error": repr(exc)}, update=update))
    log.info(
        "update.received",
        **build_log_extra(
            {"callback_data": query.data},
            update=update,
            command="/faq",
        ),
    )

    data = query.data
    key = data.removeprefix(CB_FAQ_PREFIX)

    if key == "back":
        if callable(_show_main_menu):
            await _show_main_menu(update, context)
            return
        chat = query.message.chat if query.message else update.effective_chat
        if chat is not None:
            await safe_send(
                context.bot.send_message,
                method_name="send_message",
                kind="faq",
                chat_id=chat.id,
                text=_markdown_to_html("📋 Главное меню"),
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
            )
        return

    text = FAQ_SECTIONS.get(key, "Раздел не найден.")
    if text != "Раздел не найден." and callable(_on_section_view):
        try:
            _on_section_view(key)
        except Exception:  # pragma: no cover - metrics should not break flow
            logger.exception("faq.section_metric_failed | section=%s", key)

    message = query.message
    if message is None:
        return
    chat = message.chat
    if chat is None:
        return
    await safe_edit(
        context.bot,
        chat.id,
        message.message_id,
        _markdown_to_html(text),
        faq_keyboard(),
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True,
        state=None,
    )


_BOLD_RE = re.compile(r"\*(.+?)\*")


def _markdown_to_html(text: str) -> str:
    """Convert the FAQ subset of Markdown to HTML."""

    result_parts = []
    last = 0
    for match in _BOLD_RE.finditer(text):
        result_parts.append(html.escape(text[last:match.start()]))
        result_parts.append(f"<b>{html.escape(match.group(1))}</b>")
        last = match.end()
    result_parts.append(html.escape(text[last:]))
    return "".join(result_parts).replace("\n", "<br/>")


__all__ = ["configure_faq", "faq_callback", "faq_command"]
