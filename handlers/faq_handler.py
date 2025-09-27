"""FAQ handlers rendering HTML-safe content."""

import html
import logging
import re
from typing import Awaitable, Callable, Optional

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from keyboards import CB_FAQ_PREFIX, faq_keyboard
from texts import FAQ_INTRO, FAQ_SECTIONS
from utils.safe_send import safe_send

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

    message = update.effective_message
    if message is None or message.chat is None:
        return
    if callable(_on_root_view):
        try:
            _on_root_view()
        except Exception:  # pragma: no cover - metrics should not break flow
            logger.exception("faq.root_metric_failed")
    await safe_send(
        context.bot,
        message.chat_id,
        _markdown_to_html(FAQ_INTRO),
        reply_markup=faq_keyboard(),
    )


async def faq_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle FAQ inline button clicks."""

    query = update.callback_query
    if query is None or query.data is None:
        return

    user_id = update.effective_user.id if update.effective_user else None
    logger.info("faq.callback | user_id=%s data=%s", user_id, query.data)

    data = query.data
    key = data.removeprefix(CB_FAQ_PREFIX)

    if key == "back":
        await query.answer()
        if callable(_show_main_menu):
            await _show_main_menu(update, context)
            return
        chat = query.message.chat if query.message else update.effective_chat
        if chat is not None:
            await safe_send(context.bot, chat.id, _markdown_to_html("📋 Главное меню"))
        return

    text = FAQ_SECTIONS.get(key, "Раздел не найден.")
    if text != "Раздел не найден." and callable(_on_section_view):
        try:
            _on_section_view(key)
        except Exception:  # pragma: no cover - metrics should not break flow
            logger.exception("faq.section_metric_failed | section=%s", key)

    await query.answer()
    await query.edit_message_text(
        _markdown_to_html(text),
        reply_markup=faq_keyboard(),
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True,
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
