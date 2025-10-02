"""Handlers for the /help command and its aliases."""

from __future__ import annotations

from typing import Tuple

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from logging_utils import build_log_extra, get_context_logger
from settings import SUPPORT_USERNAME
from texts import HELP_COPY
from utils.telegram_safe import safe_send_text

_FALLBACK_LANG = "ru"
_SUPPORTED_PREFIXES = {"en": "en"}
_SUPPORT_USERNAME = SUPPORT_USERNAME.lstrip("@") or "BestAi_Support"
_SUPPORT_URL = f"https://t.me/{_SUPPORT_USERNAME}"


def _resolve_language(language_code: str | None) -> str:
    if not language_code:
        return _FALLBACK_LANG
    lowered = language_code.lower()
    for prefix, lang in _SUPPORTED_PREFIXES.items():
        if lowered.startswith(prefix):
            return lang
    return _FALLBACK_LANG


def _help_text(language_code: str | None) -> Tuple[str, str]:
    lang = _resolve_language(language_code)
    data = HELP_COPY.get(lang, HELP_COPY[_FALLBACK_LANG])
    text_template = data.get("text", "")
    button_caption = data.get("button", "")
    text = text_template.format(support_username=_SUPPORT_USERNAME)
    return text, button_caption


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a localized support message with a deep link to support chat."""

    logger = get_context_logger(context)
    log_extra = build_log_extra(update=update, command="/help")
    logger.info("update.received", **log_extra)
    logger.debug("command.dispatch", **log_extra)

    user = getattr(update, "effective_user", None)
    language_code = getattr(user, "language_code", None)
    text, button_caption = _help_text(language_code)
    markup = InlineKeyboardMarkup(
        [[InlineKeyboardButton(button_caption, url=_SUPPORT_URL)]]
    )

    result = await safe_send_text(
        update,
        context,
        text=text,
        reply_markup=markup,
        disable_web_page_preview=True,
    )

    payload: dict[str, object] = {}
    if result.message_id is not None:
        payload["message_id"] = result.message_id
    if result.description:
        payload["error"] = result.description
    if result.error_code is not None:
        payload["error_code"] = result.error_code

    result_extra = build_log_extra(payload, update=update, command="/help")
    if result.ok:
        logger.info("send.ok", **result_extra)
    else:
        logger.warning("send.fail", **result_extra)


__all__ = ["help_command"]
