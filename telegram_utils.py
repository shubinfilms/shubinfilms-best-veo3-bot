"""Helpers for resilient Telegram API calls with retries and metrics."""
from __future__ import annotations

import asyncio
import hashlib
import html
import json
import logging
import os
import random
import re
from asyncio.subprocess import PIPE
from contextlib import suppress
from dataclasses import dataclass
from html.parser import HTMLParser
from functools import wraps
from typing import Any, Awaitable, Callable, Mapping, MutableMapping, Optional, Sequence

import requests
from requests import Response

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Message
from telegram.constants import ParseMode
from telegram.error import BadRequest, Forbidden, NetworkError, RetryAfter, TelegramError, TimedOut

from metrics import telegram_send_total
from redis_utils import redis_delete_keys
from settings import REDIS_PREFIX

log = logging.getLogger("telegram.utils")
BOT_LOG = logging.getLogger("bot")

_ENV = (os.getenv("APP_ENV") or "prod").strip() or "prod"
_BOT_LABELS = {"env": _ENV, "service": "bot"}
_RETRY_SCHEDULE = (0.6, 1.0, 1.6)
_TEMP_ERROR_CODES = {409, 420, 429}
_PERM_ERROR_CODES = {400, 403, 404}


def mask_tokens(text: Any) -> str:
    """Mask known sensitive tokens in ``text`` before logging."""

    if text is None:
        return ""
    secrets = [
        os.getenv("SUNO_CALLBACK_SECRET") or "",
        os.getenv("SUNO_API_TOKEN") or "",
        os.getenv("TELEGRAM_TOKEN") or "",
    ]
    cleaned = str(text)
    for secret in secrets:
        token = secret.strip()
        if token:
            cleaned = cleaned.replace(token, "***")
    return cleaned


def _context_logger(context: Any) -> logging.Logger:
    if context is None:
        return log
    application = getattr(context, "application", None)
    logger = getattr(application, "logger", None)
    if isinstance(logger, logging.Logger):
        return logger
    return log


def _purge_wait_hints(context: Any) -> None:
    if context is None:
        return
    for attr in ("chat_data", "user_data"):
        container = getattr(context, attr, None)
        if not isinstance(container, MutableMapping):
            continue
        for key in list(container.keys()):
            if not isinstance(key, str):
                continue
            lowered = key.lower()
            if lowered in {"wait", "waiting", "wait_state", "waiting_state"} or lowered.startswith(
                ("wait_", "_wait")
            ):
                container.pop(key, None)


def extract_user_id(*sources: Any) -> Optional[int]:
    for src in sources:
        if not src:
            continue
        user = getattr(src, "effective_user", None)
        user_id = getattr(user, "id", None)
        if user_id:
            try:
                return int(user_id)
            except (TypeError, ValueError):
                continue
        from_user = getattr(getattr(src, "message", None), "from_user", None)
        if from_user is not None:
            user_id = getattr(from_user, "id", None)
            if user_id:
                try:
                    return int(user_id)
                except (TypeError, ValueError):
                    continue
    return None


async def reset_user_state(update: Any = None, context: Any = None) -> int:
    """Remove wait/fsm flags for the user associated with ``update``/``context``."""

    user_id = extract_user_id(update, context)
    if not user_id:
        _purge_wait_hints(context)
        return 0

    redis_client = None
    redis_prefix = REDIS_PREFIX

    candidates = []
    if context is not None:
        candidates.append(getattr(context, "bot_data", None))
        application = getattr(context, "application", None)
        candidates.append(getattr(application, "bot_data", None))
    if update is not None:
        candidates.append(getattr(update, "bot_data", None))
        application = getattr(update, "application", None)
        candidates.append(getattr(application, "bot_data", None))

    for candidate in candidates:
        if isinstance(candidate, MutableMapping):
            redis_client = candidate.get("redis") if redis_client is None else redis_client
            redis_prefix = candidate.get("redis_prefix", redis_prefix)

    patterns = [
        f"{redis_prefix}:wait:{user_id}:*",
        f"{redis_prefix}:fsm:{user_id}:*",
    ]

    deleted = await redis_delete_keys(redis_client, patterns) if redis_client else 0
    BOT_LOG.debug("state.reset | user=%s deleted=%s", user_id, deleted)
    _purge_wait_hints(context)
    return deleted


def with_state_reset(
    handler: Callable[[Any, Any], Awaitable[Any]]
) -> Callable[[Any, Any], Awaitable[Any]]:
    """Wrap ``handler`` to reset wait-state flags and provide unified error handling."""

    @wraps(handler)
    async def wrapped(update: Any, context: Any, *args: Any, **kwargs: Any) -> Any:
        logger = _context_logger(context)

        query = getattr(update, "callback_query", None)
        if query is not None:
            answer = getattr(query, "answer", None)
            if callable(answer):
                try:
                    await answer()
                except BadRequest:
                    with suppress(BadRequest):
                        await answer()
                except Exception:
                    logger.debug("callback.answer_failed", exc_info=True)

        try:
            await reset_user_state(update, context)
        except Exception:
            logger.warning("state_reset_failed", exc_info=True)

        user_id = extract_user_id(update, context)
        BOT_LOG.debug("handler.invoke | name=%s user=%s", getattr(handler, "__name__", "handler"), user_id)

        try:
            result = handler(update, context, *args, **kwargs)
            if asyncio.iscoroutine(result):
                return await result
            return result
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("handler error")
            query = getattr(update, "callback_query", None)
            if query is not None:
                answer = getattr(query, "answer", None)
                if callable(answer):
                    try:
                        await answer("Ошибка. Попробуйте ещё раз.", show_alert=False)
                    except Exception:
                        pass
            chat = getattr(update, "effective_chat", None)
            if chat is not None:
                send = getattr(chat, "send_message", None)
                if callable(send):
                    try:
                        await send("Упс… Попробуйте /menu")
                    except Exception:
                        pass
            return None

    setattr(wrapped, "__with_state_reset__", True)
    return wrapped

_ALLOWED_HTML_TAGS = {"b", "i", "u", "s", "a", "code", "pre", "blockquote", "tg-spoiler"}
_ALLOWED_HTML_ATTRS = {"a": {"href"}}
_HTML_REMAP = {"strong": "b", "em": "i"}


class _TelegramHTMLSanitizer(HTMLParser):
    """Sanitize limited HTML subset supported by Telegram."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self._parts: list[str] = []
        self._stack: list[str] = []

    def _append(self, value: str) -> None:
        if value:
            self._parts.append(value)

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[override]
        normalized = _HTML_REMAP.get(tag.lower(), tag.lower())
        if normalized == "br":
            self._append("\n")
            return
        if normalized not in _ALLOWED_HTML_TAGS:
            return
        filtered = []
        allowed_attrs = _ALLOWED_HTML_ATTRS.get(normalized, set())
        for name, value in attrs:
            if value is None:
                continue
            attr_name = (name or "").lower()
            if attr_name not in allowed_attrs:
                continue
            if normalized == "a" and value.lower().startswith("javascript:"):
                continue
            filtered.append((attr_name, value))
        attr_payload = "".join(
            f' {name}="{html.escape(value, quote=True)}"' for name, value in filtered
        )
        self._append(f"<{normalized}{attr_payload}>")
        self._stack.append(normalized)

    def handle_startendtag(self, tag: str, attrs) -> None:  # type: ignore[override]
        normalized = _HTML_REMAP.get(tag.lower(), tag.lower())
        if normalized == "br":
            self._append("\n")
            return
        # Treat <tag/> as <tag></tag> when allowed.
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:  # type: ignore[override]
        normalized = _HTML_REMAP.get(tag.lower(), tag.lower())
        if normalized == "br":
            return
        if normalized not in _ALLOWED_HTML_TAGS:
            return
        while self._stack:
            top = self._stack.pop()
            self._append(f"</{top}>")
            if top == normalized:
                break

    def handle_data(self, data: str) -> None:  # type: ignore[override]
        if not data:
            return
        self._append(html.escape(data))

    def handle_entityref(self, name: str) -> None:  # type: ignore[override]
        self._append(f"&{name};")

    def handle_charref(self, name: str) -> None:  # type: ignore[override]
        self._append(f"&#{name};")

    def get_text(self) -> str:
        return "".join(self._parts)


def _should_sanitize(parse_mode: Any) -> bool:
    if parse_mode is None:
        return False
    if isinstance(parse_mode, ParseMode):
        return parse_mode == ParseMode.HTML
    if isinstance(parse_mode, str):
        return parse_mode.upper() == "HTML"
    return False


def sanitize_html(text: str) -> str:
    """Convert HTML into Telegram-safe subset."""

    if not text:
        return ""
    normalized = text.replace("\r", "")
    normalized = re.sub(r"<br\s*/?>", "\n", normalized, flags=re.IGNORECASE)
    parser = _TelegramHTMLSanitizer()
    try:
        parser.feed(normalized)
        parser.close()
    except Exception:  # pragma: no cover - defensive fallback
        log.exception("sanitize_html.failure")
        return html.escape(normalized)
    return parser.get_text()


def build_hub_text(user_balance: int) -> str:
    """Render the main hub text with the current balance."""

    return "<b>🏠 Главное меню</b>\nВыберите нужный раздел:"


def build_hub_keyboard() -> InlineKeyboardMarkup:
    """Return the default inline keyboard for the main hub."""

    rows = [
        [InlineKeyboardButton("🧠 Prompt-Master", callback_data="hub:prompt")],
        [
            InlineKeyboardButton("🎬 Генерация видео", callback_data="hub:video"),
            InlineKeyboardButton("🎨 Генерация изображений", callback_data="hub:image"),
        ],
        [
            InlineKeyboardButton("🎵 Генерация музыки", callback_data="hub:music"),
            InlineKeyboardButton("💬 Обычный чат", callback_data="hub:chat"),
        ],
        [
            InlineKeyboardButton("💎 Баланс", callback_data="hub:balance"),
            InlineKeyboardButton("🌐 Язык", callback_data="hub:lang"),
        ],
        [InlineKeyboardButton("🆘 Поддержка", callback_data="hub:help")],
    ]
    return InlineKeyboardMarkup(rows)


async def send_html(
    target: Any,
    text: str,
    reply_markup: InlineKeyboardMarkup | None = None,
    *,
    bot: Any | None = None,
    disable_web_page_preview: bool = True,
    **kwargs: Any,
) -> Any:
    """Send a Telegram message rendered as HTML.

    The ``target`` argument can be a chat identifier, an ``Update`` object or a
    PTB ``CallbackContext``.  When an integer is provided ``bot`` must also be
    supplied.  This helper normalises parse mode and disables link previews by
    default.
    """

    resolved_bot = None
    chat_id = None

    if hasattr(target, "bot") and hasattr(target, "application"):
        resolved_bot = target.bot
        chat = getattr(target, "effective_chat", None)
        chat_id = getattr(chat, "id", None)
    elif hasattr(target, "bot") and bot is None:
        resolved_bot = target.bot
        chat_id = getattr(target, "chat_id", None)
    elif hasattr(target, "send_message") and bot is None:
        resolved_bot = target
    elif isinstance(target, (int, str)):
        chat_id = int(target)
        resolved_bot = bot
    else:
        resolved_bot = bot

    if resolved_bot is None:
        raise ValueError("send_html requires a bot instance")

    if chat_id is None:
        chat_id = getattr(target, "chat_id", None)

    if chat_id is None:
        chat = getattr(target, "effective_chat", None)
        chat_id = getattr(chat, "id", None)

    if chat_id is None:
        raise ValueError("send_html requires a chat identifier")

    payload = dict(kwargs)
    if reply_markup is not None:
        payload["reply_markup"] = reply_markup

    payload.setdefault("parse_mode", ParseMode.HTML)
    if disable_web_page_preview:
        payload.setdefault("disable_web_page_preview", True)

    return await resolved_bot.send_message(chat_id=chat_id, text=text, **payload)


def escape(value: str) -> str:
    """Escape ``value`` for safe HTML rendering without touching quotes."""

    return html.escape(value or "", quote=False)




def _extract_status(exc: BaseException) -> Optional[int]:
    for attr in ("status_code", "code", "error_code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    return None


def _is_message_not_modified(exc: BadRequest) -> bool:
    message = str(exc)
    return "message is not modified" in message.lower()


def _should_retry(exc: BaseException, status: Optional[int]) -> bool:
    if isinstance(exc, RetryAfter):
        return True
    if isinstance(exc, (TimedOut, NetworkError)):
        return True
    if status is None:
        return False
    if status in _TEMP_ERROR_CODES:
        return True
    if status >= 500:
        return True
    return False


def _retry_delay(attempt: int) -> float:
    base = _RETRY_SCHEDULE[min(attempt - 1, len(_RETRY_SCHEDULE) - 1)]
    return base * random.uniform(0.6, 1.6)


async def _call_telegram(method: Callable[..., Awaitable[Any]], **kwargs: Any) -> Any:
    return await method(**kwargs)


async def safe_send(
    method: Callable[..., Awaitable[Any]],
    *,
    method_name: str,
    kind: str,
    req_id: Optional[str] = None,
    max_attempts: int = 4,
    **kwargs: Any,
) -> Any:
    """Call a Telegram Bot API coroutine with retry handling."""

    if _should_sanitize(kwargs.get("parse_mode")) and "text" in kwargs:
        text_value = kwargs.get("text")
        if text_value is not None:
            kwargs = dict(kwargs)
            kwargs["text"] = sanitize_html(str(text_value))

    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        try:
            result = await _call_telegram(method, **kwargs)
            telegram_send_total.labels(kind=kind, result="ok", **_BOT_LABELS).inc()
            if attempt > 1:
                log.info(
                    "telegram send succeeded",
                    extra={
                        "meta": {
                            "method": method_name,
                            "attempt": attempt,
                            "req_id": req_id,
                        }
                    },
                )
            return result
        except BadRequest as exc:
            if _is_message_not_modified(exc):
                telegram_send_total.labels(kind=kind, result="ok", **_BOT_LABELS).inc()
                log.info(
                    "telegram send noop",
                    extra={
                        "meta": {
                            "method": method_name,
                            "attempt": attempt,
                            "req_id": req_id,
                            "reason": "message_not_modified",
                        }
                    },
                )
                return None
            status = 400
            if _should_retry(exc, status) and attempt < max_attempts:
                telegram_send_total.labels(kind=kind, result="retry", **_BOT_LABELS).inc()
                delay = _retry_delay(attempt)
                log.warning(
                    "telegram retry",
                    extra={
                        "meta": {
                            "method": method_name,
                            "attempt": attempt,
                            "delay": round(delay, 3),
                            "status": status,
                            "req_id": req_id,
                        }
                    },
                )
                await asyncio.sleep(delay)
                continue
            telegram_send_total.labels(kind=kind, result="fail", **_BOT_LABELS).inc()
            log.warning(
                "telegram send permanent failure",
                extra={
                    "meta": {
                        "method": method_name,
                        "status": status,
                        "attempt": attempt,
                        "req_id": req_id,
                        "error": str(exc),
                    }
                },
            )
            raise
        except Forbidden as exc:
            status = 403
            telegram_send_total.labels(kind=kind, result="fail", **_BOT_LABELS).inc()
            log.warning(
                "telegram forbidden",
                extra={
                    "meta": {
                        "method": method_name,
                        "status": status,
                        "attempt": attempt,
                        "req_id": req_id,
                        "error": str(exc),
                    }
                },
            )
            raise
        except RetryAfter as exc:
            delay_base = getattr(exc, "retry_after", None)
            if delay_base is None:
                delay = _retry_delay(attempt)
            else:
                delay = float(delay_base) * random.uniform(0.6, 1.6)
            if attempt < max_attempts:
                telegram_send_total.labels(kind=kind, result="retry", **_BOT_LABELS).inc()
                log.warning(
                    "telegram retry",
                    extra={
                        "meta": {
                            "method": method_name,
                            "attempt": attempt,
                            "delay": round(delay, 3),
                            "status": getattr(exc, "status_code", 429),
                            "req_id": req_id,
                        }
                    },
                )
                await asyncio.sleep(delay)
                continue
            telegram_send_total.labels(kind=kind, result="fail", **_BOT_LABELS).inc()
            log.warning(
                "telegram send failure",
                extra={
                    "meta": {
                        "method": method_name,
                        "status": getattr(exc, "status_code", 429),
                        "attempt": attempt,
                        "req_id": req_id,
                        "error": str(exc),
                    }
                },
            )
            raise
        except TelegramError as exc:
            status = _extract_status(exc)
            if status in _PERM_ERROR_CODES:
                telegram_send_total.labels(kind=kind, result="fail", **_BOT_LABELS).inc()
                log.warning(
                    "telegram send permanent failure",
                    extra={
                        "meta": {
                            "method": method_name,
                            "status": status,
                            "attempt": attempt,
                            "req_id": req_id,
                            "error": str(exc),
                        }
                    },
                )
                raise
            if _should_retry(exc, status) and attempt < max_attempts:
                telegram_send_total.labels(kind=kind, result="retry", **_BOT_LABELS).inc()
                delay = _retry_delay(attempt)
                log.warning(
                    "telegram retry",
                    extra={
                        "meta": {
                            "method": method_name,
                            "attempt": attempt,
                            "delay": round(delay, 3),
                            "status": status,
                            "req_id": req_id,
                        }
                    },
                )
                await asyncio.sleep(delay)
                continue
            telegram_send_total.labels(kind=kind, result="fail", **_BOT_LABELS).inc()
            log.warning(
                "telegram send failure",
                extra={
                    "meta": {
                        "method": method_name,
                        "status": status,
                        "attempt": attempt,
                        "req_id": req_id,
                        "error": str(exc),
                    }
                },
            )
            raise
        except (TimedOut, NetworkError) as exc:
            if attempt < max_attempts:
                telegram_send_total.labels(kind=kind, result="retry", **_BOT_LABELS).inc()
                delay = _retry_delay(attempt)
                log.warning(
                    "telegram retry",
                    extra={
                        "meta": {
                            "method": method_name,
                            "attempt": attempt,
                            "delay": round(delay, 3),
                            "req_id": req_id,
                            "error": str(exc),
                        }
                    },
                )
                await asyncio.sleep(delay)
                continue
            telegram_send_total.labels(kind=kind, result="fail", **_BOT_LABELS).inc()
            log.warning(
                "telegram send failure",
                extra={
                    "meta": {
                        "method": method_name,
                        "attempt": attempt,
                        "req_id": req_id,
                        "error": str(exc),
                    }
                },
            )
            raise
    telegram_send_total.labels(kind=kind, result="fail", **_BOT_LABELS).inc()
    raise RuntimeError(f"Telegram send exceeded retries for {method_name}")


_MD2_SPECIALS = set("_*[]()~`>#+-=|{}.!")


def md2_escape(text: str) -> str:
    result = []
    for ch in text or "":
        if ch in _MD2_SPECIALS:
            result.append("\\" + ch)
        else:
            result.append(ch)
    return "".join(result)


def build_inline_kb(
    rows: Sequence[Sequence[tuple[str, str]]]
) -> InlineKeyboardMarkup:
    """Construct an :class:`InlineKeyboardMarkup` from label/callback rows."""

    keyboard_rows: list[list[InlineKeyboardButton]] = []
    for row in rows:
        buttons = [InlineKeyboardButton(label, callback_data=callback) for label, callback in row]
        keyboard_rows.append(buttons)
    return InlineKeyboardMarkup(keyboard_rows)


@dataclass
class SafeEditResult:
    message_id: Optional[int]
    status: str
    reason: Optional[str] = None

    @property
    def changed(self) -> bool:
        return self.status.startswith(("edited", "sent", "resent"))


def _hash_value(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _bust_cache_text(text: str) -> str:
    """Append an invisible character to force Telegram to treat text as changed."""

    suffix = "\u2060"
    if text.endswith(suffix):
        return text + suffix
    return text + suffix


async def safe_edit(
    bot: Any,
    chat_id: int,
    message_id: Optional[int],
    text: str,
    reply_markup: Optional[InlineKeyboardMarkup],
    parse_mode: str = "HTML",
    *,
    disable_web_page_preview: bool = True,
    state: Optional[MutableMapping[str, Any]] = None,
    resend_on_not_modified: bool = False,
) -> SafeEditResult:
    if _should_sanitize(parse_mode):
        text_to_send = sanitize_html(text)
    else:
        text_to_send = text

    state_obj: MutableMapping[str, Any]
    if isinstance(state, MutableMapping):
        state_obj = state
    else:
        state_obj = {}

    if reply_markup is None:
        markup_payload: Any = None
    else:
        try:
            markup_payload = reply_markup.to_dict()
        except AttributeError:
            markup_payload = reply_markup
    markup_json = json.dumps(markup_payload, ensure_ascii=False, sort_keys=True)
    text_hash = _hash_value(text_to_send)
    markup_hash = _hash_value(markup_json)

    last_text_hash = state_obj.get("last_text_hash") if isinstance(state_obj, MutableMapping) else None
    last_markup_hash = state_obj.get("last_markup_hash") if isinstance(state_obj, MutableMapping) else None

    if (
        isinstance(last_text_hash, str)
        and isinstance(last_markup_hash, str)
        and last_text_hash == text_hash
        and last_markup_hash == markup_hash
        and message_id is not None
    ):
        log.info(
            "safe_edit.skipped",
            extra={"meta": {"chat_id": chat_id, "message_id": message_id}},
        )
        state_obj["last_text_hash"] = text_hash
        state_obj["last_markup_hash"] = markup_hash
        if isinstance(state_obj, MutableMapping):
            state_obj.setdefault("msg_id", message_id)
        return SafeEditResult(message_id=message_id, status="skipped")

    async def _send_new() -> SafeEditResult:
        sent = await bot.send_message(
            chat_id=chat_id,
            text=text_to_send,
            parse_mode=parse_mode,
            reply_markup=reply_markup,
            disable_web_page_preview=disable_web_page_preview,
        )
        new_id = getattr(sent, "message_id", None)
        if isinstance(state_obj, MutableMapping):
            state_obj["msg_id"] = new_id
        state_obj["last_text_hash"] = text_hash
        state_obj["last_markup_hash"] = markup_hash
        return SafeEditResult(message_id=new_id, status="sent")

    if message_id is None:
        return await _send_new()

    try:
        edited = await bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text=text_to_send,
            parse_mode=parse_mode,
            reply_markup=reply_markup,
            disable_web_page_preview=disable_web_page_preview,
        )
        new_id = getattr(edited, "message_id", message_id)
        if isinstance(state_obj, MutableMapping):
            state_obj["msg_id"] = new_id
        state_obj["last_text_hash"] = text_hash
        state_obj["last_markup_hash"] = markup_hash
        return SafeEditResult(message_id=new_id, status="edited")
    except BadRequest as exc:
        lowered = str(exc).lower()
        if "message is not modified" in lowered:
            log_func = log.warning if resend_on_not_modified else log.info
            log_func(
                "safe_edit.noop",
                extra={
                    "meta": {
                        "chat_id": chat_id,
                        "message_id": message_id,
                        "last_text_hash": state_obj.get("last_text_hash"),
                        "new_text_hash": text_hash,
                        "last_markup_hash": state_obj.get("last_markup_hash"),
                        "new_markup_hash": markup_hash,
                    }
                },
            )
            state_obj["last_text_hash"] = text_hash
            state_obj["last_markup_hash"] = markup_hash
            if resend_on_not_modified:
                busted_text = _bust_cache_text(text_to_send)
                try:
                    edited = await bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=message_id,
                        text=busted_text,
                        parse_mode=parse_mode,
                        reply_markup=reply_markup,
                        disable_web_page_preview=disable_web_page_preview,
                    )
                except BadRequest as second_exc:
                    lowered_second = str(second_exc).lower()
                    if "message is not modified" not in lowered_second:
                        raise
                    try:
                        await bot.delete_message(chat_id, message_id)
                    except Exception as delete_exc:  # pragma: no cover - network issues
                        log.warning(
                            "safe_edit.delete_failed",
                            extra={
                                "meta": {
                                    "chat_id": chat_id,
                                    "message_id": message_id,
                                    "error": str(delete_exc),
                                }
                            },
                        )
                    if isinstance(state_obj, MutableMapping):
                        state_obj["msg_id"] = None
                    resend = await _send_new()
                    return SafeEditResult(
                        message_id=resend.message_id,
                        status="resent",
                        reason="not_modified",
                    )
                else:
                    new_id = getattr(edited, "message_id", message_id)
                    if isinstance(state_obj, MutableMapping):
                        state_obj["msg_id"] = new_id
                    state_obj["last_text_hash"] = text_hash
                    state_obj["last_markup_hash"] = markup_hash
                    return SafeEditResult(
                        message_id=new_id,
                        status="edited",
                        reason="bust",
                    )
            return SafeEditResult(message_id=message_id, status="skipped", reason="not_modified")
        if "message to edit not found" in lowered:
            log.info(
                "safe_edit.missing_message",
                extra={"meta": {"chat_id": chat_id, "message_id": message_id}},
            )
            if isinstance(state_obj, MutableMapping):
                state_obj["msg_id"] = None
            resend = await _send_new()
            return SafeEditResult(message_id=resend.message_id, status="resent", reason="missing_msg")
        raise


async def safe_edit_markdown_v2(
    bot: Any,
    chat_id: int,
    message_id: int,
    text: str,
    *,
    reply_markup: Optional[InlineKeyboardMarkup] = None,
) -> Optional[Any]:
    try:
        return await bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text=text,
            parse_mode=ParseMode.MARKDOWN_V2,
            disable_web_page_preview=True,
            reply_markup=reply_markup,
        )
    except BadRequest as exc:
        err_text = str(exc)
        if _is_message_not_modified(exc):
            return None
        if "can't" in err_text.lower() and "edit" in err_text.lower():
            return await bot.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode=ParseMode.MARKDOWN_V2,
                disable_web_page_preview=True,
                reply_markup=reply_markup,
            )
        raise


async def safe_edit_text(bot: Any, chat_id: int, message_id: int, text: str) -> Optional[Any]:
    return await safe_edit_markdown_v2(bot, chat_id, message_id, text)


async def safe_send_text(
    bot: Any,
    chat_id: int,
    text: str,
    *,
    parse_mode: Optional[str] = ParseMode.MARKDOWN_V2,
    disable_web_page_preview: bool = True,
) -> Optional[Any]:
    payload: dict[str, Any] = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": disable_web_page_preview,
    }
    if parse_mode:
        payload["parse_mode"] = parse_mode
    return await bot.send_message(**payload)


async def safe_send_placeholder(bot: Any, chat_id: int, text: str) -> Optional[Message]:
    return await safe_send(
        bot.send_message,
        method_name="send_message",
        kind="placeholder",
        chat_id=chat_id,
        text=text,
        parse_mode=ParseMode.MARKDOWN_V2,
        disable_web_page_preview=True,
    )


async def safe_send_sticker(
    bot: Any,
    chat_id: int,
    sticker: str,
    *,
    req_id: Optional[str] = None,
    max_attempts: int = 4,
    **kwargs: Any,
) -> Optional[Any]:
    return await safe_send(
        bot.send_sticker,
        method_name="send_sticker",
        kind="sticker",
        req_id=req_id,
        max_attempts=max_attempts,
        chat_id=chat_id,
        sticker=sticker,
        **kwargs,
    )


async def run_ffmpeg(input_bytes: bytes, args: list[str], timeout: float = 40.0) -> bytes:
    ffmpeg_bin = (os.getenv("FFMPEG_BIN") or "ffmpeg").strip() or "ffmpeg"
    cmd = [ffmpeg_bin, *args]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=PIPE,
        stdout=PIPE,
        stderr=PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(input_bytes), timeout=timeout)
    except asyncio.TimeoutError as exc:
        proc.kill()
        with suppress(Exception):
            await proc.communicate()
        raise RuntimeError("ffmpeg timeout") from exc

    if proc.returncode != 0:
        err_text = (stderr or b"").decode("utf-8", "ignore")
        log.warning(
            "ffmpeg failed",
            extra={"meta": {"code": proc.returncode, "stderr": err_text[:400]}},
        )
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode}")
    return stdout


def _telegram_post_json(
    session: requests.Session,
    url: str,
    payload: Mapping[str, Any],
    *,
    timeout: float = 60.0,
) -> tuple[bool, Optional[str], Optional[int]]:
    try:
        response = session.post(url, json=payload, timeout=timeout)
    except requests.RequestException as exc:
        reason = f"network:{exc.__class__.__name__}"
        return False, reason, None
    if response.ok:
        return True, None, response.status_code
    return False, _extract_description(response), response.status_code


def _extract_description(response: Response) -> Optional[str]:
    try:
        data = response.json()
    except ValueError:
        data = None
    if isinstance(data, Mapping):
        for key in ("description", "error", "message"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    text = response.text or response.reason
    if isinstance(text, str) and text.strip():
        return text.strip()
    return None


def send_photo_request(
    session: requests.Session,
    url: str,
    *,
    chat_id: int,
    photo: str,
    caption: Optional[str] = None,
    reply_to: Optional[int] = None,
    timeout: float = 60.0,
) -> tuple[bool, Optional[str], Optional[int]]:
    payload: dict[str, Any] = {"chat_id": chat_id, "photo": photo}
    if caption:
        payload["caption"] = caption
    if reply_to:
        payload["reply_to_message_id"] = reply_to
    return _telegram_post_json(session, url, payload, timeout=timeout)


def send_audio_request(
    session: requests.Session,
    url: str,
    *,
    chat_id: int,
    audio: str,
    caption: Optional[str] = None,
    reply_to: Optional[int] = None,
    title: Optional[str] = None,
    thumb: Optional[str] = None,
    timeout: float = 60.0,
) -> tuple[bool, Optional[str], Optional[int]]:
    payload: dict[str, Any] = {"chat_id": chat_id, "audio": audio}
    if caption:
        payload["caption"] = caption
    if reply_to:
        payload["reply_to_message_id"] = reply_to
    if title:
        payload["title"] = title
    if thumb:
        payload["thumb"] = thumb
    return _telegram_post_json(session, url, payload, timeout=timeout)


def is_remote_file_error(status: Optional[int], reason: Optional[str]) -> bool:
    if status != 400:
        return False
    if not reason:
        return False
    lowered = reason.lower()
    keywords = [
        "file reference expired",
        "failed to get http url content",
        "wrong file identifier",
        "http url content",
        "wrong remote file id",
        "remote url not valid",
    ]
    return any(keyword in lowered for keyword in keywords)


__all__ = [
    "safe_send",
    "safe_send_text",
    "safe_send_placeholder",
    "safe_send_sticker",
    "safe_edit_text",
    "safe_edit_markdown_v2",
    "run_ffmpeg",
    "md2_escape",
    "build_inline_kb",
    "safe_edit",
    "SafeEditResult",
    "sanitize_html",
    "mask_tokens",
    "send_photo_request",
    "send_audio_request",
    "is_remote_file_error",
]
