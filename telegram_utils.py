"""Helpers for resilient Telegram API calls with retries and metrics."""
from __future__ import annotations

import asyncio
import copy
import hashlib
import html
import io
import json
import logging
import os
import random
import re
from asyncio.subprocess import PIPE
from contextlib import suppress
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any, Awaitable, Callable, Mapping, MutableMapping, Optional, Sequence, Union

from pathlib import Path
from io import BytesIO
from typing import Literal

from PIL import Image

import requests
from requests import Response

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, InputFile, Message
from telegram.constants import ParseMode
from telegram.error import BadRequest, Forbidden, NetworkError, RetryAfter, TelegramError, TimedOut

from metrics import telegram_send_total
from core.balance_provider import BalanceSnapshot
from keyboards import CB_VIDEO_MENU, main_menu_kb

log = logging.getLogger("telegram.utils")

_ENV = (os.getenv("APP_ENV") or "prod").strip() or "prod"
_BOT_LABELS = {"env": _ENV, "service": "bot"}
_RETRY_SCHEDULE = (0.6, 1.0, 1.6)
_TEMP_ERROR_CODES = {409, 420, 429}
_PERM_ERROR_CODES = {400, 403, 404}

_ALLOWED_IMAGE_MIME = {"image/png", "image/jpeg", "image/webp"}
_ALLOWED_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp"}
_IMAGE_FORMAT_TO_MIME = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}
_MAX_DOCUMENT_BYTES = 20 * 1024 * 1024
_MAX_IMAGE_DIMENSION = 12000
_MAX_IMAGE_PIXELS = 12000 * 12000


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


def build_hub_text(balance: Union[int, BalanceSnapshot]) -> str:
    """Return the formatted text for the main menu card."""

    return "<b>📋 Главное меню</b>\n<i>Выберите раздел:</i>"


def build_hub_keyboard() -> InlineKeyboardMarkup:
    """Return the inline keyboard markup for the main menu card."""

    return main_menu_kb()


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


def _memory_filename(file_path: str, fallback: str, ext: str) -> str:
    candidate = Path(file_path or "").name
    if candidate:
        return candidate
    suffix = ext if ext.startswith(".") else f".{ext}" if ext else ""
    return f"{fallback}{suffix}"


def _validate_image_bytes(data: bytes) -> tuple[int, int, str]:
    try:
        with Image.open(BytesIO(data)) as im:
            im.verify()
        with Image.open(BytesIO(data)) as im2:
            width, height = im2.size
            fmt = (im2.format or "").upper()
    except Exception as exc:
        raise TelegramImageError("invalid_image", "invalid image data") from exc

    if width <= 0 or height <= 0:
        raise TelegramImageError("invalid_image", "invalid dimensions")
    if width > _MAX_IMAGE_DIMENSION or height > _MAX_IMAGE_DIMENSION or (width * height) > _MAX_IMAGE_PIXELS:
        raise TelegramImageError("too_large", "image dimensions are too large")

    return width, height, fmt


async def _download_file_bytes(file: Any) -> tuple[bytes, str]:
    downloader = getattr(file, "download_as_bytearray", None)
    if callable(downloader):
        data = await downloader()
        return bytes(data or b""), getattr(file, "file_path", "")

    downloader = getattr(file, "download_as_bytes", None)
    if callable(downloader):
        data = await downloader()
        return bytes(data or b""), getattr(file, "file_path", "")

    downloader = getattr(file, "download_to_memory", None)
    if callable(downloader):
        buffer = await downloader()
        if hasattr(buffer, "read"):
            return buffer.read(), getattr(file, "file_path", "")
        if isinstance(buffer, (bytes, bytearray)):
            return bytes(buffer), getattr(file, "file_path", "")

    downloader = getattr(file, "download", None)
    if callable(downloader):
        stream = BytesIO()
        result = downloader(out=stream)
        if asyncio.iscoroutine(result):
            await result
        return stream.getvalue(), getattr(file, "file_path", "")

    raise TelegramImageError("download_failed", "no download method available")


async def download_image_from_update(update: Any, bot: Any) -> TelegramImage:
    """Return raw bytes and metadata for a Telegram photo/document update."""

    message = getattr(update, "effective_message", None) or getattr(update, "message", None)
    if message is None:
        raise TelegramImageError("no_message", "update has no message")

    document = getattr(message, "document", None)
    photos = list(getattr(message, "photo", []) or [])

    if document is not None:
        mime_type = (getattr(document, "mime_type", "") or "").lower()
        file_name = getattr(document, "file_name", "") or ""
        extension = Path(file_name).suffix.lower()
        if mime_type not in _ALLOWED_IMAGE_MIME and extension not in _ALLOWED_IMAGE_EXT:
            raise TelegramImageError("invalid_type", "unsupported document type")
        file_size = getattr(document, "file_size", None)
        if isinstance(file_size, int) and file_size > _MAX_DOCUMENT_BYTES:
            raise TelegramImageError("too_large", "document too large")
        try:
            file = await bot.get_file(document.file_id)
        except Exception as exc:
            raise TelegramImageError("download_failed", "failed to fetch document") from exc
        data, file_path = await _download_file_bytes(file)
        if not data:
            raise TelegramImageError("download_failed", "empty document")
        width, height, fmt = _validate_image_bytes(data)
        detected_mime = _IMAGE_FORMAT_TO_MIME.get(fmt, mime_type or "image/octet-stream")
        ext = _ALLOWED_IMAGE_EXT.intersection({Path(file_path or "").suffix.lower()})
        ext_value = next(iter(ext), f".{fmt.lower()}" if fmt else "")
        filename = _memory_filename(file_path, Path(file_name or "telegram_document").stem or "document", ext_value)
        return TelegramImage(
            data=data,
            file_path=file_path,
            mime_type=detected_mime,
            filename=filename,
            source="document",
            width=width,
            height=height,
        )

    if photos:
        photo = photos[-1]
        try:
            file = await bot.get_file(photo.file_id)
        except Exception as exc:
            raise TelegramImageError("download_failed", "failed to fetch photo") from exc
        data, file_path = await _download_file_bytes(file)
        if not data:
            raise TelegramImageError("download_failed", "empty photo")
        width, height, fmt = _validate_image_bytes(data)
        detected_mime = _IMAGE_FORMAT_TO_MIME.get(fmt, "image/jpeg")
        ext_value = Path(file_path or "").suffix or f".{fmt.lower()}" if fmt else ".jpg"
        fallback = getattr(photo, "file_unique_id", None) or "photo"
        filename = _memory_filename(file_path, fallback, ext_value)
        return TelegramImage(
            data=data,
            file_path=file_path,
            mime_type=detected_mime,
            filename=filename,
            source="photo",
            width=width,
            height=height,
        )

    raise TelegramImageError("no_image", "message has no photo or document")


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


async def safe_send_text(bot: Any, chat_id: int, text: str) -> Optional[Any]:
    return await bot.send_message(
        chat_id=chat_id,
        text=text,
        parse_mode=ParseMode.MARKDOWN_V2,
        disable_web_page_preview=True,
    )


def _sanitize_caption_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    parse_mode = payload.get("parse_mode")
    caption = payload.get("caption")
    if caption is None or not _should_sanitize(parse_mode):
        return dict(payload)
    sanitized = dict(payload)
    sanitized["caption"] = sanitize_html(str(caption))
    return sanitized


def _sanitize_media_group_payload(
    media_items: Sequence[Any],
    parse_mode: Any,
) -> list[Any]:
    if not _should_sanitize(parse_mode):
        return list(media_items)
    sanitized: list[Any] = []
    for item in media_items:
        caption = getattr(item, "caption", None)
        if caption is None:
            sanitized.append(item)
            continue
        sanitized_caption = sanitize_html(str(caption))
        item_kwargs: dict[str, Any] = {"media": getattr(item, "media", None), "caption": sanitized_caption}
        parse_mode = getattr(item, "parse_mode", None)
        if parse_mode is not None:
            item_kwargs["parse_mode"] = parse_mode
        caption_entities = getattr(item, "caption_entities", None)
        if caption_entities is not None:
            item_kwargs["caption_entities"] = caption_entities
        has_spoiler = getattr(item, "has_spoiler", None)
        if has_spoiler:
            item_kwargs["has_spoiler"] = has_spoiler
        sanitized.append(type(item)(**item_kwargs))
    return sanitized


async def safe_send_photo(
    bot: Any,
    *,
    chat_id: int,
    photo: Any,
    caption: Optional[str] = None,
    reply_markup: Optional[Any] = None,
    parse_mode: Optional[ParseMode] = None,
    kind: str = "photo",
    req_id: Optional[str] = None,
    max_attempts: int = 4,
    **kwargs: Any,
) -> Any:
    payload: dict[str, Any] = {
        "chat_id": chat_id,
        "photo": photo,
        "caption": caption,
        "reply_markup": reply_markup,
        "parse_mode": parse_mode,
    }
    payload.update(kwargs)
    sanitized = _sanitize_caption_payload(payload)
    return await safe_send(
        bot.send_photo,
        method_name="send_photo",
        kind=kind,
        req_id=req_id,
        max_attempts=max_attempts,
        **sanitized,
    )


async def safe_send_document(
    bot: Any,
    *,
    chat_id: int,
    document: Any,
    caption: Optional[str] = None,
    reply_markup: Optional[Any] = None,
    parse_mode: Optional[ParseMode] = None,
    kind: str = "document",
    req_id: Optional[str] = None,
    max_attempts: int = 4,
    **kwargs: Any,
) -> Any:
    payload: dict[str, Any] = {
        "chat_id": chat_id,
        "document": document,
        "caption": caption,
        "reply_markup": reply_markup,
        "parse_mode": parse_mode,
    }
    payload.update(kwargs)
    sanitized = _sanitize_caption_payload(payload)
    return await safe_send(
        bot.send_document,
        method_name="send_document",
        kind=kind,
        req_id=req_id,
        max_attempts=max_attempts,
        **sanitized,
    )


async def send_image_as_document(
    bot: Any,
    chat_id: int,
    data: bytes,
    filename: str,
    *,
    reply_markup: Optional[Any] = None,
    caption: Optional[str] = None,
    req_id: Optional[str] = None,
) -> Any:
    buffer = io.BytesIO(data)
    buffer.name = filename
    buffer.seek(0)
    input_file = InputFile(buffer, filename=filename)
    return await safe_send_document(
        bot,
        chat_id=chat_id,
        document=input_file,
        caption=caption,
        reply_markup=reply_markup,
        kind="image_document",
        req_id=req_id,
    )


async def safe_send_media_group(
    bot: Any,
    *,
    chat_id: int,
    media: Sequence[Any],
    parse_mode: Optional[ParseMode] = None,
    kind: str = "media_group",
    req_id: Optional[str] = None,
    max_attempts: int = 4,
    **kwargs: Any,
) -> Any:
    media_items = _sanitize_media_group_payload(media, parse_mode)
    payload: dict[str, Any] = {
        "chat_id": chat_id,
        "media": media_items,
        "parse_mode": parse_mode,
    }
    payload.update(kwargs)
    return await safe_send(
        bot.send_media_group,
        method_name="send_media_group",
        kind=kind,
        req_id=req_id,
        max_attempts=max_attempts,
        **payload,
    )


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
    "safe_send_photo",
    "safe_send_document",
    "safe_send_media_group",
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
class TelegramImageError(RuntimeError):
    """Raised when Telegram image download fails."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(slots=True)
class TelegramImage:
    """Container for Telegram photo/document payload."""

    data: bytes
    file_path: str
    mime_type: str
    filename: str
    source: Literal["photo", "document"]
    width: Optional[int]
    height: Optional[int]

