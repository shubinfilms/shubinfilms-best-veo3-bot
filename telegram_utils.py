"""Helpers for resilient Telegram API calls with retries and metrics."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
from asyncio.subprocess import PIPE
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, MutableMapping, Optional, Sequence

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Message
from telegram.constants import ParseMode
from telegram.error import BadRequest, Forbidden, NetworkError, RetryAfter, TelegramError, TimedOut

from metrics import telegram_send_total

log = logging.getLogger("telegram.utils")

_ENV = (os.getenv("APP_ENV") or "prod").strip() or "prod"
_BOT_LABELS = {"env": _ENV, "service": "bot"}
_RETRY_SCHEDULE = (0.6, 1.0, 1.6)
_TEMP_ERROR_CODES = {409, 420, 429}
_PERM_ERROR_CODES = {400, 403, 404}

_DEFAULT_PROMPTS_URL = "https://t.me/bestveo3promts"
_HUB_PROMPTS_URL = (os.getenv("PROMPTS_CHANNEL_URL") or _DEFAULT_PROMPTS_URL).strip() or _DEFAULT_PROMPTS_URL


def build_hub_text(user_balance: int) -> str:
    """Render the main hub text with the current balance."""

    try:
        balance_value = int(user_balance)
    except (TypeError, ValueError):
        balance_value = 0

    return (
        "👋 Добро пожаловать!\n\n"
        f"💎 Ваш баланс: {balance_value}\n"
        f"📈 Больше идей и примеров — [канал с промптами]({_HUB_PROMPTS_URL})\n\n"
        "Выберите, что хотите сделать:"
    )


def build_hub_keyboard() -> InlineKeyboardMarkup:
    """Return a compact 2x3 inline keyboard for the emoji hub."""

    rows = [
        [
            InlineKeyboardButton("🎬", callback_data="hub:video"),
            InlineKeyboardButton("🎨", callback_data="hub:image"),
            InlineKeyboardButton("🎵", callback_data="hub:music"),
        ],
        [
            InlineKeyboardButton("🧠", callback_data="hub:prompt"),
            InlineKeyboardButton("💬", callback_data="hub:chat"),
            InlineKeyboardButton("💎", callback_data="hub:balance"),
        ],
    ]
    return InlineKeyboardMarkup(rows)


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

    @property
    def changed(self) -> bool:
        return self.status in {"edited", "sent", "resent"}


def _hash_value(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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
) -> SafeEditResult:
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
    text_hash = _hash_value(text)
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
            text=text,
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
            text=text,
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
            log.info(
                "safe_edit.noop",
                extra={"meta": {"chat_id": chat_id, "message_id": message_id}},
            )
            state_obj["last_text_hash"] = text_hash
            state_obj["last_markup_hash"] = markup_hash
            return SafeEditResult(message_id=message_id, status="skipped")
        if "message to edit not found" in lowered:
            log.info(
                "safe_edit.missing_message",
                extra={"meta": {"chat_id": chat_id, "message_id": message_id}},
            )
            if isinstance(state_obj, MutableMapping):
                state_obj["msg_id"] = None
            resend = await _send_new()
            return SafeEditResult(message_id=resend.message_id, status="resent")
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


__all__ = [
    "safe_send",
    "safe_send_text",
    "safe_send_placeholder",
    "safe_edit_text",
    "safe_edit_markdown_v2",
    "run_ffmpeg",
    "md2_escape",
    "build_inline_kb",
    "safe_edit",
    "SafeEditResult",
]
