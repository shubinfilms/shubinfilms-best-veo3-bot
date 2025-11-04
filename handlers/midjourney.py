from __future__ import annotations

import asyncio
import html
import logging
import time
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence

from telegram import Update
from telegram.ext import ContextTypes

from error_utils import handle_async_error
from helpers.errors import send_user_error
from services.db_async import get_user_balance_async
from services.kie_api_async import (
    KieAPIAsync,
    KieAPIHTTPError,
    KieAPITimeoutError,
    KieAPITransportError,
)

logger = logging.getLogger("midjourney.async")

_DEFAULT_GENERATE_PATHS: Sequence[str] = (
    "/api/v1/mj/generate",
    "/api/v1/mj/createTask",
    "/api/v1/mj/create-task",
)
_DEFAULT_STATUS_PATHS: Sequence[str] = (
    "/api/v1/mj/recordInfo",
    "/api/v1/mj/record-info",
    "/api/v1/mj/status",
)

_POLL_INTERVAL = 6.0
_POLL_TIMEOUT = 6 * 60.0

_client = KieAPIAsync()


class MidjourneyError(RuntimeError):
    """Base class for Midjourney async handler errors."""


class MidjourneyBadRequest(MidjourneyError):
    """Raised when the backend rejects a request due to invalid input."""


class MidjourneyTimeout(MidjourneyError):
    """Raised when polling timed out."""


class MidjourneyBackendError(MidjourneyError):
    """Raised for transport or unexpected backend errors."""


def _log_task_exception(task: asyncio.Task[Any]) -> None:
    try:
        exc = task.exception()
    except asyncio.CancelledError:
        return
    except Exception:  # pragma: no cover - defensive guard
        logger.exception("midjourney.task_exception_fetch_failed")
        return
    if exc is not None:
        logger.exception("midjourney.task_failed", exc_info=exc)


def _aspect_value(aspect: str) -> str:
    return "9:16" if aspect == "9:16" else "16:9"


def _short_prompt(prompt: str, limit: int = 120) -> str:
    text = prompt.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _extract_task_id(payload: Mapping[str, Any]) -> Optional[str]:
    for key in ("taskId", "task_id", "id"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    data_section = payload.get("data")
    if isinstance(data_section, Mapping):
        return _extract_task_id(data_section)
    return None


def _extract_flag(payload: Mapping[str, Any]) -> Optional[int]:
    for key in ("flag", "status", "statusFlag"):
        value = payload.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    data_section = payload.get("data")
    if isinstance(data_section, Mapping):
        return _extract_flag(data_section)
    return None


def _extract_result_urls(payload: Mapping[str, Any]) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()

    def _add_from(candidate: Iterable[Any]) -> None:
        for item in candidate:
            if not isinstance(item, str):
                continue
            normalized = item.strip()
            if not normalized:
                continue
            if normalized not in seen:
                urls.append(normalized)
                seen.add(normalized)

    direct_keys = (
        "imageUrls",
        "imageUrl",
        "resultUrls",
        "urls",
        "image_urls",
        "result_urls",
    )

    for key in direct_keys:
        value = payload.get(key)
        if isinstance(value, str):
            _add_from([value])
        elif isinstance(value, Sequence):
            _add_from(value)

    nested_candidates = (
        payload.get("resultInfo"),
        payload.get("resultInfoJson"),
        payload.get("result_json"),
        payload.get("data"),
    )
    for candidate in nested_candidates:
        if isinstance(candidate, Mapping):
            urls.extend(_extract_result_urls(candidate))
        elif isinstance(candidate, Sequence):
            for item in candidate:
                if isinstance(item, Mapping):
                    urls.extend(_extract_result_urls(item))

    return urls


def _resolve_error_reason(payload: Mapping[str, Any]) -> str:
    for key in ("message", "msg", "error", "reason", "detail"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "Midjourney service error"


async def _submit_job(prompt: str, aspect: str) -> str:
    payload = {
        "taskType": "mj_txt2img",
        "prompt": prompt,
        "speed": "fast",
        "aspectRatio": _aspect_value(aspect),
        "enableTranslation": True,
        "input": {
            "prompt": prompt,
            "aspectRatio": _aspect_value(aspect),
        },
    }

    last_error: Optional[Exception] = None
    for path in _DEFAULT_GENERATE_PATHS:
        try:
            response = await _client.request_json("POST", path, json_payload=payload)
        except KieAPIHTTPError as exc:
            message = _resolve_error_reason(exc.payload)
            if exc.status in {400, 401, 402, 403, 404, 422}:
                raise MidjourneyBadRequest(message) from exc
            last_error = MidjourneyBackendError(f"HTTP {exc.status}: {message}")
            continue
        except (KieAPITimeoutError, KieAPITransportError) as exc:
            last_error = MidjourneyBackendError(str(exc))
            continue

        task_id = _extract_task_id(response)
        if task_id:
            logger.info(
                "midjourney.submit",
                extra={"task_id": task_id, "path": path, "aspect": payload["aspectRatio"]},
            )
            return task_id
        last_error = MidjourneyBackendError("Midjourney response missing task id")

    if last_error:
        raise last_error
    raise MidjourneyBackendError("Midjourney submission failed")


async def _request_status(task_id: str) -> Mapping[str, Any]:
    params = {"taskId": task_id}
    last_error: Optional[Exception] = None
    for path in _DEFAULT_STATUS_PATHS:
        try:
            response = await _client.request_json("GET", path, params=params)
        except KieAPIHTTPError as exc:
            message = _resolve_error_reason(exc.payload)
            if exc.status == 404:
                last_error = MidjourneyBackendError("Midjourney task not found")
                continue
            last_error = MidjourneyBackendError(f"HTTP {exc.status}: {message}")
            continue
        except (KieAPITimeoutError, KieAPITransportError) as exc:
            last_error = MidjourneyBackendError(str(exc))
            continue
        return response
    if last_error:
        raise last_error
    raise MidjourneyBackendError("Midjourney status unavailable")


async def _poll_result(task_id: str) -> Sequence[str]:
    deadline = time.monotonic() + _POLL_TIMEOUT
    attempt = 0
    while True:
        attempt += 1
        if time.monotonic() > deadline:
            raise MidjourneyTimeout("Midjourney polling timed out")
        status_payload = await _request_status(task_id)
        flag = _extract_flag(status_payload)
        logger.debug(
            "midjourney.poll",
            extra={"task_id": task_id, "flag": flag, "attempt": attempt},
        )
        if flag == 1:
            urls = _extract_result_urls(status_payload)
            if urls:
                return urls
            raise MidjourneyBackendError("Midjourney returned empty result")
        if flag in (2, 3) or flag is None:
            reason = _resolve_error_reason(status_payload)
            raise MidjourneyBackendError(reason)
        await asyncio.sleep(_POLL_INTERVAL)


async def _send_gallery(context: ContextTypes.DEFAULT_TYPE, chat_id: int, urls: Sequence[str], prompt: str, aspect: str) -> None:
    caption = (
        "✅ Midjourney готово\n"
        f"Формат: <b>{html.escape(_aspect_value(aspect))}</b>\n"
        f"Промпт: <code>{html.escape(_short_prompt(prompt))}</code>"
    )
    try:
        media = urls[:4]
        if getattr(context.bot, "send_media_group", None) and len(media) > 1:
            photos = [
                {"type": "photo", "media": url} if idx else {"type": "photo", "media": url, "caption": caption, "parse_mode": "HTML"}
                for idx, url in enumerate(media)
            ]
            await context.bot.send_media_group(chat_id=chat_id, media=photos)
        elif getattr(context.bot, "send_photo", None):
            await context.bot.send_photo(chat_id=chat_id, photo=media[0], caption=caption, parse_mode="HTML")
        else:
            await context.bot.send_message(chat_id=chat_id, text=f"✅ Midjourney: {media[0]}")
    except Exception as exc:  # pragma: no cover - Telegram edge cases
        await handle_async_error(exc, "Midjourney.send_result")
        await context.bot.send_message(chat_id=chat_id, text=f"✅ Midjourney: {urls[0]}")


async def _run_job(
    *,
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: Optional[int],
    user_id: Optional[int],
    prompt: str,
    aspect: str,
    progress: Optional[MutableMapping[str, Any]],
) -> None:
    db_balance: Optional[int] = None
    if user_id is not None:
        try:
            db_balance = await get_user_balance_async(int(user_id))
        except Exception:  # pragma: no cover - diagnostics only
            logger.debug(
                "midjourney.balance_failed",
                exc_info=True,
                extra={"user_id": user_id},
            )

    try:
        task_id = await _submit_job(prompt, aspect)
    except MidjourneyBadRequest as exc:
        if progress is not None:
            progress["success"] = False
        await send_user_error(
            context,
            "content_policy",
            details={
                "chat_id": chat_id,
                "user_id": user_id,
                "mode": "midjourney",
                "reason": str(exc),
            },
        )
        return
    except MidjourneyError as exc:
        if progress is not None:
            progress["success"] = False
        await send_user_error(
            context,
            "backend_fail",
            details={
                "chat_id": chat_id,
                "user_id": user_id,
                "mode": "midjourney",
                "reason": str(exc),
            },
        )
        return

    logger.info(
        "midjourney.job_started",
        extra={"task_id": task_id, "chat_id": chat_id, "user_id": user_id, "balance": db_balance},
    )
    if progress is not None:
        progress["task_id"] = task_id

    try:
        urls = await _poll_result(task_id)
    except MidjourneyTimeout as exc:
        if progress is not None:
            progress["success"] = False
        await send_user_error(
            context,
            "timeout",
            details={
                "chat_id": chat_id,
                "user_id": user_id,
                "mode": "midjourney",
                "reason": str(exc),
                "req_id": task_id,
            },
        )
        return
    except MidjourneyError as exc:
        if progress is not None:
            progress["success"] = False
        await send_user_error(
            context,
            "backend_fail",
            details={
                "chat_id": chat_id,
                "user_id": user_id,
                "mode": "midjourney",
                "reason": str(exc),
                "req_id": task_id,
            },
        )
        return

    if chat_id is not None:
        await _send_gallery(context, chat_id, urls, prompt, aspect)
    if progress is not None:
        progress["success"] = True


async def handle_midjourney_prompt(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    aspect: str = "16:9",
) -> None:
    message = update.effective_message
    user = update.effective_user
    chat = update.effective_chat

    if message is None or not isinstance(message.text, str):
        await send_user_error(
            context,
            "invalid_input",
            details={
                "chat_id": chat.id if chat else None,
                "user_id": user.id if user else None,
                "mode": "midjourney",
                "reason": "no_prompt",
            },
        )
        return

    prompt = message.text.strip()
    if not prompt:
        await send_user_error(
            context,
            "invalid_input",
            details={
                "chat_id": chat.id if chat else None,
                "user_id": user.id if user else None,
                "mode": "midjourney",
                "reason": "empty_prompt",
            },
        )
        return

    chat_id = chat.id if chat else None
    user_id = user.id if user else None

    if getattr(message, "reply_text", None):
        await message.reply_text("🎨 Принято! Начинаю генерацию Midjourney…")
    elif chat_id is not None:
        await context.bot.send_message(chat_id, "🎨 Принято! Начинаю генерацию Midjourney…")

    progress: Optional[MutableMapping[str, Any]] = None
    chat_data = getattr(context, "chat_data", None)
    if isinstance(chat_data, MutableMapping) and chat_id is not None:
        progress = {
            "chat_id": chat_id,
            "user_id": user_id,
            "mode": "midjourney",
            "prompt": prompt,
            "success": False,
        }
        chat_data["midjourney_progress"] = progress

    task = asyncio.create_task(
        _run_job(
            context=context,
            chat_id=chat_id,
            user_id=user_id,
            prompt=prompt,
            aspect=aspect,
            progress=progress,
        ),
        name=f"mj:{chat_id}:{time.time_ns()}",
    )
    task.add_done_callback(_log_task_exception)


__all__ = ["handle_midjourney_prompt", "MidjourneyError", "MidjourneyTimeout", "MidjourneyBadRequest"]
