"""Handlers for Banana image editing workflow."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import os
import time
import json
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from helpers.errors import send_user_error
from services.db_async import get_user_balance_async
from services.kie_api_async import (
    KieAPIAsync,
    KieAPIHTTPError,
    KieAPITimeoutError,
    KieAPITransportError,
)
from services.redis_cache import get_cached_result, set_cached_result
from ui.renderers.banana import banana_card_kb, banana_card_text
from utils.banana_state import BananaState, clear, load, save
from utils.files import validate_image

from error_utils import handle_async_error
from logging_utils import get_logger

log = get_logger("handlers.banana")

MAX_IMAGES = 4
_BANANA_MODEL = os.getenv("KIE_BANANA_MODEL", "google/nano-banana-edit")
_BANANA_CREATE_PATH = "/api/v1/jobs/createTask"
_BANANA_STATUS_PATH = "/api/v1/jobs/recordInfo"
_UPLOAD_PATH = "/api/v1/upload/base64"

_WAIT_STATES = {"waiting", "queuing", "queued", "generating", "processing", "running", "pending", "0", 0}
_SUCCESS_STATES = {"success", "1", 1, "done", "finished", "completed", "ready"}
_FAIL_STATES = {"fail", "failed", "error", "2", 2, "3", 3, "canceled", "cancelled", "timeout"}

_POLL_INTERVAL = 4.0
_POLL_TIMEOUT = 8 * 60.0

_client = KieAPIAsync()


class BananaError(RuntimeError):
    """Base error for Banana async workflow."""


class BananaBadRequest(BananaError):
    """Raised when the backend rejects the payload as invalid."""


class BananaTimeout(BananaError):
    """Raised when polling timed out."""


class BananaBackendError(BananaError):
    """Raised for unexpected backend failures."""


def _log_task_exception(task: asyncio.Task[Any]) -> None:
    try:
        exc = task.exception()
    except asyncio.CancelledError:
        return
    except Exception:  # pragma: no cover - defensive guard
        log.exception("banana.task_exception_fetch_failed")
        return
    if exc is not None:
        log.exception("banana.task_failed", exc_info=exc)


async def _require_redis(context) -> object:
    redis = getattr(context, "redis", None)
    if redis is None:
        raise RuntimeError("context.redis is not configured")
    return redis


async def _get_balance_value(user_id: int) -> int:
    try:
        balance = await get_user_balance_async(int(user_id))
    except Exception:  # pragma: no cover - diagnostics only
        log.debug("banana.balance_failed", exc_info=True, extra={"user_id": user_id})
        return 0
    if balance is None:
        return 0
    return int(balance)


async def _fetch_file_bytes(bot, file_id: str) -> bytes:
    tg_file = await bot.get_file(file_id)
    return await tg_file.download_as_bytearray()


def _normalize_state(value: Any) -> str:
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, (int, float)):
        return str(int(value))
    return ""


def _extract_task_id(payload: Mapping[str, Any]) -> Optional[str]:
    candidates = ("taskId", "task_id", "id")
    for key in candidates:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    data = payload.get("data")
    if isinstance(data, Mapping):
        return _extract_task_id(data)
    return None


def _collect_urls(value: Any) -> list[str]:
    urls: list[str] = []
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            urls.append(stripped)
        return urls
    if isinstance(value, Mapping):
        for key in ("url", "resultUrl", "originUrl"):
            v = value.get(key)
            if isinstance(v, str) and v.strip():
                urls.append(v.strip())
        return urls
    if isinstance(value, Iterable):
        for item in value:
            urls.extend(_collect_urls(item))
    return urls


def _extract_result_urls(payload: Mapping[str, Any]) -> list[str]:
    data = payload.get("data")
    if isinstance(data, Mapping):
        for key in ("resultUrls", "originUrls", "urls", "imageUrls"):
            urls = _collect_urls(data.get(key))
            if urls:
                return urls
        result_json = data.get("resultJson")
        if isinstance(result_json, str):
            try:
                parsed = json.loads(result_json)
            except Exception:  # pragma: no cover - defensive guard
                log.debug("banana.result_json_parse_fail", exc_info=True)
            else:
                if isinstance(parsed, Mapping):
                    for key in ("resultUrls", "urls", "originUrls"):
                        urls = _collect_urls(parsed.get(key))
                        if urls:
                            return urls
    return []


def _extract_state(payload: Mapping[str, Any]) -> str:
    data = payload.get("data")
    if isinstance(data, Mapping):
        for key in ("state", "successFlag", "status", "flag"):
            value = data.get(key)
            normalized = _normalize_state(value)
            if normalized:
                return normalized
    return ""


def _extract_error_reason(payload: Mapping[str, Any]) -> str:
    data = payload.get("data") if isinstance(payload.get("data"), Mapping) else {}
    fields = ("failMsg", "message", "error", "detail", "reason")
    for source in (payload, data):
        if isinstance(source, Mapping):
            for key in fields:
                value = source.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return "Banana service error"


async def _upload_image_bytes(data: bytes, *, filename: str) -> str:
    payload = {
        "fileName": filename,
        "base64": base64.b64encode(data).decode("ascii"),
    }
    try:
        response = await _client.request_json("POST", _UPLOAD_PATH, json_payload=payload)
    except (KieAPITimeoutError, KieAPITransportError) as exc:
        raise BananaBackendError(str(exc)) from exc
    except KieAPIHTTPError as exc:
        raise BananaBackendError(f"HTTP {exc.status}") from exc

    if isinstance(response, Mapping):
        for key in ("public_url", "publicUrl", "url"):
            value = response.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        data_section = response.get("data")
        if isinstance(data_section, Mapping):
            return await _ensure_public_url(data_section)
    raise BananaBackendError("Upload response missing url")


async def _ensure_public_url(payload: Mapping[str, Any]) -> str:
    for key in ("public_url", "publicUrl", "url"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    data = payload.get("data")
    if isinstance(data, Mapping):
        return await _ensure_public_url(data)
    raise BananaBackendError("Upload response missing url")


async def _submit_job(prompt: str, image_urls: Sequence[str]) -> str:
    payload = {
        "model": _BANANA_MODEL,
        "input": {
            "prompt": prompt,
            "image_urls": list(image_urls)[:4],
            "output_format": "png",
            "image_size": "auto",
        },
    }
    try:
        response = await _client.request_json("POST", _BANANA_CREATE_PATH, json_payload=payload)
    except KieAPIHTTPError as exc:
        message = _extract_error_reason(exc.payload)
        if exc.status in {400, 401, 402, 403, 404, 422}:
            raise BananaBadRequest(message) from exc
        raise BananaBackendError(f"HTTP {exc.status}: {message}") from exc
    except (KieAPITimeoutError, KieAPITransportError) as exc:
        raise BananaBackendError(str(exc)) from exc

    task_id = _extract_task_id(response)
    if not task_id:
        raise BananaBackendError("Banana response missing task id")
    return task_id


async def _poll_result(task_id: str) -> list[str]:
    deadline = time.monotonic() + _POLL_TIMEOUT
    while True:
        if time.monotonic() > deadline:
            raise BananaTimeout("Banana polling timed out")
        try:
            status_payload = await _client.request_json(
                "GET",
                _BANANA_STATUS_PATH,
                params={"taskId": task_id},
            )
        except KieAPIHTTPError as exc:
            raise BananaBackendError(f"HTTP {exc.status}: {_extract_error_reason(exc.payload)}") from exc
        except (KieAPITimeoutError, KieAPITransportError) as exc:
            raise BananaBackendError(str(exc)) from exc

        state = _extract_state(status_payload)
        if state in _SUCCESS_STATES:
            urls = _extract_result_urls(status_payload)
            if urls:
                return urls
            raise BananaBackendError("Banana returned empty result")
        if state in _FAIL_STATES:
            raise BananaBackendError(_extract_error_reason(status_payload))
        await asyncio.sleep(_POLL_INTERVAL)


def _build_cache_key(user_id: int, prompt: str, images: Sequence[str]) -> str:
    normalized_prompt = (prompt or "").strip().lower()
    digest = hashlib.sha256(
        "|".join(
            [
                "banana",
                str(int(user_id)),
                normalized_prompt,
                *sorted(str(img) for img in images),
            ]
        ).encode("utf-8")
    ).hexdigest()
    return f"banana:result:{digest}"


async def _send_cached_result(
    *,
    context,
    user_id: int,
    cache_entry: Mapping[str, Any],
) -> None:
    file_id = cache_entry.get("file_id") if isinstance(cache_entry, Mapping) else None
    if not isinstance(file_id, str) or not file_id:
        return
    try:
        await context.bot.send_document(user_id, file_id)
        await context.bot.send_message(
            user_id,
            "Нашёл готовый результат — отправляю сохранённый файл. ✅",
            reply_markup=InlineKeyboardMarkup(
                [
                    [InlineKeyboardButton("🔁 Сгенерировать ещё", callback_data="banana:again")],
                    [InlineKeyboardButton("🆕 Новая генерация", callback_data="banana:new")],
                ]
            ),
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        await handle_async_error(exc, "Banana.send_cached_result")


async def _prepare_image_urls(bot, file_ids: Sequence[str]) -> list[str]:
    uploads: list[str] = []
    for idx, file_id in enumerate(file_ids):
        try:
            data = await _fetch_file_bytes(bot, file_id)
        except Exception:  # pragma: no cover - fetch failure
            log.warning(
                "banana.upload_prepare_fail",
                exc_info=True,
                extra={"file_id": file_id},
            )
            continue

        ok, fmt, mime = validate_image(data)
        if not ok:
            log.warning(
                "banana.upload_prepare_invalid_image",
                extra={
                    "file_id": file_id,
                    "format": fmt,
                    "mime": mime,
                    "size": len(data),
                },
            )
            continue
        filename = f"banana_input_{idx + 1}.png"
        try:
            url = await _upload_image_bytes(bytes(data), filename=filename)
        except BananaBackendError:
            log.warning(
                "banana.upload_fail",
                exc_info=True,
                extra={"file_id": file_id},
            )
            continue
        uploads.append(url)
    return uploads


async def open_banana_card(update, context) -> None:
    user_id = update.effective_user.id
    redis = await _require_redis(context)
    state = await load(redis, user_id)
    balance_value = await _get_balance_value(user_id)
    text = banana_card_text(balance_value, state)
    await context.bot.send_message(
        user_id, text, reply_markup=banana_card_kb(state)
    )


async def on_banana_photo(update, context) -> None:
    message = update.effective_message
    if message is None:
        return
    user_id = update.effective_user.id
    redis = await _require_redis(context)
    state = await load(redis, user_id)
    if len(state.images) >= MAX_IMAGES:
        await context.bot.send_message(
            user_id,
            f"Максимум {MAX_IMAGES} фото. Удалите лишнее или начните новую генерацию.",
        )
        return
    file_id: Optional[str] = None
    if message.photo:
        file_id = message.photo[-1].file_id
    elif getattr(message, "document", None) is not None:
        document = message.document
        if document and (document.mime_type or "").startswith("image/"):
            file_id = document.file_id
    if not file_id:
        await context.bot.send_message(
            user_id,
            "Пришлите фото как изображение или файл без сжатия.",
        )
        return
    state.images.append(file_id)
    await save(redis, user_id, state)
    balance_value = await _get_balance_value(user_id)
    await message.reply_text(
        banana_card_text(balance_value, state),
        reply_markup=banana_card_kb(state),
    )


async def on_banana_text(update, context) -> None:
    message = update.effective_message
    if message is None or not message.text:
        return
    user_id = update.effective_user.id
    redis = await _require_redis(context)
    state = await load(redis, user_id)
    state.prompt = message.text.strip()
    await save(redis, user_id, state)
    balance_value = await _get_balance_value(user_id)
    await message.reply_text(
        banana_card_text(balance_value, state),
        reply_markup=banana_card_kb(state),
    )


async def on_banana_start(update, context) -> None:
    query = update.callback_query
    if query is None:
        return
    await query.answer()
    user_id = query.from_user.id
    redis = await _require_redis(context)
    state = await load(redis, user_id)
    if not state.ready:
        await context.bot.send_message(user_id, "Добавьте фото и промпт для редактирования.")
        return
    prompt = state.prompt or ""
    cache_key = _build_cache_key(user_id, prompt, state.images)
    cache_hit = await get_cached_result(cache_key, redis=redis)
    if cache_hit:
        await _send_cached_result(context=context, user_id=user_id, cache_entry=cache_hit)
        return

    image_urls = await _prepare_image_urls(context.bot, state.images)
    if not image_urls:
        await send_user_error(
            context,
            "backend_fail",
            details={
                "chat_id": user_id,
                "user_id": user_id,
                "mode": "banana",
                "reason": "upload_failed",
            },
        )
        return

    state.last_payload = {"photos": list(state.photos), "prompt": prompt}
    await save(redis, user_id, state)

    try:
        await context.bot.send_message(
            user_id,
            "🍌 Принято! Начинаю редактирование — пришлю результат, как только будет готово.",
        )
    except Exception:  # pragma: no cover - acknowledgement best-effort
        log.debug("banana.ack_failed", exc_info=True, extra={"user_id": user_id})

    task = asyncio.create_task(
        _run_banana_job(
            context=context,
            redis=redis,
            user_id=user_id,
            prompt=prompt,
            image_urls=image_urls,
            cache_key=cache_key,
        ),
        name=f"banana:{user_id}:{time.time_ns()}",
    )
    task.add_done_callback(_log_task_exception)


async def _run_banana_job(
    *,
    context,
    redis,
    user_id: int,
    prompt: str,
    image_urls: Sequence[str],
    cache_key: str,
) -> None:
    try:
        db_balance = await get_user_balance_async(int(user_id))
    except Exception:  # pragma: no cover - diagnostics only
        log.debug("banana.balance_lookup_fail", exc_info=True, extra={"user_id": user_id})
        db_balance = None

    try:
        task_id = await _submit_job(prompt, image_urls)
    except BananaBadRequest as exc:
        await send_user_error(
            context,
            "content_policy",
            details={
                "chat_id": user_id,
                "user_id": user_id,
                "mode": "banana",
                "reason": str(exc),
            },
            retry_cb="banana:again",
        )
        return
    except BananaBackendError as exc:
        await send_user_error(
            context,
            "backend_fail",
            details={
                "chat_id": user_id,
                "user_id": user_id,
                "mode": "banana",
                "reason": str(exc),
            },
            retry_cb="banana:again",
        )
        return

    log.info(
        "banana.job_started",
        extra={
            "user_id": user_id,
            "task_id": task_id,
            "balance": db_balance,
        },
    )

    try:
        state = await load(redis, user_id)
        state.last_job_id = task_id
        state.last_payload = {"photos": list(state.photos), "prompt": prompt}
        await save(redis, user_id, state)
    except Exception:  # pragma: no cover - best-effort persistence
        log.debug("banana.state_update_fail", exc_info=True, extra={"user_id": user_id})

    try:
        result_urls = await _poll_result(task_id)
    except BananaTimeout as exc:
        await send_user_error(
            context,
            "timeout",
            details={
                "chat_id": user_id,
                "user_id": user_id,
                "mode": "banana",
                "reason": str(exc),
                "req_id": task_id,
            },
            retry_cb="banana:again",
        )
        return
    except BananaBackendError as exc:
        await send_user_error(
            context,
            "backend_fail",
            details={
                "chat_id": user_id,
                "user_id": user_id,
                "mode": "banana",
                "reason": str(exc),
                "req_id": task_id,
            },
            retry_cb="banana:again",
        )
        return

    if not result_urls:
        await send_user_error(
            context,
            "backend_fail",
            details={
                "chat_id": user_id,
                "user_id": user_id,
                "mode": "banana",
                "reason": "empty_result",
                "req_id": task_id,
            },
            retry_cb="banana:again",
        )
        return

    result_url = result_urls[0]
    try:
        message = await context.bot.send_document(user_id, result_url)
    except Exception:
        log.exception("banana.telegram_send_fail", extra={"user_id": user_id, "req_id": task_id})
        await send_user_error(
            context,
            "backend_fail",
            details={
                "chat_id": user_id,
                "user_id": user_id,
                "mode": "banana",
                "reason": "telegram_send_fail",
                "req_id": task_id,
            },
            retry_cb="banana:again",
        )
        return

    file_id: Optional[str] = None
    document = getattr(message, "document", None)
    if document is not None:
        file_id = getattr(document, "file_id", None)
    if file_id is None and getattr(message, "photo", None):
        sizes = message.photo
        if sizes:
            file_id = getattr(sizes[-1], "file_id", None)

    if file_id:
        try:
            await set_cached_result(cache_key, {"file_id": file_id}, redis=redis)
        except Exception:  # pragma: no cover - cache best effort
            log.debug(
                "banana.cache_store_fail",
                exc_info=True,
                extra={"user_id": user_id, "req_id": task_id},
            )

    try:
        state = await load(redis, user_id)
        state.last_result_id = str(getattr(message, "message_id", "")) or None
        state.last_result_msg_id = getattr(message, "message_id", None)
        await save(redis, user_id, state)
    except Exception:  # pragma: no cover - best-effort persistence
        log.debug("banana.state_store_fail", exc_info=True, extra={"user_id": user_id})

    keyboard = InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("🔁 Сгенерировать ещё", callback_data="banana:again")],
            [InlineKeyboardButton("🆕 Новая генерация", callback_data="banana:new")],
        ]
    )
    await context.bot.send_message(user_id, "Готово ✅", reply_markup=keyboard)


async def on_banana_again(update, context) -> None:
    query = update.callback_query
    if query is None:
        return
    await query.answer()
    await on_banana_start(update, context)


async def on_banana_new(update, context) -> None:
    query = update.callback_query
    if query is None:
        return
    await query.answer()
    user_id = query.from_user.id
    redis = await _require_redis(context)
    await clear(redis, user_id)
    state = BananaState(photos=[], prompt=None)
    await save(redis, user_id, state)
    balance_value = await _get_balance_value(user_id)
    await context.bot.send_message(
        user_id,
        "Новая карточка 🆕 Отправьте фото и промпт.",
        reply_markup=banana_card_kb(state),
    )


__all__ = [
    "open_banana_card",
    "on_banana_photo",
    "on_banana_text",
    "on_banana_start",
    "on_banana_again",
    "on_banana_new",
]
