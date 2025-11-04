"""Async Banana handler integrated with Telegram workflow."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Optional, Sequence

from telegram import InputMediaDocument, Update
from telegram.ext import ContextTypes

from bot_logger import BotLogger
from cache_helper import get_or_set_cache
from error_utils import handle_async_error
from services.kie_api_async import (
    KieAPIAsync,
    KieAPIHTTPError,
    KieAPITimeoutError,
    KieAPITransportError,
)
from keyboards import banana_result_kb
from utils.banana_state import BananaState, load, save
from utils.files import validate_image

from handlers.banana import (
    BananaBackendError,
    BananaBadRequest,
    BananaTimeout,
)

log = logging.getLogger("handlers.banana_async")

_ACK_TEXT = "🟡 Processing your Banana edit... please wait."
_SUCCESS_CAPTION = "✅ Banana edit ready!"
_FAILURE_TEXT = "⚠️ Something went wrong on Banana servers. Please try again later."

_BANANA_MODEL = "google/nano-banana-edit"
_BANANA_CREATE_PATH = "/api/v1/jobs/createTask"
_BANANA_STATUS_PATH = "/api/v1/jobs/recordInfo"
_UPLOAD_PATH = "/api/v1/upload/base64"

_WAIT_STATES = {"waiting", "queuing", "queued", "generating", "processing", "running", "pending", "0", 0}
_SUCCESS_STATES = {"success", "1", 1, "done", "finished", "completed", "ready"}
_FAIL_STATES = {"fail", "failed", "error", "2", 2, "3", 3, "canceled", "cancelled", "timeout"}


@dataclass(slots=True)
class BananaResult:
    file_id: str
    task_id: str
    caption: str = _SUCCESS_CAPTION
    url: Optional[str] = None

    def to_mapping(self) -> MutableMapping[str, Any]:
        payload: MutableMapping[str, Any] = {
            "file_id": self.file_id,
            "task_id": self.task_id,
            "caption": self.caption,
        }
        if self.url:
            payload["url"] = self.url
        return payload


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
            raw = value.get(key)
            if isinstance(raw, str) and raw.strip():
                urls.append(raw.strip())
        return urls
    if isinstance(value, Sequence):
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
                nested = json.loads(result_json)
            except Exception:
                return []
            if isinstance(nested, Mapping):
                for key in ("resultUrls", "urls", "originUrls"):
                    urls = _collect_urls(nested.get(key))
                    if urls:
                        return urls
    return []


def _extract_state(payload: Mapping[str, Any]) -> str:
    data = payload.get("data")
    if isinstance(data, Mapping):
        for key in ("state", "successFlag", "status", "flag"):
            normalized = _normalize_state(data.get(key))
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


class BananaAsyncHandler:
    """Asynchronous Banana pipeline orchestrator."""

    def __init__(
        self,
        *,
        redis: Any = None,
        client: Optional[KieAPIAsync] = None,
        bot_logger: Optional[BotLogger] = None,
        cache_ttl: int = 3600,
        poll_interval: float = 4.0,
        poll_timeout: float = 8 * 60.0,
    ) -> None:
        self._redis = redis
        self._client = client or KieAPIAsync()
        self._cache_ttl = cache_ttl
        self._poll_interval = poll_interval
        self._poll_timeout = poll_timeout
        self._logger = bot_logger or BotLogger(redis=redis)

    async def run(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat = update.effective_chat
        user = update.effective_user
        if chat is None or user is None:
            return

        ack_message = await context.bot.send_message(chat.id, _ACK_TEXT)

        try:
            redis = self._resolve_redis(context)
        except RuntimeError as exc:
            await self._handle_failure(context, chat.id, ack_message.message_id, exc, user_id=user.id)
            return

        try:
            state = await load(redis, user.id)
        except Exception as exc:
            await handle_async_error(exc, "BananaAsync.load_state")
            await self._handle_failure(context, chat.id, ack_message.message_id, exc, user_id=user.id)
            return

        if not isinstance(state, BananaState) or not state.ready:
            await self._notify_invalid_state(context, chat.id, ack_message.message_id)
            return

        prompt = state.prompt or ""
        photos = list(state.photos)
        cache_key = _build_cache_key(user.id, prompt, photos)

        try:
            result = await get_or_set_cache(
                cache_key,
                lambda: self._generate(
                    context=context,
                    chat_id=chat.id,
                    user_id=user.id,
                    prompt=prompt,
                    photos=photos,
                    ack_message_id=ack_message.message_id,
                ),
                redis=redis,
                ttl=self._cache_ttl,
            )
        except BananaBadRequest as exc:
            await handle_async_error(exc, "BananaAsync.bad_request")
            await self._handle_failure(context, chat.id, ack_message.message_id, exc, user_id=user.id)
            return
        except (BananaBackendError, BananaTimeout, KieAPITimeoutError) as exc:
            await handle_async_error(exc, "BananaAsync.backend_error")
            await self._handle_failure(context, chat.id, ack_message.message_id, exc, user_id=user.id)
            return
        except Exception as exc:
            await handle_async_error(exc, "BananaAsync.run")
            await self._handle_failure(context, chat.id, ack_message.message_id, exc, user_id=user.id)
            return

        cached = bool(result.get("_cached"))
        if cached:
            await self._logger.info(f"[BananaAsync] cache hit  | key: {cache_key}", user_id=user.id)
            await self._send_cached_result(
                context=context,
                chat_id=chat.id,
                message_id=ack_message.message_id,
                payload=result,
            )

        try:
            state.last_job_id = str(result.get("task_id") or state.last_job_id or "") or None
            state.last_result_id = str(result.get("file_id") or state.last_result_id or "") or None
            await save(redis, user.id, state)
        except Exception as exc:
            await handle_async_error(exc, "BananaAsync.save_state")

    def _resolve_redis(self, context: ContextTypes.DEFAULT_TYPE) -> Any:
        if self._redis is not None:
            return self._redis
        redis = getattr(context, "redis", None)
        if redis is None:
            raise RuntimeError("Redis client is not configured")
        return redis

    async def _generate(
        self,
        *,
        context: ContextTypes.DEFAULT_TYPE,
        chat_id: int,
        user_id: int,
        prompt: str,
        photos: Sequence[str],
        ack_message_id: int,
    ) -> MutableMapping[str, Any]:
        image_urls = await self._prepare_image_urls(context.bot, photos)
        if not image_urls:
            raise BananaBackendError("no_uploads")

        payload = {
            "taskType": "banana_img2img",
            "model": _BANANA_MODEL,
            "prompt": prompt,
            "enableTranslation": True,
            "input": {"prompt": prompt, "images": image_urls},
        }

        try:
            response = await self._client.request_json("POST", _BANANA_CREATE_PATH, json_payload=payload)
        except KieAPIHTTPError as exc:
            if exc.status in {400, 401, 402, 403, 404, 422}:
                raise BananaBadRequest(_extract_error_reason(exc.payload)) from exc
            raise BananaBackendError(f"HTTP {exc.status}") from exc
        except (KieAPITimeoutError, KieAPITransportError) as exc:
            raise BananaBackendError(str(exc)) from exc

        task_id = _extract_task_id(response)
        if not task_id:
            raise BananaBackendError("Banana task id missing")

        await self._logger.info(
            f"[BananaAsync] start job  | user_id: {user_id}",
            task_id=task_id,
            prompt=prompt,
        )

        async for status_payload in self._client.poll_job(
            _BANANA_STATUS_PATH,
            task_id=task_id,
            interval=self._poll_interval,
            timeout=self._poll_timeout,
        ):
            state = _extract_state(status_payload)
            if state in _WAIT_STATES:
                await asyncio.sleep(0)
                continue
            if state in _SUCCESS_STATES:
                urls = _extract_result_urls(status_payload)
                if not urls:
                    raise BananaBackendError("Banana returned empty result")
                url = urls[0]
                message, caption = await self._publish_result(
                    context=context,
                    chat_id=chat_id,
                    message_id=ack_message_id,
                    media=url,
                    prompt=prompt,
                )
                file_id = self._extract_file_id(message) or url
                result = BananaResult(file_id=file_id, task_id=task_id, caption=caption, url=url)
                return result.to_mapping()
            if state in _FAIL_STATES:
                raise BananaBackendError(_extract_error_reason(status_payload))
            raise BananaBackendError(f"Unexpected Banana state: {state or 'unknown'}")

        raise BananaTimeout("Banana polling timed out")

    async def _prepare_image_urls(self, bot, file_ids: Sequence[str]) -> list[str]:
        uploads: list[str] = []
        for idx, file_id in enumerate(file_ids):
            try:
                data = await self._fetch_file_bytes(bot, file_id)
            except Exception as exc:
                await handle_async_error(exc, "BananaAsync.prepare_image")
                continue

            ok, fmt, mime = validate_image(data)
            if not ok:
                log.warning(
                    "banana_async.image_validation_failed",
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
                url = await self._upload_image_bytes(bytes(data), filename=filename)
            except BananaBackendError as exc:
                await handle_async_error(exc, "BananaAsync.upload_image")
                continue
            uploads.append(url)
        return uploads

    async def _fetch_file_bytes(self, bot, file_id: str) -> bytes:
        tg_file = await bot.get_file(file_id)
        return await tg_file.download_as_bytearray()

    async def _upload_image_bytes(self, data: bytes, *, filename: str) -> str:
        payload = {
            "fileName": filename,
            "base64": base64.b64encode(data).decode("ascii"),
        }
        try:
            response = await self._client.request_json("POST", _UPLOAD_PATH, json_payload=payload)
        except (KieAPITimeoutError, KieAPITransportError) as exc:
            raise BananaBackendError(str(exc)) from exc
        except KieAPIHTTPError as exc:
            raise BananaBackendError(f"HTTP {exc.status}") from exc

        if isinstance(response, Mapping):
            for key in ("public_url", "publicUrl", "url"):
                value = response.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            data = response.get("data")
            if isinstance(data, Mapping):
                for key in ("public_url", "publicUrl", "url"):
                    value = data.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()
        raise BananaBackendError("upload_failed")

    async def _publish_result(
        self,
        *,
        context: ContextTypes.DEFAULT_TYPE,
        chat_id: int,
        message_id: int,
        media: str,
        prompt: str,
    ) -> tuple[Any, str]:
        caption = _SUCCESS_CAPTION if not prompt else f"{_SUCCESS_CAPTION}\n{prompt}"
        try:
            message = await context.bot.edit_message_media(
                chat_id=chat_id,
                message_id=message_id,
                media=InputMediaDocument(media=media, caption=caption),
                reply_markup=banana_result_kb(),
            )
            return message, caption
        except Exception as exc:
            await handle_async_error(exc, "BananaAsync.publish_result")
            await context.bot.send_message(chat_id, caption)
            return None, caption

    async def _send_cached_result(
        self,
        *,
        context: ContextTypes.DEFAULT_TYPE,
        chat_id: int,
        message_id: int,
        payload: Mapping[str, Any],
    ) -> None:
        file_id = payload.get("file_id") if isinstance(payload, Mapping) else None
        caption = payload.get("caption") if isinstance(payload, Mapping) else None
        if not isinstance(file_id, str) or not file_id:
            await context.bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=_FAILURE_TEXT)
            return
        try:
            await context.bot.edit_message_media(
                chat_id=chat_id,
                message_id=message_id,
                media=InputMediaDocument(media=file_id, caption=caption or _SUCCESS_CAPTION),
                reply_markup=banana_result_kb(),
            )
        except Exception as exc:
            await handle_async_error(exc, "BananaAsync.cached_edit")
            await context.bot.send_message(chat_id, caption or _SUCCESS_CAPTION)

    async def _handle_failure(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        chat_id: int,
        message_id: int,
        exc: Exception,
        *,
        user_id: Optional[int] = None,
    ) -> None:
        await self._logger.error(
            f"[BananaAsync] error      | message: {exc}",
            user_id=user_id,
        )
        try:
            await context.bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=_FAILURE_TEXT)
        except Exception as edit_exc:
            await handle_async_error(edit_exc, "BananaAsync.edit_failure")
            await context.bot.send_message(chat_id, _FAILURE_TEXT)

    async def _notify_invalid_state(self, context, chat_id: int, message_id: int) -> None:
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text="Добавьте фото и промпт для редактирования.",
            )
        except Exception as exc:
            await handle_async_error(exc, "BananaAsync.invalid_state")
            await context.bot.send_message(chat_id, "Добавьте фото и промпт для редактирования.")

    @staticmethod
    def _extract_file_id(message: Any) -> Optional[str]:
        if message is None:
            return None
        document = getattr(message, "document", None)
        if document is not None:
            file_id = getattr(document, "file_id", None)
            if isinstance(file_id, str) and file_id:
                return file_id
        photos = getattr(message, "photo", None)
        if photos:
            try:
                candidate = photos[-1]
            except Exception:
                candidate = None
            if candidate is not None:
                file_id = getattr(candidate, "file_id", None)
                if isinstance(file_id, str) and file_id:
                    return file_id
        return None


__all__ = ["BananaAsyncHandler"]
