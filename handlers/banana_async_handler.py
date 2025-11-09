"""Async Banana handler integrated with Telegram workflow."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import mimetypes
import time
from io import BytesIO
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, Sequence

import httpx
from telegram import Update
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
from utils.async_input_state import input_state as async_input_state
from utils.banana_state import BananaState, ensure, load, save
from PIL import Image

from settings import (
    BANANA_CARD_REUSE,
    TG_FILE_DIRECT_URL,
    UPLOAD_BASE64_PATH,
    UPLOAD_BASE_URL,
    UPLOAD_STREAM_PATH,
)
from utils.files import identify_image, validate_image
from utils.redis_client import get_redis

from ui.renderers.banana import banana_card_kb as build_banana_card_kb, banana_card_text

from handlers.banana import (
    BananaBackendError,
    BananaBadRequest,
    BananaTimeout,
)

log = logging.getLogger("handlers.banana_async")


_MAX_CARD_IMAGES = 4
_DEFAULT_HANDLER: Optional["BananaAsyncHandler"] = None

_ACK_TEXT = "🟡 Processing your Banana edit... please wait."
_SUCCESS_CAPTION = "✅ Banana edit ready!"
_FAILURE_TEXT = "⚠️ Something went wrong on Banana servers. Please try again later."

_BANANA_MODEL = "google/nano-banana-edit"
_BANANA_CREATE_PATH = "/api/v1/jobs/createTask"
_BANANA_STATUS_PATH = "/api/v1/jobs/recordInfo"
_UPLOAD_PATH = "/api/v1/upload/base64"

_CARD_MESSAGE_KEY = "banana:card_msg_id:{chat_id}:{user_id}"
_CARD_SNAPSHOT_KEY = "banana:card_snapshot:{chat_id}:{user_id}"
_SESSION_KEY = "banana:session:{user_id}"
_CARD_STORAGE_TTL = 60 * 60 * 24
_SESSION_TTL = 60 * 60 * 24
_UPLOAD_TIMEOUT = httpx.Timeout(20.0, connect=20.0)
_UPLOAD_RETRIES = (2, 5, 10)
_EXTENSION_MAP = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}
_IMAGE_DOC_MIME_WHITELIST = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/heic",
    "image/heif",
}
_ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP", "HEIC", "HEIF"}

_WAIT_STATES = {"waiting", "queuing", "queued", "generating", "processing", "running", "pending", "0", 0}
_SUCCESS_STATES = {"success", "1", 1, "done", "finished", "completed", "ready"}
_FAIL_STATES = {"fail", "failed", "error", "2", 2, "3", 3, "canceled", "cancelled", "timeout"}


_CARD_REUSE_ENABLED = bool(BANANA_CARD_REUSE)
_TG_DIRECT_URL_ALLOWED = bool(TG_FILE_DIRECT_URL)

_ALBUM_BUFFER_KEY = "banana:album:{media_group_id}:{user_id}"
_ALBUM_LOCK_KEY = "banana:album:lock:{media_group_id}:{user_id}"
_ALBUM_BUFFER_TTL = 60
_ALBUM_DEBOUNCE_SECONDS = 0.9


def _guess_filename(url: str) -> str:
    base = (url or "").split("?")[0].lower()
    for ext in ("mp4", "gif", "jpg", "jpeg", "png", "webp"):
        if base.endswith(f".{ext}"):
            return f"result.{ext}"
    return "result.png"


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


def _build_upload_url(path: Optional[str]) -> Optional[str]:
    base = (UPLOAD_BASE_URL or "").strip()
    normalized_path = (path or "").strip()
    if normalized_path.startswith("http://") or normalized_path.startswith("https://"):
        return normalized_path
    if not base and not normalized_path:
        return None
    if not base:
        return None
    if not normalized_path.startswith("/"):
        normalized_path = f"/{normalized_path}"
    return f"{base}{normalized_path}" if normalized_path else None


def _extract_upload_url(payload: Mapping[str, Any]) -> Optional[str]:
    for key in ("public_url", "publicUrl", "url"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    data = payload.get("data") if isinstance(payload.get("data"), Mapping) else None
    if isinstance(data, Mapping):
        return _extract_upload_url(data)
    return None


def _resolve_extension(filename: str, fmt: Optional[str]) -> str:
    path_ext = Path(filename).suffix.lower().lstrip(".")
    if path_ext:
        return path_ext
    if fmt:
        mapped = _EXTENSION_MAP.get(fmt.upper())
        if mapped:
            return mapped
    return "png"


def _convert_to_jpeg(data: bytes) -> bytes:
    with Image.open(BytesIO(data)) as image:
        converted = image.convert("RGB")
        buffer = BytesIO()
        converted.save(buffer, format="JPEG", quality=95)
        return buffer.getvalue()


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
        uploads: list[str] = []
        payload_snapshot = state.last_payload if isinstance(state.last_payload, Mapping) else {}
        if isinstance(payload_snapshot, Mapping):
            snapshot_prompt = str(payload_snapshot.get("prompt") or "").strip()
            snapshot_images = [str(img) for img in (payload_snapshot.get("images") or payload_snapshot.get("photos") or [])]
            stored_uploads: list[str] = []
            for item in payload_snapshot.get("uploads") or []:
                if isinstance(item, Mapping):
                    url = str(item.get("url") or "").strip()
                else:
                    url = str(item or "").strip()
                if url:
                    stored_uploads.append(url)
            if snapshot_images == photos and snapshot_prompt == (prompt or "").strip():
                uploads = stored_uploads
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
                    uploads=uploads,
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
                user_id=user.id,
            )

        try:
            state.last_job_id = str(result.get("task_id") or state.last_job_id or "") or None
            state.last_result_id = str(result.get("file_id") or state.last_result_id or "") or None
            await save(redis, user.id, state)
            await _store_session_state(redis, user.id, state)
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
        uploads: Sequence[str],
        ack_message_id: int,
    ) -> MutableMapping[str, Any]:
        upload_candidates = [str(url).strip() for url in uploads if str(url or "").strip()]
        if upload_candidates:
            image_urls = upload_candidates
        else:
            image_urls = await self.prepare_uploads(context.bot, photos, user_id=user_id)
        if not image_urls and not (prompt or "").strip():
            raise BananaBackendError("no_uploads")

        payload = {
            "taskType": "banana_img2img",
            "model": _BANANA_MODEL,
            "prompt": prompt,
            "enableTranslation": True,
            "input": {"prompt": prompt, "images": image_urls},
        }

        log.info(
            "banana.gen.start",
            extra={
                "user_id": user_id,
                "payload": {"images": len(image_urls), "has_prompt": bool(prompt)},
                "source": "worker",
            },
        )

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
                    user_id=user_id,
                    task_id=task_id,
                )
                file_id = self._extract_file_id(message) or url
                result = BananaResult(file_id=file_id, task_id=task_id, caption=caption, url=url)
                return result.to_mapping()
            if state in _FAIL_STATES:
                raise BananaBackendError(_extract_error_reason(status_payload))
            raise BananaBackendError(f"Unexpected Banana state: {state or 'unknown'}")

        raise BananaTimeout("Banana polling timed out")

    async def upload_image(
        self,
        bot,
        file_id: str,
        *,
        user_id: Optional[int] = None,
    ) -> Optional[dict[str, Any]]:
        try:
            tg_file = await bot.get_file(file_id)
        except Exception as exc:
            await handle_async_error(exc, "BananaAsync.upload_get_file")
            return None
        file_path = getattr(tg_file, "file_path", "") or ""
        if (
            file_path
            and not _TG_DIRECT_URL_ALLOWED
            and str(file_path).startswith(("http://", "https://"))
        ):
            file_path = Path(file_path).name
        try:
            data = await tg_file.download_as_bytearray()
        except Exception as exc:
            await handle_async_error(exc, "BananaAsync.upload_download")
            return None

        raw_bytes = bytes(data)
        ok, fmt, mime = validate_image(raw_bytes, allowed=_ALLOWED_IMAGE_FORMATS)
        if not ok:
            detected_fmt, detected_mime = identify_image(raw_bytes)
            log.warning(
                "banana.upload.err",
                extra={
                    "user_id": user_id,
                    "status": 0,
                    "reason": "invalid-image",
                    "format": detected_fmt or fmt,
                    "mime": detected_mime or mime,
                },
            )
            return None

        fmt_upper = (fmt or "").upper()
        source_name = Path(file_path or file_id).name or f"banana_{file_id}"
        target_ext = _resolve_extension(source_name, fmt_upper)
        target_mime = mime or mimetypes.guess_type(source_name)[0] or "application/octet-stream"
        upload_bytes = raw_bytes

        if fmt_upper in {"WEBP", "HEIC", "HEIF"} or target_ext.lower() in {"webp", "heic", "heif"}:
            try:
                upload_bytes = _convert_to_jpeg(raw_bytes)
            except Exception as exc:
                await handle_async_error(exc, "BananaAsync.upload_convert")
                log.warning(
                    "banana.upload.err",
                    extra={
                        "user_id": user_id,
                        "status": 0,
                        "reason": "convert-failed",
                        "format": fmt_upper,
                    },
                )
                return None
            target_ext = "jpg"
            target_mime = "image/jpeg"
        elif fmt_upper in {"JPEG", "JPG"}:
            target_ext = "jpg"
            target_mime = "image/jpeg"
        elif fmt_upper == "PNG":
            target_ext = "png"
            target_mime = "image/png"

        size = len(upload_bytes)
        filename = f"{Path(source_name).stem or 'banana_input'}.{target_ext}"

        url = await self._upload_image_bytes(
            upload_bytes,
            filename=filename,
            mime_type=target_mime,
            ext=target_ext,
            size=size,
            user_id=user_id,
        )
        log.info(
            "banana.upload.ok",
            extra={"user_id": user_id, "ext": target_ext.lower(), "size": size},
        )
        return {"type": "image", "url": url}

    async def _prepare_image_urls(
        self, bot, file_ids: Sequence[str], *, user_id: Optional[int] = None
    ) -> list[str]:
        uploads: list[str] = []
        for file_id in file_ids:
            try:
                result = await self.upload_image(bot, file_id, user_id=user_id)
            except BananaBackendError as exc:
                await handle_async_error(exc, "BananaAsync.upload_image")
                continue
            if result and isinstance(result, Mapping):
                url = str(result.get("url") or "").strip()
                if url:
                    uploads.append(url)
        return uploads

    async def prepare_uploads(
        self, bot, file_ids: Sequence[str], *, user_id: Optional[int] = None
    ) -> list[str]:
        try:
            return await self._prepare_image_urls(bot, file_ids, user_id=user_id)
        except TypeError as exc:
            if "user_id" not in str(exc):
                raise
            return await self._prepare_image_urls(bot, file_ids)

    async def _fetch_file_bytes(self, bot, file_id: str) -> bytes:
        tg_file = await bot.get_file(file_id)
        return await tg_file.download_as_bytearray()

    async def _upload_image_bytes(
        self,
        data: bytes,
        *,
        filename: str,
        mime_type: str,
        ext: str,
        size: int,
        user_id: Optional[int] = None,
    ) -> str:
        last_error: Optional[BananaBackendError] = None
        stream_endpoint = _build_upload_url(UPLOAD_STREAM_PATH or "/api/file-stream-upload")
        if stream_endpoint:
            try:
                url = await self._upload_via_stream(
                    data,
                    filename=filename,
                    mime_type=mime_type,
                    endpoint=stream_endpoint,
                    user_id=user_id,
                )
            except BananaBackendError as exc:
                last_error = exc
            else:
                return url

        base64_endpoint = _build_upload_url(UPLOAD_BASE64_PATH or _UPLOAD_PATH)
        if base64_endpoint:
            try:
                url = await self._upload_via_base64(
                    data,
                    filename=filename,
                    endpoint=base64_endpoint,
                    user_id=user_id,
                )
            except BananaBackendError as exc:
                last_error = exc
            else:
                return url

        if last_error is not None:
            raise last_error
        raise BananaBackendError("upload_failed")

    async def _upload_via_stream(
        self,
        data: bytes,
        *,
        filename: str,
        mime_type: str,
        endpoint: str,
        user_id: Optional[int] = None,
    ) -> str:
        for attempt, delay in enumerate(_UPLOAD_RETRIES, start=1):
            retry = False
            try:
                async with httpx.AsyncClient(timeout=_UPLOAD_TIMEOUT) as client:
                    response = await client.post(
                        endpoint,
                        files={"file": (filename, data, mime_type)},
                    )
            except httpx.RequestError as exc:
                log.warning(
                    "banana.upload.err",
                    extra={
                        "user_id": user_id,
                        "status": 0,
                        "path": endpoint,
                        "attempt": attempt,
                        "reason": str(exc),
                    },
                )
                retry = attempt < len(_UPLOAD_RETRIES)
            else:
                status = response.status_code
                body_preview = (response.text or "")[:200]
                if 200 <= status < 300:
                    try:
                        payload = response.json()
                    except ValueError:
                        payload = {}
                    url = _extract_upload_url(payload if isinstance(payload, Mapping) else {})
                    if url:
                        return url
                    log.warning(
                        "banana.upload.err",
                        extra={
                            "user_id": user_id,
                            "status": status,
                            "path": endpoint,
                            "attempt": attempt,
                            "reason": "missing-url",
                        },
                    )
                    break
                if status >= 500 or status == 429:
                    log.warning(
                        "banana.upload.err",
                        extra={
                            "user_id": user_id,
                            "status": status,
                            "path": endpoint,
                            "attempt": attempt,
                            "reason": body_preview,
                        },
                    )
                    retry = attempt < len(_UPLOAD_RETRIES)
                else:
                    log.warning(
                        "banana.upload.err",
                        extra={
                            "user_id": user_id,
                            "status": status,
                            "path": endpoint,
                            "attempt": attempt,
                            "reason": body_preview,
                        },
                    )
                    break
            if retry and attempt < len(_UPLOAD_RETRIES):
                await asyncio.sleep(delay)
                continue
            if not retry:
                break
        raise BananaBackendError("upload_failed")

    async def _upload_via_base64(
        self,
        data: bytes,
        *,
        filename: str,
        endpoint: str,
        user_id: Optional[int] = None,
    ) -> str:
        payload = {
            "fileName": filename,
            "base64": base64.b64encode(data).decode("ascii"),
        }
        try:
            async with httpx.AsyncClient(timeout=_UPLOAD_TIMEOUT) as client:
                response = await client.post(endpoint, json=payload)
        except httpx.RequestError as exc:
            log.warning(
                "banana.upload.err",
                extra={
                    "user_id": user_id,
                    "status": 0,
                    "path": endpoint,
                    "reason": str(exc),
                },
            )
            raise BananaBackendError("upload_failed") from exc
        if 200 <= response.status_code < 300:
            try:
                payload_json: Any = response.json()
            except ValueError as exc:
                raise BananaBackendError("upload_failed") from exc
            if isinstance(payload_json, Mapping):
                url = _extract_upload_url(payload_json)
            else:
                url = None
            if url:
                return url
            log.warning(
                "banana.upload.err",
                extra={
                    "user_id": user_id,
                    "status": response.status_code,
                    "path": endpoint,
                    "reason": "missing-url",
                },
            )
            raise BananaBackendError("upload_failed")
        log.warning(
            "banana.upload.err",
            extra={
                "user_id": user_id,
                "status": response.status_code,
                "path": endpoint,
                "reason": (response.text or "")[:200],
            },
        )
        raise BananaBackendError("upload_failed")

    async def _publish_result(
        self,
        *,
        context: ContextTypes.DEFAULT_TYPE,
        chat_id: int,
        message_id: int,
        media: str,
        prompt: str,
        user_id: int,
        task_id: str,
    ) -> tuple[Any, str]:
        caption = _SUCCESS_CAPTION if not prompt else f"{_SUCCESS_CAPTION}\n{prompt}"
        try:
            await context.bot.delete_message(chat_id, message_id)
        except Exception:
            pass
        try:
            raw_filename = _guess_filename(media)
            suffix = Path(raw_filename).suffix or ".png"
            safe_task = (task_id or "").strip()
            if safe_task:
                filename = f"banana_result_{safe_task[:8]}{suffix}"
            else:
                filename = raw_filename
            message = await context.bot.send_document(
                chat_id,
                media,
                caption=caption,
                filename=filename,
                reply_markup=banana_result_kb(task_id),
            )
            log.info(
                "banana.gen.done",
                extra={"chat_id": chat_id, "user_id": user_id, "task_id": task_id, "cached": False},
            )
            return message, caption
        except Exception as exc:
            await handle_async_error(exc, "BananaAsync.publish_result")
            await context.bot.send_message(chat_id, caption)
            log.info(
                "banana.gen.done",
                extra={
                    "chat_id": chat_id,
                    "user_id": user_id,
                    "task_id": task_id,
                    "cached": False,
                    "fallback": True,
                },
            )
            return None, caption

    async def _send_cached_result(
        self,
        *,
        context: ContextTypes.DEFAULT_TYPE,
        chat_id: int,
        message_id: int,
        payload: Mapping[str, Any],
        user_id: int,
    ) -> None:
        file_id = payload.get("file_id") if isinstance(payload, Mapping) else None
        caption = payload.get("caption") if isinstance(payload, Mapping) else None
        task_id = payload.get("task_id") if isinstance(payload, Mapping) else None
        if not isinstance(file_id, str) or not file_id:
            await context.bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=_FAILURE_TEXT)
            return
        try:
            await context.bot.delete_message(chat_id, message_id)
        except Exception:
            pass
        try:
            await context.bot.send_document(
                chat_id,
                file_id,
                caption=caption or _SUCCESS_CAPTION,
                reply_markup=banana_result_kb(task_id if isinstance(task_id, str) else None),
            )
            log.info(
                "banana.gen.done",
                extra={"chat_id": chat_id, "user_id": user_id, "task_id": task_id, "cached": True},
            )
        except Exception as exc:
            await handle_async_error(exc, "BananaAsync.cached_edit")
            await context.bot.send_message(chat_id, caption or _SUCCESS_CAPTION)
            log.info(
                "banana.gen.done",
                extra={"chat_id": chat_id, "user_id": user_id, "task_id": task_id, "cached": True, "fallback": True},
            )

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
        log.info(
            "banana.gen.fail",
            extra={
                "chat_id": chat_id,
                "user_id": user_id,
                "error": str(exc),
            },
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


def _get_default_handler() -> "BananaAsyncHandler":
    global _DEFAULT_HANDLER
    if _DEFAULT_HANDLER is None:
        _DEFAULT_HANDLER = BananaAsyncHandler()
    return _DEFAULT_HANDLER


def _resolve_redis(context: ContextTypes.DEFAULT_TYPE):
    redis = getattr(context, "redis", None)
    if redis is None:
        redis = get_redis()
        try:
            setattr(context, "redis", redis)
        except Exception:
            pass
    return redis


async def _load_state(context: ContextTypes.DEFAULT_TYPE, user_id: int) -> BananaState:
    redis = _resolve_redis(context)
    return await load(redis, user_id)


async def _save_state(
    context: ContextTypes.DEFAULT_TYPE, user_id: int, state: BananaState
) -> None:
    redis = _resolve_redis(context)
    await save(redis, user_id, state)
    await _store_session_state(redis, user_id, state)


def _session_key_for(user_id: int) -> str:
    return _SESSION_KEY.format(user_id=int(user_id))


def _session_payload_from_state(state: BananaState) -> dict[str, Any]:
    prompt = (state.prompt or "").strip()
    payload: dict[str, Any] = {
        "prompt": prompt,
        "images": list(state.photos),
    }
    if isinstance(state.last_payload, Mapping):
        payload["last_payload"] = state.last_payload
    return payload


async def _store_session_state(redis, user_id: int, state: BananaState) -> None:
    payload = _session_payload_from_state(state)
    payload["prompt"] = payload.get("prompt") or ""
    payload["images"] = [str(fid) for fid in payload.get("images", []) if str(fid or "").strip()]
    try:
        await redis.set(
            _session_key_for(user_id),
            json.dumps(payload, ensure_ascii=False),
            ex=_SESSION_TTL,
        )
    except Exception as exc:
        await handle_async_error(exc, "BananaAsync.session_store")


async def _load_session_payload(
    ctx: ContextTypes.DEFAULT_TYPE, user_id: int
) -> dict[str, Any]:
    try:
        redis = _resolve_redis(ctx)
    except Exception as exc:
        await handle_async_error(exc, "BananaAsync.session_load_redis")
        return {"prompt": "", "images": [], "last_payload": None}
    try:
        raw = await redis.get(_session_key_for(user_id))
    except Exception as exc:
        await handle_async_error(exc, "BananaAsync.session_load")
        return {"prompt": "", "images": [], "last_payload": None}
    if not raw:
        return {"prompt": "", "images": [], "last_payload": None}
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "ignore")
    try:
        data = json.loads(raw)
    except Exception as exc:
        await handle_async_error(exc, "BananaAsync.session_decode")
        return {"prompt": "", "images": [], "last_payload": None}
    prompt = str(data.get("prompt") or "").strip()
    images = [
        str(fid).strip()
        for fid in data.get("images") or []
        if isinstance(fid, str) and fid.strip()
    ]
    last_payload = data.get("last_payload") if isinstance(data.get("last_payload"), Mapping) else None
    return {"prompt": prompt, "images": images, "last_payload": last_payload}


def _extract_photo_file_id(message) -> Optional[str]:
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
    document = getattr(message, "document", None)
    if document is not None:
        mime = (getattr(document, "mime_type", "") or "").lower()
        if mime in _IMAGE_DOC_MIME_WHITELIST:
            file_id = getattr(document, "file_id", None)
            if isinstance(file_id, str) and file_id:
                return file_id
    return None


def _message_has_image_document(message) -> bool:
    document = getattr(message, "document", None)
    if document is None:
        return False
    mime = (getattr(document, "mime_type", "") or "").lower()
    return mime in _IMAGE_DOC_MIME_WHITELIST


def _extract_prompt_text(message) -> Optional[str]:
    def _normalize(value: Any) -> Optional[str]:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                return stripped
        return None

    caption_prompt = _normalize(getattr(message, "caption", None))
    text_prompt = _normalize(getattr(message, "text", None))
    if caption_prompt:
        return caption_prompt
    has_media = bool(getattr(message, "photo", None)) or _message_has_image_document(message)
    if has_media:
        return text_prompt
    return text_prompt


async def _acknowledge_callback(query, ctx, *, text: str = "✓", show_alert: bool = False) -> None:
    if query is None:
        return
    try:
        await ctx.bot.answer_callback_query(query.id, text=text, show_alert=show_alert)
    except Exception:
        pass


def _card_key(key_tmpl: str, *, chat_id: int, user_id: int) -> str:
    return key_tmpl.format(chat_id=int(chat_id), user_id=int(user_id))


def _album_buffer_key_for(media_group_id: str, user_id: int) -> str:
    return _ALBUM_BUFFER_KEY.format(media_group_id=media_group_id, user_id=int(user_id))


def _album_lock_key_for(media_group_id: str, user_id: int) -> str:
    return _ALBUM_LOCK_KEY.format(media_group_id=media_group_id, user_id=int(user_id))


async def _store_card_message_id(
    ctx: ContextTypes.DEFAULT_TYPE, chat_id: int, user_id: int, message_id: int
) -> None:
    try:
        redis = _resolve_redis(ctx)
        await redis.set(
            _card_key(_CARD_MESSAGE_KEY, chat_id=chat_id, user_id=user_id),
            int(message_id),
            ex=_CARD_STORAGE_TTL,
        )
    except Exception as exc:
        await handle_async_error(exc, "BananaAsync.card_store_id")


async def _load_card_message_id(
    ctx: ContextTypes.DEFAULT_TYPE, chat_id: int, user_id: int
) -> Optional[int]:
    try:
        redis = _resolve_redis(ctx)
    except Exception as exc:
        await handle_async_error(exc, "BananaAsync.card_load_id_redis")
        return None
    try:
        raw = await redis.get(
            _card_key(_CARD_MESSAGE_KEY, chat_id=chat_id, user_id=user_id)
        )
    except Exception as exc:
        await handle_async_error(exc, "BananaAsync.card_load_id")
        return None
    if not raw:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


async def _clear_card_storage(
    ctx: ContextTypes.DEFAULT_TYPE, chat_id: int, user_id: int
) -> None:
    try:
        redis = _resolve_redis(ctx)
    except Exception as exc:
        await handle_async_error(exc, "BananaAsync.card_clear_id_redis")
        return
    try:
        await redis.delete(
            _card_key(_CARD_MESSAGE_KEY, chat_id=chat_id, user_id=user_id),
            _card_key(_CARD_SNAPSHOT_KEY, chat_id=chat_id, user_id=user_id),
        )
    except Exception as exc:
        await handle_async_error(exc, "BananaAsync.card_clear_id")


async def _schedule_album_finalize(
    ctx: ContextTypes.DEFAULT_TYPE,
    *,
    chat_id: int,
    user_id: int,
    media_group_id: str,
) -> None:
    await asyncio.sleep(_ALBUM_DEBOUNCE_SECONDS)
    try:
        redis = _resolve_redis(ctx)
    except Exception as exc:
        await handle_async_error(exc, "BananaAsync.album_redis")
        return
    lock_key = _album_lock_key_for(media_group_id, user_id)
    try:
        acquired = await redis.set(lock_key, "1", ex=5, nx=True)
    except Exception as exc:
        await handle_async_error(exc, "BananaAsync.album_lock")
        return
    if not acquired:
        return
    buffer_key = _album_buffer_key_for(media_group_id, user_id)
    try:
        payload_raw = await redis.get(buffer_key)
    except Exception as exc:
        await handle_async_error(exc, "BananaAsync.album_fetch")
        payload_raw = None
    try:
        payload = json.loads(payload_raw) if payload_raw else {}
    except Exception:
        payload = {}
    updated_at = float(payload.get("updated_at") or 0.0)
    wait_for = (updated_at + _ALBUM_DEBOUNCE_SECONDS) - time.monotonic()
    if wait_for and wait_for > 0:
        await asyncio.sleep(min(wait_for, _ALBUM_DEBOUNCE_SECONDS))
        try:
            payload_raw = await redis.get(buffer_key)
        except Exception as exc:
            await handle_async_error(exc, "BananaAsync.album_fetch")
            payload_raw = None
        try:
            payload = json.loads(payload_raw) if payload_raw else {}
        except Exception:
            payload = {}
    try:
        await redis.delete(buffer_key)
    except Exception:
        pass
    try:
        await redis.delete(lock_key)
    except Exception:
        pass
    if not payload:
        return
    await _apply_album_payload(ctx, chat_id=chat_id, user_id=user_id, payload=payload)


async def _apply_album_payload(
    ctx: ContextTypes.DEFAULT_TYPE,
    *,
    chat_id: int,
    user_id: int,
    payload: Mapping[str, Any],
) -> None:
    raw_photos = payload.get("photos") or []
    photos: list[str] = []
    if isinstance(raw_photos, Sequence):
        for item in raw_photos:
            if isinstance(item, str):
                stripped = item.strip()
            else:
                stripped = str(item or "").strip()
            if stripped:
                photos.append(stripped)
    prompt_raw = payload.get("prompt")
    prompt_text = prompt_raw.strip() if isinstance(prompt_raw, str) else ""
    if not photos and not prompt_text:
        return
    try:
        state = await _load_state(ctx, user_id)
    except Exception as exc:
        await handle_async_error(exc, "BananaAsync.album_state_load")
        return
    added = 0
    overflow = False
    for fid in photos:
        if fid in state.photos:
            continue
        if len(state.photos) >= _MAX_CARD_IMAGES:
            overflow = True
            break
        state.photos.append(fid)
        added += 1
    prompt_updated = False
    if prompt_text:
        if (state.prompt or "").strip() != prompt_text:
            state.prompt = prompt_text
            prompt_updated = True
    if not added and not prompt_updated:
        return
    try:
        await _save_state(ctx, user_id, state)
    except Exception as exc:
        await handle_async_error(exc, "BananaAsync.album_state_save")
    await _render_card(ctx=ctx, chat_id=chat_id, state=state, user_id=user_id)
    if overflow:
        try:
            await ctx.bot.send_message(chat_id, "Можно до 4 фото.")
        except Exception:
            pass
    log.info(
        "banana.album.flushed",
        extra={
            "user_id": user_id,
            "added": added,
            "total": len(state.photos),
            "prompt": bool(prompt_text),
            "overflow": overflow,
        },
    )


async def _buffer_media_group_message(
    ctx: ContextTypes.DEFAULT_TYPE,
    *,
    chat_id: int,
    user_id: int,
    media_group_id: str,
    file_id: Optional[str],
    prompt_text: Optional[str],
) -> bool:
    if not media_group_id or not file_id:
        return False
    try:
        redis = _resolve_redis(ctx)
    except Exception as exc:
        await handle_async_error(exc, "BananaAsync.album_redis")
        return False
    buffer_key = _album_buffer_key_for(media_group_id, user_id)
    try:
        payload_raw = await redis.get(buffer_key)
    except Exception as exc:
        await handle_async_error(exc, "BananaAsync.album_fetch")
        payload_raw = None
    try:
        payload = json.loads(payload_raw) if payload_raw else {}
    except Exception:
        payload = {}
    photos = payload.get("photos")
    if not isinstance(photos, list):
        photos = []
    if file_id not in photos:
        photos.append(file_id)
    payload["photos"] = photos
    if prompt_text and not payload.get("prompt"):
        payload["prompt"] = prompt_text
    payload["updated_at"] = time.monotonic()
    try:
        await redis.set(buffer_key, json.dumps(payload), ex=_ALBUM_BUFFER_TTL)
    except Exception as exc:
        await handle_async_error(exc, "BananaAsync.album_store")
        return False
    asyncio.create_task(
        _schedule_album_finalize(
            ctx,
            chat_id=chat_id,
            user_id=user_id,
            media_group_id=media_group_id,
        )
    )
    log.info(
        "banana.album.buffered",
        extra={
            "user_id": user_id,
            "media_group_id": media_group_id,
            "count": len(photos),
            "prompt": bool(prompt_text),
        },
    )
    return True


def _serialize_markup(markup) -> str:
    if markup is None:
        return "null"
    try:
        payload = markup.to_dict()
    except Exception:
        payload = None
    if not payload:
        return "null"
    try:
        return json.dumps(payload, sort_keys=True, ensure_ascii=False)
    except Exception:
        return str(payload)


def _snapshot_payload(text: str, markup_repr: str, generating: bool) -> str:
    return json.dumps(
        {
            "text": text,
            "markup": markup_repr,
            "generating": bool(generating),
        },
        sort_keys=True,
        ensure_ascii=False,
    )


async def _load_card_snapshot(
    ctx: ContextTypes.DEFAULT_TYPE, chat_id: int, user_id: int
) -> Optional[str]:
    try:
        redis = _resolve_redis(ctx)
    except Exception as exc:
        await handle_async_error(exc, "BananaAsync.card_snapshot_load_redis")
        return None
    try:
        raw = await redis.get(
            _card_key(_CARD_SNAPSHOT_KEY, chat_id=chat_id, user_id=user_id)
        )
    except Exception as exc:
        await handle_async_error(exc, "BananaAsync.card_snapshot_load")
        return None
    if not raw:
        return None
    if isinstance(raw, bytes):
        return raw.decode("utf-8", "ignore")
    return str(raw)


async def _store_card_snapshot(
    ctx: ContextTypes.DEFAULT_TYPE, chat_id: int, user_id: int, snapshot: str
) -> None:
    try:
        redis = _resolve_redis(ctx)
        await redis.set(
            _card_key(_CARD_SNAPSHOT_KEY, chat_id=chat_id, user_id=user_id),
            snapshot,
            ex=_CARD_STORAGE_TTL,
        )
    except Exception as exc:
        await handle_async_error(exc, "BananaAsync.card_snapshot_store")


async def _render_card(
    *,
    ctx: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    state: BananaState,
    user_id: Optional[int],
    message=None,
    generating: bool = False,
    text_override: Optional[str] = None,
) -> Optional[int]:
    text = text_override or banana_card_text(0, state)
    markup = build_banana_card_kb(state, generating=generating)
    markup_repr = _serialize_markup(markup)
    snapshot = _snapshot_payload(text, markup_repr, generating)

    message_id = None
    reuse_enabled = _CARD_REUSE_ENABLED
    if reuse_enabled and message is not None and getattr(message, "message_id", None):
        message_id = message.message_id
    elif reuse_enabled and user_id is not None:
        message_id = await _load_card_message_id(ctx, chat_id, user_id)

    if reuse_enabled and user_id is not None:
        previous_snapshot = await _load_card_snapshot(ctx, chat_id, user_id)
        if previous_snapshot == snapshot and message_id is not None:
            return message_id

    if reuse_enabled and message_id is not None:
        try:
            await ctx.bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=text,
                reply_markup=markup,
            )
            if user_id is not None:
                await _store_card_message_id(ctx, chat_id, user_id, message_id)
                await _store_card_snapshot(ctx, chat_id, user_id, snapshot)
            log.info(
                "banana.card.update",
                extra={
                    "user_id": user_id,
                    "photos": len(state.photos),
                    "has_prompt": bool(state.prompt),
                    "generating": generating,
                    "message_id": message_id,
                },
            )
            return message_id
        except Exception as exc:
            await handle_async_error(exc, "BananaAsync.card_edit")
            if user_id is not None:
                await _clear_card_storage(ctx, chat_id, user_id)
            message_id = None

    try:
        sent = await ctx.bot.send_message(chat_id, text, reply_markup=markup)
    except Exception as exc:
        await handle_async_error(exc, "BananaAsync.card_send")
        return None

    if reuse_enabled and user_id is not None:
        await _store_card_message_id(ctx, chat_id, user_id, sent.message_id)
        await _store_card_snapshot(ctx, chat_id, user_id, snapshot)

    log.info(
        "banana.card.update",
        extra={
            "user_id": user_id,
            "photos": len(state.photos),
            "has_prompt": bool(state.prompt),
            "generating": generating,
            "message_id": getattr(sent, "message_id", None),
        },
    )
    return getattr(sent, "message_id", None)


async def open_card(update: Update, ctx: ContextTypes.DEFAULT_TYPE, payload=None) -> None:
    query = update.callback_query
    if query is None:
        return
    try:
        await query.answer()
    except Exception:
        pass
    user = update.effective_user
    message = query.message
    chat = getattr(message, "chat", None) or update.effective_chat
    if user is None or chat is None:
        return
    redis = _resolve_redis(ctx)
    try:
        await ensure(redis, user.id)
    except Exception as exc:
        await handle_async_error(exc, "BananaAsync.card_init")
        return
    try:
        state = await _load_state(ctx, user.id)
    except Exception as exc:
        await handle_async_error(exc, "BananaAsync.card_load")
        return
    stored_message_id: Optional[int] = None
    reused = False
    if _CARD_REUSE_ENABLED and user is not None:
        stored_message_id = await _load_card_message_id(ctx, chat.id, user.id)
        reused = stored_message_id is not None
    if (
        _CARD_REUSE_ENABLED
        and message is not None
        and getattr(message, "message_id", None)
    ):
        await _store_card_message_id(ctx, chat.id, user.id, message.message_id)
        if stored_message_id == message.message_id:
            reused = True
    await _render_card(
        ctx=ctx,
        chat_id=chat.id,
        state=state,
        user_id=user.id,
        message=message,
    )
    try:
        await async_input_state.set(user.id, mode="banana")
    except Exception as exc:
        await handle_async_error(exc, "BananaAsync.input_state_set")
    log.info(
        "banana.card.opened",
        extra={
            "user_id": user.id,
            "photos": len(state.photos),
            "has_prompt": bool(state.prompt),
            "reused_msg": reused,
        },
    )


async def on_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    chat = update.effective_chat
    user = update.effective_user
    if message is None or chat is None or user is None:
        return
    file_id = _extract_photo_file_id(message)
    prompt_text = _extract_prompt_text(message)
    media_group_id = getattr(message, "media_group_id", None)
    has_updates = False
    try:
        state = await _load_state(ctx, user.id)
    except Exception as exc:
        await handle_async_error(exc, "BananaAsync.photo_load")
        return
    if media_group_id and file_id:
        buffered = await _buffer_media_group_message(
            ctx,
            chat_id=chat.id,
            user_id=user.id,
            media_group_id=str(media_group_id),
            file_id=file_id,
            prompt_text=prompt_text,
        )
        if buffered:
            try:
                await ctx.bot.delete_message(chat.id, message.message_id)
            except Exception:
                pass
            return
    if file_id:
        if len(state.photos) >= _MAX_CARD_IMAGES:
            if prompt_text is not None:
                state.prompt = prompt_text or None
                try:
                    await _save_state(ctx, user.id, state)
                except Exception as exc:
                    await handle_async_error(exc, "BananaAsync.photo_save")
                await _render_card(ctx=ctx, chat_id=chat.id, state=state, user_id=user.id)
            await ctx.bot.send_message(
                chat.id,
                "Можно до 4 фото.",
            )
            try:
                await ctx.bot.delete_message(chat.id, message.message_id)
            except Exception:
                pass
            return
        state.photos.append(file_id)
        has_updates = True
    if prompt_text is not None:
        state.prompt = prompt_text or None
        has_updates = True
    if not has_updates:
        return
    try:
        await _save_state(ctx, user.id, state)
    except Exception as exc:
        await handle_async_error(exc, "BananaAsync.photo_save")
    try:
        await ctx.bot.delete_message(chat.id, message.message_id)
    except Exception:
        pass
    await _render_card(ctx=ctx, chat_id=chat.id, state=state, user_id=user.id)


async def on_prompt(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    chat = update.effective_chat
    user = update.effective_user
    text = getattr(message, "text", None)
    if message is None or chat is None or user is None or text is None:
        return
    prompt = text.strip()
    try:
        state = await _load_state(ctx, user.id)
    except Exception as exc:
        await handle_async_error(exc, "BananaAsync.text_load")
        return
    state.prompt = prompt or None
    try:
        await _save_state(ctx, user.id, state)
    except Exception as exc:
        await handle_async_error(exc, "BananaAsync.text_save")
    try:
        await ctx.bot.delete_message(chat.id, message.message_id)
    except Exception:
        pass
    await _render_card(ctx=ctx, chat_id=chat.id, state=state, user_id=user.id)


async def clear_card(update: Update, ctx: ContextTypes.DEFAULT_TYPE, payload=None) -> None:
    query = update.callback_query
    user = update.effective_user
    chat = update.effective_chat
    message = query.message if query else None
    if query is not None:
        await _acknowledge_callback(query, ctx)
    if user is None or chat is None:
        return
    try:
        state = await _load_state(ctx, user.id)
    except Exception as exc:
        await handle_async_error(exc, "BananaAsync.clear_load")
        return
    state.reset()
    try:
        await _save_state(ctx, user.id, state)
    except Exception as exc:
        await handle_async_error(exc, "BananaAsync.clear_save")
    await _render_card(
        ctx=ctx,
        chat_id=chat.id,
        state=state,
        user_id=user.id,
        message=message if getattr(message, "text", None) else None,
    )
    try:
        await async_input_state.set(user.id, mode="banana")
    except Exception as exc:
        await handle_async_error(exc, "BananaAsync.clear_input_state")
    log.info("banana.clear", extra={"user_id": user.id})


async def start_generation(update: Update, ctx: ContextTypes.DEFAULT_TYPE, payload=None) -> None:
    query = update.callback_query
    user = update.effective_user
    message = query.message if query else None
    chat = getattr(message, "chat", None) or update.effective_chat
    if query is None or user is None or chat is None:
        return
    await _acknowledge_callback(query, ctx, text="Запускаю…")
    try:
        state = await _load_state(ctx, user.id)
    except Exception as exc:
        await handle_async_error(exc, "BananaAsync.start_load")
        return
    session_payload = await _load_session_payload(ctx, user.id)
    session_images = session_payload.get("images") or list(state.photos)
    session_images = [str(fid) for fid in session_images if str(fid or "").strip()][: _MAX_CARD_IMAGES]
    if not session_images:
        session_images = list(state.photos)[:_MAX_CARD_IMAGES]
    prompt_text = session_payload.get("prompt") or state.prompt or ""
    normalized_prompt = (prompt_text or "").strip()
    has_prompt = bool(normalized_prompt)
    handler = _get_default_handler()
    uploads: list[dict[str, Any]] = []
    upload_failures = False
    for file_id in session_images:
        try:
            upload_result = await handler.upload_image(ctx.bot, file_id, user_id=user.id)
        except BananaBackendError as exc:
            await handle_async_error(exc, "BananaAsync.start_upload_image")
            upload_failures = True
            continue
        if not upload_result:
            upload_failures = True
            continue
        uploads.append(upload_result)
    if session_images and not uploads:
        feedback = "Не удалось загрузить фото. Попробуйте ещё раз" if upload_failures else "Добавьте фото или промпт"
        await _render_card(
            ctx=ctx,
            chat_id=chat.id,
            state=state,
            user_id=user.id,
            message=message if getattr(message, "text", None) else None,
            text_override=feedback,
        )
        return
    if not uploads and not has_prompt:
        await _render_card(
            ctx=ctx,
            chat_id=chat.id,
            state=state,
            user_id=user.id,
            message=message if getattr(message, "text", None) else None,
            text_override="Добавьте фото или промпт",
        )
        return
    state.photos = list(session_images)
    state.prompt = normalized_prompt or None
    state.last_payload = {
        "images": list(state.photos),
        "prompt": normalized_prompt,
        "uploads": list(uploads),
    }
    try:
        await _save_state(ctx, user.id, state)
    except Exception as exc:
        await handle_async_error(exc, "BananaAsync.start_snapshot")
    try:
        await async_input_state.set(user.id, mode="banana")
    except Exception as exc:
        await handle_async_error(exc, "BananaAsync.start_input_state")

    await _render_card(
        ctx=ctx,
        chat_id=chat.id,
        state=state,
        user_id=user.id,
        message=message if getattr(message, "text", None) else None,
        generating=True,
    )

    log.info(
        "banana.gen.start",
        extra={
            "user_id": user.id,
            "payload": {"images": len(uploads), "has_prompt": has_prompt},
            "source": "start",
        },
    )

    try:
        await handler.run(update, ctx)
    finally:
        try:
            refreshed_state = await _load_state(ctx, user.id)
        except Exception as exc:
            await handle_async_error(exc, "BananaAsync.start_reload_state")
            refreshed_state = state
        await _render_card(
            ctx=ctx,
            chat_id=chat.id,
            state=refreshed_state,
            user_id=user.id,
        )


async def generation_busy(update: Update, ctx: ContextTypes.DEFAULT_TYPE, payload=None) -> None:
    query = update.callback_query
    if query is None:
        return
    try:
        await query.answer("Генерация уже идёт")
    except Exception:
        pass


async def restart_generation(update: Update, ctx: ContextTypes.DEFAULT_TYPE, payload=None) -> None:
    query = update.callback_query
    user = update.effective_user
    message = query.message if query else None
    chat = getattr(message, "chat", None) or update.effective_chat
    if query is None or user is None or chat is None:
        return
    try:
        state = await _load_state(ctx, user.id)
    except Exception as exc:
        await handle_async_error(exc, "BananaAsync.restart_load")
        return
    payload_snapshot = state.last_payload or {}
    photos = payload_snapshot.get("images") or payload_snapshot.get("photos") or []
    prompt = payload_snapshot.get("prompt") or ""
    uploads = payload_snapshot.get("uploads") or []
    uploads_list: list[str] = []
    for item in uploads:
        if isinstance(item, Mapping):
            url = str(item.get("url") or "").strip()
        else:
            url = str(item or "").strip()
        if url:
            uploads_list.append(url)
    if not uploads_list and not photos:
        await _acknowledge_callback(
            query,
            ctx,
            text="Нет сохранённого запроса для повторной генерации.",
            show_alert=True,
        )
        return
    await _acknowledge_callback(query, ctx, text="Запускаю…")
    state.photos = list(map(str, photos))[:_MAX_CARD_IMAGES]
    uploads_list = uploads_list[: len(state.photos)]
    normalized_prompt = (prompt or "").strip()
    state.prompt = normalized_prompt or None
    state.last_payload = {
        "images": list(state.photos),
        "prompt": normalized_prompt,
        "uploads": list(uploads),
    }
    try:
        await _save_state(ctx, user.id, state)
    except Exception as exc:
        await handle_async_error(exc, "BananaAsync.restart_save")
    data = query.data or ""
    parts = data.split(":", 2)
    job_id = parts[2] if len(parts) == 3 else None
    try:
        await async_input_state.set(user.id, mode="banana")
    except Exception as exc:
        await handle_async_error(exc, "BananaAsync.restart_input_state")
    log.info(
        "banana.restart",
        extra={
            "user_id": user.id,
            "photos": len(state.photos),
            "has_prompt": bool(state.prompt),
            "job_id": job_id,
        },
    )
    handler = _get_default_handler()
    await _render_card(
        ctx=ctx,
        chat_id=chat.id,
        state=state,
        user_id=user.id,
        message=message if getattr(message, "text", None) else None,
        generating=True,
    )

    log.info(
        "banana.gen.start",
        extra={
            "user_id": user.id,
            "payload": {"images": len(uploads_list), "has_prompt": bool(state.prompt)},
            "source": "restart",
        },
    )

    try:
        await handler.run(update, ctx)
    finally:
        try:
            refreshed_state = await _load_state(ctx, user.id)
        except Exception as exc:
            await handle_async_error(exc, "BananaAsync.restart_reload_state")
            refreshed_state = state
        await _render_card(
            ctx=ctx,
            chat_id=chat.id,
            state=refreshed_state,
            user_id=user.id,
        )


async def new_card(update: Update, ctx: ContextTypes.DEFAULT_TYPE, payload=None) -> None:
    query = update.callback_query
    user = update.effective_user
    chat = update.effective_chat
    message = query.message if query else None
    if query is None or user is None or chat is None:
        return
    try:
        state = await _load_state(ctx, user.id)
    except Exception as exc:
        await handle_async_error(exc, "BananaAsync.new_load")
        return
    state.reset()
    try:
        await _save_state(ctx, user.id, state)
    except Exception as exc:
        await handle_async_error(exc, "BananaAsync.new_save")
    try:
        await query.answer()
    except Exception:
        pass
    try:
        await query.edit_message_reply_markup(reply_markup=None)
    except Exception:
        pass
    try:
        await async_input_state.clear(user.id, reason="banana_exit")
    except Exception as exc:
        await handle_async_error(exc, "BananaAsync.new_clear_state")
    editable_message = message if getattr(message, "text", None) else None
    await _render_card(
        ctx=ctx,
        chat_id=chat.id,
        state=state,
        user_id=user.id,
        message=editable_message,
    )
    try:
        await async_input_state.set(user.id, mode="banana")
    except Exception as exc:
        await handle_async_error(exc, "BananaAsync.new_set_state")


__all__ = [
    "BananaAsyncHandler",
    "open_card",
    "on_photo",
    "on_prompt",
    "clear_card",
    "start_generation",
    "generation_busy",
    "restart_generation",
    "new_card",
]
