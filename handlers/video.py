"""Handlers for the simplified VEO animate flow."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Mapping, MutableMapping, Optional, Sequence
from urllib.parse import urlparse

import httpx
from telegram import Update
from telegram.ext import ContextTypes
from telegram.error import TelegramError

from settings import (
    HTTP_TIMEOUT_CONNECT,
    HTTP_TIMEOUT_READ,
    HTTP_TIMEOUT_TOTAL,
    KIE_API_KEY,
    KIE_BASE_URL,
    KIE_GEN_PATH,
    KIE_HD_PATH,
    KIE_STATUS_PATH,
)

from redis_utils import remember_veo_anim_job

logger = logging.getLogger("veo.anim")

_POLL_INTERVAL = 2.0
_POLL_TIMEOUT = 240.0
_MAX_VIDEO_BYTES = 50 * 1024 * 1024
_WAIT_FLAG = "veo_animate_waiting_photo"

_ERROR_TIMEOUT = "Не удалось оживить изображение вовремя. Попробуйте ещё раз."
_ERROR_BAD_REQUEST = (
    "Запрос отклонён. Измените описание и избегайте запрещённого контента."
)
_ERROR_GENERIC = "Что-то пошло не так. Попробуйте другой кадр или запрос."
_PROMPT_PHOTO = "📸 Отправьте фото, и я сразу попробую его оживить."


class VeoAnimateError(RuntimeError):
    """Base class for VEO animate errors."""


class VeoAnimateBadRequest(VeoAnimateError):
    """Raised when backend rejects the request due to bad input."""


class VeoAnimateTimeout(VeoAnimateError):
    """Raised when polling timed out."""


class VeoAnimateHTTPError(VeoAnimateError):
    """Raised for non-2xx HTTP responses."""

    def __init__(self, status_code: int, payload: Mapping[str, Any]):
        super().__init__(f"HTTP {status_code}")
        self.status_code = int(status_code)
        self.payload = payload


def _http_timeout() -> httpx.Timeout:
    total = float(HTTP_TIMEOUT_TOTAL or 75.0)
    connect = float(HTTP_TIMEOUT_CONNECT or 10.0)
    read = float(HTTP_TIMEOUT_READ or 60.0)
    return httpx.Timeout(timeout=total, connect=connect, read=read)


def _headers(method: str) -> Mapping[str, str]:
    token = (KIE_API_KEY or "").strip()
    if token and not token.lower().startswith("bearer "):
        token = f"Bearer {token}"
    headers = {"Authorization": token} if token else {}
    if method.upper() == "POST":
        headers = {**headers, "Content-Type": "application/json"}
    return headers


async def _request_json(
    method: str,
    path: str,
    *,
    json_payload: Optional[Mapping[str, Any]] = None,
    params: Optional[Mapping[str, Any]] = None,
    client: Optional[httpx.AsyncClient] = None,
) -> Mapping[str, Any]:
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(base_url=KIE_BASE_URL, timeout=_http_timeout())
    try:
        response = await client.request(
            method,
            path,
            json=json_payload,
            params=params,
            headers=_headers(method),
            follow_redirects=True,
        )
    except httpx.TimeoutException as exc:  # pragma: no cover - network guard
        raise VeoAnimateError("timeout") from exc
    except httpx.RequestError as exc:  # pragma: no cover - network guard
        raise VeoAnimateError("network") from exc
    finally:
        if own_client:
            await client.aclose()
    try:
        payload = response.json()
    except ValueError:
        payload = {"raw": response.text}
    if response.status_code >= 400:
        raise VeoAnimateHTTPError(response.status_code, payload)
    if not isinstance(payload, Mapping):
        return {"value": payload}
    return payload


def _ensure_state(context: ContextTypes.DEFAULT_TYPE) -> MutableMapping[str, Any]:
    user_data = getattr(context, "user_data", None)
    if not isinstance(user_data, MutableMapping):
        user_data = {}
        setattr(context, "user_data", user_data)
    state = user_data.get("state")
    if not isinstance(state, MutableMapping):
        state = {}
        user_data["state"] = state
    return state


def _extract_mapping(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    data = payload.get("data")
    if isinstance(data, Mapping):
        return data
    return payload


def _normalize_status(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    mapping = {
        "success": "done",
        "succeeded": "done",
        "completed": "done",
        "finish": "done",
        "finished": "done",
        "ok": "done",
        "1": "done",
        "true": "done",
        "pending": "pending",
        "queued": "pending",
        "queue": "pending",
        "wait": "pending",
        "waiting": "pending",
        "processing": "processing",
        "in_progress": "processing",
    }
    if text in mapping:
        return mapping[text]
    if text.startswith("done") or text.startswith("success"):
        return "done"
    if text.startswith("pend") or text.startswith("wait"):
        return "pending"
    if text.startswith("process"):
        return "processing"
    if text in {"0", "false"}:
        return "pending"
    return text or None


def _extract_status(payload: Mapping[str, Any]) -> Optional[str]:
    data = _extract_mapping(payload)
    for key in ("status", "state", "jobStatus", "successFlag"):
        value = data.get(key) if key in data else payload.get(key)
        normalized = _normalize_status(value)
        if normalized:
            return normalized
    return None


def _normalize_url_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except ValueError:
                pass
            else:
                return _normalize_url_values(parsed)
        if text.lower().startswith("http"):
            return [text]
        return []
    if isinstance(value, Mapping):
        collected: list[str] = []
        for key in ("url", "resultUrl", "originUrl", "videoUrl"):
            collected.extend(_normalize_url_values(value.get(key)))
        if "urls" in value:
            collected.extend(_normalize_url_values(value.get("urls")))
        return collected
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        collected: list[str] = []
        for item in value:
            collected.extend(_normalize_url_values(item))
        return collected
    return []


def _extract_result_candidates(payload: Mapping[str, Any]) -> list[str]:
    urls: list[str] = []
    if isinstance(payload, Mapping):
        urls.extend(_normalize_url_values(payload.get("result")))
        urls.extend(_normalize_url_values(payload.get("resultUrl")))
        urls.extend(_normalize_url_values(payload.get("resultUrls")))
        urls.extend(_normalize_url_values(payload.get("videoUrl")))
        urls.extend(_normalize_url_values(payload.get("videoUrls")))
        urls.extend(_normalize_url_values(payload.get("urls")))
    data = _extract_mapping(payload)
    if data is not payload:
        urls.extend(_extract_result_candidates(data))
    unique: list[str] = []
    seen: set[str] = set()
    for url in urls:
        if url and url not in seen:
            unique.append(url)
            seen.add(url)
    return unique


def _extract_hd_task_id(payload: Mapping[str, Any]) -> Optional[str]:
    data = _extract_mapping(payload)
    for key in (
        "hdTaskId",
        "hd_task_id",
        "hdTaskID",
        "taskHdId",
        "1080TaskId",
    ):
        value = data.get(key) if key in data else payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


async def _fetch_hd_urls(
    client: httpx.AsyncClient,
    task_id: str,
) -> list[str]:
    try:
        payload = await _request_json("GET", KIE_HD_PATH, params={"taskId": task_id}, client=client)
    except VeoAnimateHTTPError:
        return []
    except VeoAnimateError:  # pragma: no cover - defensive fallback
        return []
    return _extract_result_candidates(payload)


async def _start_animation(image_url: str, prompt: Optional[str]) -> str:
    payload = {"image_url": image_url}
    if prompt:
        payload["prompt"] = prompt
    try:
        data = await _request_json("POST", KIE_GEN_PATH, json_payload=payload)
    except VeoAnimateHTTPError as exc:
        if 400 <= exc.status_code < 500:
            raise VeoAnimateBadRequest("start rejected") from exc
        raise VeoAnimateError("start failed") from exc
    job_id = None
    for key in ("job_id", "jobId", "taskId", "id"):
        if key in data:
            job_id = data[key]
            break
        inner = data.get("data") if isinstance(data.get("data"), Mapping) else None
        if inner and key in inner:
            job_id = inner[key]
            break
    if job_id is None:
        raise VeoAnimateError("missing job id")
    return str(job_id)


async def _wait_for_result(job_id: str) -> tuple[list[str], Mapping[str, Any]]:
    deadline = time.monotonic() + _POLL_TIMEOUT
    async with httpx.AsyncClient(base_url=KIE_BASE_URL, timeout=_http_timeout()) as client:
        while time.monotonic() < deadline:
            try:
                payload = await _request_json(
                    "GET",
                    KIE_STATUS_PATH,
                    params={"taskId": job_id},
                    client=client,
                )
            except VeoAnimateHTTPError as exc:
                if 400 <= exc.status_code < 500:
                    raise VeoAnimateBadRequest("status rejected") from exc
                raise VeoAnimateError("status failed") from exc
            status = _extract_status(payload)
            normalized = status or "pending"
            if normalized in {"pending", "queue", "waiting"}:
                await asyncio.sleep(_POLL_INTERVAL)
                continue
            if normalized == "processing":
                await asyncio.sleep(_POLL_INTERVAL)
                continue
            if normalized in {"done", "success", "succeed", "completed"}:
                urls = _extract_result_candidates(payload)
                if urls:
                    return urls, payload
                hd_task = _extract_hd_task_id(payload)
                if hd_task:
                    hd_urls = await _fetch_hd_urls(client, hd_task)
                    if hd_urls:
                        return hd_urls, payload
                raise VeoAnimateError("result missing")
            if normalized in {"failed", "error", "blocked", "rejected"}:
                raise VeoAnimateBadRequest("generation rejected")
            await asyncio.sleep(_POLL_INTERVAL)
    raise VeoAnimateTimeout("poll timeout")


async def _fetch_content_length(url: str) -> Optional[int]:
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            response = await client.head(url, follow_redirects=True)
    except Exception:  # pragma: no cover - network guard
        return None
    value = response.headers.get("Content-Length") or response.headers.get("content-length")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):  # pragma: no cover - defensive guard
        return None


def _shorten_url(url: str) -> str:
    try:
        parsed = urlparse(url)
    except ValueError:  # pragma: no cover - defensive guard
        return url[:60]
    host = parsed.netloc
    tail = parsed.path.rstrip("/").split("/")[-1]
    if host and tail:
        return f"{host}/{tail}"[-60:]
    if host:
        return host
    return url[:60]


async def _send_error(bot: Any, chat_id: Optional[int], text: str) -> None:
    if chat_id is None:
        return
    try:
        await bot.send_message(chat_id=chat_id, text=text)
    except TelegramError:  # pragma: no cover - defensive guard
        logger.exception("veo.anim.error_message_fail chat=%s", chat_id)
    except Exception:  # pragma: no cover - defensive guard
        logger.exception("veo.anim.error_message_fail chat=%s", chat_id)


async def veo_animate(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    image_url: Optional[str] = None,
    auto_started: bool = False,
) -> None:
    chat = update.effective_chat
    chat_id = chat.id if chat else None
    user = update.effective_user
    user_id = user.id if user else None
    state = _ensure_state(context)
    source_url = image_url or state.get("last_image_url")
    prompt = state.get("last_prompt") if isinstance(state.get("last_prompt"), str) else None

    if not source_url:
        state[_WAIT_FLAG] = True
        if not auto_started:
            await _send_error(context.bot, chat_id, _PROMPT_PHOTO)
        return

    state[_WAIT_FLAG] = False

    try:
        job_id = await _start_animation(source_url, prompt)
    except VeoAnimateBadRequest:
        logger.info("veo.anim.fail user=%s reason=bad_request", user_id)
        await _send_error(context.bot, chat_id, _ERROR_BAD_REQUEST)
        return
    except VeoAnimateError:
        logger.info("veo.anim.fail user=%s reason=error", user_id)
        await _send_error(context.bot, chat_id, _ERROR_GENERIC)
        return

    if user_id is not None:
        remember_veo_anim_job(user_id, job_id)
    logger.info("veo.anim.request user=%s job=%s", user_id, job_id)

    try:
        urls, _payload = await _wait_for_result(job_id)
    except VeoAnimateTimeout:
        logger.info("veo.anim.fail user=%s reason=timeout", user_id)
        await _send_error(context.bot, chat_id, _ERROR_TIMEOUT)
        return
    except VeoAnimateBadRequest:
        logger.info("veo.anim.fail user=%s reason=bad_request", user_id)
        await _send_error(context.bot, chat_id, _ERROR_BAD_REQUEST)
        return
    except VeoAnimateError:
        logger.info("veo.anim.fail user=%s reason=error", user_id)
        await _send_error(context.bot, chat_id, _ERROR_GENERIC)
        return

    if not urls:
        logger.info("veo.anim.fail user=%s reason=error", user_id)
        await _send_error(context.bot, chat_id, _ERROR_GENERIC)
        return

    result_url = urls[0]
    logger.info("veo.anim.done user=%s url=%s", user_id, _shorten_url(result_url))
    if chat_id is None:
        return

    size = await _fetch_content_length(result_url)
    try:
        if size is not None and size > _MAX_VIDEO_BYTES:
            await context.bot.send_document(chat_id=chat_id, document=result_url)
        else:
            await context.bot.send_video(chat_id=chat_id, video=result_url)
    except TelegramError:
        logger.exception("veo.anim.telegram_send_fail user=%s", user_id)
        await _send_error(context.bot, chat_id, _ERROR_GENERIC)
    except Exception:  # pragma: no cover - defensive guard
        logger.exception("veo.anim.telegram_send_fail user=%s", user_id)
        await _send_error(context.bot, chat_id, _ERROR_GENERIC)


async def veo_animate_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await veo_animate(update, context)


async def handle_veo_animate_photo(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    image_url: str,
) -> None:
    state = _ensure_state(context)
    if not state.get(_WAIT_FLAG):
        return
    state[_WAIT_FLAG] = False
    await veo_animate(update, context, image_url=image_url, auto_started=True)


__all__ = [
    "VeoAnimateBadRequest",
    "VeoAnimateError",
    "VeoAnimateTimeout",
    "handle_veo_animate_photo",
    "veo_animate",
    "veo_animate_command",
]
