"""Helpers for validating and uploading Suno cover sources."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Mapping, MutableMapping, Optional
from urllib.parse import urljoin, urlparse

import httpx

from settings import (
    SUNO_API_TOKEN,
    SUNO_TIMEOUT_SEC,
    UPLOAD_BASE_URL,
    UPLOAD_STREAM_PATH,
    UPLOAD_URL_PATH,
)

MAX_AUDIO_MB = 50
_ALLOWED_EXTENSIONS = {".mp3", ".wav"}


class CoverSourceError(Exception):
    """Base error for cover source operations."""


class CoverSourceValidationError(CoverSourceError):
    """Raised when the provided source is invalid."""


class CoverSourceClientError(CoverSourceError):
    """Raised when the upload service rejects the request."""


class CoverSourceUnavailableError(CoverSourceError):
    """Raised when the upload service is temporarily unavailable."""


class _RetryableError(Exception):
    """Internal helper to signal retryable errors."""

    def __init__(self, message: str, response: Optional[httpx.Response] = None) -> None:
        super().__init__(message)
        self.response = response


def _compose_upload_url(path: str) -> str:
    base = (UPLOAD_BASE_URL or "").rstrip("/") + "/"
    return urljoin(base, path.lstrip("/"))


def _auth_header() -> dict[str, str]:
    token = (SUNO_API_TOKEN or "").strip()
    if not token:
        raise CoverSourceUnavailableError("missing-token")
    if not token.lower().startswith("bearer "):
        token = f"Bearer {token}"
    return {"Authorization": token}


def _extract_kie_file_id(payload: Mapping[str, Any]) -> Optional[str]:
    for key in ("kie_file_id", "file_id", "id"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    data = payload.get("data")
    if isinstance(data, Mapping):
        return _extract_kie_file_id(data)
    return None


async def _request_with_retries(
    operation: str,
    request_cb,
    *,
    logger: Optional[logging.Logger],
    request_id: str,
) -> httpx.Response:
    delays = [1, 2, 4]
    last_exc: Optional[Exception] = None
    for attempt, delay in enumerate(delays, start=1):
        try:
            return await request_cb()
        except _RetryableError as exc:  # pragma: no cover - simple control flow
            last_exc = exc
            if logger:
                logger.warning(
                    "cover.upload_retry",
                    extra={
                        "request_id": request_id,
                        "operation": operation,
                        "attempt": attempt,
                        "error": str(exc),
                    },
                )
        except httpx.TimeoutException as exc:
            last_exc = exc
            if logger:
                logger.warning(
                    "cover.upload_timeout",
                    extra={
                        "request_id": request_id,
                        "operation": operation,
                        "attempt": attempt,
                        "error": str(exc),
                    },
                )
        except httpx.HTTPError as exc:
            last_exc = exc
            if logger:
                logger.warning(
                    "cover.upload_http_error",
                    extra={
                        "request_id": request_id,
                        "operation": operation,
                        "attempt": attempt,
                        "error": str(exc),
                    },
                )
        if attempt < len(delays):
            await asyncio.sleep(delay)
    if isinstance(last_exc, _RetryableError) and getattr(last_exc, "response", None):
        response = last_exc.response  # type: ignore[assignment]
        assert response is not None
        return response
    raise CoverSourceUnavailableError(operation) from last_exc


def _timeout() -> httpx.Timeout:
    total = max(float(SUNO_TIMEOUT_SEC or 60), 1.0)
    return httpx.Timeout(total=total)


async def upload_stream(
    data: bytes,
    filename: str,
    mime_type: Optional[str],
    *,
    request_id: str,
    logger: Optional[logging.Logger] = None,
) -> str:
    """Upload audio bytes and return the KIE file identifier."""

    if not data:
        raise CoverSourceValidationError("empty-data")

    headers = _auth_header()
    url = _compose_upload_url(UPLOAD_STREAM_PATH)
    timeout = _timeout()

    async def _do_request() -> httpx.Response:
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                response = await client.post(
                    url,
                    headers=headers,
                    files={
                        "file": (
                            filename or "audio.mp3",
                            data,
                            mime_type or "application/octet-stream",
                        )
                    },
                )
            except httpx.TimeoutException:
                raise
            except httpx.HTTPError as exc:
                raise _RetryableError(str(exc)) from exc
        if response.status_code in {429} or 500 <= response.status_code < 600:
            raise _RetryableError(f"status:{response.status_code}", response=response)
        if 400 <= response.status_code < 500:
            raise CoverSourceClientError(f"status:{response.status_code}")
        return response

    response = await _request_with_retries(
        "stream",
        _do_request,
        logger=logger,
        request_id=request_id,
    )

    status = response.status_code
    if status in {429} or 500 <= status < 600:
        raise CoverSourceUnavailableError(f"status:{status}")

    try:
        payload = response.json()
    except ValueError:
        raise CoverSourceUnavailableError("invalid-json") from None

    if not isinstance(payload, Mapping):
        if isinstance(payload, MutableMapping):
            payload = dict(payload)
        else:
            payload = {"data": payload}

    kie_file_id = _extract_kie_file_id(payload)
    if not kie_file_id:
        raise CoverSourceUnavailableError("missing-kie-file-id")
    return kie_file_id


async def upload_url(
    source_url: str,
    *,
    request_id: str,
    logger: Optional[logging.Logger] = None,
) -> str:
    """Upload a remote URL and return the KIE file identifier."""

    headers = {"Content-Type": "application/json"}
    headers.update(_auth_header())
    url = _compose_upload_url(UPLOAD_URL_PATH)
    timeout = _timeout()

    async def _do_request() -> httpx.Response:
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                response = await client.post(
                    url,
                    headers=headers,
                    json={"url": source_url},
                )
            except httpx.TimeoutException:
                raise
            except httpx.HTTPError as exc:
                raise _RetryableError(str(exc)) from exc
        if response.status_code in {429} or 500 <= response.status_code < 600:
            raise _RetryableError(f"status:{response.status_code}", response=response)
        if 400 <= response.status_code < 500:
            raise CoverSourceClientError(f"status:{response.status_code}")
        return response

    response = await _request_with_retries(
        "url",
        _do_request,
        logger=logger,
        request_id=request_id,
    )

    status = response.status_code
    if status in {429} or 500 <= status < 600:
        raise CoverSourceUnavailableError(f"status:{status}")

    try:
        payload = response.json()
    except ValueError:
        raise CoverSourceUnavailableError("invalid-json") from None

    if not isinstance(payload, Mapping):
        payload = {"data": payload}

    kie_file_id = _extract_kie_file_id(payload)
    if not kie_file_id:
        raise CoverSourceUnavailableError("missing-kie-file-id")
    return kie_file_id


def validate_url(url: str) -> str:
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"}:
        raise CoverSourceValidationError("invalid-scheme")
    if not parsed.netloc:
        raise CoverSourceValidationError("missing-host")
    return url.strip()


async def ensure_audio_url(url: str) -> str:
    """Validate that the URL points to an audio resource."""

    cleaned = validate_url(url)
    parsed = urlparse(cleaned)
    path = parsed.path.lower()
    if any(path.endswith(ext) for ext in _ALLOWED_EXTENSIONS):
        return cleaned

    timeout = _timeout()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.head(cleaned, follow_redirects=True)
    except httpx.HTTPError:
        raise CoverSourceValidationError("head-failed")

    if response.status_code >= 400:
        raise CoverSourceValidationError("head-status")

    content_type = response.headers.get("Content-Type", "").lower()
    if not content_type.startswith("audio/"):
        raise CoverSourceValidationError("invalid-content-type")
    return cleaned


def validate_audio_file(
    mime_type: Optional[str],
    file_name: Optional[str],
    file_size: Optional[int],
) -> tuple[str, str]:
    """Validate Telegram file metadata and return filename & mime."""

    size = int(file_size or 0)
    if size <= 0:
        raise CoverSourceValidationError("empty-file")
    if size > MAX_AUDIO_MB * 1024 * 1024:
        raise CoverSourceValidationError("too-large")

    mime = (mime_type or "").strip().lower()
    name = (file_name or "").strip()

    if mime and not mime.startswith("audio/"):
        raise CoverSourceValidationError("invalid-mime")

    if not mime:
        lowered = name.lower()
        if lowered.endswith(".mp3"):
            mime = "audio/mpeg"
        elif lowered.endswith(".wav"):
            mime = "audio/wav"
        else:
            raise CoverSourceValidationError("unknown-mime")

    if not name:
        extension = ".mp3" if mime.endswith("mpeg") else ".wav"
        name = f"cover{extension}"

    return name, mime


__all__ = [
    "MAX_AUDIO_MB",
    "CoverSourceError",
    "CoverSourceValidationError",
    "CoverSourceClientError",
    "CoverSourceUnavailableError",
    "upload_stream",
    "upload_url",
    "ensure_audio_url",
    "validate_audio_file",
]

