"""Helpers for validating and uploading Suno cover sources."""

from __future__ import annotations

import asyncio
import logging
from typing import IO, Any, Mapping, MutableMapping, Optional, Union
from urllib.parse import urlparse

import httpx

from settings import (
    MAX_IN_LOG_BODY,
    SUNO_API_BASE,
    SUNO_API_TOKEN,
    SUNO_TIMEOUT_SEC,
    UPLOAD_BASE_URL,
    UPLOAD_FALLBACK_ENABLED,
    UPLOAD_STREAM_PATH,
    UPLOAD_URL_PATH,
)

MAX_AUDIO_MB = 50
_ALLOWED_EXTENSIONS = {".mp3", ".wav"}
_RETRY_DELAYS = (1, 2, 4)
_CONNECT_TIMEOUT = 10.0
_MAX_FAIL_BODY = min(1024, int(MAX_IN_LOG_BODY))


class CoverSourceError(Exception):
    """Base error for cover source operations."""


class CoverSourceValidationError(CoverSourceError):
    """Raised when the provided source is invalid."""


class CoverSourceClientError(CoverSourceError):
    """Raised when the upload service rejects the request."""


class CoverSourceUnavailableError(CoverSourceError):
    """Raised when the upload service is temporarily unavailable."""


def _normalize_base(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    text = str(url).strip()
    if not text:
        return None
    return text.rstrip("/")


def _normalize_path(path: str) -> str:
    text = str(path or "/").strip()
    if not text:
        return "/"
    if text.startswith("http://") or text.startswith("https://"):
        return text
    if not text.startswith("/"):
        text = f"/{text}"
    return text


def _auth_header() -> dict[str, str]:
    token = (SUNO_API_TOKEN or "").strip()
    if not token:
        raise CoverSourceUnavailableError("missing-token")
    if token.lower().startswith("bearer "):
        return {"Authorization": token}
    return {"Authorization": f"Bearer {token}"}


def _timeout() -> httpx.Timeout:
    read_timeout = max(float(SUNO_TIMEOUT_SEC or 60.0), 1.0)
    write_timeout = max(read_timeout, 1.0)
    return httpx.Timeout(
        connect=_CONNECT_TIMEOUT,
        read=read_timeout,
        write=write_timeout,
        pool=_CONNECT_TIMEOUT,
    )


def _extract_kie_file_id(payload: Mapping[str, Any]) -> Optional[str]:
    for key in ("kie_file_id", "file_id", "id"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    data = payload.get("data")
    if isinstance(data, Mapping):
        return _extract_kie_file_id(data)
    return None


def _trim_body(response: httpx.Response) -> str:
    try:
        body = response.text
    except Exception:  # pragma: no cover - defensive
        body = response.content.decode("utf-8", errors="replace")
    if len(body) > _MAX_FAIL_BODY:
        return body[:_MAX_FAIL_BODY]
    return body


def _log_try(
    logger: Optional[logging.Logger],
    *,
    request_id: str,
    kind: str,
    host: str,
    path: str,
    attempt: int,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    if not logger:
        return
    payload = {"request_id": request_id, "kind": kind, "host": host, "path": path, "attempt": attempt}
    if extra:
        payload.update(extra)
    logger.info("cover_upload_try", extra=payload)


def _log_fail(
    logger: Optional[logging.Logger],
    *,
    request_id: str,
    kind: str,
    host: str,
    path: str,
    attempt: int,
    status: Optional[int] = None,
    reason: Optional[str] = None,
    body: Optional[str] = None,
) -> None:
    if not logger:
        return
    payload: dict[str, Any] = {
        "request_id": request_id,
        "kind": kind,
        "host": host,
        "path": path,
        "attempt": attempt,
    }
    if status is not None:
        payload["status"] = int(status)
    if reason:
        payload["reason"] = reason
    if body:
        payload["body"] = body
    logger.warning("cover_upload_fail", extra=payload)


def _log_ok(
    logger: Optional[logging.Logger],
    *,
    request_id: str,
    kind: str,
    host: str,
    kie_file_id: str,
) -> None:
    if not logger:
        return
    logger.info(
        "cover_upload_ok",
        extra={"request_id": request_id, "kind": kind, "host": host, "kie_file_id": kie_file_id},
    )


async def _perform_upload(
    *,
    request_id: str,
    kind: str,
    path: str,
    request_cb,
    logger: Optional[logging.Logger],
    try_extra: Optional[dict[str, Any]] = None,
) -> tuple[httpx.Response, str]:
    path = _normalize_path(path)
    primary = _normalize_base(UPLOAD_BASE_URL) or _normalize_base(SUNO_API_BASE)
    fallback = _normalize_base(SUNO_API_BASE)
    if fallback == primary:
        fallback = None
    if primary is None and fallback is None:
        raise CoverSourceUnavailableError("missing-host")

    hosts: list[tuple[str, bool]] = []
    if primary is not None:
        hosts.append((primary, False))
    elif fallback is not None:
        hosts.append((fallback, True))
        fallback = None
    if fallback is not None:
        hosts.append((fallback, True))

    last_error: Optional[BaseException] = None
    last_status: Optional[int] = None
    for index, (host, is_fallback) in enumerate(hosts):
        trigger_fallback = False
        for attempt, delay in enumerate(_RETRY_DELAYS, start=1):
            if path.startswith("http://") or path.startswith("https://"):
                target_url = path
                parsed = urlparse(target_url)
                log_host = parsed.netloc or host
                log_path = parsed.path or "/"
            else:
                target_url = f"{host}{path}"
                log_host = host
                log_path = path

            _log_try(
                logger,
                request_id=request_id,
                kind=kind,
                host=log_host,
                path=log_path,
                attempt=attempt,
                extra=try_extra,
            )
            try:
                response: httpx.Response = await request_cb(target_url)
            except httpx.TimeoutException as exc:
                last_error = exc
                last_status = None
                _log_fail(
                    logger,
                    request_id=request_id,
                    kind=kind,
                    host=log_host,
                    path=log_path,
                    attempt=attempt,
                    reason="timeout",
                )
            except httpx.RequestError as exc:
                last_error = exc
                last_status = None
                _log_fail(
                    logger,
                    request_id=request_id,
                    kind=kind,
                    host=log_host,
                    path=log_path,
                    attempt=attempt,
                    reason=str(exc),
                )
            else:
                status = response.status_code
                last_status = status
                if status == 429 or 500 <= status < 600:
                    body = _trim_body(response)
                    _log_fail(
                        logger,
                        request_id=request_id,
                        kind=kind,
                        host=log_host,
                        path=log_path,
                        attempt=attempt,
                        status=status,
                        body=body,
                    )
                    if attempt < len(_RETRY_DELAYS):
                        await asyncio.sleep(delay)
                        continue
                    last_error = CoverSourceUnavailableError(f"status:{status}")
                elif 400 <= status < 500:
                    body = _trim_body(response)
                    _log_fail(
                        logger,
                        request_id=request_id,
                        kind=kind,
                        host=log_host,
                        path=log_path,
                        attempt=attempt,
                        status=status,
                        body=body,
                    )
                    raise CoverSourceClientError(f"status:{status}")
                else:
                    return response, log_host

            if attempt < len(_RETRY_DELAYS):
                await asyncio.sleep(delay)
                continue

            trigger_fallback = (
                index == 0
                and len(hosts) > 1
                and not is_fallback
                and UPLOAD_FALLBACK_ENABLED
                and (last_status is None or (last_status >= 500))
            )
            if trigger_fallback:
                break
        else:
            trigger_fallback = False
        if trigger_fallback:
            continue
        break

    if isinstance(last_error, CoverSourceUnavailableError):
        raise last_error
    raise CoverSourceUnavailableError(str(last_error or "upload")) from last_error


async def upload_stream(
    file_stream: Union[bytes, bytearray, memoryview, IO[bytes]],
    filename: str,
    mime_type: Optional[str],
    *,
    request_id: str,
    logger: Optional[logging.Logger] = None,
) -> str:
    """Upload audio data to the Suno cover service."""

    if isinstance(file_stream, (bytes, bytearray, memoryview)):
        payload = bytes(file_stream)
    else:
        payload = file_stream.read()
    if not payload:
        raise CoverSourceValidationError("empty-data")

    headers = _auth_header()
    timeout = _timeout()
    safe_filename = filename or "audio.bin"
    content_type = mime_type or "application/octet-stream"

    async def _request(url: str) -> httpx.Response:
        async with httpx.AsyncClient(timeout=timeout) as client:
            return await client.post(
                url,
                headers=headers,
                files={"file": (safe_filename, payload, content_type)},
            )

    response, host = await _perform_upload(
        request_id=request_id,
        kind="stream",
        path=UPLOAD_STREAM_PATH,
        request_cb=_request,
        logger=logger,
        try_extra={"mime": content_type, "size": len(payload)},
    )

    try:
        payload_json: Any = response.json()
    except ValueError as exc:  # pragma: no cover - defensive
        raise CoverSourceUnavailableError("invalid-json") from exc

    if not isinstance(payload_json, Mapping):
        if isinstance(payload_json, MutableMapping):
            payload_json = dict(payload_json)
        else:
            payload_json = {"data": payload_json}

    kie_file_id = _extract_kie_file_id(payload_json)
    if not kie_file_id:
        raise CoverSourceUnavailableError("missing-kie-file-id")

    _log_ok(logger, request_id=request_id, kind="stream", host=host, kie_file_id=kie_file_id)
    return kie_file_id


async def upload_url(
    source_url: str,
    *,
    request_id: str,
    logger: Optional[logging.Logger] = None,
) -> str:
    """Upload a remote audio URL to the Suno cover service."""

    headers = _auth_header()
    headers["Content-Type"] = "application/json"
    timeout = _timeout()

    async def _request(url: str) -> httpx.Response:
        async with httpx.AsyncClient(timeout=timeout) as client:
            return await client.post(url, headers=headers, json={"url": source_url})

    response, host = await _perform_upload(
        request_id=request_id,
        kind="url",
        path=UPLOAD_URL_PATH,
        request_cb=_request,
        logger=logger,
        try_extra={"source": source_url},
    )

    try:
        payload_json: Any = response.json()
    except ValueError as exc:  # pragma: no cover - defensive
        raise CoverSourceUnavailableError("invalid-json") from exc

    if not isinstance(payload_json, Mapping):
        payload_json = {"data": payload_json}

    kie_file_id = _extract_kie_file_id(payload_json)
    if not kie_file_id:
        raise CoverSourceUnavailableError("missing-kie-file-id")

    _log_ok(logger, request_id=request_id, kind="url", host=host, kie_file_id=kie_file_id)
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
    *,
    user_id: Optional[int] = None,
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
        suffix = "mp3" if mime.endswith("mpeg") else "wav"
        user_part = str(user_id) if user_id is not None else "user"
        name = f"cover-{user_part}.{suffix}"

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
