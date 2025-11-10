"""Utilities to upload Telegram files to Banana storage using presigned URLs."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import httpx

from settings import KIE_API_KEY, UPLOAD_BASE_URL, UPLOAD_URL_PATH

__all__ = [
    "BananaUploadError",
    "BananaUploadExpiredError",
    "BananaUploadFailedError",
    "BananaUploader",
    "UploadResult",
]


_LOGGER = logging.getLogger("banana.uploader")
_RETRY_DELAYS = (0.8, 2.0)
_TIMEOUT = httpx.Timeout(15.0, connect=15.0)


class BananaUploadError(RuntimeError):
    """Base class for upload errors."""

    def __init__(self, message: str, *, status: Optional[int] = None) -> None:
        super().__init__(message)
        self.status = status


class BananaUploadExpiredError(BananaUploadError):
    """Raised when a presigned URL is no longer valid (HTTP 403/404)."""


class BananaUploadFailedError(BananaUploadError):
    """Raised when the upload failed after retries."""


@dataclass(slots=True)
class UploadTicket:
    """Ticket describing a single upload attempt."""

    upload_url: str
    upload_id: Optional[str]
    confirm_url: Optional[str]
    public_url: Optional[str]


@dataclass(slots=True)
class UploadResult:
    """Result of a successful upload."""

    public_url: str
    upload_id: Optional[str]


def _auth_header() -> Optional[str]:
    token = (KIE_API_KEY or "").strip()
    if not token:
        return None
    if not token.lower().startswith("bearer "):
        token = f"Bearer {token}"
    return token


def _build_endpoint() -> Optional[str]:
    base = (UPLOAD_BASE_URL or "").strip()
    path = (UPLOAD_URL_PATH or "").strip() or "/api/v1/upload/url"
    if path.startswith("http://") or path.startswith("https://"):
        return path
    if not base:
        return None
    if not path.startswith("/"):
        path = f"/{path}"
    return f"{base}{path}"


def _extract(payload: Mapping[str, Any], *candidates: str) -> Optional[str]:
    for key in candidates:
        value = payload.get(key)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                return stripped
    return None


def _extract_nested(payload: Mapping[str, Any], *candidates: str) -> Optional[str]:
    value = _extract(payload, *candidates)
    if value:
        return value
    data = payload.get("data")
    if isinstance(data, Mapping):
        return _extract_nested(data, *candidates)
    return None


class BananaUploader:
    """Upload helper interacting with Banana presigned upload API."""

    def __init__(self) -> None:
        endpoint = _build_endpoint()
        if not endpoint:
            raise RuntimeError("Banana upload endpoint is not configured")
        self._endpoint = endpoint
        self._auth = _auth_header()

    async def _request_ticket(
        self,
        *,
        filename: str,
        content_type: str,
        size: int,
        user_id: Optional[int],
    ) -> UploadTicket:
        payload = {
            "filename": filename,
            "contentType": content_type,
            "size": int(size),
        }
        headers = {"Content-Type": "application/json"}
        if self._auth:
            headers["Authorization"] = self._auth

        _LOGGER.info(
            "ns=banana action=start_upload file=%s size=%s", filename, size, extra={"user_id": user_id}
        )

        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            try:
                response = await client.post(self._endpoint, headers=headers, json=payload)
            except httpx.RequestError as exc:  # pragma: no cover - network/timeout
                raise BananaUploadFailedError(str(exc)) from exc

        if response.status_code >= 400:
            raise BananaUploadFailedError(
                f"HTTP {response.status_code} while requesting upload ticket",
                status=response.status_code,
            )

        try:
            payload_json: Mapping[str, Any] = response.json()
        except json.JSONDecodeError as exc:  # pragma: no cover - unexpected
            raise BananaUploadFailedError("invalid-json") from exc

        upload_url = _extract_nested(payload_json, "uploadUrl", "upload_url", "url", "presignedUrl")
        if not upload_url:
            raise BananaUploadFailedError("missing-upload-url")

        upload_id = _extract_nested(payload_json, "uploadId", "upload_id", "id")
        confirm_url = _extract_nested(payload_json, "confirmUrl", "confirm_url", "completeUrl")
        public_url = _extract_nested(payload_json, "publicUrl", "public_url", "fileUrl", "resultUrl")

        return UploadTicket(
            upload_url=upload_url,
            upload_id=upload_id,
            confirm_url=confirm_url,
            public_url=public_url,
        )

    async def _confirm_upload(
        self,
        *,
        ticket: UploadTicket,
        filename: str,
        content_type: str,
        size: int,
        user_id: Optional[int],
    ) -> Optional[str]:
        if not ticket.confirm_url:
            return ticket.public_url

        payload = {
            "uploadId": ticket.upload_id,
            "filename": filename,
            "contentType": content_type,
            "size": int(size),
        }
        headers = {"Content-Type": "application/json"}
        if self._auth:
            headers["Authorization"] = self._auth

        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            try:
                response = await client.post(ticket.confirm_url, headers=headers, json=payload)
            except httpx.RequestError as exc:  # pragma: no cover - network
                raise BananaUploadFailedError(str(exc)) from exc

        if response.status_code >= 400:
            raise BananaUploadFailedError(
                f"HTTP {response.status_code} while confirming upload",
                status=response.status_code,
            )

        try:
            payload_json: Mapping[str, Any] = response.json()
        except json.JSONDecodeError as exc:  # pragma: no cover - unexpected
            raise BananaUploadFailedError("invalid-json") from exc

        public_url = _extract_nested(payload_json, "publicUrl", "public_url", "fileUrl", "url")
        if public_url:
            return public_url
        return ticket.public_url

    async def upload(
        self,
        data: bytes,
        *,
        filename: str,
        content_type: str,
        user_id: Optional[int],
    ) -> UploadResult:
        size = len(data)
        ticket = await self._request_ticket(
            filename=filename,
            content_type=content_type,
            size=size,
            user_id=user_id,
        )

        attempt = 0
        refreshes_left = 1
        while True:
            attempt += 1
            try:
                await self._put_bytes(ticket.upload_url, data, content_type)
            except BananaUploadExpiredError as exc:
                if refreshes_left <= 0:
                    raise
                refreshes_left -= 1
                _LOGGER.info(
                    "ns=banana action=upload_refresh file=%s attempt=%s reason=%s",
                    filename,
                    attempt,
                    str(exc),
                    extra={"user_id": user_id},
                )
                ticket = await self._request_ticket(
                    filename=filename,
                    content_type=content_type,
                    size=size,
                    user_id=user_id,
                )
                attempt = 0
                continue
            except BananaUploadFailedError as exc:
                if attempt > len(_RETRY_DELAYS) + 1:
                    raise
                delay = _RETRY_DELAYS[min(attempt - 1, len(_RETRY_DELAYS) - 1)]
                _LOGGER.warning(
                    "ns=banana action=upload_retry file=%s attempt=%s delay=%.1f reason=%s",
                    filename,
                    attempt,
                    delay,
                    str(exc),
                    extra={"user_id": user_id},
                )
                await asyncio.sleep(delay)
                continue
            else:
                break

        public_url = await self._confirm_upload(
            ticket=ticket,
            filename=filename,
            content_type=content_type,
            size=size,
            user_id=user_id,
        )

        if not public_url:
            raise BananaUploadFailedError("missing-public-url")

        _LOGGER.info(
            "ns=banana action=upload_done file=%s size=%s",
            filename,
            size,
            extra={"user_id": user_id},
        )
        return UploadResult(public_url=public_url, upload_id=ticket.upload_id)

    @staticmethod
    async def _put_bytes(url: str, data: bytes, content_type: str) -> None:
        headers = {"Content-Type": content_type}
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            try:
                response = await client.put(url, headers=headers, content=data)
            except httpx.TimeoutException as exc:  # pragma: no cover - network
                raise BananaUploadFailedError("timeout") from exc
            except httpx.RequestError as exc:  # pragma: no cover - network
                raise BananaUploadFailedError(str(exc)) from exc

        status = response.status_code
        if status in {403, 404}:
            raise BananaUploadExpiredError(f"HTTP {status}", status=status)
        if status >= 400:
            raise BananaUploadFailedError(f"HTTP {status}", status=status)
