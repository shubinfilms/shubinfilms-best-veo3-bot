"""Asynchronous client for interacting with the KIE API."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Optional

import aiohttp

from settings import (
    HTTP_TIMEOUT_CONNECT,
    HTTP_TIMEOUT_READ,
    HTTP_TIMEOUT_TOTAL,
    KIE_API_KEY,
    KIE_BASE_URL,
)

__all__ = [
    "KieAPIAsync",
    "KieAPIError",
    "KieAPIHTTPError",
    "KieAPITimeoutError",
    "KieAPITransportError",
]


class KieAPIError(RuntimeError):
    """Base exception for KIE API errors."""


class KieAPITimeoutError(KieAPIError):
    """Raised when a request exceeds the configured timeout."""


class KieAPITransportError(KieAPIError):
    """Raised for network/transport layer errors."""


@dataclass(slots=True)
class KieAPIHTTPError(KieAPIError):
    """Raised for non-successful HTTP responses."""

    status: int
    payload: Mapping[str, Any]


def _auth_header() -> Optional[str]:
    token = (KIE_API_KEY or "").strip()
    if not token:
        return None
    if not token.lower().startswith("bearer "):
        token = f"Bearer {token}"
    return token


def _build_timeout() -> aiohttp.ClientTimeout:
    total = float(HTTP_TIMEOUT_TOTAL or 60.0)
    connect = float(HTTP_TIMEOUT_CONNECT or 15.0)
    read = float(HTTP_TIMEOUT_READ or 60.0)
    return aiohttp.ClientTimeout(total=total, connect=connect, sock_read=read)


class KieAPIAsync:
    """Thin asynchronous KIE API client based on :mod:`aiohttp`."""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> None:
        self._base_url = (base_url or KIE_BASE_URL or "https://api.kie.ai").rstrip("/")
        self._session: Optional[aiohttp.ClientSession] = session
        self._own_session = session is None
        self._timeout = _build_timeout()
        self._default_headers = self._init_headers()
        self._lock = asyncio.Lock()

    @staticmethod
    def _init_headers() -> MutableMapping[str, str]:
        headers: MutableMapping[str, str] = {}
        token = _auth_header()
        if token:
            headers["Authorization"] = token
        return headers

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            async with self._lock:
                if self._session is None:
                    self._session = aiohttp.ClientSession(
                        base_url=self._base_url,
                        timeout=self._timeout,
                        raise_for_status=False,
                    )
        return self._session

    async def close(self) -> None:
        if self._own_session and self._session is not None:
            await self._session.close()
            self._session = None

    async def __aenter__(self) -> "KieAPIAsync":
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    def _prepare_url(self, path: str) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            return path
        return f"{self._base_url}/{path.lstrip('/')}"

    @staticmethod
    def _merge_headers(
        base: Mapping[str, str],
        extra: Optional[Mapping[str, str]] = None,
    ) -> MutableMapping[str, str]:
        merged: MutableMapping[str, str] = dict(base)
        if extra:
            merged.update({k: v for k, v in extra.items() if v is not None})
        return merged

    async def request_json(
        self,
        method: str,
        path: str,
        *,
        json_payload: Optional[Mapping[str, Any]] = None,
        params: Optional[Mapping[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> Mapping[str, Any]:
        session = await self._ensure_session()
        url = self._prepare_url(path)
        extra_headers = dict(headers or {})
        if json_payload is not None:
            extra_headers.setdefault("Content-Type", "application/json")
        request_headers = self._merge_headers(self._default_headers, extra_headers)

        try:
            async with session.request(
                method,
                url,
                json=json_payload,
                params=params,
                headers=request_headers,
            ) as response:
                status = response.status
                content_type = response.headers.get("Content-Type", "")
                if "application/json" in content_type:
                    try:
                        payload = await response.json(content_type=None)
                    except json.JSONDecodeError:
                        payload = {"raw": await response.text()}
                else:
                    payload = {"raw": await response.text()}
        except asyncio.TimeoutError as exc:
            raise KieAPITimeoutError("KIE request timed out") from exc
        except aiohttp.ClientError as exc:
            raise KieAPITransportError(str(exc)) from exc

        if status >= 400:
            raise KieAPIHTTPError(status=status, payload=payload)

        if isinstance(payload, Mapping):
            return payload
        return {"value": payload}

    async def poll_job(
        self,
        path: str,
        *,
        task_id: Optional[str] = None,
        params: Optional[Mapping[str, Any]] = None,
        interval: float = 4.0,
        timeout: float = 8 * 60.0,
    ):
        """Yield status payloads for ``task_id`` until cancelled."""

        loop = asyncio.get_running_loop()
        deadline = loop.time() + float(timeout)
        query = dict(params or {})
        if task_id is not None:
            query.setdefault("taskId", task_id)

        while True:
            if loop.time() >= deadline:
                raise KieAPITimeoutError("KIE polling timed out")
            payload = await self.request_json("GET", path, params=query)
            yield payload
            if interval > 0:
                await asyncio.sleep(float(interval))

