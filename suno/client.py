"""HTTP client wrapper for the Suno API."""
from __future__ import annotations

import json
import logging
import random
import time
from typing import Any, Mapping, MutableMapping, Optional
from urllib.parse import urljoin

import requests
from requests import RequestException, Response, Session
from requests.adapters import HTTPAdapter
from urllib3.util import Timeout

from settings import (
    HTTP_POOL_CONNECTIONS,
    HTTP_POOL_PER_HOST,
    HTTP_RETRY_ATTEMPTS,
    HTTP_TIMEOUT_CONNECT,
    HTTP_TIMEOUT_READ,
    HTTP_TIMEOUT_TOTAL,
    SUNO_API_BASE,
    SUNO_API_TOKEN,
    SUNO_CALLBACK_SECRET,
    SUNO_CALLBACK_URL,
    SUNO_GEN_PATH,
    SUNO_MAX_RETRIES,
    SUNO_TASK_STATUS_PATH,
)

log = logging.getLogger("suno.client")

_RETRYABLE_CODES = {408, 429}
_BACKOFF_SCHEDULE = (1, 3, 7)


class SunoAPIError(RuntimeError):
    """Raised when the Suno API responds with an error."""

    def __init__(self, message: str, *, status: Optional[int] = None, payload: Any = None) -> None:
        super().__init__(message)
        self.status = status
        self.payload = payload


class SunoClient:
    """Thin wrapper around :mod:`requests` with retries/backoff."""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        token: Optional[str] = None,
        session: Optional[Session] = None,
        max_retries: Optional[int] = None,
        timeout: Optional[tuple[int, int] | Timeout] = None,
    ) -> None:
        raw_base = (base_url or SUNO_API_BASE or "").strip()
        if not raw_base:
            raise RuntimeError("SUNO_API_BASE is not configured")
        self.base_url = raw_base.rstrip("/") + "/"
        self.token = (token or SUNO_API_TOKEN or "").strip()
        if not self.token:
            raise RuntimeError("SUNO_API_TOKEN is not configured")
        self.session = session or requests.Session()
        adapter = HTTPAdapter(pool_connections=HTTP_POOL_CONNECTIONS, pool_maxsize=HTTP_POOL_PER_HOST)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        retries = max_retries if max_retries is not None else SUNO_MAX_RETRIES or HTTP_RETRY_ATTEMPTS
        self.max_attempts = max(1, int(retries))
        if timeout is None:
            self.timeout = Timeout(
                total=HTTP_TIMEOUT_TOTAL,
                connect=HTTP_TIMEOUT_CONNECT,
                read=HTTP_TIMEOUT_READ,
            )
        elif isinstance(timeout, tuple):
            connect, read = timeout
            self.timeout = Timeout(
                total=max(HTTP_TIMEOUT_TOTAL, float(connect), float(read)),
                connect=float(connect),
                read=float(read),
            )
        else:
            self.timeout = timeout

    # ------------------------------------------------------------------ helpers
    def _headers(self) -> MutableMapping[str, str]:
        headers: MutableMapping[str, str] = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
        }
        if SUNO_CALLBACK_URL:
            headers["X-Callback-Url"] = SUNO_CALLBACK_URL
        if SUNO_CALLBACK_SECRET:
            headers["X-Callback-Token"] = SUNO_CALLBACK_SECRET
        return headers

    def _url(self, path: str) -> str:
        return urljoin(self.base_url, path.lstrip("/"))

    @staticmethod
    def _parse_json(response: Response) -> Mapping[str, Any]:
        if not response.content:
            return {}
        try:
            payload = response.json()
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise SunoAPIError("Invalid JSON from Suno", status=response.status_code) from exc
        if isinstance(payload, Mapping):
            return payload
        return {"data": payload}

    def _maybe_backoff(self, code: Optional[int], attempt: int) -> bool:
        if attempt >= self.max_attempts:
            return False
        retryable = code is None or code in _RETRYABLE_CODES or (code is not None and code >= 500)
        if retryable:
            base = _BACKOFF_SCHEDULE[min(attempt - 1, len(_BACKOFF_SCHEDULE) - 1)]
            jitter = random.uniform(0.0, 1.0)
            delay = base + jitter
            log.warning(
                "suno.http retry",
                extra={"meta": {"code": code or "error", "attempt": attempt, "delay": round(delay, 3)}},
            )
            time.sleep(delay)
            return True
        return False

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_payload: Optional[Mapping[str, Any]] = None,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        url = self._url(path)
        attempt = 0
        last_error: Optional[BaseException] = None
        while attempt < self.max_attempts:
            attempt += 1
            try:
                response = self.session.request(
                    method.upper(),
                    url,
                    headers=self._headers(),
                    json=json_payload,
                    params=params,
                    timeout=self.timeout,
                )
            except RequestException as exc:
                last_error = exc
                if not self._maybe_backoff(None, attempt):
                    log.error("suno.http failed attempt=%s error=%s", attempt, exc)
                    raise SunoAPIError("Network error talking to Suno") from exc
                continue

            status = response.status_code
            if self._maybe_backoff(status, attempt):
                continue

            payload = self._parse_json(response)
            if status >= 400:
                log.error("suno.http error code=%s attempt=%s", status, attempt)
                raise SunoAPIError(
                    payload.get("message") or payload.get("msg") or f"HTTP {status}",
                    status=status,
                    payload=payload,
                )

            log.info("suno.http success code=%s attempt=%s", status, attempt)
            return payload

        raise SunoAPIError("Suno request exhausted retries", payload=getattr(last_error, "response", None))

    # ------------------------------------------------------------------ public API
    def create_music(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        body = {key: value for key, value in payload.items() if value is not None}
        return self._request("POST", SUNO_GEN_PATH, json_payload=body)

    def get_task_status(self, task_id: str) -> Mapping[str, Any]:
        return self._request("GET", SUNO_TASK_STATUS_PATH, params={"task_id": task_id})


__all__ = ["SunoClient", "SunoAPIError"]
