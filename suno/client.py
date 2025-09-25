"""Lightweight HTTP client for the Suno API."""
from __future__ import annotations

import json
import logging
import os
import random
import time
from typing import Any, Mapping, MutableMapping, Optional
from urllib.parse import urljoin

import requests
from requests import Response, Session

from settings import (
    SUNO_API_BASE,
    SUNO_API_TOKEN,
    SUNO_CALLBACK_SECRET,
    SUNO_CALLBACK_URL,
    SUNO_GEN_PATH,
    SUNO_HTTP_RETRIES,
    SUNO_RETRY_BACKOFF_BASE,
    SUNO_TASK_STATUS_PATH,
    SUNO_TIMEOUT_SEC,
)

log = logging.getLogger("suno.client")


class SunoAPIError(RuntimeError):
    """Raised when the Suno API returns an error response."""

    def __init__(self, message: str, *, status: Optional[int] = None, payload: Any = None) -> None:
        super().__init__(message)
        self.status = status
        self.payload = payload


class SunoClient:
    """HTTP wrapper with retry/backoff and default headers."""

    _RETRY_STATUSES = {408, 429}

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        token: Optional[str] = None,
        timeout: Optional[float] = None,
        retries: Optional[int] = None,
        backoff_base: Optional[float] = None,
        session: Optional[Session] = None,
    ) -> None:
        self.base_url = (base_url or SUNO_API_BASE or "").rstrip("/") + "/"
        if not self.base_url.startswith("http"):
            raise RuntimeError("SUNO_API_BASE must include protocol")
        self.token = token or SUNO_API_TOKEN
        if not self.token:
            raise RuntimeError("SUNO_API_TOKEN is not configured")
        self.timeout = timeout or max(float(SUNO_TIMEOUT_SEC or 60), 1.0)
        self.retries = max(1, int(retries or SUNO_HTTP_RETRIES or 1))
        self.backoff_base = max(0.1, float(backoff_base or SUNO_RETRY_BACKOFF_BASE or 0.6))
        self.session = session or requests.Session()

    # ------------------------------------------------------------------ helpers
    def _headers(self) -> MutableMapping[str, str]:
        headers: MutableMapping[str, str] = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
            "User-Agent": "best-veo3-bot/1.0",
        }
        callback_url = SUNO_CALLBACK_URL or os.getenv("SUNO_CALLBACK_URL")
        callback_token = SUNO_CALLBACK_SECRET or os.getenv("SUNO_CALLBACK_SECRET")
        if callback_url:
            headers["X-Callback-Url"] = callback_url
        if callback_token:
            headers["X-Callback-Token"] = callback_token
        return headers

    def _full_url(self, path: str) -> str:
        return urljoin(self.base_url, path.lstrip("/"))

    def _should_retry(self, status: Optional[int]) -> bool:
        if status is None:
            return True
        if status in self._RETRY_STATUSES:
            return True
        return 500 <= status < 600

    def _sleep(self, attempt: int) -> None:
        delay = self.backoff_base * (2 ** attempt)
        delay += random.uniform(0, self.backoff_base)
        time.sleep(delay)

    def _parse_json(self, response: Response) -> Mapping[str, Any]:
        if not response.content:
            return {}
        try:
            payload = response.json()
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise SunoAPIError("Invalid JSON from Suno", status=response.status_code) from exc
        if isinstance(payload, Mapping):
            return payload
        return {"data": payload}

    def _log_attempt(self, method: str, url: str, status: Any, tries: int, started_at: float) -> None:
        elapsed_ms = round((time.perf_counter() - started_at) * 1000)
        log.info(
            "SUNO HTTP | method=%s url=%s code=%s ms=%s tries=%s",
            method.upper(),
            url,
            status,
            elapsed_ms,
            tries,
        )

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_payload: Optional[Mapping[str, Any]] = None,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        url = self._full_url(path)
        attempt = 0
        last_error: Optional[Exception] = None
        status: Optional[int] = None
        started_at = time.perf_counter()
        while attempt < self.retries:
            attempt += 1
            try:
                response = self.session.request(
                    method.upper(),
                    url,
                    json=json_payload,
                    params=params,
                    headers=self._headers(),
                    timeout=self.timeout,
                )
                status = response.status_code
                if self._should_retry(status) and attempt < self.retries:
                    self._sleep(attempt - 1)
                    continue
                payload = self._parse_json(response)
                if status >= 400:
                    self._log_attempt(method, url, status, attempt, started_at)
                    raise SunoAPIError(
                        payload.get("message") or payload.get("msg") or f"HTTP {status}",
                        status=status,
                        payload=payload,
                    )
                self._log_attempt(method, url, status, attempt, started_at)
                return payload
            except requests.RequestException as exc:
                last_error = exc
                status = None
                if attempt >= self.retries:
                    break
                self._sleep(attempt - 1)
        self._log_attempt(method, url, status or "error", attempt, started_at)
        if isinstance(last_error, Exception):
            raise SunoAPIError("Network error talking to Suno") from last_error
        raise SunoAPIError(f"Suno API error: HTTP {status}", status=status)

    # ---------------------------------------------------------------- public API
    def create_music(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        body = {key: value for key, value in payload.items() if value is not None}
        callback_url = SUNO_CALLBACK_URL or os.getenv("SUNO_CALLBACK_URL")
        callback_token = SUNO_CALLBACK_SECRET or os.getenv("SUNO_CALLBACK_SECRET")
        if callback_url:
            body.setdefault("callback_url", callback_url)
        if callback_token:
            body.setdefault("callback_token", callback_token)
        return self._request("POST", SUNO_GEN_PATH, json_payload=body)

    def get_task_status(self, task_id: str) -> Mapping[str, Any]:
        return self._request("GET", SUNO_TASK_STATUS_PATH, params={"task_id": task_id})


__all__ = ["SunoClient", "SunoAPIError"]
