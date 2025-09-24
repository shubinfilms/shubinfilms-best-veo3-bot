"""HTTP client for interacting with the Suno API."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional
from urllib.parse import urljoin

import requests
from requests import Response, Session

from ._retry import RetryError, Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger("suno.client")


class SunoAPIError(RuntimeError):
    """Raised when the Suno API returns an error."""


class SunoRetryableError(Exception):
    """Internal helper to trigger retry for retryable responses."""


_DEFAULT_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT_S", "20"))


def _should_retry(response: Response) -> bool:
    if response.status_code >= 500 or response.status_code == 429:
        return True
    try:
        payload = response.json()
    except ValueError:
        return False
    code = payload.get("code")
    return isinstance(code, int) and code >= 500


class SunoClient:
    """Simple HTTP client for Suno API."""

    def __init__(
        self,
        base_url: str,
        token: str,
        timeout: int = _DEFAULT_TIMEOUT,
        session: Optional[Session] = None,
    ) -> None:
        if not base_url:
            raise ValueError("base_url must be provided")
        if not token:
            raise ValueError("token must be provided")
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.session = session or requests.Session()
        self._retryer = Retrying(
            stop=stop_after_attempt(4),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type((requests.RequestException, SunoRetryableError)),
            reraise=True,
        )

    def _request(self, method: str, path: str, *, json: Optional[dict] = None, params: Optional[dict] = None) -> dict:
        if not path.startswith("/"):
            raise ValueError("path must start with '/'")
        url = urljoin(self.base_url + "/", path.lstrip("/"))
        headers = {"Authorization": f"Bearer {self.token}"}

        def _do_request() -> dict:
            try:
                response = self.session.request(
                    method,
                    url,
                    json=json,
                    params=params,
                    headers=headers,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:  # pragma: no cover - handled by retryer
                logger.warning("Request error on %s %s: %s", method, url, exc)
                raise exc
            if response.status_code >= 400:
                if _should_retry(response):
                    raise SunoRetryableError(f"retryable status {response.status_code}")
                raise SunoAPIError(f"HTTP {response.status_code}: {response.text}")
            try:
                payload = response.json()
            except ValueError as exc:
                raise SunoAPIError("invalid JSON response") from exc
            code = payload.get("code")
            if isinstance(code, int) and code >= 400:
                message = payload.get("msg") or "unexpected error"
                raise SunoAPIError(f"Suno API error {code}: {message}")
            return payload

        try:
            return self._retryer(_do_request)
        except RetryError as exc:  # pragma: no cover - defensive
            raise SunoAPIError("maximum retries exceeded") from exc

    def post(self, path: str, payload: Dict[str, Any]) -> dict:
        if not isinstance(payload, dict):
            raise ValueError("payload must be a dict")
        return self._request("POST", path, json=payload)

    def get(self, path: str, params: Dict[str, Any]) -> dict:
        if not isinstance(params, dict):
            raise ValueError("params must be a dict")
        return self._request("GET", path, params=params)

    # Specialized wrappers
    def _ensure_fields(self, payload: Dict[str, Any], required: tuple[str, ...]) -> None:
        missing = [field for field in required if not payload.get(field)]
        if missing:
            raise ValueError(f"missing required fields: {', '.join(missing)}")

    def add_vocals(self, payload: Dict[str, Any]) -> dict:
        self._ensure_fields(payload, ("uploadUrl", "callBackUrl"))
        return self.post("/api/v1/generate/add-vocals", payload)

    def add_instrumental(self, payload: Dict[str, Any]) -> dict:
        self._ensure_fields(payload, ("uploadUrl", "callBackUrl"))
        return self.post("/api/v1/generate/add-instrumental", payload)

    def upload_extend(self, payload: Dict[str, Any]) -> dict:
        self._ensure_fields(payload, ("uploadUrl", "callBackUrl"))
        return self.post("/api/v1/generate/upload-extend", payload)

    def wav_generate(self, payload: Dict[str, Any]) -> dict:
        self._ensure_fields(payload, ("uploadUrl", "callBackUrl"))
        return self.post("/api/v1/wav/generate", payload)

    def vocal_removal_generate(self, payload: Dict[str, Any]) -> dict:
        self._ensure_fields(payload, ("uploadUrl", "callBackUrl"))
        return self.post("/api/v1/vocal-removal/generate", payload)

    def cover_generate(self, payload: Dict[str, Any]) -> dict:
        self._ensure_fields(payload, ("callBackUrl",))
        return self.post("/api/v1/suno/cover/generate", payload)

    def mp4_generate(self, payload: Dict[str, Any]) -> dict:
        self._ensure_fields(payload, ("callBackUrl",))
        return self.post("/api/v1/mp4/generate", payload)

    def style_generate(self, content: str) -> dict:
        if not content or not content.strip():
            raise ValueError("content must not be empty")
        return self.post("/api/v1/style/generate", {"content": content})

    # Status endpoints
    def music_record_info(self, task_id: str) -> dict:
        if not task_id:
            raise ValueError("task_id is required")
        return self.get("/api/v1/generate/record-info", {"taskId": task_id})

    def get_timestamped_lyrics(self, task_id: str, audio_id: str) -> dict:
        if not task_id or not audio_id:
            raise ValueError("task_id and audio_id are required")
        return self.post(
            "/api/v1/generate/get-timestamped-lyrics",
            {"taskId": task_id, "audioId": audio_id},
        )

    def wav_record_info(self, task_id: str) -> dict:
        if not task_id:
            raise ValueError("task_id is required")
        return self.get("/api/v1/wav/record-info", {"taskId": task_id})

    def vocal_removal_record_info(self, task_id: str) -> dict:
        if not task_id:
            raise ValueError("task_id is required")
        return self.get("/api/v1/vocal-removal/record-info", {"taskId": task_id})

    def cover_record_info(self, task_id: str) -> dict:
        if not task_id:
            raise ValueError("task_id is required")
        return self.get("/api/v1/suno/cover/record-info", {"taskId": task_id})

    def mp4_record_info(self, task_id: str) -> dict:
        if not task_id:
            raise ValueError("task_id is required")
        return self.get("/api/v1/mp4/record-info", {"taskId": task_id})
