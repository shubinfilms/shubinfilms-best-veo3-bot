"""HTTP client for interacting with the Suno API."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional
from urllib.parse import urlparse, urlunparse

import requests
from requests import Response, Session

from ._retry import RetryError, Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

log = logging.getLogger("suno")


def _join_url(*parts: str) -> str:
    """Join URL parts safely, normalizing duplicate segments."""

    cleaned: list[str] = []
    for part in parts:
        if not part:
            continue
        cleaned.append(part.strip().replace("\\", "/"))

    if not cleaned:
        return ""

    joined = "/".join(segment.strip("/") for segment in cleaned)

    # Ensure we have a scheme for urlparse to work reliably
    if "://" not in joined:
        joined = f"https://{joined}"

    parsed = urlparse(joined)
    scheme = parsed.scheme or "https"
    if parsed.netloc:
        netloc = parsed.netloc
        path = parsed.path
    else:
        # Recover netloc from the first cleaned part when scheme was missing
        first = cleaned[0]
        if "://" in first:
            first_parsed = urlparse(first)
            netloc = first_parsed.netloc
            path_prefix = first_parsed.path
        else:
            netloc = first.split("/", 1)[0]
            path_prefix = "/" + first[len(netloc):]
        path = (path_prefix.rstrip("/") + "/" + parsed.path.lstrip("/")).replace("//", "/")

    # Normalize path and drop duplicate "suno-api" prefixes
    path = path.replace("//", "/")
    path = path.replace("/suno-api/api/", "/api/")
    path = path.replace("/suno-api/v1/", "/api/v1/")

    return urlunparse((scheme, netloc, path, "", "", ""))


BASE = os.getenv("SUNO_API_BASE", "https://api.kie.ai")

ENDPOINTS: dict[str, str] = {
    "generate_music": "/api/v1/generate/music",
    "extend_music": "/api/v1/generate/extend",
    "upload_extend": "/api/v1/generate/upload-extend",
    "add_instrumental": "/api/v1/generate/add-instrumental",
    "add_vocals": "/api/v1/generate/add-vocals",
    "record_info": "/api/v1/generate/record-info",
    "timestamped_lyrics": "/api/v1/generate/get-timestamped-lyrics",
    "style_generate": "/api/v1/style/generate",
    "cover_generate": "/api/v1/suno/cover/generate",
    "cover_record_info": "/api/v1/suno/cover/record-info",
    "wav_generate": "/api/v1/wav/generate",
    "wav_record_info": "/api/v1/wav/record-info",
    "vocal_sep_generate": "/api/v1/vocal-removal/generate",
    "vocal_sep_record_info": "/api/v1/vocal-removal/record-info",
    "mp4_generate": "/api/v1/mp4/generate",
    "mp4_record_info": "/api/v1/mp4/record-info",
}

_RETRYABLE_CODES = {408, 429, 455, 500}


class SunoAPIError(RuntimeError):
    """Raised when the Suno API returns an error."""


class SunoRetryableError(Exception):
    """Internal helper to trigger retry for retryable responses."""


_DEFAULT_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT_S", "20"))


def _should_retry(response: Response) -> bool:
    if response.status_code in _RETRYABLE_CODES:
        return True
    try:
        payload = response.json()
    except ValueError:
        return False
    code = payload.get("code")
    return isinstance(code, int) and code in _RETRYABLE_CODES


class SunoClient:
    """Simple HTTP client for Suno API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        token: Optional[str] = None,
        timeout: int = _DEFAULT_TIMEOUT,
        session: Optional[Session] = None,
    ) -> None:
        env_base = (os.environ.get("SUNO_API_BASE") or BASE or "").strip()
        env_token = (os.environ.get("SUNO_API_TOKEN") or "").strip()
        self._prefix = (os.environ.get("SUNO_API_PREFIX") or "").strip()

        resolved_base = (base_url or env_base or "").strip()
        if base_url is None:
            if not resolved_base or not resolved_base.startswith("http"):
                raise RuntimeError(f"Invalid SUNO_API_BASE: {resolved_base!r}")
        else:
            if not resolved_base:
                raise ValueError("base_url must be provided")
            if not resolved_base.startswith("http"):
                raise ValueError("base_url must start with 'http'")

        resolved_token = token if token is not None else env_token
        if token is None:
            if not resolved_token:
                raise RuntimeError("SUNO_API_TOKEN is empty")
        elif not resolved_token:
            raise ValueError("token must be provided")

        self.base_url = resolved_base.rstrip("/")
        self.token = resolved_token
        self.timeout = timeout
        self.session = session or requests.Session()
        self._retryer = Retrying(
            stop=stop_after_attempt(4),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type((requests.RequestException, SunoRetryableError)),
            reraise=True,
        )

    def _resolve_path(self, endpoint: str) -> str:
        if not endpoint:
            raise ValueError("endpoint must be provided")
        path = ENDPOINTS.get(endpoint, endpoint)
        if not path.startswith("/"):
            raise ValueError("path must start with '/'")
        return path

    def _url(self, path: str) -> str:
        url = _join_url(self.base_url, self._prefix, path)
        log.debug("Suno request → %s", url)
        return url

    def _url_for(self, endpoint: str) -> str:
        path = self._resolve_path(endpoint)
        return self._url(path)

    def _request(
        self,
        method: str,
        endpoint: str,
        *,
        json: Optional[dict] = None,
        params: Optional[dict] = None,
    ) -> dict:
        path = self._resolve_path(endpoint)
        url = self._url(path)
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

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
                log.warning("Request error on %s %s: %s", method, url, exc)
                raise exc
            if response.status_code != 200:
                log.warning(
                    "Suno API HTTP error %s on %s %s: %s",
                    response.status_code,
                    method,
                    path,
                    response.text,
                )
                if _should_retry(response):
                    raise SunoRetryableError(f"retryable status {response.status_code}")
                raise SunoAPIError(f"HTTP {response.status_code}: {response.text}")
            try:
                payload = response.json()
            except ValueError as exc:
                raise SunoAPIError("invalid JSON response") from exc
            code = payload.get("code")
            if isinstance(code, int) and code != 200:
                message = payload.get("msg") or "unexpected error"
                log.warning(
                    "Suno API logical error on %s %s: code=%s msg=%s payload=%s",
                    method,
                    path,
                    code,
                    message,
                    payload,
                )
                if code in _RETRYABLE_CODES:
                    raise SunoRetryableError(f"retryable code {code}")
                raise SunoAPIError(f"Suno API error {code}: {message}")
            return payload

        try:
            return self._retryer(_do_request)
        except RetryError as exc:  # pragma: no cover - defensive
            raise SunoAPIError("maximum retries exceeded") from exc

    def post(self, endpoint: str, payload: Dict[str, Any]) -> dict:
        if not isinstance(payload, dict):
            raise ValueError("payload must be a dict")
        return self._request("POST", endpoint, json=payload)

    def get(self, endpoint: str, params: Dict[str, Any]) -> dict:
        if not isinstance(params, dict):
            raise ValueError("params must be a dict")
        return self._request("GET", endpoint, params=params)

    # Specialized wrappers
    def _ensure_fields(self, payload: Dict[str, Any], required: tuple[str, ...]) -> None:
        missing = [field for field in required if not payload.get(field)]
        if missing:
            raise ValueError(f"missing required fields: {', '.join(missing)}")

    def generate_music(
        self,
        *,
        prompt: str,
        model: str,
        title: str,
        style: str,
        callBackUrl: str,
        instrumental: bool = True,
        negativeTags: Optional[str] = None,
        vocalGender: Optional[str] = None,
        styleWeight: Optional[float] = None,
        weirdnessConstraint: Optional[float] = None,
        audioWeight: Optional[float] = None,
    ) -> dict:
        """Submit a generate music task."""

        payload: Dict[str, Any] = {
            "prompt": prompt,
            "model": model,
            "title": title,
            "style": style,
            "callBackUrl": callBackUrl,
            "instrumental": instrumental,
        }
        self._ensure_fields(
            payload,
            ("prompt", "model", "title", "style", "callBackUrl"),
        )
        optional_fields = {
            "negativeTags": negativeTags,
            "vocalGender": vocalGender,
            "styleWeight": styleWeight,
            "weirdnessConstraint": weirdnessConstraint,
            "audioWeight": audioWeight,
        }
        for key, value in optional_fields.items():
            if value is not None:
                payload[key] = value
        return self.post("generate_music", payload)

    def add_vocals(self, payload: Dict[str, Any]) -> dict:
        self._ensure_fields(payload, ("uploadUrl", "callBackUrl"))
        return self.post("add_vocals", payload)

    def add_instrumental(self, payload: Dict[str, Any]) -> dict:
        self._ensure_fields(payload, ("uploadUrl", "callBackUrl"))
        return self.post("add_instrumental", payload)

    def upload_extend(self, payload: Dict[str, Any]) -> dict:
        self._ensure_fields(payload, ("uploadUrl", "callBackUrl"))
        return self.post("upload_extend", payload)

    def wav_generate(self, payload: Dict[str, Any]) -> dict:
        self._ensure_fields(payload, ("uploadUrl", "callBackUrl"))
        return self.post("wav_generate", payload)

    def vocal_removal_generate(self, payload: Dict[str, Any]) -> dict:
        self._ensure_fields(payload, ("uploadUrl", "callBackUrl"))
        return self.post("vocal_sep_generate", payload)

    def cover_generate(self, payload: Dict[str, Any]) -> dict:
        self._ensure_fields(payload, ("callBackUrl",))
        return self.post("cover_generate", payload)

    def mp4_generate(self, payload: Dict[str, Any]) -> dict:
        self._ensure_fields(payload, ("callBackUrl",))
        return self.post("mp4_generate", payload)

    def style_generate(self, content: str) -> dict:
        if not content or not content.strip():
            raise ValueError("content must not be empty")
        return self.post("style_generate", {"content": content})

    # Status endpoints
    def record_info(self, task_id: str) -> dict:
        if not task_id:
            raise ValueError("task_id is required")
        return self.get("record_info", {"taskId": task_id})

    def get_timestamped_lyrics(self, task_id: str, audio_id: str) -> dict:
        if not task_id or not audio_id:
            raise ValueError("task_id and audio_id are required")
        return self.post(
            "timestamped_lyrics",
            {"taskId": task_id, "audioId": audio_id},
        )

    def wav_record_info(self, task_id: str) -> dict:
        if not task_id:
            raise ValueError("task_id is required")
        return self.get("wav_record_info", {"taskId": task_id})

    def vocal_removal_record_info(self, task_id: str) -> dict:
        if not task_id:
            raise ValueError("task_id is required")
        return self.get("vocal_sep_record_info", {"taskId": task_id})

    def cover_record_info(self, task_id: str) -> dict:
        if not task_id:
            raise ValueError("task_id is required")
        return self.get("cover_record_info", {"taskId": task_id})

    def mp4_record_info(self, task_id: str) -> dict:
        if not task_id:
            raise ValueError("task_id is required")
        return self.get("mp4_record_info", {"taskId": task_id})
