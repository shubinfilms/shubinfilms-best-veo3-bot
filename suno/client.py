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
    SUNO_MODEL,
    SUNO_TASK_STATUS_PATH,
)

log = logging.getLogger("suno.client")

_RETRYABLE_CODES = {408, 429}
_BACKOFF_SCHEDULE = (1, 3, 7)
_LEGACY_GEN_PATH = "/api/v1/generate/music"
_LEGACY_TASK_STATUS_PATH = "/api/v1/generate/record-info"
_API_V5 = "v5"
_API_LEGACY = "legacy"


class SunoAPIError(RuntimeError):
    """Raised when the Suno API responds with an error."""

    def __init__(self, message: str, *, status: Optional[int] = None, payload: Any = None) -> None:
        super().__init__(message)
        self.status = status
        self.payload = payload
        self.api_version: Optional[str] = None


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
        self._primary_gen_path = self._normalize_path(SUNO_GEN_PATH or "/suno-api/generate")
        self._primary_status_path = self._normalize_path(SUNO_TASK_STATUS_PATH or "/suno-api/record-info")
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
    def _headers(self, *, req_id: Optional[str] = None) -> MutableMapping[str, str]:
        headers: MutableMapping[str, str] = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
        }
        if SUNO_CALLBACK_URL:
            headers["X-Callback-Url"] = SUNO_CALLBACK_URL
        if SUNO_CALLBACK_SECRET:
            headers["X-Callback-Token"] = SUNO_CALLBACK_SECRET
        if req_id:
            headers["X-Request-ID"] = req_id
        return headers

    def _url(self, path: str) -> str:
        return urljoin(self.base_url, path.lstrip("/"))

    @staticmethod
    def _normalize_path(path: str) -> str:
        text = (path or "").strip()
        if not text:
            return "/"
        if text.startswith("http://") or text.startswith("https://"):
            return text
        if not text.startswith("/"):
            text = f"/{text}"
        return text

    @staticmethod
    def _normalize_model(value: Optional[str]) -> str:
        raw = (value or SUNO_MODEL or "suno-v5" or "").strip()
        if not raw:
            return "suno-v5"
        normalized = raw.replace("_", "-")
        if normalized.lower() in {"v5", "suno-v5"}:
            return "suno-v5"
        return normalized

    @staticmethod
    def _legacy_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
        allowed = {"title", "style", "lyrics", "model", "instrumental"}
        result: dict[str, Any] = {}
        for key in allowed:
            if key in payload and payload[key] is not None:
                result[key] = payload[key]
        return result

    def _build_v5_payload(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self._normalize_model(str(payload.get("model") or "")),
            "input": {},
        }
        input_block = body["input"]
        prompt = payload.get("prompt") or payload.get("lyrics") or payload.get("title")
        if prompt is not None and str(prompt).strip():
            input_block["prompt"] = str(prompt)
        style = payload.get("style")
        if style is not None and str(style).strip():
            input_block["style"] = str(style)
        if "instrumental" in payload:
            input_block["instrumental"] = bool(payload.get("instrumental"))
        title = payload.get("title")
        if title is not None and str(title).strip():
            input_block["title"] = str(title)
        if SUNO_CALLBACK_URL:
            body["callbackUrl"] = SUNO_CALLBACK_URL
        return body

    @staticmethod
    def _is_not_found_code(value: Any) -> bool:
        try:
            return int(value) == 404
        except (TypeError, ValueError):
            return False

    def _should_fallback_error(self, error: SunoAPIError) -> bool:
        if error.status == 404:
            return True
        payload = error.payload
        if isinstance(payload, Mapping) and self._is_not_found_code(payload.get("code")):
            return True
        return False

    def _should_fallback_payload(self, payload: Mapping[str, Any]) -> bool:
        if not isinstance(payload, Mapping):
            return False
        if self._is_not_found_code(payload.get("code")):
            return True
        nested = payload.get("data")
        if isinstance(nested, Mapping) and self._is_not_found_code(nested.get("code")):
            return True
        return False

    def _log_fallback(self, *, target: str, req_id: Optional[str], reason: str) -> None:
        log.info(
            "suno.http fallback",
            extra={"meta": {"target": target, "reason": reason, "req_id": req_id}},
        )

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

    def _maybe_backoff(self, code: Optional[int], attempt: int, *, req_id: Optional[str] = None) -> bool:
        if attempt >= self.max_attempts:
            return False
        retryable = code is None or code in _RETRYABLE_CODES or (code is not None and code >= 500)
        if retryable:
            base = _BACKOFF_SCHEDULE[min(attempt - 1, len(_BACKOFF_SCHEDULE) - 1)]
            jitter = random.uniform(0.0, 1.0)
            delay = base + jitter
            log.warning(
                "suno.http retry",
                extra={
                    "meta": {
                        "code": code or "error",
                        "attempt": attempt,
                        "delay": round(delay, 3),
                        "req_id": req_id,
                    }
                },
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
        req_id: Optional[str] = None,
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
                    headers=self._headers(req_id=req_id),
                    json=json_payload,
                    params=params,
                    timeout=self.timeout,
                )
            except RequestException as exc:
                last_error = exc
                if not self._maybe_backoff(None, attempt, req_id=req_id):
                    log.error(
                        "suno.http failed",
                        extra={"meta": {"attempt": attempt, "req_id": req_id, "error": str(exc)}},
                    )
                    raise SunoAPIError("Network error talking to Suno") from exc
                continue

            status = response.status_code
            if self._maybe_backoff(status, attempt, req_id=req_id):
                continue

            payload = self._parse_json(response)
            if status >= 400:
                log.error(
                    "suno.http error",
                    extra={"meta": {"code": status, "attempt": attempt, "req_id": req_id}},
                )
                raise SunoAPIError(
                    payload.get("message") or payload.get("msg") or f"HTTP {status}",
                    status=status,
                    payload=payload,
                )

            log.info(
                "suno.http success",
                extra={"meta": {"code": status, "attempt": attempt, "req_id": req_id}},
            )
            return payload

        raise SunoAPIError("Suno request exhausted retries", payload=getattr(last_error, "response", None))

    # ------------------------------------------------------------------ public API
    def create_music(
        self, payload: Mapping[str, Any], *, req_id: Optional[str] = None
    ) -> tuple[Mapping[str, Any], str]:
        legacy_payload = self._legacy_payload(payload)
        body_v5 = self._build_v5_payload(payload)
        api_version = _API_V5
        try:
            response = self._request("POST", self._primary_gen_path, json_payload=body_v5, req_id=req_id)
        except SunoAPIError as exc:
            if self._should_fallback_error(exc):
                self._log_fallback(target=_LEGACY_GEN_PATH, req_id=req_id, reason="not_found")
                api_version = _API_LEGACY
                try:
                    response = self._request(
                        "POST",
                        _LEGACY_GEN_PATH,
                        json_payload=legacy_payload,
                        req_id=req_id,
                    )
                except SunoAPIError as legacy_exc:
                    legacy_exc.api_version = api_version
                    raise
            else:
                exc.api_version = api_version
                raise
        else:
            if self._should_fallback_payload(response):
                self._log_fallback(target=_LEGACY_GEN_PATH, req_id=req_id, reason="code_404")
                api_version = _API_LEGACY
                try:
                    response = self._request(
                        "POST",
                        _LEGACY_GEN_PATH,
                        json_payload=legacy_payload,
                        req_id=req_id,
                    )
                except SunoAPIError as legacy_exc:
                    legacy_exc.api_version = api_version
                    raise
        return response, api_version

    def get_task_status(self, task_id: str, *, req_id: Optional[str] = None) -> Mapping[str, Any]:
        params_v5 = {"taskId": task_id}
        try:
            response = self._request(
                "GET",
                self._primary_status_path,
                params=params_v5,
                req_id=req_id,
            )
        except SunoAPIError as exc:
            if self._should_fallback_error(exc):
                self._log_fallback(target=_LEGACY_TASK_STATUS_PATH, req_id=req_id, reason="not_found")
                try:
                    response = self._request(
                        "GET",
                        _LEGACY_TASK_STATUS_PATH,
                        params={"task_id": task_id},
                        req_id=req_id,
                    )
                except SunoAPIError as legacy_exc:
                    legacy_exc.api_version = _API_LEGACY
                    raise
            else:
                exc.api_version = _API_V5
                raise
        else:
            if self._should_fallback_payload(response):
                self._log_fallback(target=_LEGACY_TASK_STATUS_PATH, req_id=req_id, reason="code_404")
                response = self._request(
                    "GET",
                    _LEGACY_TASK_STATUS_PATH,
                    params={"task_id": task_id},
                    req_id=req_id,
                )
        return response


__all__ = ["SunoClient", "SunoAPIError"]
