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
    SUNO_COVER_INFO_PATH,
    SUNO_GEN_PATH,
    SUNO_INSTR_PATH,
    SUNO_LYRICS_PATH,
    SUNO_MAX_RETRIES,
    SUNO_MODEL,
    SUNO_MP4_INFO_PATH,
    SUNO_MP4_PATH,
    SUNO_STEM_INFO_PATH,
    SUNO_STEM_PATH,
    SUNO_TASK_STATUS_PATH,
    SUNO_UPLOAD_EXTEND_PATH,
    SUNO_WAV_INFO_PATH,
    SUNO_WAV_PATH,
)

log = logging.getLogger("suno.client")

_RETRYABLE_CODES = {408, 429}
_BACKOFF_SCHEDULE = (1, 3, 7)
_API_V5 = "v5"


class SunoAPIError(RuntimeError):
    """Raised when the Suno API responds with an error."""

    def __init__(self, message: str, *, status: Optional[int] = None, payload: Any = None) -> None:
        super().__init__(message)
        self.status = status
        self.payload = payload
        self.api_version: Optional[str] = None


class SunoClientError(SunoAPIError):
    """Represents a 4xx response from the Suno API."""


class SunoServerError(SunoAPIError):
    """Represents retry-exhausted network issues or 5xx responses."""


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
        self._primary_gen_path = self._normalize_path(SUNO_GEN_PATH or "/api/v1/generate/music")
        self._primary_status_path = self._normalize_path(SUNO_TASK_STATUS_PATH or "/api/v1/generate/record-info")
        self._wav_path = self._normalize_path(SUNO_WAV_PATH or "/api/v1/wav/generate")
        self._wav_info_path = self._normalize_path(SUNO_WAV_INFO_PATH or "/api/v1/wav/record-info")
        self._mp4_path = self._normalize_path(SUNO_MP4_PATH or "/api/v1/mp4/generate")
        self._mp4_info_path = self._normalize_path(SUNO_MP4_INFO_PATH or "/api/v1/mp4/record-info")
        self._cover_info_path = self._normalize_path(SUNO_COVER_INFO_PATH or "/api/v1/suno/cover/record-info")
        self._lyrics_path = self._normalize_path(SUNO_LYRICS_PATH or "/api/v1/generate/get-timestamped-lyrics")
        self._stem_path = self._normalize_path(SUNO_STEM_PATH or "/api/v1/vocal-removal/generate")
        self._stem_info_path = self._normalize_path(SUNO_STEM_INFO_PATH or "/api/v1/vocal-removal/record-info")
        self._instrumental_path = self._normalize_path(SUNO_INSTR_PATH or "/api/v1/generate/add-instrumental")
        self._extend_path = self._normalize_path(SUNO_UPLOAD_EXTEND_PATH or "/api/v1/generate/upload-extend")
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
            headers["X-Callback-Secret"] = SUNO_CALLBACK_SECRET
            headers.setdefault("X-Callback-Token", SUNO_CALLBACK_SECRET)
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

    def _raise_if_not_found(self, payload: Mapping[str, Any]) -> None:
        if not isinstance(payload, Mapping):
            return
        if self._is_not_found_code(payload.get("code")):
            error = SunoClientError("Suno reported not found", status=404, payload=payload)
            error.api_version = _API_V5
            raise error
        nested = payload.get("data")
        if isinstance(nested, Mapping) and self._is_not_found_code(nested.get("code")):
            error = SunoClientError("Suno reported not found", status=404, payload=payload)
            error.api_version = _API_V5
            raise error

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

    @staticmethod
    def _response_request_id(response: Response) -> Optional[str]:
        for header in ("X-Request-ID", "X-Request-Id", "X-Req-Id", "X-Req-ID"):
            value = response.headers.get(header)
            if value:
                return str(value)
        return None

    def _log_request(
        self,
        op: str,
        *,
        level: int,
        method: str,
        url: str,
        status: Any,
        req_id: Optional[str],
        duration_ms: float,
        attempt: int,
        context: Optional[Mapping[str, Any]] = None,
    ) -> None:
        fields: MutableMapping[str, Any] = {
            "method": method.upper(),
            "url": url,
            "status": status,
            "req_id": req_id,
            "ms": round(duration_ms, 3),
        }
        if attempt > 1:
            fields["attempt"] = attempt
        if context:
            for key, value in context.items():
                if value is not None and key not in fields:
                    fields[key] = value
        message = " ".join(f"{key}={value}" for key, value in fields.items() if value not in (None, ""))
        log.log(level, "[SUNO][%s] %s", op, message, extra={"meta": {"op": op, **fields}})

    @staticmethod
    def _task_payload(task_id: str, extra: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"taskId": str(task_id)}
        if not extra:
            return payload
        for key, value in extra.items():
            if value is None:
                continue
            payload[key] = value
        return payload

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_payload: Optional[Mapping[str, Any]] = None,
        params: Optional[Mapping[str, Any]] = None,
        req_id: Optional[str] = None,
        op: str = "request",
        log_context: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        url = self._url(path)
        attempt = 0
        last_error: Optional[BaseException] = None
        last_status: Any = None
        last_req_id = req_id
        last_duration = 0.0
        while attempt < self.max_attempts:
            attempt += 1
            start_ts = time.monotonic()
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
                duration_ms = max(0.0, (time.monotonic() - start_ts) * 1000.0)
                last_error = exc
                last_status = "network_error"
                last_duration = duration_ms
                if not self._maybe_backoff(None, attempt, req_id=req_id):
                    context = dict(log_context or {})
                    context.setdefault("error", str(exc))
                    self._log_request(
                        op,
                        level=logging.ERROR,
                        method=method,
                        url=url,
                        status="network_error",
                        req_id=req_id,
                        duration_ms=duration_ms,
                        attempt=attempt,
                        context=context,
                    )
                    raise SunoServerError("Network error talking to Suno") from exc
                continue

            status = response.status_code
            duration_ms = max(0.0, (time.monotonic() - start_ts) * 1000.0)
            response_req_id = self._response_request_id(response) or req_id
            if self._maybe_backoff(status, attempt, req_id=req_id):
                last_status = status
                last_req_id = response_req_id
                last_duration = duration_ms
                continue

            payload = self._parse_json(response)
            if status >= 400:
                message = payload.get("message") or payload.get("msg") or f"HTTP {status}"
                context = dict(log_context or {})
                context.setdefault("error", message)
                level = logging.WARNING if 400 <= status < 500 else logging.ERROR
                self._log_request(
                    op,
                    level=level,
                    method=method,
                    url=url,
                    status=status,
                    req_id=response_req_id,
                    duration_ms=duration_ms,
                    attempt=attempt,
                    context=context,
                )
                error_cls = SunoClientError if 400 <= status < 500 else SunoServerError
                error = error_cls(message, status=status, payload=payload)
                error.api_version = _API_V5
                raise error

            context = dict(log_context or {})
            self._log_request(
                op,
                level=logging.INFO,
                method=method,
                url=url,
                status=status,
                req_id=response_req_id,
                duration_ms=duration_ms,
                attempt=attempt,
                context=context,
            )
            return payload

        context = dict(log_context or {})
        context.setdefault("error", "exhausted")
        self._log_request(
            op,
            level=logging.ERROR,
            method=method,
            url=url,
            status=last_status or "exhausted",
            req_id=last_req_id,
            duration_ms=last_duration,
            attempt=attempt,
            context=context,
        )
        raise SunoServerError("Suno request exhausted retries", payload=getattr(last_error, "response", None))

    # ------------------------------------------------------------------ public API
    def create_music(
        self, payload: Mapping[str, Any], *, req_id: Optional[str] = None
    ) -> tuple[Mapping[str, Any], str]:
        body_v5 = self._build_v5_payload(payload)
        prompt_source = (
            payload.get("prompt")
            or payload.get("lyrics")
            or payload.get("title")
            or ""
        )
        context = {
            "model": body_v5.get("model"),
            "prompt_len": len(str(prompt_source or "")),
        }
        if "instrumental" in payload:
            context["instrumental"] = bool(payload.get("instrumental"))
        try:
            response = self._request(
                "POST",
                self._primary_gen_path,
                json_payload=body_v5,
                req_id=req_id,
                op="enqueue",
                log_context=context,
            )
        except SunoAPIError as exc:
            exc.api_version = _API_V5
            raise
        self._raise_if_not_found(response)
        return response, _API_V5

    def get_task_status(self, task_id: str, *, req_id: Optional[str] = None) -> Mapping[str, Any]:
        params_v5 = {"taskId": task_id}
        try:
            response = self._request(
                "GET",
                self._primary_status_path,
                params=params_v5,
                req_id=req_id,
                op="status",
                log_context={"task_id": task_id},
            )
        except SunoAPIError as exc:
            exc.api_version = _API_V5
            raise
        self._raise_if_not_found(response)
        return response

    # ------------------------------------------------------------- WAV helpers
    def build_wav(
        self,
        task_id: str,
        *,
        req_id: Optional[str] = None,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        body = self._task_payload(task_id, payload)
        response = self._request(
            "POST",
            self._wav_path,
            json_payload=body,
            req_id=req_id,
            op="build_wav",
            log_context={"task_id": task_id},
        )
        self._raise_if_not_found(response)
        return response

    def get_wav_info(
        self,
        task_id: str,
        *,
        req_id: Optional[str] = None,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        params = self._task_payload(task_id, payload)
        response = self._request(
            "GET",
            self._wav_info_path,
            params=params,
            req_id=req_id,
            op="wav_info",
            log_context={"task_id": task_id},
        )
        self._raise_if_not_found(response)
        return response

    # ------------------------------------------------------------- MP4 helpers
    def build_mp4(
        self,
        task_id: str,
        *,
        req_id: Optional[str] = None,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        body = self._task_payload(task_id, payload)
        response = self._request(
            "POST",
            self._mp4_path,
            json_payload=body,
            req_id=req_id,
            op="build_mp4",
            log_context={"task_id": task_id},
        )
        self._raise_if_not_found(response)
        return response

    def get_mp4_info(
        self,
        task_id: str,
        *,
        req_id: Optional[str] = None,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        params = self._task_payload(task_id, payload)
        response = self._request(
            "GET",
            self._mp4_info_path,
            params=params,
            req_id=req_id,
            op="mp4_info",
            log_context={"task_id": task_id},
        )
        self._raise_if_not_found(response)
        return response

    # ------------------------------------------------------------- ancillary info
    def get_cover_info(
        self,
        task_id: str,
        *,
        req_id: Optional[str] = None,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        params = self._task_payload(task_id, payload)
        response = self._request(
            "GET",
            self._cover_info_path,
            params=params,
            req_id=req_id,
            op="cover_info",
            log_context={"task_id": task_id},
        )
        self._raise_if_not_found(response)
        return response

    def get_lyrics(
        self,
        task_id: str,
        *,
        req_id: Optional[str] = None,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        params = self._task_payload(task_id, payload)
        response = self._request(
            "GET",
            self._lyrics_path,
            params=params,
            req_id=req_id,
            op="lyrics",
            log_context={"task_id": task_id},
        )
        self._raise_if_not_found(response)
        return response

    def build_stem(
        self,
        task_id: str,
        *,
        req_id: Optional[str] = None,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        body = self._task_payload(task_id, payload)
        response = self._request(
            "POST",
            self._stem_path,
            json_payload=body,
            req_id=req_id,
            op="stem",
            log_context={"task_id": task_id},
        )
        self._raise_if_not_found(response)
        return response

    def get_stem_info(
        self,
        task_id: str,
        *,
        req_id: Optional[str] = None,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        params = self._task_payload(task_id, payload)
        response = self._request(
            "GET",
            self._stem_info_path,
            params=params,
            req_id=req_id,
            op="stem_info",
            log_context={"task_id": task_id},
        )
        self._raise_if_not_found(response)
        return response

    def add_instrumental(
        self,
        task_id: str,
        *,
        req_id: Optional[str] = None,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        body = self._task_payload(task_id, payload)
        response = self._request(
            "POST",
            self._instrumental_path,
            json_payload=body,
            req_id=req_id,
            op="instrumental",
            log_context={"task_id": task_id},
        )
        self._raise_if_not_found(response)
        return response

    def upload_extend(
        self,
        payload: Mapping[str, Any],
        *,
        req_id: Optional[str] = None,
    ) -> Mapping[str, Any]:
        context = {}
        task_id = payload.get("taskId") or payload.get("task_id")
        if task_id:
            context["task_id"] = task_id
        response = self._request(
            "POST",
            self._extend_path,
            json_payload=dict(payload),
            req_id=req_id,
            op="extend",
            log_context=context,
        )
        self._raise_if_not_found(response)
        return response


__all__ = [
    "SunoClient",
    "SunoAPIError",
    "SunoClientError",
    "SunoServerError",
]
