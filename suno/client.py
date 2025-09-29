"""HTTP client wrapper for the Suno API."""
from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Optional, Sequence
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
    KIE_BASE_URL,
    resolve_outbound_ip,
    SUNO_VOCAL_PATH,
)

log = logging.getLogger("suno.client")

_API_V5 = "v5"


@dataclass(frozen=True)
class EnqueueResult:
    """Result returned by :meth:`SunoClient.enqueue_music`."""

    body: Mapping[str, Any]
    req_id: Optional[str]
    task_id: Optional[str]
    status: Optional[str]
    path: str
    custom_mode: bool


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
        raw_base = (base_url or SUNO_API_BASE or KIE_BASE_URL or "https://api.kie.ai").strip()
        if not raw_base:
            raw_base = "https://api.kie.ai"
        self.base_url = raw_base.rstrip("/") + "/"
        self.token = (token or SUNO_API_TOKEN or "").strip()
        if not self.token:
            log.warning("SunoClient initialized without API token; requests will fail")
        self._primary_gen_path = self._normalize_path(SUNO_GEN_PATH or "/api/v1/generate")
        self._primary_status_path = self._normalize_path(
            SUNO_TASK_STATUS_PATH or "/api/v1/generate/record-info"
        )
        self._wav_path = self._normalize_path(SUNO_WAV_PATH or "/api/v1/wav/generate")
        self._wav_info_path = self._normalize_path(SUNO_WAV_INFO_PATH or "/api/v1/wav/record-info")
        self._mp4_path = self._normalize_path(SUNO_MP4_PATH or "/api/v1/mp4/generate")
        self._mp4_info_path = self._normalize_path(SUNO_MP4_INFO_PATH or "/api/v1/mp4/record-info")
        self._cover_info_path = self._normalize_path(SUNO_COVER_INFO_PATH or "/api/v1/suno/cover/record-info")
        self._lyrics_path = self._normalize_path(SUNO_LYRICS_PATH or "/api/v1/generate/get-timestamped-lyrics")
        self._stem_path = self._normalize_path(SUNO_STEM_PATH or "/api/v1/vocal-removal/generate")
        self._stem_info_path = self._normalize_path(SUNO_STEM_INFO_PATH or "/api/v1/vocal-removal/record-info")
        self._instrumental_path = self._normalize_path(SUNO_INSTR_PATH or "/api/v1/generate/add-instrumental")
        self._vocal_path = self._normalize_path(SUNO_VOCAL_PATH or "/api/v1/generate/add-vocals")
        self._extend_path = self._normalize_path(SUNO_UPLOAD_EXTEND_PATH or "/api/v1/suno/upload-extend")
        self.session = session or requests.Session()
        adapter = HTTPAdapter(pool_connections=HTTP_POOL_CONNECTIONS, pool_maxsize=HTTP_POOL_PER_HOST)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        retries = max_retries if max_retries is not None else SUNO_MAX_RETRIES or HTTP_RETRY_ATTEMPTS
        self.max_attempts = max(1, int(retries))
        self._retry_total_cap = 40.0
        self._retry_max_delay = 12.0
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
    def _missing_fields(payload: Mapping[str, Any]) -> list[str]:
        missing: list[str] = []
        detail = payload.get("detail")
        if isinstance(detail, list):
            for item in detail:
                if not isinstance(item, Mapping):
                    continue
                message = str(item.get("msg") or item.get("message") or "").lower()
                if "field required" not in message and "missing" not in message:
                    continue
                loc = item.get("loc")
                if isinstance(loc, (list, tuple)) and loc:
                    missing.append(str(loc[-1]))
                    continue
                field = item.get("field")
                if isinstance(field, str):
                    missing.append(field)
        field_list = payload.get("missing_fields")
        if isinstance(field_list, (list, tuple)):
            missing.extend(str(value) for value in field_list)
        return missing

    @staticmethod
    def _payload_message(payload: Mapping[str, Any]) -> Optional[str]:
        for key in ("msg", "message"):
            value = payload.get(key)
            if value:
                return str(value)
        detail = payload.get("detail")
        if isinstance(detail, str) and detail.strip():
            return detail.strip()
        if isinstance(detail, Mapping):
            nested_message = SunoClient._payload_message(detail)
            if nested_message:
                return nested_message
        if isinstance(detail, Sequence) and not isinstance(detail, (str, bytes, bytearray)):
            for item in detail:
                if isinstance(item, Mapping):
                    nested_message = SunoClient._payload_message(item)
                    if nested_message:
                        return nested_message
                elif isinstance(item, str) and item.strip():
                    return item.strip()
        data = payload.get("data")
        if isinstance(data, Mapping):
            nested_message = SunoClient._payload_message(data)
            if nested_message:
                return nested_message
        return None

    @staticmethod
    def _is_user_id_empty_error(payload: Any) -> bool:
        if not isinstance(payload, Mapping):
            return False
        texts: list[str] = []
        for key in ("message", "msg"):
            value = payload.get(key)
            if value:
                texts.append(str(value))
        detail = payload.get("detail")
        if isinstance(detail, str):
            texts.append(detail)
        elif isinstance(detail, Mapping):
            nested = SunoClient._payload_message(detail)
            if nested:
                texts.append(nested)
        elif isinstance(detail, Sequence) and not isinstance(detail, (str, bytes, bytearray)):
            for item in detail:
                if isinstance(item, Mapping):
                    nested = SunoClient._payload_message(item)
                    if nested:
                        texts.append(nested)
                elif isinstance(item, str):
                    texts.append(item)
        combined = " ".join(text.lower() for text in texts if text)
        return "userid" in combined and "empty" in combined

    @staticmethod
    def _format_422_error(payload: Mapping[str, Any]) -> str:
        detail_messages: list[str] = []
        detail = payload.get("detail")
        if isinstance(detail, list):
            for item in detail:
                if not isinstance(item, Mapping):
                    continue
                msg = str(item.get("msg") or item.get("message") or "").strip()
                field_name: Optional[str] = None
                loc = item.get("loc")
                if isinstance(loc, (list, tuple)) and loc:
                    field_name = str(loc[-1])
                elif isinstance(item.get("field"), str):
                    field_name = str(item["field"])
                if field_name and msg:
                    detail_messages.append(f"{field_name}: {msg}")
                elif field_name:
                    detail_messages.append(field_name)
                elif msg:
                    detail_messages.append(msg)
        missing = SunoClient._missing_fields(payload)
        segments: list[str] = []
        base_message = str(payload.get("message") or payload.get("msg") or "").strip()
        if base_message:
            segments.append(base_message)
        if detail_messages:
            segments.append("; ".join(detail_messages))
        if missing:
            unique_missing = ", ".join(sorted({field for field in missing if field}))
            if unique_missing:
                segments.append(f"missing: {unique_missing}")
        summary = " | ".join(segment for segment in segments if segment)
        if not summary:
            summary = "Unprocessable request"
        return f"Suno validation error: {summary}"

    @staticmethod
    def _extract_identifier(payload: Mapping[str, Any], keys: Sequence[str]) -> Optional[str]:
        for key in keys:
            value = payload.get(key)
            if value not in (None, ""):
                text = str(value).strip()
                if text:
                    return text
        nested = payload.get("data")
        if isinstance(nested, Mapping):
            found = SunoClient._extract_identifier(nested, keys)
            if found:
                return found
        nested = payload.get("payload")
        if isinstance(nested, Mapping):
            found = SunoClient._extract_identifier(nested, keys)
            if found:
                return found
        return None

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

    def _compute_backoff(self, code: Optional[int], attempt: int) -> Optional[float]:
        if attempt >= self.max_attempts:
            return None
        retryable = code is None or code == 429 or (code is not None and code >= 500)
        if not retryable:
            return None
        base_delay = 1.0 * (2 ** max(attempt - 1, 0))
        capped_base = min(base_delay, self._retry_max_delay)
        jitter = random.uniform(0.3, 1.3)
        return min(max(capped_base * jitter, 0.1), self._retry_max_delay)

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

    def _log_retry(
        self,
        *,
        code: Optional[int],
        attempt: int,
        delay: float,
        req_id: Optional[str],
        path: str,
    ) -> None:
        log.warning(
            "suno.http retry",
            extra={
                "meta": {
                    "code": code or "error",
                    "attempt": attempt,
                    "delay": round(delay, 3),
                    "req_id": req_id,
                    "path": path,
                }
            },
        )

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
        log_context = dict(log_context or {})
        log_context.setdefault("path", path)
        total_backoff = 0.0
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
                delay = self._compute_backoff(None, attempt)
                remaining = self._retry_total_cap - total_backoff
                if delay is not None and remaining > 0:
                    actual_delay = min(delay, remaining)
                    if actual_delay > 0:
                        self._log_retry(code=None, attempt=attempt, delay=actual_delay, req_id=req_id, path=path)
                        time.sleep(actual_delay)
                        total_backoff += actual_delay
                        continue
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

            status = response.status_code
            duration_ms = max(0.0, (time.monotonic() - start_ts) * 1000.0)
            response_req_id = self._response_request_id(response) or req_id
            delay = self._compute_backoff(status, attempt)
            remaining = self._retry_total_cap - total_backoff
            if delay is not None and remaining > 0:
                actual_delay = min(delay, remaining)
                if actual_delay > 0:
                    self._log_retry(
                        code=status,
                        attempt=attempt,
                        delay=actual_delay,
                        req_id=req_id,
                        path=path,
                    )
                    time.sleep(actual_delay)
                    total_backoff += actual_delay
                    last_status = status
                    last_req_id = response_req_id
                    last_duration = duration_ms
                    continue

            payload = self._parse_json(response)
            task_hint = self._extract_identifier(payload, ("task_id", "taskId"))
            payload_message = self._payload_message(payload)
            if status >= 400:
                message = payload.get("message") or payload.get("msg") or f"HTTP {status}"
                context = dict(log_context or {})
                path_hint = context.get("path", path)
                if task_hint and "taskId" not in context:
                    context["taskId"] = task_hint
                if payload_message and "msg" not in context:
                    context["msg"] = payload_message
                if status == 401:
                    outbound_ip = resolve_outbound_ip() or "unknown"
                    context["outbound_ip"] = outbound_ip
                    context["base"] = self.base_url
                    log.error(
                        "401 KIE. OutboundIP=%s Add this IP to the KIE whitelist.",
                        outbound_ip,
                        extra={"meta": {"outbound_ip": outbound_ip, "base": self.base_url}},
                    )
                    message = "Suno: invalid credentials or IP not whitelisted"
                elif status == 404:
                    context["path"] = path_hint
                    context["base"] = self.base_url
                    log.error(
                        "Suno endpoint not found base=%s path=%s",
                        self.base_url,
                        path_hint,
                        extra={"meta": {"base": self.base_url, "path": path_hint}},
                    )
                    message = "Suno endpoint not found (check paths)"
                elif status == 422:
                    missing_fields = self._missing_fields(payload)
                    if missing_fields:
                        unique_missing = sorted({field for field in missing_fields if field})
                        if unique_missing:
                            context["missing"] = unique_missing
                    message = self._format_422_error(payload)
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
            if task_hint and "taskId" not in context:
                context["taskId"] = task_hint
            if payload_message and "msg" not in context:
                context["msg"] = payload_message
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
    def enqueue_music(
        self,
        *,
        user_id: int | str,
        title: str,
        prompt: str,
        instrumental: bool,
        has_lyrics: bool,
        lyrics: Optional[str],
        prompt_len: int = 16,
        model: Optional[str] = None,
        req_id: Optional[str] = None,
        call_back_url: Optional[str] = None,
        call_back_secret: Optional[str] = None,
    ) -> EnqueueResult:
        if not self.token:
            raise SunoClientError("SUNO_API_TOKEN is not configured", status=401)
        callback_url = (call_back_url or SUNO_CALLBACK_URL or "").strip()
        callback_secret = (call_back_secret or SUNO_CALLBACK_SECRET or "").strip()
        if not callback_url or not callback_secret:
            raise SunoClientError("Suno callback configuration missing", status=422)
        title_text = str(title or "").strip()
        prompt_text = str(prompt or "").strip()
        if not title_text:
            title_text = prompt_text or "Untitled track"
        if not prompt_text:
            prompt_text = title_text or "Untitled track"
        try:
            prompt_length = max(1, int(prompt_len))
        except (TypeError, ValueError):
            prompt_length = 16
        normalized_model = str(model or SUNO_MODEL or "V5").strip()
        if normalized_model.lower() in {"v5", "suno-v5"}:
            normalized_model = "V5"
        payload: dict[str, Any] = {
            "model": normalized_model or "V5",
            "title": title_text,
            "prompt": prompt_text,
            "instrumental": bool(instrumental),
            "has_lyrics": bool(has_lyrics),
            "prompt_len": prompt_length,
            "userId": str(user_id),
            "callBackUrl": callback_url,
            "tags": [],
            "negativeTags": [],
            "customMode": False,
        }
        if has_lyrics:
            payload["lyrics"] = str(lyrics or "")
        path = self._primary_gen_path
        context = {
            "phase": "enqueue",
            "path": path,
            "custom_mode": False,
            "instrumental": bool(instrumental),
            "has_lyrics": bool(has_lyrics),
            "prompt_len": prompt_length,
            "model": payload["model"],
        }
        try:
            response = self._request(
                "POST",
                path,
                json_payload=payload,
                req_id=req_id,
                op="enqueue",
                log_context=context,
            )
        except SunoAPIError as exc:
            exc.api_version = _API_V5
            raise
        task_identifier = self._extract_identifier(response, ("task_id", "taskId"))
        req_identifier = task_identifier or req_id or self._extract_identifier(
            response, ("req_id", "requestId", "request_id")
        )
        status_label = response.get("status") or response.get("code")
        self._raise_if_not_found(response)
        return EnqueueResult(
            body=response,
            req_id=req_identifier,
            task_id=task_identifier,
            status=str(status_label) if status_label is not None else None,
            path=path,
            custom_mode=False,
        )

    def create_music(
        self, payload: Mapping[str, Any], *, req_id: Optional[str] = None
    ) -> tuple[Mapping[str, Any], str]:
        user_identifier = payload.get("userId") or payload.get("user_id")
        if user_identifier is None:
            raise SunoClientError("userId is required", status=422)
        title = str(payload.get("title") or "").strip()
        style = str(payload.get("style") or "").strip()
        base_prompt = payload.get("prompt") or style or payload.get("input_text") or payload.get("lyrics")
        prompt = str(base_prompt or title or "").strip() or "Untitled track"
        if not title:
            title = prompt
        has_lyrics = bool(payload.get("has_lyrics") or payload.get("lyrics"))
        instrumental = bool(payload.get("instrumental"))
        lyrics_value = payload.get("lyrics") if has_lyrics else None
        try:
            prompt_length = int(payload.get("prompt_len") or 16)
        except (TypeError, ValueError):
            prompt_length = 16
        model_value = payload.get("model") or SUNO_MODEL
        result = self.enqueue_music(
            user_id=user_identifier,
            title=title,
            prompt=prompt,
            instrumental=instrumental,
            has_lyrics=has_lyrics,
            lyrics=str(lyrics_value) if lyrics_value is not None else None,
            prompt_len=prompt_length,
            model=str(model_value) if model_value is not None else None,
            req_id=req_id,
        )
        return result.body, _API_V5

    def get_task_status(
        self,
        req_id: str,
        *,
        user_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> Mapping[str, Any]:
        lookup_id = task_id or req_id
        if not lookup_id:
            raise SunoClientError("req_id is required for status check", status=422)
        params: dict[str, Any] = {"taskId": str(lookup_id)}
        user_param = str(user_id).strip() if user_id is not None else None
        if user_param:
            params["userId"] = user_param
        context = {
            "req_id": req_id or lookup_id,
            "userId": user_param,
            "taskId": str(lookup_id),
            "path": self._primary_status_path,
        }
        identifier = req_id or task_id or str(lookup_id)
        try:
            response = self._request(
                "GET",
                self._primary_status_path,
                params=params,
                req_id=identifier,
                op="status",
                log_context=context,
            )
        except SunoClientError as exc:
            if user_param and exc.status == 422 and self._is_user_id_empty_error(exc.payload):
                trimmed_context = dict(context)
                trimmed_context["userId"] = None
                trimmed_context["hint"] = "retry_without_userId"
                retry_params = {"taskId": str(lookup_id)}
                response = self._request(
                    "GET",
                    self._primary_status_path,
                    params=retry_params,
                    req_id=identifier,
                    op="status",
                    log_context=trimmed_context,
                )
            else:
                exc.api_version = _API_V5
                raise
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
    "EnqueueResult",
    "SunoClient",
    "SunoAPIError",
    "SunoClientError",
    "SunoServerError",
]
