"""HTTP client for interacting with the Suno API."""
from __future__ import annotations

import json
import logging
import random
import time
from typing import Any, Dict, Mapping, Optional

import requests
from requests import Response, Session

from settings import (
    SUNO_API_BASE,
    SUNO_API_TOKEN,
    SUNO_CALLBACK_SECRET,
    SUNO_TIMEOUT_SEC,
)

log = logging.getLogger("suno.http")


class SunoError(RuntimeError):
    """Base class for all Suno HTTP errors."""

    def __init__(
        self,
        message: str,
        *,
        safe_message: Optional[str] = None,
        details: Optional[Mapping[str, Any]] = None,
        status: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.safe_message = safe_message or message or "Suno service error"
        self.details: Mapping[str, Any] = details or {}
        self.status = status


class SunoBadRequest(SunoError):
    def __init__(self, message: str, *, details: Optional[Mapping[str, Any]] = None) -> None:
        super().__init__(
            message or "Bad request to Suno",
            safe_message="Запрос к Suno отклонён. Проверьте параметры.",
            details=details,
            status=400,
        )


class SunoAuthError(SunoError):
    def __init__(self, message: str, status: int = 401) -> None:
        super().__init__(
            message or "Authentication failed for Suno API",
            safe_message="Не удалось авторизоваться в Suno. Проверьте токен.",
            status=status,
        )


class SunoNotFound(SunoError):
    def __init__(self, message: str, path: str) -> None:
        super().__init__(
            message or "Resource not found at Suno",
            safe_message="Не удалось найти задачу в Suno.",
            details={"path": path},
            status=404,
        )
        self.path = path


class SunoConflict(SunoError):
    def __init__(self, message: str, status: int) -> None:
        super().__init__(
            message or "Conflict while calling Suno",
            safe_message="Запрос конфликтует с текущим состоянием Suno.",
            status=status,
        )


class SunoUnprocessable(SunoError):
    def __init__(self, message: str, status: int) -> None:
        super().__init__(
            message or "Unprocessable request for Suno",
            safe_message="Suno не смог обработать запрос.",
            status=status,
        )


class SunoRateLimited(SunoError):
    def __init__(self, message: str, retry_after: Optional[float]) -> None:
        safe = "Слишком много запросов к Suno. Попробуйте позже."
        details: Dict[str, Any] = {}
        if retry_after is not None:
            details["retry_after"] = retry_after
        super().__init__(
            message or "Rate limited by Suno",
            safe_message=safe,
            details=details,
            status=429,
        )
        self.retry_after = retry_after


class SunoServerError(SunoError):
    def __init__(self, message: str, *, status: Optional[int] = None) -> None:
        super().__init__(
            message or "Suno server error",
            safe_message="Suno временно недоступен. Попробуйте ещё раз позже.",
            status=status or 503,
        )


def _validate_base_url(base_url: Optional[str]) -> str:
    if not base_url:
        raise RuntimeError("SUNO_API_BASE is not configured")
    normalized = base_url.rstrip("/")
    if not normalized.startswith("http"):
        raise RuntimeError("SUNO_API_BASE must start with http/https")
    return normalized


def _validate_token(token: Optional[str]) -> str:
    if not token:
        raise RuntimeError("SUNO_API_TOKEN is not configured")
    return token


class SunoHttp:
    """Low level HTTP client with retry and error mapping."""

    _max_attempts = 3
    _base_delay = 0.5
    _jitter_max = 0.25

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        token: Optional[str] = None,
        timeout: Optional[int] = None,
        session: Optional[Session] = None,
        callback_secret: Optional[str] = None,
    ) -> None:
        self.base_url = _validate_base_url(base_url or SUNO_API_BASE)
        self.token = _validate_token(token or SUNO_API_TOKEN)
        self.timeout = timeout or SUNO_TIMEOUT_SEC or 45
        self.session = session or requests.Session()
        self.callback_secret = callback_secret or SUNO_CALLBACK_SECRET or None

    # ------------------------------------------------------------------ utils
    def _full_url(self, path: str) -> str:
        if not path:
            raise ValueError("path must be provided")
        if not path.startswith("/"):
            path = "/" + path
        return f"{self.base_url}{path}"

    def _build_headers(self, *, has_files: bool, extra: Optional[Mapping[str, str]]) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
            "User-Agent": "best-veo3-bot/1.0 (+render)",
        }
        if not has_files:
            headers["Content-Type"] = "application/json"
        if self.callback_secret:
            headers["X-Callback-Token"] = self.callback_secret
        if extra:
            headers.update(extra)
        if has_files:
            headers.pop("Content-Type", None)
        return headers

    def _should_retry(self, status: Optional[int]) -> bool:
        if status is None:
            return True
        if status == 429:
            return True
        return 500 <= status < 600

    def _sleep(self, attempt: int) -> None:
        delay = self._base_delay * (2 ** attempt)
        delay += random.random() * self._jitter_max
        time.sleep(delay)

    def _extract_task_id(self, payload: Any) -> Optional[str]:
        if not isinstance(payload, Mapping):
            return None
        data = payload.get("data")
        if isinstance(data, Mapping):
            task_id = data.get("taskId") or data.get("task_id")
            if task_id:
                return str(task_id)
        task_id = payload.get("taskId") or payload.get("task_id")
        if task_id:
            return str(task_id)
        return None

    def _parse_json(self, response: Response) -> Optional[Dict[str, Any]]:
        if response.content is None or not response.content:
            return None
        try:
            parsed = response.json()
        except json.JSONDecodeError as exc:
            raise SunoServerError("Invalid JSON returned by Suno", status=response.status_code) from exc
        if isinstance(parsed, dict):
            return parsed
        return {"data": parsed}

    def _error_message(self, payload: Optional[Mapping[str, Any]]) -> str:
        if not payload:
            return ""
        for key in ("message", "msg", "error", "detail"):
            value = payload.get(key)
            if isinstance(value, str):
                return value
        return ""

    def _map_error(
        self,
        status: int,
        path: str,
        payload: Optional[Mapping[str, Any]],
        response: Optional[Response] = None,
    ) -> SunoError:
        message = self._error_message(payload) or f"HTTP {status}"
        if status == 400:
            return SunoBadRequest(message, details=payload)
        if status in (401, 403):
            return SunoAuthError(message, status=status)
        if status == 404:
            return SunoNotFound(message, path)
        if status in (409,):
            return SunoConflict(message, status)
        if status in (422,):
            return SunoUnprocessable(message, status)
        if status == 429:
            retry_after: Optional[float] = None
            if response is not None:
                raw = response.headers.get("Retry-After") if response.headers else None
                if raw:
                    try:
                        retry_after = float(raw)
                    except ValueError:
                        retry_after = None
            if payload:
                retry_after_payload = payload.get("retry_after")
                if retry_after is None and isinstance(retry_after_payload, (int, float)):
                    retry_after = float(retry_after_payload)
            return SunoRateLimited(message, retry_after)
        if status >= 500:
            return SunoServerError(message, status=status)
        return SunoError(message, status=status, safe_message="Неизвестная ошибка Suno", details=payload)

    # ---------------------------------------------------------------- requests
    def request(
        self,
        method: str,
        path: str,
        *,
        json: Optional[Mapping[str, Any]] = None,
        params: Optional[Mapping[str, Any]] = None,
        data: Optional[Mapping[str, Any]] = None,
        files: Optional[Mapping[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
        timeout: Optional[int] = None,
        op: Optional[str] = None,
    ) -> Dict[str, Any]:
        url = self._full_url(path)
        payload_keys = list(json.keys()) if isinstance(json, Mapping) else []
        attempt_error: Optional[Exception] = None
        for attempt in range(1, self._max_attempts + 1):
            start = time.perf_counter()
            status: Optional[int] = None
            task_id: Optional[str] = None
            try:
                response = self.session.request(
                    method,
                    url,
                    json=json,
                    params=params,
                    data=data,
                    files=files,
                    headers=self._build_headers(has_files=files is not None, extra=headers),
                    timeout=timeout or self.timeout,
                )
                status = response.status_code
                payload = None
                if status != 204:
                    try:
                        payload = self._parse_json(response)
                    except SunoError as exc:
                        attempt_error = exc
                        if attempt >= self._max_attempts:
                            raise
                        self._log_attempt(op or method.upper(), path, "invalid_json", attempt, time.perf_counter() - start, task_id, payload_keys)
                        self._sleep(attempt)
                        continue
                task_id = self._extract_task_id(payload)
                elapsed = time.perf_counter() - start
                self._log_attempt(op or method.upper(), path, status, attempt, elapsed, task_id, payload_keys)
                if self._should_retry(status):
                    if attempt >= self._max_attempts:
                        raise self._map_error(status or 503, path, payload, response)
                    self._sleep(attempt)
                    continue
                if status and status >= 400:
                    raise self._map_error(status, path, payload, response)
                if isinstance(payload, Mapping):
                    code = payload.get("code")
                    if isinstance(code, int) and code >= 400:
                        raise self._map_error(code, path, payload, response)
                    return dict(payload)
                return {}
            except requests.RequestException as exc:
                elapsed = time.perf_counter() - start
                attempt_error = exc
                self._log_attempt(op or method.upper(), path, "network_error", attempt, elapsed, task_id, payload_keys)
                if attempt >= self._max_attempts:
                    raise SunoServerError("Network error talking to Suno") from exc
                self._sleep(attempt)
                continue
        if attempt_error:
            raise SunoServerError(str(attempt_error)) from attempt_error
        raise SunoServerError("Unknown error calling Suno")

    def get(
        self,
        path: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
        op: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.request("GET", path, params=params, headers=headers, op=op)

    def post(
        self,
        path: str,
        *,
        json: Optional[Mapping[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
        op: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.request("POST", path, json=json, headers=headers, op=op)

    # ---------------------------------------------------------------- logging
    def _log_attempt(
        self,
        op: str,
        path: str,
        status: Any,
        attempt: int,
        elapsed: float,
        task_id: Optional[str],
        payload_keys: Optional[list[str]],
    ) -> None:
        elapsed_ms = round(elapsed * 1000, 1)
        extra = {
            "op": op,
            "path": path,
            "status": status,
            "elapsed_ms": elapsed_ms,
            "try": attempt,
        }
        if task_id:
            extra["task_id"] = task_id
        if payload_keys:
            extra["payload_keys"] = payload_keys
        log.info(
            "suno request op=%s path=%s status=%s elapsed_ms=%.1f try=%s task_id=%s keys=%s",
            op,
            path,
            status,
            elapsed_ms,
            attempt,
            task_id or "-",
            ",".join(payload_keys or []),
            extra=extra,
        )


# Backwards compatibility ----------------------------------------------------
SunoClient = SunoHttp
SunoAPIError = SunoError

__all__ = [
    "SunoHttp",
    "SunoClient",
    "SunoError",
    "SunoAPIError",
    "SunoBadRequest",
    "SunoAuthError",
    "SunoNotFound",
    "SunoConflict",
    "SunoUnprocessable",
    "SunoRateLimited",
    "SunoServerError",
]
