from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional

import httpx

from settings import (
    KIE_API_KEY,
    SORA2,
    SORA2_API_KEY,
    UPLOAD_BASE_URL,
    UPLOAD_URL_PATH,
)


_LOG = logging.getLogger(__name__)
_RETRYABLE_STATUS_CODES = {500, 502, 503, 504}
_MAX_RESPONSE_LOG = 512


def _auth_token() -> str:
    token = SORA2_API_KEY or KIE_API_KEY
    if not token:
        raise RuntimeError("Sora 2 API key is not configured")
    return token


def _timeout() -> httpx.Timeout:
    return httpx.Timeout(connect=20.0, read=30.0, write=30.0)


def _log_response(method: str, url: str, response: httpx.Response) -> None:
    try:
        text = response.text
    except Exception:  # pragma: no cover - defensive
        text = response.content.decode("utf-8", errors="replace")
    snippet = text[:_MAX_RESPONSE_LOG]
    _LOG.info(
        "sora2.http.response",
        extra={"method": method, "url": url, "status": response.status_code, "body": snippet},
    )


def _should_retry(response: httpx.Response) -> bool:
    return response.status_code in _RETRYABLE_STATUS_CODES


def _perform_request(
    method: str,
    url: str,
    *,
    headers: Optional[Mapping[str, str]] = None,
    json_body: Optional[Mapping[str, Any]] = None,
    params: Optional[Mapping[str, Any]] = None,
    attempts: int = 3,
) -> httpx.Response:
    last_exc: Optional[BaseException] = None
    for attempt in range(1, attempts + 1):
        try:
            with httpx.Client(timeout=_timeout()) as client:
                response = client.request(
                    method,
                    url,
                    headers=dict(headers or {}),
                    json=json_body,
                    params=dict(params or {}),
                )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last_exc = exc
            _LOG.warning(
                "sora2.http.error",
                extra={"method": method, "url": url, "attempt": attempt, "error": str(exc)},
            )
        else:
            _log_response(method, url, response)
            if response.status_code >= 400 and _should_retry(response) and attempt < attempts:
                _LOG.warning(
                    "sora2.http.retry",
                    extra={
                        "method": method,
                        "url": url,
                        "attempt": attempt,
                        "status": response.status_code,
                    },
                )
                time.sleep(2 ** (attempt - 1))
                continue
            return response

        if attempt < attempts:
            time.sleep(2 ** (attempt - 1))

    if last_exc is not None:
        raise last_exc
    raise httpx.HTTPError(f"Failed to call {url}")


def sora2_create_task(payload: Dict[str, Any]) -> Dict[str, Any]:
    url = SORA2["GEN_PATH"]
    headers = {
        "Authorization": f"Bearer {_auth_token()}",
        "Content-Type": "application/json",
    }
    _LOG.info("sora2.http.request", extra={"method": "POST", "url": url})
    response = _perform_request("POST", url, headers=headers, json_body=payload)
    response.raise_for_status()
    try:
        return response.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError("Invalid JSON from Sora 2 createTask") from exc


def sora2_get_task(task_id: str) -> Dict[str, Any]:
    url = SORA2["STATUS_PATH"]
    headers = {"Authorization": f"Bearer {_auth_token()}"}
    params = {"taskId": task_id}
    _LOG.info("sora2.http.request", extra={"method": "GET", "url": url, "taskId": task_id})
    response = _perform_request("GET", url, headers=headers, params=params)
    response.raise_for_status()
    try:
        return response.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError("Invalid JSON from Sora 2 queryTask") from exc


def _build_upload_url() -> Optional[str]:
    base = (UPLOAD_BASE_URL or "").strip()
    if not base:
        return None
    path = (UPLOAD_URL_PATH or "").strip() or "/api/v1/upload/url"
    if not path.startswith("http"):
        if not path.startswith("/"):
            path = f"/{path}"
        return f"{base}{path}"
    return path


def _extract_public_url(payload: Mapping[str, Any]) -> Optional[str]:
    for key in ("public_url", "publicUrl", "url"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip().lower().startswith("http"):
            return value.strip()
    data = payload.get("data")
    if isinstance(data, Mapping):
        return _extract_public_url(data)
    return None


def sora2_upload_image_urls(image_urls: Iterable[str]) -> List[str]:
    upload_endpoint = _build_upload_url()
    if not upload_endpoint:
        return [str(url).strip() for url in image_urls if str(url).strip()]

    headers = {
        "Authorization": f"Bearer {_auth_token()}",
        "Content-Type": "application/json",
    }
    results: List[str] = []
    for original in image_urls:
        cleaned = str(original or "").strip()
        if not cleaned:
            continue
        _LOG.info("sora2.upload.request", extra={"url": cleaned})
        response = _perform_request(
            "POST",
            upload_endpoint,
            headers=headers,
            json_body={"url": cleaned},
        )
        if response.status_code >= 400:
            response.raise_for_status()
        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise RuntimeError("Invalid JSON from upload service") from exc
        public_url = _extract_public_url(payload)
        if not public_url:
            raise RuntimeError("Upload service response missing public URL")
        results.append(public_url)
    return results


__all__ = ["sora2_create_task", "sora2_get_task", "sora2_upload_image_urls"]
