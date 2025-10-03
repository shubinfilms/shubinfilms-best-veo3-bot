"""HTTP client for interacting with the KIE Sora2 API."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import httpx

import settings


logger = logging.getLogger(__name__)
_MAX_LOG_BODY = 512
_INVALID_422_MARKERS = ("model cannot be null", "page does not exist")


class Sora2Error(RuntimeError):
    """Base error for Sora2 client failures."""


class Sora2UnavailableError(Sora2Error):
    """Raised when the Sora2 model is disabled for the current API key."""


class Sora2AuthError(Sora2Error):
    """Raised when authentication with KIE Sora2 fails."""


@dataclass(slots=True)
class CreateTaskResponse:
    task_id: str
    raw: Mapping[str, Any]


@dataclass(slots=True)
class QueryTaskResponse:
    status: str
    result_url: Optional[str]
    raw: Mapping[str, Any]


def _timeout() -> httpx.Timeout:
    return httpx.Timeout(
        connect=settings.SORA2_TIMEOUT_CONNECT,
        read=settings.SORA2_TIMEOUT_READ,
        write=settings.SORA2_TIMEOUT_WRITE,
        pool=settings.SORA2_TIMEOUT_POOL,
    )


def _mask_token(token: Optional[str]) -> str:
    tail = settings.token_tail(token)
    if not tail:
        return ""
    return f"***{tail}"


def _log_request(url: str, payload: Mapping[str, Any]) -> None:
    logger.info(
        "sora2.http.request",
        extra={"path": url, "token_tail": _mask_token(settings.SORA2_API_KEY)},
    )
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "sora2.http.payload",
            extra={"path": url, "payload": json.dumps(payload, ensure_ascii=False)},
        )


def _log_response(url: str, response: httpx.Response) -> None:
    try:
        body = response.text
    except Exception:  # pragma: no cover - defensive fallback
        body = response.content.decode("utf-8", errors="replace")
    snippet = (body or "")[:_MAX_LOG_BODY]
    logger.info(
        "sora2.http.response",
        extra={
            "path": url,
            "token_tail": _mask_token(settings.SORA2_API_KEY),
            "status": response.status_code,
            "body": snippet,
        },
    )


def _sanitize_payload_for_log(payload: Mapping[str, Any]) -> Dict[str, Any]:
    sample: Dict[str, Any] = {}
    for key, value in payload.items():
        if key == "input" and isinstance(value, Mapping):
            input_copy: Dict[str, Any] = {}
            for inner_key, inner_value in value.items():
                if inner_key == "prompt":
                    text = str(inner_value or "")
                    input_copy["prompt_length"] = len(text)
                elif inner_key == "image_urls" and isinstance(inner_value, Sequence) and not isinstance(inner_value, (str, bytes, bytearray)):
                    urls = [str(url) for url in list(inner_value)[:4]]
                    input_copy["image_urls"] = urls
                else:
                    input_copy[inner_key] = inner_value
            sample[key] = input_copy
        else:
            sample[key] = value
    return sample


def _send_once(url: str, payload: Mapping[str, Any]) -> httpx.Response:
    api_key = settings.SORA2_API_KEY
    if not api_key:
        raise Sora2Error("Sora2 API key is not configured")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    _log_request(url, payload)
    with httpx.Client(timeout=_timeout()) as client:
        response = client.post(url, headers=headers, json=payload)
    _log_response(url, response)
    return response


def _request_with_retries(
    url: str,
    payload: Mapping[str, Any],
    *,
    auto_model: bool,
) -> Mapping[str, Any]:
    network_retries = [1, 2]
    server_retries = [1, 3]
    invalid_retry_allowed = bool(auto_model)

    while True:
        try:
            response = _send_once(url, payload)
        except httpx.TimeoutException as exc:  # pragma: no cover - network
            if network_retries:
                delay = network_retries.pop(0)
                logger.warning(
                    "sora2.http.timeout",
                    extra={"path": url, "delay": delay, "error": str(exc)},
                )
                time.sleep(delay)
                continue
            raise Sora2Error("Sora2 request timed out") from exc
        except httpx.RequestError as exc:
            if network_retries:
                delay = network_retries.pop(0)
                logger.warning(
                    "sora2.http.failure",
                    extra={"path": url, "delay": delay, "error": str(exc)},
                )
                time.sleep(delay)
                continue
            raise Sora2Error("Sora2 request failed") from exc

        status = response.status_code
        body_text = response.text or ""
        lowered = body_text.lower()

        if status in {401, 403}:
            logger.warning(
                "sora2_auth_error",
                extra={"path": url, "status": status},
            )
            raise Sora2AuthError("authentication with Sora2 API failed")

        if status == 422 and any(marker in lowered for marker in _INVALID_422_MARKERS):
            logger.warning(
                "sora2.invalid_request",
                extra={
                    "path": url,
                    "status": status,
                    "token_tail": _mask_token(settings.SORA2_API_KEY),
                    "payload_sample": _sanitize_payload_for_log(payload),
                },
            )
            if invalid_retry_allowed:
                invalid_retry_allowed = False
                time.sleep(0.75)
                continue
            if "page does not exist" in lowered:
                raise Sora2UnavailableError("Sora2 model is disabled for this API key")
            raise Sora2Error("Sora2 request failed with status 422")

        if 500 <= status < 600:
            if server_retries:
                delay = server_retries.pop(0)
                logger.warning(
                    "sora2.http.server_error",
                    extra={
                        "path": url,
                        "status": status,
                        "delay": delay,
                        "token_tail": _mask_token(settings.SORA2_API_KEY),
                    },
                )
                time.sleep(delay)
                continue
            raise Sora2Error(f"Sora2 request failed with status {status}")

        if not response.is_success:
            logger.error(
                "sora2.http.error",
                extra={
                    "path": url,
                    "status": status,
                    "token_tail": _mask_token(settings.SORA2_API_KEY),
                    "body": (body_text or "")[:_MAX_LOG_BODY],
                },
            )
            raise Sora2Error(f"Sora2 request failed with status {status}")

        try:
            data = response.json()
        except Exception as exc:
            raise Sora2Error("Sora2 response is not valid JSON") from exc
        if not isinstance(data, Mapping):
            raise Sora2Error("Sora2 response is not a JSON object")
        return data


def create_task(payload: Mapping[str, Any]) -> CreateTaskResponse:
    """Submit a new generation task to Sora2."""

    task_type = str(payload.get("task_type") or "").strip().lower()
    normalized: Dict[str, Any] = dict(payload)
    input_payload = dict(normalized.get("input") or {})
    if task_type == "text2video":
        normalized.setdefault("model", "sora2-text-to-video")
        input_payload.setdefault("prompt", "")
    elif task_type == "image2video":
        normalized.setdefault("model", "sora2-image-to-video")
        input_payload.setdefault("image_urls", [])
    normalized["input"] = input_payload

    response = _request_with_retries(
        settings.SORA2_GEN_PATH,
        normalized,
        auto_model="model" not in payload,
    )
    task_id = (
        str(response.get("taskId") or response.get("task_id") or "").strip()
    )
    if not task_id:
        raise Sora2Error("Sora2 response did not include taskId")
    return CreateTaskResponse(task_id=task_id, raw=response)


def _extract_status(payload: Mapping[str, Any]) -> str:
    for key in ("status", "task_status", "state"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    data = payload.get("data")
    if isinstance(data, Mapping):
        nested = _extract_status(data)
        if nested:
            return nested
    return "unknown"


def _extract_result_url(payload: Mapping[str, Any]) -> Optional[str]:
    for key in ("result_url", "resultUrl", "video_url", "videoUrl", "url"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip().lower().startswith("http"):
            return value.strip()
    data = payload.get("result")
    if isinstance(data, Mapping):
        nested = _extract_result_url(data)
        if nested:
            return nested
    data = payload.get("data")
    if isinstance(data, Mapping):
        nested = _extract_result_url(data)
        if nested:
            return nested
    return None


def query_task(task_id: str) -> QueryTaskResponse:
    """Fetch current status for a Sora2 task."""

    payload = {"task_id": str(task_id)}
    response = _request_with_retries(
        settings.SORA2_STATUS_PATH,
        payload,
        auto_model=False,
    )
    status = _extract_status(response)
    return QueryTaskResponse(status=status, result_url=_extract_result_url(response), raw=response)


def _build_upload_url() -> Optional[str]:
    base = (settings.UPLOAD_BASE_URL or "").strip()
    if not base:
        return None
    path = (settings.UPLOAD_URL_PATH or "").strip() or "/api/v1/upload/url"
    if path.startswith("http"):
        return path
    if not path.startswith("/"):
        path = f"/{path}"
    return f"{base}{path}"


def _extract_public_url(payload: Mapping[str, Any]) -> Optional[str]:
    for key in ("public_url", "publicUrl", "url"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip().lower().startswith("http"):
            return value.strip()
    data = payload.get("data")
    if isinstance(data, Mapping):
        return _extract_public_url(data)
    return None


def upload_image_urls(image_urls: Iterable[str]) -> List[str]:
    endpoint = _build_upload_url()
    cleaned = [str(url or "").strip() for url in image_urls if str(url or "").strip()]
    if not endpoint:
        return cleaned
    if not cleaned:
        return []

    api_key = settings.SORA2_API_KEY
    if not api_key:
        raise Sora2Error("Sora2 API key is not configured")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    results: List[str] = []
    for original in cleaned:
        payload = {"url": original}
        _log_request(endpoint, payload)
        try:
            with httpx.Client(timeout=_timeout()) as client:
                response = client.post(endpoint, headers=headers, json=payload)
        except httpx.RequestError as exc:  # pragma: no cover - network
            logger.error(
                "sora2.upload_http_failure",
                extra={"path": endpoint, "error": str(exc)},
            )
            raise Sora2Error("Failed to upload image URL for Sora2") from exc
        _log_response(endpoint, response)
        if not response.is_success:
            logger.error(
                "sora2.upload_http_error",
                extra={"path": endpoint, "status": response.status_code},
            )
            raise Sora2Error("Failed to upload image URL for Sora2")
        try:
            payload = response.json()
        except Exception as exc:  # pragma: no cover - defensive
            raise Sora2Error("Upload service response is not valid JSON") from exc
        public_url = _extract_public_url(payload)
        if not public_url:
            raise Sora2Error("Upload service response missing public URL")
        results.append(public_url)
    return results


__all__ = [
    "CreateTaskResponse",
    "QueryTaskResponse",
    "Sora2Error",
    "Sora2AuthError",
    "Sora2UnavailableError",
    "create_task",
    "query_task",
    "upload_image_urls",
]
