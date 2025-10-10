"""HTTP client for interacting with the KIE Sora2 API."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

import httpx

import settings


logger = logging.getLogger(__name__)
_MAX_LOG_BODY = 512
_BAD_REQUEST_MARKERS = (
    "the input cannot be null",
    "the page does not exist or is not published",
)


class Sora2Error(RuntimeError):
    """Base error for Sora2 client failures."""


class Sora2UnavailableError(Sora2Error):
    """Raised when the Sora2 model is disabled for the current API key."""


class Sora2AuthError(Sora2Error):
    """Raised when authentication with KIE Sora2 fails."""


class Sora2BadRequestError(Sora2Error):
    """Raised when Sora2 rejects the payload with a validation error."""


@dataclass(slots=True)
class CreateTaskResponse:
    task_id: str
    raw: Mapping[str, Any]


@dataclass(slots=True)
class QueryTaskResponse:
    status: str
    result_urls: List[str]
    result_payload: Optional[Mapping[str, Any]]
    error_message: Optional[str]
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


def _payload_summary(payload: Mapping[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    model = payload.get("model")
    if isinstance(model, str) and model.strip():
        summary["model"] = model.strip()
    input_payload = payload.get("input")
    if isinstance(input_payload, Mapping):
        prompt = input_payload.get("prompt")
        if isinstance(prompt, str):
            summary["prompt_length"] = len(prompt)
        image_urls = input_payload.get("image_urls")
        if isinstance(image_urls, Sequence) and not isinstance(image_urls, (str, bytes, bytearray)):
            summary["image_count"] = len([u for u in image_urls if isinstance(u, str) and u.strip()])
            summary["has_image"] = any(
                isinstance(u, str) and u.strip() for u in image_urls
            )
        aspect = input_payload.get("aspect_ratio")
        if isinstance(aspect, str) and aspect.strip():
            summary["aspect_ratio"] = aspect.strip()
        quality = input_payload.get("quality")
        if isinstance(quality, str) and quality.strip():
            summary["quality"] = quality.strip()
    callback_url = payload.get("callBackUrl")
    if isinstance(callback_url, str) and callback_url.strip():
        summary["has_callback"] = True
    return summary


def _log_request(url: str, payload: Mapping[str, Any]) -> None:
    logger.info(
        "sora2.http.request",
        extra={"path": url, "token_tail": _mask_token(settings.SORA2_API_KEY)},
    )
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "sora2.http.payload",
            extra={"path": url, "payload": json.dumps(_payload_summary(payload), ensure_ascii=False)},
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
    return _payload_summary(payload)


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

        if status == 422:
            if any(marker in lowered for marker in _BAD_REQUEST_MARKERS):
                logger.warning(
                    "sora2.invalid_request",
                    extra={
                        "path": url,
                        "status": status,
                        "token_tail": _mask_token(settings.SORA2_API_KEY),
                        "payload_sample": _sanitize_payload_for_log(payload),
                        "body": (body_text or "")[:_MAX_LOG_BODY],
                    },
                )
                raise Sora2BadRequestError("Sora2 rejected the request payload (422)")
            if invalid_retry_allowed:
                invalid_retry_allowed = False
                time.sleep(0.75)
                continue
            logger.error(
                "sora2.http.unexpected_422",
                extra={
                    "path": url,
                    "status": status,
                    "token_tail": _mask_token(settings.SORA2_API_KEY),
                    "body": (body_text or "")[:_MAX_LOG_BODY],
                },
            )
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

    normalized: Dict[str, Any] = {}
    model = payload.get("model")
    if isinstance(model, str) and model.strip():
        normalized["model"] = model.strip()
    callback_url = payload.get("callBackUrl")
    if isinstance(callback_url, str) and callback_url.strip():
        normalized["callBackUrl"] = callback_url.strip()
    input_payload: Dict[str, Any] = {}
    raw_input = payload.get("input")
    if isinstance(raw_input, Mapping):
        for key, value in raw_input.items():
            if value is None:
                continue
            input_payload[key] = value
    normalized["input"] = input_payload

    response = _request_with_retries(
        settings.SORA2_GEN_PATH,
        normalized,
        auto_model=False,
    )
    data = response.get("data") if isinstance(response, Mapping) else None
    task_id = ""
    if isinstance(data, Mapping):
        task_id = str(data.get("taskId") or data.get("task_id") or "").strip()
    if not task_id:
        task_id = str(response.get("taskId") or response.get("task_id") or "").strip()
    if not task_id:
        raise Sora2Error("Sora2 response did not include taskId")
    return CreateTaskResponse(task_id=task_id, raw=response)


def _log_status_request(url: str, params: Mapping[str, Any]) -> None:
    logger.info(
        "sora2.http.status_request",
        extra={"path": url, "token_tail": _mask_token(settings.SORA2_API_KEY)},
    )
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "sora2.http.status_params",
            extra={"path": url, "params": dict(params)},
        )


def _collect_urls(value: Any) -> List[str]:
    urls: List[str] = []
    seen: Set[str] = set()

    def add(candidate: str) -> None:
        trimmed = candidate.strip()
        if not trimmed or not trimmed.lower().startswith("http"):
            return
        if trimmed in seen:
            return
        seen.add(trimmed)
        urls.append(trimmed)

    def walk(node: Any, depth: int = 0) -> None:
        if depth > 6:
            return
        if isinstance(node, str):
            text = node.strip()
            if not text:
                return
            if text.startswith("[") or text.startswith("{"):
                try:
                    parsed = json.loads(text)
                except Exception:
                    add(text)
                else:
                    walk(parsed, depth + 1)
            else:
                add(text)
            return
        if isinstance(node, Mapping):
            for value in node.values():
                walk(value, depth + 1)
            return
        if isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
            for item in node:
                walk(item, depth + 1)
            return

    walk(value)
    return urls


def _parse_result_json(raw: Any) -> Optional[Mapping[str, Any]]:
    if raw is None:
        return None
    try:
        parsed: Any
        if isinstance(raw, str):
            text = raw.strip()
            if not text:
                return None
            parsed = json.loads(text)
        else:
            parsed = raw
    except Exception as exc:
        logger.debug(
            "sora2.result_json.parse_failed",
            extra={"error": str(exc)},
        )
        return None
    if isinstance(parsed, Mapping):
        return parsed
    if isinstance(parsed, Sequence) and not isinstance(parsed, (str, bytes, bytearray)):
        return {"items": list(parsed)}
    return None


def _extract_error_message(*payloads: Mapping[str, Any]) -> Optional[str]:
    error_keys = ("error", "errorMsg", "errorMessage", "message", "reason", "failReason")
    for payload in payloads:
        for key in error_keys:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def query_task(task_id: str) -> QueryTaskResponse:
    """Fetch current status for a Sora2 task."""

    api_key = settings.SORA2_API_KEY
    if not api_key:
        raise Sora2Error("Sora2 API key is not configured")

    params = {"taskId": str(task_id)}
    url = settings.SORA2_STATUS_PATH
    headers = {"Authorization": f"Bearer {api_key}"}
    _log_status_request(url, params)
    with httpx.Client(timeout=_timeout()) as client:
        response = client.get(url, headers=headers, params=params)
    _log_response(url, response)

    status_code = response.status_code
    body_text = response.text or ""
    if status_code in {401, 403}:
        logger.warning(
            "sora2_auth_error",
            extra={"path": url, "status": status_code},
        )
        raise Sora2AuthError("authentication with Sora2 API failed")
    if status_code == 404:
        logger.warning(
            "sora2.status.not_found",
            extra={"path": url, "task_id": task_id},
        )
        raise Sora2Error("Sora2 task not found")
    if status_code >= 500:
        logger.warning(
            "sora2.status.server_error",
            extra={"path": url, "status": status_code},
        )
        raise Sora2Error(f"Sora2 status endpoint returned {status_code}")
    if not response.is_success:
        logger.error(
            "sora2.status.http_error",
            extra={
                "path": url,
                "status": status_code,
                "token_tail": _mask_token(settings.SORA2_API_KEY),
                "body": body_text[:_MAX_LOG_BODY],
            },
        )
        raise Sora2Error(f"Sora2 status endpoint returned {status_code}")

    try:
        payload = response.json()
    except Exception as exc:
        raise Sora2Error("Sora2 status response is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise Sora2Error("Sora2 status response is not a JSON object")

    data = payload.get("data")
    data_map: Mapping[str, Any] = data if isinstance(data, Mapping) else {}
    state_raw = data_map.get("state")
    state = str(state_raw or "").strip().lower() or "unknown"

    result_urls: List[str] = []
    direct_urls = _collect_urls(data_map.get("resultUrls"))
    if direct_urls:
        result_urls.extend(direct_urls)

    result_payload = _parse_result_json(data_map.get("resultJson"))
    if result_payload:
        for url_value in _collect_urls(result_payload):
            if url_value not in result_urls:
                result_urls.append(url_value)

    error_message = _extract_error_message(data_map, payload)

    return QueryTaskResponse(
        status=state,
        result_urls=result_urls,
        result_payload=result_payload,
        error_message=error_message,
        raw=payload,
    )


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


def _kie_async_base() -> str:
    base = getattr(settings, "KIE_BASE_URL", "") or "https://api.kie.ai/api/v1"
    return str(base).rstrip("/")


def _kie_create_endpoint() -> str:
    return f"{_kie_async_base()}/jobs/createTask"


def _kie_status_endpoint() -> str:
    return f"{_kie_async_base()}/jobs/recordInfo"


def _kie_headers() -> Dict[str, str]:
    token = (settings.KIE_API_KEY or "").strip()
    if not token:
        raise RuntimeError("KIE API key is not configured")
    return {"Authorization": f"Bearer {token}"}


def _elapsed_ms(response: httpx.Response) -> Optional[int]:
    try:
        elapsed = response.elapsed
    except Exception:
        return None
    if not elapsed:
        return None
    return int(elapsed.total_seconds() * 1000)


async def kie_create_sora2_task(
    ctx: Any,
    *,
    prompt: str,
    aspect_ratio: Optional[str] = None,
    quality: Optional[str] = None,
    callback_url: Optional[str] = None,
) -> str:
    payload: Dict[str, Any] = {
        "model": "sora-2-text-to-video",
        "input": {"prompt": prompt},
    }
    if aspect_ratio:
        payload["input"]["aspect_ratio"] = aspect_ratio
    if quality:
        payload["input"]["quality"] = quality
    if callback_url:
        payload["callBackUrl"] = callback_url

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            _kie_create_endpoint(),
            json=payload,
            headers=_kie_headers(),
        )
    logger.info(
        "kie.http.create",
        extra={
            "status": response.status_code,
            "elapsed_ms": _elapsed_ms(response),
        },
    )
    response.raise_for_status()
    data = response.json()
    task_id = data.get("taskId") or data.get("task_id")
    if not task_id:
        raise RuntimeError("KIE: taskId is empty")
    return str(task_id)


async def kie_poll_sora2(
    ctx: Any,
    task_id: str,
    *,
    timeout_s: int = 600,
    step_s: int = 5,
) -> Mapping[str, Any]:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + max(int(timeout_s), 1)
    endpoint = _kie_status_endpoint()
    headers = _kie_headers()

    async with httpx.AsyncClient(timeout=30.0) as client:
        while True:
            response = await client.get(endpoint, params={"taskId": task_id}, headers=headers)
            state_payload: Mapping[str, Any] = response.json()
            state = str(state_payload.get("state") or "").lower()
            logger.debug(
                "kie.http.poll",
                extra={"task_id": task_id, "state": state, "status": response.status_code},
            )
            response.raise_for_status()
            if state in {"success"}:
                result_json = state_payload.get("resultJson")
                if not (isinstance(result_json, Mapping) and result_json.get("resultUrls")):
                    raise RuntimeError("KIE: empty resultUrls")
                return state_payload
            if state in {"fail", "failed", "error"}:
                raise RuntimeError(f"KIE: state={state}")
            if loop.time() >= deadline:
                break
            await asyncio.sleep(max(step_s, 1))
    raise TimeoutError("KIE: polling timeout")


__all__ = [
    "CreateTaskResponse",
    "QueryTaskResponse",
    "Sora2Error",
    "Sora2BadRequestError",
    "Sora2AuthError",
    "Sora2UnavailableError",
    "create_task",
    "query_task",
    "upload_image_urls",
    "kie_create_sora2_task",
    "kie_poll_sora2",
]
