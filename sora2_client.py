from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional

import httpx

from settings import (
    KIE_API_KEY,
    SORA2_API_KEY,
    SORA2_GEN_PATH,
    SORA2_STATUS_PATH,
    UPLOAD_BASE_URL,
    UPLOAD_URL_PATH,
)


logger = logging.getLogger(__name__)
_MAX_RESPONSE_LOG = 512


def _auth_token() -> str:
    token = SORA2_API_KEY or KIE_API_KEY
    if not token:
        raise RuntimeError("Sora 2 API key is not configured")
    return token


def _timeout() -> httpx.Timeout:
    if os.getenv("SORA2_TIMEOUT"):
        return httpx.Timeout(float(os.getenv("SORA2_TIMEOUT", "30")))
    connect = float(os.getenv("SORA2_TIMEOUT_CONNECT", "20"))
    read = float(os.getenv("SORA2_TIMEOUT_READ", "30"))
    write = float(os.getenv("SORA2_TIMEOUT_WRITE", "30"))
    pool = float(os.getenv("SORA2_TIMEOUT_POOL", "30"))
    return httpx.Timeout(connect=connect, read=read, write=write, pool=pool)


def _log_response(method: str, url: str, response: httpx.Response) -> None:
    try:
        text = response.text
    except Exception:  # pragma: no cover - defensive
        text = response.content.decode("utf-8", errors="replace")
    snippet = text[:_MAX_RESPONSE_LOG]
    logger.info(
        "sora2.http.response",
        extra={"method": method, "url": url, "status": response.status_code, "body": snippet},
    )


class Sora2RequestError(Exception):
    pass


def _perform_request(method: str, url: str, headers: Dict[str, str], json_body: Dict[str, Any]) -> Dict[str, Any]:
    tout = _timeout()
    last_exc: Optional[BaseException] = None
    for attempt in range(3):
        try:
            with httpx.Client(timeout=tout) as client:
                response = client.request(method, url, headers=headers, json=json_body)
                response.raise_for_status()
                _log_response(method, url, response)
                return response.json()
        except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as exc:
            last_exc = exc
            logger.warning(
                "sora2.http.retry",
                extra={"attempt": attempt + 1, "error": str(exc)},
            )
            time.sleep(0.5 * (2 ** attempt))
    raise Sora2RequestError(f"request failed after retries: {last_exc}")


def sora2_create_task(payload: Dict[str, Any]) -> str:
    url = SORA2_GEN_PATH
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {_auth_token()}",
    }
    logger.info("sora2.http.request", extra={"endpoint": "createTask"})
    data = _perform_request("POST", url, headers, payload)
    task_id = (data.get("taskId") or data.get("task_id") or "").strip()
    if not task_id:
        raise Sora2RequestError(f"missing task id in response: {data}")
    return task_id


def sora2_query_task(task_id: str) -> Dict[str, Any]:
    url = SORA2_STATUS_PATH
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {_auth_token()}",
    }
    payload = {"taskId": task_id}
    logger.info("sora2.http.request", extra={"endpoint": "queryTask", "task_id": task_id})
    return _perform_request("POST", url, headers, payload)


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
        logger.info("sora2.upload.request", extra={"url": cleaned})
        payload = _perform_request(
            "POST",
            upload_endpoint,
            headers,
            {"url": cleaned},
        )
        public_url = _extract_public_url(payload)
        if not public_url:
            raise RuntimeError("Upload service response missing public URL")
        results.append(public_url)
    return results


__all__ = ["sora2_create_task", "sora2_query_task", "sora2_upload_image_urls", "Sora2RequestError"]
