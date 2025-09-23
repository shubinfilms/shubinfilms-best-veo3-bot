# -*- coding: utf-8 -*-
"""
kie_banana.py — надёжная обёртка для KIE Nano Banana (edit)
Версия: 2025-09-13

Функции:
  create_banana_task(prompt, image_urls, output_format="png", image_size="auto",
                     callback_url=None, extra_input=None, timeout=60) -> str
  wait_for_banana_result(task_id, timeout_sec=480, poll_sec=3) -> List[str]

ENV:
  KIE_BASE_URL       (default: https://api.kie.ai)
  KIE_API_KEY        (Bearer ... или чистый токен)
  KIE_BANANA_MODEL   (default: google/nano-banana-edit)
"""

from __future__ import annotations
import os
import json
import time
import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple

import aiohttp
import requests

KIE_BASE_URL = (os.getenv("KIE_BASE_URL", "https://api.kie.ai") or "").strip()
KIE_API_KEY  = (os.getenv("KIE_API_KEY", "") or "").strip()
MODEL_BANANA = (os.getenv("KIE_BANANA_MODEL", "google/nano-banana-edit").strip()
                or "google/nano-banana-edit")

class KieBananaError(Exception):
    pass

def _auth_header() -> Dict[str, str]:
    tok = KIE_API_KEY
    if tok and not tok.lower().startswith("bearer "):
        tok = f"Bearer {tok}"
    return {"Authorization": tok} if tok else {}

def _headers_json() -> Dict[str, str]:
    return {**_auth_header(), "Content-Type": "application/json"}

def _join(base: str, path: str) -> str:
    u = f"{base.rstrip('/')}/{path.lstrip('/')}"
    return u.replace("://", "§§").replace("//", "/").replace("§§", "://")

def _post_json(url: str, payload: Dict[str, Any], timeout: int = 60) -> Tuple[int, Dict[str, Any]]:
    r = requests.post(url, headers=_headers_json(), json=payload, timeout=timeout)
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, {"error": r.text}

def _get_json(url: str, params: Dict[str, Any], timeout: int = 60) -> Tuple[int, Dict[str, Any]]:
    r = requests.get(url, headers=_auth_header(), params=params, timeout=timeout)
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, {"error": r.text}

def _take_task_id(j: Dict[str, Any]) -> Optional[str]:
    data = j.get("data") or {}
    for k in ("taskId", "taskid", "id"):
        if j.get(k):
            return str(j[k])
        if data.get(k):
            return str(data[k])
    return None

def _coerce_urls(val) -> List[str]:
    out: List[str] = []

    def add(u: str):
        if isinstance(u, str):
            s = u.strip()
            if s.startswith("http"):
                out.append(s)

    if not val:
        return out
    if isinstance(val, str):
        s = val.strip()
        if s.startswith("["):
            try:
                arr = json.loads(s)
                if isinstance(arr, list):
                    for v in arr:
                        if isinstance(v, str):
                            add(v)
                return out
            except Exception:
                add(s); return out
        add(s); return out
    if isinstance(val, list):
        for v in val:
            if isinstance(v, str):
                add(v)
            elif isinstance(v, dict):
                u = v.get("url") or v.get("resultUrl") or v.get("originUrl")
                if isinstance(u, str):
                    add(u)
        return out
    if isinstance(val, dict):
        for k in ("url", "resultUrl", "originUrl"):
            u = val.get(k)
            if isinstance(u, str):
                add(u)
        return out
    return out

def create_banana_task(
    prompt: str,
    image_urls: List[str],
    output_format: str = "png",
    image_size: str = "auto",
    callback_url: Optional[str] = None,
    extra_input: Optional[Dict[str, Any]] = None,
    timeout: int = 60,
) -> str:
    """
    Создаёт задачу для google/nano-banana-edit.
    Вход строго соответствует документации KIE (snake_case поля в 'input').
    """
    if not KIE_API_KEY:
        raise KieBananaError("KIE_API_KEY is missing")

    imgs = [u for u in (image_urls or []) if isinstance(u, str) and u.startswith("http")]
    if not imgs:
        raise KieBananaError("image_urls is empty")

    payload: Dict[str, Any] = {
        "model": MODEL_BANANA,
        "input": {
            "prompt": prompt or "",
            "image_urls": imgs[:4],           # до 4
            "output_format": output_format,   # png/jpeg
            "image_size": image_size,         # auto | 1:1 | 3:4 | 9:16 | 4:3 | 16:9
        }
    }
    if extra_input:
        payload["input"].update(extra_input)
    if callback_url:
        payload["callBackUrl"] = callback_url

    url = _join(KIE_BASE_URL, "/api/v1/jobs/createTask")
    status, j = _post_json(url, payload, timeout=timeout)
    code = j.get("code", status)
    if status != 200 or code != 200:
        raise KieBananaError(f"createTask failed: status={status}, code={code}, resp={j}")

    task_id = _take_task_id(j)
    if not task_id:
        raise KieBananaError(f"createTask: empty taskId in resp={j}")
    return str(task_id)

def get_banana_record(task_id: str, timeout: int = 60) -> Dict[str, Any]:
    url = _join(KIE_BASE_URL, "/api/v1/jobs/recordInfo")
    status, j = _get_json(url, {"taskId": task_id}, timeout=timeout)
    if status != 200:
        raise KieBananaError(f"recordInfo http {status}: {j}")
    return j

def _parse_result_urls(record_json: Dict[str, Any]) -> Tuple[Optional[List[str]], Optional[str]]:
    data = record_json.get("data") or {}
    state = data.get("state") or data.get("successFlag")

    # прямые поля
    for field in ("resultUrls", "originUrls", "videoUrls"):
        urls = _coerce_urls(data.get(field))
        if urls:
            return urls, str(state) if state is not None else None

    # через resultJson
    rj = data.get("resultJson")
    if rj:
        try:
            obj = json.loads(rj) if isinstance(rj, str) else rj
            for field in ("resultUrls", "urls", "originUrls"):
                urls = _coerce_urls(obj.get(field))
                if urls:
                    return urls, str(state) if state is not None else None
        except Exception:
            logging.exception("banana: resultJson parse fail")

    return None, str(state) if state is not None else None

def wait_for_banana_result(task_id: str, timeout_sec: int = 480, poll_sec: int = 3) -> List[str]:
    """
    Ждём success/fail. Возвращаем список URL'ов.
    """
    deadline = time.time() + timeout_sec
    last_state = None
    while time.time() < deadline:
        j = get_banana_record(task_id)
        code = j.get("code", 200)
        data = j.get("data") or {}
        state = data.get("state") or data.get("successFlag")
        last_state = state if state is not None else last_state

        if code != 200:
            time.sleep(poll_sec); continue

        # ожидание
        if state in (None, "waiting", "queuing", "generating") or str(state) == "0":
            time.sleep(poll_sec); continue

        # успех
        if state in ("success", "1"):
            urls, _ = _parse_result_urls(j)
            if not urls:
                raise KieBananaError(f"success without resultUrls: {j}")
            return urls

        # провал
        if state in ("fail", "2", "3"):
            raise KieBananaError(f"banana fail: code={data.get('failCode')} msg={data.get('failMsg')}")

        time.sleep(poll_sec)

    raise KieBananaError(f"banana timeout; last_state={last_state}")


log = logging.getLogger("kie")

KIE_OK_STATES = {"done", "finished", "success", "completed", "ready"}
KIE_WAIT_STATES = {"queued", "processing", "running", "pending", "started"}
KIE_BAD_STATES = {"failed", "error", "canceled", "cancelled", "timeout"}


async def poll_veo_status(
    session: aiohttp.ClientSession,
    base_url: str,
    status_path: str,
    task_id: str,
    api_key: str,
    *,
    timeout_sec: int = 15 * 60,
    start_delay: float = 2.0,
    max_delay: float = 20.0,
) -> str:
    """
    Ждёт готовности задачи и возвращает прямой URL видео.
    Бросает исключение при ошибке / таймауте.
    """

    headers = {"Accept": "application/json"}
    if api_key:
        token = api_key
        if not token.lower().startswith("bearer "):
            token = f"Bearer {token}"
        headers["Authorization"] = token

    url = f"{base_url.rstrip('/')}{status_path}" if status_path.startswith("/") else _join(base_url, status_path)

    t0 = time.time()
    delay = start_delay
    last_status: Optional[str] = None

    while True:
        params = {"id": task_id, "taskId": task_id}
        async with session.get(url, params=params, headers=headers, timeout=60) as response:
            data: Dict[str, Any] = await response.json(content_type=None)

        status = (data.get("status") or data.get("state") or "").lower()

        if status != last_status:
            log.info("KIE STATUS | task=%s | %s -> %s", task_id, last_status or "?", status or "?")
            last_status = status

        if status in KIE_OK_STATES:
            result = data.get("result") or {}
            file_url = (
                result.get("file_url")
                or result.get("video_url")
                or data.get("file_url")
                or data.get("video_url")
            )
            if file_url:
                log.info("KIE READY | task=%s | url=%s", task_id, file_url)
                return file_url
            log.info("KIE READY | task=%s | empty url, waiting", task_id)
            if time.time() - t0 > timeout_sec:
                raise TimeoutError(f"KIE polling timeout after {timeout_sec}s, last={status}")
            await asyncio.sleep(delay)
            delay = min(delay * 1.5, max_delay)
            continue

        if not status or status in KIE_WAIT_STATES:
            if time.time() - t0 > timeout_sec:
                raise TimeoutError(f"KIE polling timeout after {timeout_sec}s, last={status}")
            await asyncio.sleep(delay)
            delay = min(delay * 1.5, max_delay)
            continue

        if status in KIE_BAD_STATES:
            raise RuntimeError(f"KIE failed with status '{status}': {data}")

        log.warning("Unknown KIE status '%s': %s", status, data)
        if time.time() - t0 > timeout_sec:
            raise TimeoutError(f"KIE polling timeout after {timeout_sec}s, last={status}")
        await asyncio.sleep(delay)
        delay = min(delay * 1.5, max_delay)
