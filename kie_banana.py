# -*- coding: utf-8 -*-
# kie_banana.py — KIE wrapper for google/nano-banana-edit (multi-image up to 4)
# Версия: 2025-09-12
#
# Функции:
#   create_banana_task(prompt, image_urls, ... ) -> taskId (str)
#   wait_for_banana_result(taskId, ...) -> List[str] (result urls)
#
# ENV обязательные:
#   KIE_BASE_URL (по умолчанию https://api.kie.ai)
#   KIE_API_KEY  (Bearer ...)

from __future__ import annotations
import os
import json
import time
import logging
from typing import Dict, Any, List, Optional, Tuple

import requests

KIE_BASE_URL = os.getenv("KIE_BASE_URL", "https://api.kie.ai").strip()
KIE_API_KEY  = os.getenv("KIE_API_KEY", "").strip()

MODEL_BANANA = "google/nano-banana-edit"

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

class KieBananaError(Exception):
    pass

def create_banana_task(
    prompt: str,
    image_urls: List[str],
    output_format: str = "png",
    image_size: str = "auto",
    callback_url: Optional[str] = None,
    extra_input: Optional[Dict[str, Any]] = None,
    timeout: int = 60,
) -> str:
    """Создаёт задачу nano-banana-edit. Поддерживает до 4 image_urls."""
    if not KIE_API_KEY:
        raise KieBananaError("KIE_API_KEY is missing")
    imgs = [u for u in (image_urls or []) if isinstance(u, str) and u.startswith("http")]
    if not imgs:
        raise KieBananaError("image_urls is empty")
    payload: Dict[str, Any] = {
        "model": MODEL_BANANA,
        "input": {
            "prompt": prompt or "",
            "image_urls": imgs[:4],
            "output_format": output_format,
            "image_size": image_size,
        },
    }
    if callback_url:
        payload["callBackUrl"] = callback_url
    if extra_input:
        payload["input"].update(extra_input)

    url = _join(KIE_BASE_URL, "/api/v1/jobs/createTask")
    r = requests.post(url, headers=_headers_json(), data=json.dumps(payload), timeout=timeout)
    try:
        j = r.json()
    except Exception:
        j = {"error": r.text}
    if r.status_code != 200 or (j.get("code", r.status_code) != 200):
        raise KieBananaError(f"createTask failed: status={r.status_code}, resp={j}")
    task_id = (j.get("data") or {}).get("taskId")
    if not task_id:
        raise KieBananaError(f"createTask: empty taskId in resp={j}")
    return str(task_id)

def get_banana_record(task_id: str, timeout: int = 60) -> Dict[str, Any]:
    if not KIE_API_KEY:
        raise KieBananaError("KIE_API_KEY is missing")
    url = _join(KIE_BASE_URL, "/api/v1/jobs/recordInfo")
    r = requests.get(url, headers=_auth_header(), params={"taskId": task_id}, timeout=timeout)
    try:
        j = r.json()
    except Exception:
        j = {"error": r.text}
    if r.status_code != 200:
        raise KieBananaError(f"recordInfo http {r.status_code}: {j}")
    return j

def parse_banana_result_urls(record_json: Dict[str, Any]) -> Tuple[Optional[List[str]], Optional[str]]:
    """Возвращает (urls, state). urls=None — результата нет."""
    data = record_json.get("data") or {}
    state = data.get("state")
    rj = data.get("resultJson")
    if not rj:
        return None, state
    try:
        parsed = json.loads(rj)
    except Exception:
        logging.exception("banana: failed to parse resultJson")
        return None, state
    urls = parsed.get("resultUrls") or parsed.get("urls") or []
    urls = [u for u in urls if isinstance(u, str) and u.startswith("http")]
    return (urls if urls else None), state

def wait_for_banana_result(task_id: str, timeout_sec: int = 300, poll_sec: int = 3) -> List[str]:
    """Ждём success/fail/таймаут. Возвращаем список URL'ов изображений."""
    deadline = time.time() + timeout_sec
    last_state = None
    while time.time() < deadline:
        j = get_banana_record(task_id)
        code = j.get("code")
        data = j.get("data") or {}
        state = data.get("state")
        last_state = state or last_state

        if code != 200:
            logging.warning("banana record non-200: %s", j)
            time.sleep(poll_sec)
            continue

        if state in ("waiting", "queuing", "generating", None):
            time.sleep(poll_sec); continue

        if state == "success":
            urls, _ = parse_banana_result_urls(j)
            if not urls:
                raise KieBananaError(f"success without resultUrls: {j}")
            return urls

        if state == "fail":
            raise KieBananaError(f"banana fail: code={data.get('failCode')} msg={data.get('failMsg')}")

        time.sleep(poll_sec)

    raise KieBananaError(f"banana timeout; last_state={last_state}")
