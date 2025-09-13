# -*- coding: utf-8 -*-
# kie_banana.py — обёртка для KIE google/nano-banana-edit
# Совместимо с официальными docs:
#   POST /api/v1/jobs/createTask
#   GET  /api/v1/jobs/recordInfo

from __future__ import annotations
import os
import json
import time
import logging
from typing import Dict, Any, List, Optional, Tuple

import requests

KIE_BASE_URL = os.getenv("KIE_BASE_URL", "https://api.kie.ai").strip()
KIE_API_KEY  = os.getenv("KIE_API_KEY", "").strip()
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
    return u.replace("://", "__SCHEME__").replace("//", "/").replace("__SCHEME__", "://")

def create_banana_task(
    prompt: str,
    image_urls: List[str],
    output_format: str = "png",
    image_size: str = "auto",
    callback_url: Optional[str] = None,
    extra_input: Optional[Dict[str, Any]] = None,
    timeout: int = 60,
) -> str:
    """Создаёт задачу google/nano-banana-edit (до 4 изображений)."""
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
        }
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
    url = _join(KIE_BASE_URL, "/api/v1/jobs/recordInfo")
    r = requests.get(url, headers=_auth_header(), params={"taskId": task_id}, timeout=timeout)
    try:
        j = r.json()
    except Exception:
        j = {"error": r.text}
    if r.status_code != 200:
        raise KieBananaError(f"recordInfo http {r.status_code}: {j}")
    return j

def _coerce_url_list(value) -> List[str]:
    urls: List[str] = []
    def add(u: str):
        if isinstance(u, str):
            s = u.strip()
            if s.startswith("http"):
                urls.append(s)
    if not value:
        return urls
    if isinstance(value, str):
        try:
            arr = json.loads(value) if value.strip().startswith("[") else None
        except Exception:
            arr = None
        if arr and isinstance(arr, list):
            for v in arr:
                if isinstance(v, str): add(v)
            return urls
        add(value); return urls
    if isinstance(value, list):
        for v in value:
            if isinstance(v, str): add(v)
            elif isinstance(v, dict):
                u = v.get("resultUrl") or v.get("originUrl") or v.get("url")
                if isinstance(u, str): add(u)
        return urls
    if isinstance(value, dict):
        for k in ("resultUrl", "originUrl", "url"):
            u = value.get(k)
            if isinstance(u, str): add(u)
        return urls
    return urls

def parse_banana_result_urls(record_json: Dict[str, Any]) -> Tuple[Optional[List[str]], Optional[str]]:
    data = record_json.get("data") or {}
    state = data.get("state")
    rj = data.get("resultJson")
    if not rj:
        return None, state
    try:
        parsed = json.loads(rj) if isinstance(rj, str) else (rj or {})
    except Exception:
        logging.exception("banana: failed to parse resultJson")
        return None, state
    urls = _coerce_url_list(parsed.get("resultUrls") or parsed.get("urls") or [])
    return (urls if urls else None), state

def wait_for_banana_result(task_id: str, timeout_sec: int = 300, poll_sec: int = 3) -> List[str]:
    """Ожидаем завершения и возвращаем список URL."""
    deadline = time.time() + timeout_sec
    last_state = None
    while time.time() < deadline:
        j = get_banana_record(task_id)
        code = j.get("code", 200)
        data = j.get("data") or {}
        state = data.get("state")
        last_state = state or last_state

        if code != 200:
            time.sleep(poll_sec); continue

        if state in (None, "waiting", "queuing", "generating"):
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
