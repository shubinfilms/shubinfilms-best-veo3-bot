\# -*- coding: utf-8 -*-
# kie_banana.py — KIE wrapper for google/nano-banana-edit (multi-image up to 4)
# Версия: 2025-09-12 (stable)

from __future__ import annotations
import os
import json
import time
import logging
from typing import Dict, Any, List, Optional, Tuple

import requests

KIE_BASE_URL = os.getenv("KIE_BASE_URL", "https://api.kie.ai").strip()
KIE_API_KEY  = os.getenv("KIE_API_KEY", "").strip()
MODEL_BANANA = os.getenv("KIE_BANANA_MODEL", "google/nano-banana-edit").strip() or "google/nano-banana-edit"

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

def _req_json_post(url: str, payload: Dict[str, Any], timeout: int = 60) -> Tuple[int, Dict[str, Any]]:
    r = requests.post(url, headers=_headers_json(), data=json.dumps(payload), timeout=timeout)
    try:
        j = r.json()
    except Exception:
        j = {"error": r.text}
    return r.status_code, j

def _req_json_get(url: str, params: Dict[str, Any], timeout: int = 60) -> Tuple[int, Dict[str, Any]]:
    r = requests.get(url, headers=_auth_header(), params=params, timeout=timeout)
    try:
        j = r.json()
    except Exception:
        j = {"error": r.text}
    return r.status_code, j

def _extract_task_id(j: Dict[str, Any]) -> Optional[str]:
    data = j.get("data") or {}
    for k in ("taskId", "taskid", "id"):
        if j.get(k): return str(j[k])
        if data.get(k): return str(data[k])
    return None

def _coerce_url_list(value) -> List[str]:
    urls: List[str] = []
    def add(u: str):
        if isinstance(u, str):
            s = u.strip()
            if s.startswith("http"):
                urls.append(s)
    if not value: return urls
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("["):
            try:
                arr = json.loads(s)
                for v in arr:
                    if isinstance(v, str): add(v)
                return urls
            except Exception:
                add(s); return urls
        else:
            add(s); return urls
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

# ---------- Public API ----------

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
    Создаёт задачу nano-banana-edit. Поддерживает до 4 image_urls.
    Пробует несколько совместимых роутов и форматов.
    """
    if not KIE_API_KEY:
        raise KieBananaError("KIE_API_KEY is missing")

    imgs = [u for u in (image_urls or []) if isinstance(u, str) and u.startswith("http")]
    if not imgs:
        raise KieBananaError("image_urls is empty")

    # Кандидаты: разные эндпоинты + snake/camel кейсы
    inputs = [
        {"prompt": prompt or "", "image_urls": imgs[:4], "output_format": output_format, "image_size": image_size},
        {"prompt": prompt or "", "imageUrls": imgs[:4], "outputFormat": output_format, "imageSize": image_size},
    ]
    if extra_input:
        for i in inputs:
            i.update(extra_input)

    routes = [
        "/api/v1/jobs/generate",
        "/api/v1/jobs/createTask",
        "/api/v1/banana/generate",
        "/api/v1/banana/edit",
        "/api/v1/jobs/create",
    ]

    last_err = None
    for inp in inputs:
        payload = {"model": MODEL_BANANA, "input": inp}
        if callback_url:
            payload["callBackUrl"] = callback_url
        for r in routes:
            url = _join(KIE_BASE_URL, r)
            status, j = _req_json_post(url, payload, timeout=timeout)
            code = j.get("code", status)
            if status == 200 and code == 200:
                tid = _extract_task_id(j)
                if tid:
                    return tid
            last_err = f"route={r}, status={status}, resp={j}"

    raise KieBananaError(f"create task failed: {last_err}")

def get_banana_record(task_id: str, timeout: int = 60) -> Dict[str, Any]:
    # пробуем несколько роутов
    for r in ("/api/v1/jobs/recordInfo", "/api/v1/banana/record-info"):
        url = _join(KIE_BASE_URL, r)
        status, j = _req_json_get(url, {"taskId": task_id}, timeout=timeout)
        if status == 200:
            return j
    raise KieBananaError("recordInfo: no 200 responses")

def parse_banana_result_urls(record_json: Dict[str, Any]) -> Tuple[Optional[List[str]], Optional[str]]:
    """
    Возвращает (urls, state). urls=None — результата ещё нет.
    Поддерживает: data.resultJson (str/json), data.resultInfoJson, data.resultUrls, info/result/response…
    """
    data = record_json.get("data") or {}
    state = data.get("state") or data.get("successFlag")
    # прямые поля
    for k in ("resultUrls", "originUrls", "videoUrls"):
        if k in data:
            urls = _coerce_url_list(data.get(k))
            if urls:
                return urls, str(state) if state is not None else None

    # вложенные контейнеры
    for cont in ("resultJson", "resultInfoJson", "info", "response"):
        v = data.get(cont)
        if not v:
            continue
        try:
            if isinstance(v, str):
                v = json.loads(v)
        except Exception:
            pass
        if isinstance(v, dict):
            for k in ("resultUrls", "originUrls", "videoUrls", "urls"):
                urls = _coerce_url_list(v.get(k))
                if urls:
                    return urls, str(state) if state is not None else None

    return None, str(state) if state is not None else None

def wait_for_banana_result(task_id: str, timeout_sec: int = 300, poll_sec: int = 3) -> List[str]:
    """Ждём success/fail/таймаут. Возвращаем список URL'ов изображений."""
    deadline = time.time() + timeout_sec
    last_state = None
    while time.time() < deadline:
        j = get_banana_record(task_id)
        code = j.get("code", 200)
        data = j.get("data") or {}
        state = data.get("state") or data.get("successFlag")
        last_state = state or last_state

        if code != 200:
            logging.warning("banana record non-200: %s", j)
            time.sleep(poll_sec)
            continue

        # successFlag может быть 0/1/2/3 — тоже учитываем
        if state in ("waiting", "queuing", "generating", None) or str(state) == "0":
            time.sleep(poll_sec); continue

        if state in ("success", "1"):
            urls, _ = parse_banana_result_urls(j)
            if not urls:
                raise KieBananaError(f"success without resultUrls: {j}")
            return urls

        if state in ("fail", "2", "3"):
            raise KieBananaError(f"banana fail: code={data.get('failCode')} msg={data.get('failMsg')}")

        time.sleep(poll_sec)

    raise KieBananaError(f"banana timeout; last_state={last_state}")
