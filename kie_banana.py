# -*- coding: utf-8 -*-
"""
kie_banana.py — KIE Nano Banana wrapper
Версия: 2025-09-12

✅ Что умеет
- Создаёт задачу для моделей Nano Banana:
  • google/nano-banana
  • google/nano-banana-edit
  • nano-banana-upscale
- Поддерживает до 4 изображений одновременно (объединение).
- Жёстко требует текст-промпт (не пустой).
- Пуллит статус до готовности и возвращает список URL результатов.
- Сохраняет совместимость с вызовами двух типов:
  1) create_banana_task(...) + wait_for_banana_result(...)
  2) submit_banana_job(api_key, model, images_bytes, prompt) -> (job_id, bytes|None)

⚙️ ENV:
- KIE_BASE_URL  (по умолчанию https://api.kie.ai)
- KIE_API_KEY   (Bearer ...)

🧪 Пример использования (из bot.py):
    from kie_banana import create_banana_task, wait_for_banana_result, KieBananaError
    task_id = create_banana_task(prompt="make it neon", image_urls=["https://.../1.jpg", "https://.../2.jpg"])
    urls = wait_for_banana_result(task_id)

Или совместимый короткий путь:
    from kie_banana import submit_banana_job
    job_id, maybe_bytes = await submit_banana_job(api_key=KIE_API_KEY, model="google/nano-banana", images_bytes=[b'...'], prompt="...")
"""

from __future__ import annotations

import os
import io
import json
import time
import base64
import logging
from typing import List, Dict, Any, Optional, Tuple

import requests

# ----------------------
# Конфиг и константы
# ----------------------

log = logging.getLogger("kie-banana")

DEFAULT_BASE_URL = "https://api.kie.ai"
KIE_BASE_URL = os.getenv("KIE_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
KIE_API_KEY = os.getenv("KIE_API_KEY", "").strip()

MODEL_BANANA_DEFAULT = os.getenv("KIE_MODEL", "google/nano-banana").strip()
VALID_MODELS = {
    "google/nano-banana",
    "google/nano-banana-edit",
    "nano-banana-upscale",
}

CREATE_PATH = "/api/v1/jobs/create"
RECORD_INFO_PATH = "/api/v1/jobs/recordInfo"

# ----------------------
# Исключения
# ----------------------

class KieBananaError(Exception):
    pass

# ----------------------
# Утилиты HTTP
# ----------------------

def _headers(api_key: Optional[str] = None) -> Dict[str, str]:
    key = (api_key or KIE_API_KEY or "").strip()
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers

def _post_json(path: str, payload: Dict[str, Any], api_key: Optional[str] = None, timeout: int = 60) -> Dict[str, Any]:
    url = f"{KIE_BASE_URL}{path}"
    t0 = time.time()
    resp = requests.post(url, headers=_headers(api_key), data=json.dumps(payload), timeout=timeout)
    elapsed = int((time.time() - t0) * 1000)
    try:
        resp.raise_for_status()
    except Exception as e:
        log.error("POST %s %s failed: %s (elapsed=%sms) payload=%s", url, resp.status_code, e, elapsed, _safe_preview(payload))
        raise
    data = resp.json()
    log.info("POST %s -> %s (elapsed=%sms)", url, resp.status_code, elapsed)
    return data

def _get_json(path: str, params: Dict[str, Any], api_key: Optional[str] = None, timeout: int = 60) -> Dict[str, Any]:
    url = f"{KIE_BASE_URL}{path}"
    t0 = time.time()
    resp = requests.get(url, headers=_headers(api_key), params=params, timeout=timeout)
    elapsed = int((time.time() - t0) * 1000)
    try:
        resp.raise_for_status()
    except Exception as e:
        log.error("GET %s %s failed: %s (elapsed=%sms) params=%s", url, resp.status_code, e, elapsed, params)
        raise
    data = resp.json()
    log.info("GET %s -> %s (elapsed=%sms)", url, resp.status_code, elapsed)
    return data

def _safe_preview(obj: Any, limit: int = 400) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False)[:limit]
        return s + ("…" if len(s) >= limit else "")
    except Exception:
        return str(obj)[:limit]

# ----------------------
# Валидация и нормализация
# ----------------------

def _ensure_model(model: Optional[str]) -> str:
    m = (model or MODEL_BANANA_DEFAULT).strip()
    if m not in VALID_MODELS:
        # пусть всё же пройдёт, но логнем
        log.warning("Unknown Nano Banana model '%s'. Allowed: %s", m, ", ".join(sorted(VALID_MODELS)))
    return m

def _validate_prompt(prompt: Optional[str]) -> str:
    if not prompt or not str(prompt).strip():
        raise KieBananaError("Prompt must not be empty")
    return str(prompt).strip()

def _validate_image_urls(image_urls: Optional[List[str]]) -> List[str]:
    if not image_urls or not isinstance(image_urls, list):
        raise KieBananaError("At least one image is required")
    urls = [u for u in (x.strip() for x in image_urls) if u]
    if not urls:
        raise KieBananaError("At least one non-empty image URL is required")
    if len(urls) > 4:
        log.warning("Received %d image urls; trimming to 4", len(urls))
        urls = urls[:4]
    for u in urls:
        if not (u.startswith("http://") or u.startswith("https://") or u.startswith("data:image/")):
            raise KieBananaError(f"Invalid image url: {u}")
    return urls

# ----------------------
# Основные функции
# ----------------------

def create_banana_task(
    prompt: str,
    image_urls: List[str],
    model: Optional[str] = None,
    output_format: str = "jpg",
    image_size: Optional[str] = None,
    callback_url: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
) -> str:
    """
    Создаёт задачу Nano Banana. Возвращает taskId (str).
    Не более 4 изображений.
    """
    m = _ensure_model(model)
    p = _validate_prompt(prompt)
    urls = _validate_image_urls(image_urls)

    payload: Dict[str, Any] = {
        "model": m,
        "input": {
            "prompt": p,
            "image_urls": urls,
            "output_format": output_format,
        },
    }
    if image_size:
        payload["input"]["image_size"] = image_size
    if extra:
        payload["input"]["extra"] = extra
    if callback_url:
        payload["callBackUrl"] = callback_url  # по спецификации KIE

    data = _post_json(CREATE_PATH, payload, api_key=api_key)
    # ожидаем структуру {"data": {"taskId": "..."}}
    try:
        task_id = data["data"]["taskId"]
    except Exception:
        raise KieBananaError(f"Unexpected create response: {data}")
    log.info("Nano Banana task created: %s", task_id)
    return task_id

def parse_banana_result_urls(record_info_json: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
    """
    Достаёт resultUrls из ответа recordInfo. Возвращает (urls, meta).
    """
    try:
        data = record_info_json["data"]
        result = data.get("result", {})
        urls = result.get("resultUrls") or result.get("images") or []
        if isinstance(urls, list):
            urls = [u for u in urls if isinstance(u, str) and u.strip()]
        else:
            urls = []
        return urls, result
    except Exception:
        return [], {}

def wait_for_banana_result(task_id: str, timeout_sec: int = 300, poll_sec: int = 3, api_key: Optional[str] = None) -> List[str]:
    """
    Циклически опрашивает /recordInfo пока задача не завершится, и возвращает список URL-ов результата.
    """
    if not task_id:
        raise KieBananaError("task_id is empty")
    deadline = time.time() + max(5, timeout_sec)
    last_state = None

    while time.time() < deadline:
        data = _get_json(RECORD_INFO_PATH, {"taskId": task_id}, api_key=api_key)
        try:
            state = data["data"].get("state")
        except Exception:
            state = None

        if state != last_state:
            log.info("Nano Banana %s state => %s", task_id, state)
            last_state = state

        if state in (None, "waiting", "queuing", "generating", "processing"):
            time.sleep(poll_sec)
            continue

        if state == "success":
            urls, meta = parse_banana_result_urls(data)
            if not urls:
                raise KieBananaError(f"success without result urls: {data}")
            return urls

        if state == "fail":
            code = (data.get("data") or {}).get("failCode")
            msg = (data.get("data") or {}).get("failMsg")
            raise KieBananaError(f"banana fail: code={code} msg={msg}")

        time.sleep(poll_sec)

    raise KieBananaError(f"banana timeout; last_state={last_state}")

# ----------------------
# Обратная совместимость: bytes-вызов
# ----------------------

def _to_data_url(img: bytes) -> str:
    """Преобразует байты изображения в data: URL (если у вас нет хостинга для картинок)."""
    b64 = base64.b64encode(img).decode("ascii")
    # оставим generic 'image/jpeg' — большинство сервисов примут
    return f"data:image/jpeg;base64,{b64}"

async def submit_banana_job(
    api_key: str,
    model: str,
    images_bytes: List[bytes],
    prompt: str,
    output_format: str = "jpg",
    image_size: Optional[str] = None,
) -> Tuple[str, Optional[bytes]]:
    """
    Совместимый асинхронный метод (как в прежнем варианте).
    - Конвертирует bytes -> data URLs (до 4 штук).
    - Создаёт задачу и сразу пробует дождаться результата (до 5 минут).
    - Возвращает (job_id, bytes|None). Если сервис отдаёт только URL, bytes будет None.
    """
    if not images_bytes:
        raise KieBananaError("images_bytes is empty")

    urls = [_to_data_url(b) for b in images_bytes[:4]]
    task_id = create_banana_task(
        prompt=prompt,
        image_urls=urls,
        model=model,
        output_format=output_format,
        image_size=image_size,
        api_key=api_key,
    )

    try:
        urls = wait_for_banana_result(task_id, timeout_sec=300, poll_sec=3, api_key=api_key)
    except Exception as e:
        # Отдадим job_id чтобы можно было дополучить результат позже
        log.error("submit_banana_job wait failed: %s", e)
        return task_id, None

    # Если получили URL — вернём None (байты можно скачать в bot.py при желании)
    # Либо можем попытаться скачать первый URL:
    try:
        if urls:
            r = requests.get(urls[0], timeout=60)
            r.raise_for_status()
            return task_id, r.content
    except Exception as e:
        log.warning("Could not download result by URL (%s): %s", urls[0] if urls else None, e)

    return task_id, None
