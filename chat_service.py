from __future__ import annotations

import json
import os
import random
import time
from typing import Dict, List, Optional

try:  # pragma: no cover - optional redis factory
    from redis_utils import rds as _redis_instance  # type: ignore
except Exception:  # pragma: no cover - redis optional
    _redis_instance = None

CHAT_PREFIX = os.getenv("REDIS_PREFIX", "veo3:prod")
CTX_KEY = CHAT_PREFIX + ":chat:ctx:{user_id}"
MODE_KEY = CHAT_PREFIX + ":chat:mode:{user_id}"
RATE_KEY = CHAT_PREFIX + ":chat:rate:{user_id}"

CTX_TTL_SEC = 48 * 3600
CTX_MAX_PAIRS = 12
CTX_MAX_TOKENS = 2048
INPUT_MAX_CHARS = 3000

_memory: Dict[str, Dict[str, object]] = {
    "ctx": {},
    "mode": {},
    "rate": {},
}


def _now() -> int:
    return int(time.time())


def estimate_tokens(text: str) -> int:
    text = text or ""
    return max(1, len(text) // 4)


def _get_redis():
    return _redis_instance


def append_ctx(user_id: int, role: str, content: str) -> None:
    item = {"role": role, "content": content, "ts": _now()}
    key = CTX_KEY.format(user_id=user_id)
    payload = json.dumps(item, ensure_ascii=False)
    redis = _get_redis()
    if redis:
        try:
            redis.rpush(key, payload)
            redis.expire(key, CTX_TTL_SEC)
            return
        except Exception:
            pass
    ctx_store = _memory.setdefault("ctx", {})
    existing = ctx_store.get(key)
    if not isinstance(existing, list):
        existing = []
        ctx_store[key] = existing
    existing.append(item)


def load_ctx(
    user_id: int,
    max_pairs: int = CTX_MAX_PAIRS,
    max_tokens: int = CTX_MAX_TOKENS,
) -> List[Dict[str, object]]:
    key = CTX_KEY.format(user_id=user_id)
    redis = _get_redis()
    raw_items: List[Dict[str, object]] = []
    if redis:
        try:
            entries = redis.lrange(key, 0, -1) or []
        except Exception:
            entries = []
        for entry in entries:
            if isinstance(entry, bytes):
                entry = entry.decode("utf-8")
            try:
                parsed = json.loads(entry)
            except Exception:
                continue
            if isinstance(parsed, dict):
                raw_items.append(parsed)
    else:
        ctx_store = _memory.setdefault("ctx", {})
        stored = ctx_store.get(key)
        if isinstance(stored, list):
            raw_items = list(stored)
        else:
            raw_items = []

    trimmed: List[Dict[str, object]] = []
    tokens = 0
    pairs = 0
    for item in reversed(raw_items):
        content = str(item.get("content", ""))
        token_add = estimate_tokens(content)
        if tokens + token_add > max_tokens:
            break
        trimmed.append(item)
        tokens += token_add
        role = item.get("role")
        if role in ("user", "assistant"):
            pairs = sum(1 for it in trimmed if it.get("role") in ("user", "assistant")) // 2
            if pairs >= max_pairs:
                break
    return list(reversed(trimmed))


def clear_ctx(user_id: int) -> None:
    key = CTX_KEY.format(user_id=user_id)
    redis = _get_redis()
    if redis:
        try:
            redis.delete(key)
            return
        except Exception:
            pass
    ctx_store = _memory.setdefault("ctx", {})
    ctx_store.pop(key, None)


def set_mode(user_id: int, on: bool) -> None:
    key = MODE_KEY.format(user_id=user_id)
    redis = _get_redis()
    if redis:
        try:
            if on:
                redis.setex(key, CTX_TTL_SEC, "on")
            else:
                redis.delete(key)
            return
        except Exception:
            pass
    mode_store = _memory.setdefault("mode", {})
    if on:
        mode_store[key] = {"value": "on", "exp": _now() + CTX_TTL_SEC}
    else:
        mode_store.pop(key, None)


def is_mode_on(user_id: int) -> bool:
    key = MODE_KEY.format(user_id=user_id)
    redis = _get_redis()
    if redis:
        try:
            value = redis.get(key)
        except Exception:
            value = None
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return value == "on"
    mode_store = _memory.setdefault("mode", {})
    entry = mode_store.get(key)
    if not isinstance(entry, dict):
        return False
    exp = entry.get("exp")
    if isinstance(exp, (int, float)) and exp >= _now():
        return True
    mode_store.pop(key, None)
    return False


def rate_limit_hit(user_id: int) -> bool:
    key = RATE_KEY.format(user_id=user_id)
    redis = _get_redis()
    if redis:
        try:
            created = redis.setnx(key, "1")
            if created:
                redis.expire(key, 1)
                return False
            return True
        except Exception:
            pass
    rate_store = _memory.setdefault("rate", {})
    expires_at = rate_store.get(key, 0)
    now = _now()
    if now < expires_at:
        return True
    rate_store[key] = now + 1
    return False


def build_messages(
    system_prompt: str,
    history: List[Dict[str, object]],
    user_text: str,
    answer_lang: str,
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": f"{system_prompt}\nЯзык ответа: {answer_lang}.",
        }
    ]
    for item in history:
        role = str(item.get("role", ""))
        content = str(item.get("content", ""))
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_text})
    return messages


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
LLM_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

try:  # pragma: no cover - optional dependency
    import requests
except Exception:  # pragma: no cover - fallback for type checkers
    requests = None  # type: ignore


def call_llm(
    messages: List[Dict[str, str]],
    temperature: float = 0.6,
    max_output_tokens: int = 800,
    timeout_sec: float = 20.0,
) -> str:
    if not requests:
        raise RuntimeError("requests library is required for call_llm")

    url = f"{OPENAI_API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_output_tokens,
    }

    backoffs = [1.0 + random.random() * 1.5 for _ in range(3)]
    for attempt in range(3):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=timeout_sec,
            )
        except Exception as exc:
            if attempt < 2:
                time.sleep(backoffs[attempt])
                continue
            raise RuntimeError(f"llm request failed: {exc}") from exc

        if response.status_code == 200:
            data = response.json()
            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError("llm response missing choices")
            message = choices[0].get("message") or {}
            content = message.get("content")
            if not isinstance(content, str):
                raise RuntimeError("llm response invalid content")
            return content.strip()

        if response.status_code in {429, 500, 502, 503, 504} and attempt < 2:
            time.sleep(backoffs[attempt])
            continue
        text = response.text[:400] if response.text else ""
        raise RuntimeError(f"llm http {response.status_code}: {text}")

    raise RuntimeError("llm request exhausted retries")


__all__ = [
    "append_ctx",
    "build_messages",
    "call_llm",
    "clear_ctx",
    "estimate_tokens",
    "is_mode_on",
    "load_ctx",
    "rate_limit_hit",
    "set_mode",
    "CTX_MAX_PAIRS",
    "CTX_MAX_TOKENS",
    "INPUT_MAX_CHARS",
]
