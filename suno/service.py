"""High level Suno service shared between the bot worker and web callback."""
from __future__ import annotations

import json
import difflib
import logging
import os
import time
from collections import OrderedDict
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, Dict, Iterable, Literal, Mapping, MutableMapping, Optional
from urllib.parse import urlparse, quote_plus

import requests
from requests.adapters import HTTPAdapter

from metrics import bot_telegram_send_fail_total, suno_requests_total, suno_task_store_total
from suno.client import (
    AMBIENT_NATURE_PRESET_ID,
    SunoClient,
    SunoAPIError,
    get_preset_config,
)
from suno.schemas import ApiEnvelope, CallbackEnvelope, SunoTask, SunoTrack
from suno.tempfiles import cleanup_old_directories, schedule_unlink, task_directory
from settings import (
    HTTP_POOL_CONNECTIONS,
    HTTP_POOL_PER_HOST,
    REDIS_PREFIX,
    SUNO_API_BASE,
    SUNO_API_TOKEN,
    SUNO_CALLBACK_SECRET,
    SUNO_CALLBACK_URL,
    SUNO_ENABLED,
)
from telegram_utils import mask_tokens
from utils.audio_post import prepare_audio_file_sync

try:  # pragma: no cover - optional runtime dependency
    from redis import Redis
except Exception:  # pragma: no cover - library may be unavailable
    Redis = None  # type: ignore

try:
    from redis_utils import rds as _redis_instance
except Exception:  # pragma: no cover - optional import
    _redis_instance = None

log = logging.getLogger("suno.service")

API_BASE = SUNO_API_BASE
API_KEY = SUNO_API_TOKEN
CALLBACK_URL = SUNO_CALLBACK_URL


_POLL_DEFAULT_TIMEOUT = 420.0
_POLL_DELIVERED_TTL = 15 * 60


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    text = raw.strip().lower()
    if not text:
        return default
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


_STRICT_LYRICS_ENABLED = _env_bool("SUNO_STRICT_LYRICS_ENABLED", True)
_STRICT_LYRICS_THRESHOLD = max(0.0, min(1.0, _env_float("SUNO_LYRICS_RETRY_THRESHOLD", 0.75)))
_STRICT_LYRICS_TEMPERATURE = max(0.0, _env_float("SUNO_LYRICS_STRICT_TEMPERATURE", 0.3))


def _parse_backoff_series(raw: str) -> list[float]:
    parts = (raw or "").split(",")
    values: list[float] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        try:
            value = float(part)
        except ValueError:
            continue
        if value > 0:
            values.append(value)
    return values


@dataclass(slots=True)
class RecordInfoPollResult:
    state: Literal["pending", "ready", "hard_error", "retry", "timeout", "delivered"]
    status_code: int
    payload: Mapping[str, Any]
    attempts: int = 0
    elapsed: float = 0.0
    message: Optional[str] = None
    error: Optional[str] = None


class SunoError(Exception):
    """Raised when the lightweight Suno API reports an error."""


def _check(env: ApiEnvelope, resp: requests.Response) -> ApiEnvelope:
    """Validate the response envelope and raise ``SunoError`` on failure."""

    if resp.ok and env.code == 200:
        return env
    raise SunoError(env.msg or resp.text)


def _make_headers() -> Dict[str, str]:
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    return headers


def _post(path: str, payload: Dict[str, Any]) -> ApiEnvelope:
    response = requests.post(
        f"{API_BASE}{path}",
        headers=_make_headers(),
        json=payload,
        timeout=20,
    )
    envelope = ApiEnvelope.model_validate(response.json())
    return _check(envelope, response)


def _get(path: str, params: Dict[str, Any]) -> ApiEnvelope:
    response = requests.get(
        f"{API_BASE}{path}",
        headers={"Authorization": f"Bearer {API_KEY}"} if API_KEY else None,
        params=params,
        timeout=15,
    )
    envelope = ApiEnvelope.model_validate(response.json())
    return _check(envelope, response)


def _extract_task_id(data: Any) -> str:
    if isinstance(data, Mapping) and data.get("taskId"):
        return str(data["taskId"])
    if hasattr(data, "taskId") and getattr(data, "taskId"):
        return str(getattr(data, "taskId"))
    raise SunoError("Missing taskId in response")


def suno_generate(
    *,
    prompt: str,
    model: Literal["V3_5", "V4", "V4_5", "V4_5PLUS", "V5"] = "V5",
    customMode: bool = True,
    instrumental: bool = False,
    title: Optional[str] = None,
    style: Optional[str] = None,
    negativeTags: Optional[str] = None,
    vocalGender: Optional[Literal["m", "f"]] = None,
    styleWeight: Optional[float] = None,
    weirdnessConstraint: Optional[float] = None,
    audioWeight: Optional[float] = None,
    callback_url: Optional[str] = None,
) -> str:
    payload: Dict[str, Any] = {
        "prompt": prompt,
        "model": model,
        "customMode": customMode,
        "instrumental": instrumental,
        "callBackUrl": callback_url or CALLBACK_URL,
    }
    if title:
        payload["title"] = title
    if style:
        payload["style"] = style
    if negativeTags:
        payload["negativeTags"] = negativeTags
    if vocalGender:
        payload["vocalGender"] = vocalGender
    if styleWeight is not None:
        payload["styleWeight"] = styleWeight
    if weirdnessConstraint is not None:
        payload["weirdnessConstraint"] = weirdnessConstraint
    if audioWeight is not None:
        payload["audioWeight"] = audioWeight

    envelope = _post("/api/v1/generate", payload)
    return _extract_task_id(envelope.data)


def suno_add_instrumental(
    *,
    uploadUrl: str,
    model: str = "V4_5PLUS",
    title: Optional[str] = None,
    tags: Optional[str] = None,
    negativeTags: Optional[str] = None,
) -> str:
    payload: Dict[str, Any] = {"uploadUrl": uploadUrl, "model": model}
    if title:
        payload["title"] = title
    if tags:
        payload["tags"] = tags
    if negativeTags:
        payload["negativeTags"] = negativeTags
    envelope = _post("/api/v1/generate/add-instrumental", payload)
    return _extract_task_id(envelope.data)


def suno_add_vocals(
    *,
    title: str,
    lyrics: str,
    model: str = "V5",
    has_lyrics: bool = True,
) -> str:
    payload = {
        "title": title,
        "lyrics": lyrics,
        "model": model,
        "has_lyrics": has_lyrics,
    }
    envelope = _post("/api/v1/generate/add-vocals", payload)
    return _extract_task_id(envelope.data)


def suno_record_info(task_id: str) -> Dict[str, Any]:
    envelope = _get("/api/v1/generate/record-info", {"taskId": task_id})
    data = envelope.data or {}
    if isinstance(data, Mapping):
        return dict(data)
    if hasattr(data, "model_dump"):
        return data.model_dump()
    return {"data": data}

_TASK_TTL = 24 * 60 * 60
_USER_LINK_TTL = 7 * 24 * 60 * 60
_REQ_TTL = 24 * 60 * 60
_LOG_ONCE_TTL = 48 * 60 * 60
_ENV = (os.getenv("APP_ENV") or "prod").strip() or "prod"


def _json_preview(payload: Any, *, limit: int = 700) -> str:
    """Return a trimmed JSON representation of ``payload`` for logs."""

    try:
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        text = str(payload)
    text = mask_tokens(text)
    if len(text) > limit:
        return f"{text[:limit]}…"
    return text


def _extract_enqueue_identifiers(payload: Mapping[str, Any]) -> tuple[Optional[str], Optional[str]]:
    """Return ``(request_id, task_id)`` discovered in ``payload``."""

    if not isinstance(payload, Mapping):
        return None, None
    seen: set[int] = set()
    stack: list[Mapping[str, Any]] = [payload]
    request_id: Optional[str] = None
    task_id: Optional[str] = None
    while stack:
        current = stack.pop()
        marker = id(current)
        if marker in seen:
            continue
        seen.add(marker)
        for key in ("req_id", "requestId", "request_id"):
            if request_id:
                break
            value = current.get(key)
            if value not in (None, ""):
                candidate = str(value).strip()
                if candidate:
                    request_id = candidate
        for key in ("task_id", "taskId", "id", "taskID", "job_id", "jobId"):
            if task_id:
                break
            value = current.get(key)
            if value not in (None, ""):
                candidate = str(value).strip()
                if candidate:
                    task_id = candidate
        for nested_key in ("data", "payload", "result"):
            nested = current.get(nested_key)
            if isinstance(nested, Mapping):
                stack.append(nested)
    return request_id, task_id


def _metric_labels(service: str) -> dict[str, str]:
    return {"env": _ENV, "service": service}


@dataclass(slots=True)
class TelegramMeta:
    chat_id: int
    msg_id: int
    title: Optional[str]
    ts: str
    req_id: Optional[str]
    user_title: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TaskLink:
    user_id: int
    prompt: str
    ts: str


class SunoService:
    """Facade that hides HTTP, Redis and Telegram plumbing."""

    def __init__(
        self,
        *,
        client: Optional[SunoClient] = None,
        redis: Optional[Redis] = None,
        telegram_token: Optional[str] = None,
    ) -> None:
        self.client = client or SunoClient()
        self.redis = redis or _redis_instance
        self.telegram_token = telegram_token or os.getenv("TELEGRAM_TOKEN")
        self._memory: MutableMapping[str, tuple[float, str]] = {}
        self._user_memory: MutableMapping[str, tuple[float, str]] = {}
        self._task_records_memory: MutableMapping[str, tuple[float, str]] = {}
        self._req_memory: MutableMapping[str, tuple[float, str]] = {}
        self._req_index_memory: MutableMapping[str, tuple[float, str]] = {}
        self._task_order: list[str] = []
        self._bot_session = requests.Session()
        adapter = HTTPAdapter(pool_connections=HTTP_POOL_CONNECTIONS, pool_maxsize=HTTP_POOL_PER_HOST)
        self._bot_session.mount("https://", adapter)
        self._bot_session.mount("http://", adapter)
        self._api_session = requests.Session()
        self._api_session.mount("https://", adapter)
        self._api_session.mount("http://", adapter)
        self._admin_ids = self._parse_admins(os.getenv("ADMIN_IDS"))
        self._log_once_memory: MutableMapping[str, float] = {}
        self._delivered_cache: MutableMapping[str, float] = {}
        self._delivery_seen: "OrderedDict[str, float]" = OrderedDict()
        raw_fallback = (os.getenv("TELEGRAM_DOWNLOAD_FALLBACK") or "true").strip().lower()
        self._telegram_download_fallback = raw_fallback not in {"0", "false", "no"}
        self._delivery_seen_limit = 200
        self._poll_first_delay = max(0.1, _env_float("SUNO_POLL_FIRST_DELAY_SEC", 5.0))
        backoff_raw = os.getenv("SUNO_POLL_BACKOFF_SERIES", "5,8,13,21,34") or ""
        parsed_backoff = _parse_backoff_series(backoff_raw)
        if parsed_backoff and abs(parsed_backoff[0] - self._poll_first_delay) < 1e-3:
            parsed_backoff = parsed_backoff[1:]
        self._poll_backoff_series = parsed_backoff or [8.0, 13.0, 21.0, 34.0]
        self._poll_timeout = max(
            self._poll_first_delay,
            _env_float("SUNO_POLL_TIMEOUT_SEC", _POLL_DEFAULT_TIMEOUT),
        )
        summary = {
            "suno_enabled": bool(SUNO_ENABLED),
            "api_base": SUNO_API_BASE,
            "callback_configured": bool(SUNO_CALLBACK_URL and SUNO_CALLBACK_SECRET),
        }
        log.info("configuration summary", extra={"meta": summary})
        cleanup_old_directories()

    # ------------------------------------------------------------------ storage
    def _redis_key(self, task_id: str) -> str:
        return f"{REDIS_PREFIX}:task:{task_id}"

    def _user_key(self, task_id: str) -> str:
        return f"{REDIS_PREFIX}:task-user:{task_id}"

    def _record_key(self, task_id: str) -> str:
        return f"{REDIS_PREFIX}:suno:record:{task_id}"

    def _last_tasks_key(self) -> str:
        return f"{REDIS_PREFIX}:suno:last"

    def _req_key(self, task_id: str) -> str:
        return f"{REDIS_PREFIX}:suno:req:{task_id}"

    def _req_index_key(self, req_id: str) -> str:
        return f"{REDIS_PREFIX}:suno:req-index:{req_id}"

    def _log_once_key(self, task_id: str, callback_type: Optional[str]) -> str:
        kind = (callback_type or "unknown").lower()
        return f"log:once:{task_id}:{kind}"

    def _should_log_once(self, task_id: str, callback_type: Optional[str]) -> bool:
        key = self._log_once_key(task_id, callback_type)
        if self.redis is not None:
            try:
                stored = self.redis.set(key, "1", nx=True, ex=_LOG_ONCE_TTL)
            except Exception:  # pragma: no cover - Redis failure
                log.warning("SunoService redis.set log-once failed", exc_info=True)
            else:
                if stored:
                    return True
                return False
        now = time.time()
        expires_at = now + _LOG_ONCE_TTL
        current = self._log_once_memory.get(key)
        if current and current > now:
            return False
        self._log_once_memory[key] = expires_at
        return True

    def _cleanup_delivered_cache(self) -> None:
        now = time.time()
        stale = [key for key, expires in self._delivered_cache.items() if expires <= now]
        for key in stale:
            del self._delivered_cache[key]

    def _mark_delivered(self, task_id: str) -> None:
        if not task_id:
            return
        self._cleanup_delivered_cache()
        self._delivered_cache[task_id] = time.time() + _POLL_DELIVERED_TTL

    def _recently_delivered(self, task_id: Optional[str]) -> bool:
        if not task_id:
            return False
        self._cleanup_delivered_cache()
        expires = self._delivered_cache.get(task_id)
        return bool(expires and expires > time.time())

    def _iter_poll_delays(self) -> Iterable[float]:
        base = [delay for delay in [self._poll_first_delay, *self._poll_backoff_series] if delay > 0]
        if not base:
            base = [5.0]
        index = 0
        while True:
            if index < len(base):
                yield base[index]
            else:
                yield base[-1]
            index += 1

    @staticmethod
    def _poll_section(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
        value = payload.get(key)
        if isinstance(value, Mapping):
            return value
        return {}

    @classmethod
    def _status_from_payload(cls, payload: Mapping[str, Any]) -> Optional[str]:
        for key in ("status", "taskStatus", "callbackType", "callback_type", "state"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().upper()
        data_section = cls._poll_section(payload, "data")
        for key in ("status", "taskStatus", "callbackType", "callback_type", "state"):
            value = data_section.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().upper()
        response_section = cls._poll_section(data_section, "response")
        for key in ("status", "taskStatus", "state"):
            value = response_section.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().upper()
        return None

    @classmethod
    def _error_code_from_payload(cls, payload: Mapping[str, Any]) -> Optional[str]:
        data_section = cls._poll_section(payload, "data")
        response_section = cls._poll_section(data_section, "response")
        for section in (payload, data_section, response_section):
            for key in ("errorCode", "error_code"):
                value = section.get(key)
                if value not in (None, ""):
                    return str(value)
        return None

    @classmethod
    def _tracks_from_payload(cls, payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        data_section = cls._poll_section(payload, "data")
        response_section = cls._poll_section(data_section, "response")
        tracks_candidate = response_section.get("sunoData") or response_section.get("tracks")
        if not isinstance(tracks_candidate, list):
            tracks_candidate = data_section.get("sunoData") or data_section.get("tracks")
        tracks: list[Mapping[str, Any]] = []
        if isinstance(tracks_candidate, list):
            for item in tracks_candidate:
                if isinstance(item, Mapping):
                    tracks.append(item)
        return tracks

    @staticmethod
    def _durations_from_tracks(tracks: Iterable[Mapping[str, Any]]) -> list[float]:
        durations: list[float] = []
        for track in tracks:
            value = track.get("duration")
            if isinstance(value, (int, float)):
                durations.append(float(value))
            elif isinstance(value, str):
                try:
                    durations.append(float(value))
                except ValueError:
                    continue
        return durations

    def poll_record_info_once(
        self,
        task_id: str,
        *,
        user_id: Optional[int] = None,
    ) -> RecordInfoPollResult:
        params: Dict[str, Any] = {"taskId": task_id}
        if user_id is not None:
            params["userId"] = str(user_id)
        headers: Dict[str, str] = {}
        if API_KEY:
            headers["Authorization"] = f"Bearer {API_KEY}"
        url = f"{API_BASE}/api/v1/generate/record-info"
        try:
            response = self._api_session.get(url, params=params, headers=headers or None, timeout=20)
        except requests.RequestException as exc:
            log.warning(
                "Suno record-info network error",
                extra={"meta": {"taskId": task_id, "error": str(exc)}},
            )
            return RecordInfoPollResult(state="retry", status_code=0, payload={}, error=str(exc))
        try:
            payload_raw = response.json()
        except ValueError:
            payload_raw = {}
        payload = payload_raw if isinstance(payload_raw, Mapping) else {}
        message = payload.get("message") or payload.get("msg") if isinstance(payload, Mapping) else None
        status_code = response.status_code
        if status_code == 404:
            return RecordInfoPollResult(state="pending", status_code=status_code, payload=payload, message=message)
        if status_code == 429 or 500 <= status_code < 600:
            return RecordInfoPollResult(state="retry", status_code=status_code, payload=payload, message=message)
        if 400 <= status_code < 500:
            return RecordInfoPollResult(state="hard_error", status_code=status_code, payload=payload, message=message)
        data_section = payload.get("data") if isinstance(payload, Mapping) else None
        if not data_section:
            return RecordInfoPollResult(state="pending", status_code=status_code, payload=payload, message=message)
        status_value = self._status_from_payload(payload)
        error_code = self._error_code_from_payload(payload)
        tracks = self._tracks_from_payload(payload)
        if error_code:
            return RecordInfoPollResult(
                state="hard_error",
                status_code=status_code,
                payload=payload,
                message=message,
                error=error_code,
            )
        success_states = {"SUCCESS", "SUCCEEDED", "COMPLETE", "COMPLETED", "READY"}
        failure_states = {"FAILED", "ERROR", "TIMEOUT", "CANCELLED", "CANCELED"}
        if tracks and (status_value is None or status_value in success_states):
            return RecordInfoPollResult(state="ready", status_code=status_code, payload=payload, message=message)
        if status_value in failure_states:
            return RecordInfoPollResult(state="hard_error", status_code=status_code, payload=payload, message=message)
        return RecordInfoPollResult(state="pending", status_code=status_code, payload=payload, message=message)

    def wait_for_record_info(
        self,
        task_id: str,
        *,
        user_id: Optional[int] = None,
    ) -> RecordInfoPollResult:
        start = time.monotonic()
        attempts = 0
        if self._recently_delivered(task_id):
            log.info(
                "Suno poll delivered via webhook",
                extra={"meta": {"taskId": task_id, "via": "webhook"}},
            )
            return RecordInfoPollResult(state="delivered", status_code=200, payload={}, attempts=0, elapsed=0.0)
        for delay in self._iter_poll_delays():
            if self._recently_delivered(task_id):
                log.info(
                    "Suno poll delivered via webhook",
                    extra={"meta": {"taskId": task_id, "via": "webhook"}},
                )
                return RecordInfoPollResult(
                    state="delivered",
                    status_code=200,
                    payload={},
                    attempts=attempts,
                    elapsed=time.monotonic() - start,
                )
            now = time.monotonic()
            if attempts > 0 and now - start >= self._poll_timeout:
                break
            time.sleep(delay)
            attempts += 1
            result = self.poll_record_info_once(task_id, user_id=user_id)
            result.attempts = attempts
            result.elapsed = time.monotonic() - start
            meta = {
                "taskId": task_id,
                "attempt": attempts,
                "http_status": result.status_code,
                "mapped_state": result.state,
            }
            if result.error:
                meta["error_code"] = result.error
            if result.state == "retry":
                log.warning("Suno poll retry", extra={"meta": meta})
            else:
                log.info("Suno poll step", extra={"meta": meta})
            if result.state == "ready":
                tracks = self._tracks_from_payload(result.payload)
                durations = self._durations_from_tracks(tracks)
                log.info(
                    "Suno poll ready",
                    extra={
                        "meta": {
                            "taskId": task_id,
                            "takes": len(tracks),
                            "durations": durations,
                            "http_status": result.status_code,
                        }
                    },
                )
                return result
            if result.state == "hard_error":
                log.error(
                    "Suno poll hard failure",
                    extra={"meta": {**meta, "message": result.message}},
                )
                return result
            if result.elapsed >= self._poll_timeout:
                break
        if self._recently_delivered(task_id):
            log.info(
                "Suno poll delivered via webhook",
                extra={"meta": {"taskId": task_id, "via": "webhook"}},
            )
            return RecordInfoPollResult(
                state="delivered",
                status_code=200,
                payload={},
                attempts=attempts,
                elapsed=time.monotonic() - start,
            )
        final_attempt = self.poll_record_info_once(task_id, user_id=user_id)
        final_attempt.attempts = attempts + 1
        final_attempt.elapsed = time.monotonic() - start
        if final_attempt.state == "ready":
            tracks = self._tracks_from_payload(final_attempt.payload)
            durations = self._durations_from_tracks(tracks)
            log.info(
                "Suno poll ready",
                extra={
                    "meta": {
                        "taskId": task_id,
                        "takes": len(tracks),
                        "durations": durations,
                        "http_status": final_attempt.status_code,
                    }
                },
            )
            return final_attempt
        if final_attempt.state == "hard_error":
            log.error(
                "Suno poll hard failure",
                extra={
                    "meta": {
                        "taskId": task_id,
                        "attempt": final_attempt.attempts,
                        "http_status": final_attempt.status_code,
                        "message": final_attempt.message,
                    }
                },
            )
            return final_attempt
        log.warning(
            "Suno poll timeout",
            extra={
                "meta": {
                    "taskId": task_id,
                    "attempts": final_attempt.attempts,
                    "elapsed": final_attempt.elapsed,
                }
            },
        )
        final_attempt.state = "timeout"
        return final_attempt

    @staticmethod
    def _parse_admins(raw: Optional[str]) -> set[int]:
        result: set[int] = set()
        if not raw:
            return result
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                result.add(int(part))
            except ValueError:
                log.warning("SunoService admin id invalid: %s", part)
        return result

    @staticmethod
    def _strict_normalize_text(value: Optional[str]) -> str:
        if not value:
            return ""
        text = re.sub(r"[\s]+", " ", value.strip().lower())
        text = re.sub(r"[^\w\s]", " ", text)
        return " ".join(text.split())

    @staticmethod
    def _strict_similarity(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        return difflib.SequenceMatcher(None, a, b).ratio()

    @staticmethod
    def _strict_extract_lyrics(payload: Mapping[str, Any]) -> Optional[str]:
        lines: list[str] = []

        def _collect(value: Any, hint: Optional[str] = None) -> None:
            if isinstance(value, str):
                text = value.strip()
                if text:
                    lines.append(text)
                return
            if isinstance(value, Mapping):
                for key, nested in value.items():
                    lowered = str(key).lower()
                    if lowered in {"lyrics", "text", "content", "words", "lines"}:
                        _collect(nested, lowered)
                    elif isinstance(nested, (Mapping, list, tuple, set)):
                        _collect(nested, lowered)
                return
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    _collect(item, hint)

        _collect(payload)
        if lines:
            unique_lines = []
            seen = set()
            for line in lines:
                if line not in seen:
                    seen.add(line)
                    unique_lines.append(line)
            return "\n".join(unique_lines)
        return None

    def _strict_fetch_lyrics(
        self,
        task_id: str,
        *,
        req_id: Optional[str],
        payload: Optional[Mapping[str, Any]] = None,
    ) -> Optional[str]:
        try:
            response = self.client.get_lyrics(task_id, req_id=req_id, payload=payload)
        except SunoAPIError as exc:
            log.warning(
                "Suno strict lyrics fetch failed",
                extra={"meta": {"task_id": task_id, "status": exc.status, "message": str(exc)}},
            )
            return None
        except Exception:
            log.warning("Suno strict lyrics fetch crashed", exc_info=True)
            return None
        if isinstance(response, Mapping):
            return self._strict_extract_lyrics(response)
        return None

    def _strict_context_from_meta(self, meta: Optional[TelegramMeta]) -> Optional[Dict[str, Any]]:
        if meta is None:
            return None
        extras = meta.extras or {}
        strict_info = extras.get("strict_lyrics")
        context: Dict[str, Any]
        if isinstance(strict_info, Mapping):
            context = dict(strict_info)
        else:
            context = {}
        for key in ("original", "payload", "threshold", "attempts"):
            if key not in context and key in extras and extras[key] is not None:
                context[key] = extras[key]
        if "enabled" not in context and "strict_enabled" in extras:
            context["enabled"] = extras["strict_enabled"]
        if "lyrics_source" not in context and extras.get("lyrics_source"):
            context["lyrics_source"] = extras["lyrics_source"]
        if "original" not in context and "original_lyrics" in extras and extras["original_lyrics"]:
            context["original"] = extras["original_lyrics"]
        if "payload" not in context and "strict_payload" in extras and extras["strict_payload"]:
            context["payload"] = extras["strict_payload"]
        if "threshold" not in context and "strict_threshold" in extras and extras["strict_threshold"] is not None:
            context["threshold"] = extras["strict_threshold"]
        if context:
            context.setdefault("attempts", 0)
        return context or None

    def _strict_retry_enqueue(
        self,
        *,
        task: SunoTask,
        meta: TelegramMeta,
        link: Optional[TaskLink],
        strict_context: Dict[str, Any],
        req_id: Optional[str],
        threshold: float,
    ) -> bool:
        payload = strict_context.get("payload")
        if not isinstance(payload, Mapping):
            return False
        retry_payload = dict(payload)
        current_temp = retry_payload.get("temperature")
        try:
            current_temp_value = float(current_temp) if current_temp is not None else None
        except (TypeError, ValueError):
            current_temp_value = None
        base_temp = current_temp_value if current_temp_value is not None else _STRICT_LYRICS_TEMPERATURE
        retry_payload["temperature"] = max(0.0, min(base_temp, _STRICT_LYRICS_TEMPERATURE) * 0.7)
        new_req_id = f"{req_id or task.task_id}-retry"
        try:
            new_task_id = self.client.enqueue(retry_payload, req_id=new_req_id)
        except SunoAPIError as exc:
            log.warning(
                "Suno strict retry enqueue failed",
                extra={"meta": {"task_id": task.task_id, "status": exc.status, "message": str(exc)}},
            )
            return False
        except Exception:
            log.warning("Suno strict retry enqueue crashed", exc_info=True)
            return False
        if not new_task_id:
            return False
        strict_meta = {
            "attempts": int(strict_context.get("attempts") or 0) + 1,
            "threshold": threshold,
            "original": strict_context.get("original"),
            "payload": retry_payload,
        }
        new_meta: Dict[str, Any] = {
            "chat_id": meta.chat_id,
            "msg_id": meta.msg_id,
            "title": meta.title,
            "ts": datetime.now(timezone.utc).isoformat(),
            "req_id": new_req_id,
            "user_title": meta.user_title,
            "strict_enabled": True,
            "strict_threshold": threshold,
            "strict_lyrics": strict_meta,
            "lyrics_source": meta.extras.get("lyrics_source"),
        }
        if strict_context.get("original"):
            new_meta["original_lyrics"] = strict_context.get("original")
        if strict_context.get("payload"):
            new_meta["strict_payload"] = retry_payload
        self._store_mapping(str(new_task_id), new_meta)
        self._store_req_id(str(new_task_id), new_req_id)
        if link is not None:
            self._store_user_link(
                str(new_task_id),
                {
                    "user_id": link.user_id,
                    "prompt": link.prompt,
                    "ts": datetime.now(timezone.utc).isoformat(),
                },
            )
        self._log_delivery(
            "suno.strict.retry.enqueued",
            task_id=task.task_id,
            retry_task_id=str(new_task_id),
            req_id=req_id,
            retry_req_id=new_req_id,
        )
        try:
            self._send_text(
                meta.chat_id,
                "🔁 Перезапускаем генерацию, чтобы сохранить ваш текст.",
                reply_to=meta.msg_id,
            )
        except Exception:
            log.debug("Suno strict retry notify failed", exc_info=True)
        return True

    def _process_strict_delivery(
        self,
        *,
        task: SunoTask,
        meta: TelegramMeta,
        link: Optional[TaskLink],
        strict_context: Dict[str, Any],
        req_id: Optional[str],
    ) -> Optional[Dict[str, Any] | str]:
        if not strict_context.get("enabled") or not _STRICT_LYRICS_ENABLED:
            return None
        original = str(strict_context.get("original") or "").strip()
        if not original:
            return None
        threshold_raw = strict_context.get("threshold")
        try:
            threshold = float(threshold_raw)
        except (TypeError, ValueError):
            threshold = _STRICT_LYRICS_THRESHOLD
        threshold = max(0.0, min(1.0, threshold))
        attempts = int(strict_context.get("attempts") or 0)
        lyrics_payload = strict_context.get("payload") if isinstance(strict_context.get("payload"), Mapping) else None
        generated_text = self._strict_fetch_lyrics(task.task_id, req_id=req_id, payload=lyrics_payload)
        normalized_original = self._strict_normalize_text(original)
        normalized_generated = self._strict_normalize_text(generated_text)
        similarity = self._strict_similarity(normalized_original, normalized_generated)
        strict_context["similarity"] = similarity
        source_label = str(strict_context.get("lyrics_source") or "").strip().lower()
        skip_retry = source_label == "user"
        if similarity >= threshold and generated_text:
            return {"lyrics": generated_text, "similarity": similarity, "attempts": attempts}
        if skip_retry:
            return {"lyrics": generated_text or "", "similarity": similarity, "attempts": attempts}
        if attempts < 1 and self._strict_retry_enqueue(
            task=task,
            meta=meta,
            link=link,
            strict_context=strict_context,
            req_id=req_id,
            threshold=threshold,
        ):
            return "retry"
        warning_text = None
        if generated_text:
            warning_text = "⚠️ Модель частично перефразировала текст. Ниже — фактические слова."
        return {
            "lyrics": generated_text or "",
            "warning": warning_text,
            "similarity": similarity,
            "attempts": attempts,
        }

    def _store_mapping(self, task_id: str, payload: Mapping[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False)
        key = self._redis_key(task_id)
        if self.redis is not None:
            try:
                self.redis.setex(key, _TASK_TTL, raw)
            except Exception:  # pragma: no cover - Redis failure
                log.warning("SunoService redis.setex failed", exc_info=True)
                suno_task_store_total.labels(result="redis_error").inc()
            else:
                suno_task_store_total.labels(result="redis").inc()
                return
        expires_at = time.time() + _TASK_TTL
        self._memory[key] = (expires_at, raw)
        suno_task_store_total.labels(result="memory").inc()

    def _store_user_link(self, task_id: str, payload: Mapping[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False)
        key = self._user_key(task_id)
        if self.redis is not None:
            try:
                self.redis.setex(key, _USER_LINK_TTL, raw)
            except Exception:  # pragma: no cover - Redis failure
                log.warning("SunoService redis.setex user-link failed", exc_info=True)
                suno_task_store_total.labels(result="redis_error").inc()
            else:
                suno_task_store_total.labels(result="redis").inc()
                return
        expires_at = time.time() + _USER_LINK_TTL
        self._user_memory[key] = (expires_at, raw)
        suno_task_store_total.labels(result="memory").inc()

    def _save_task_record(self, task_id: str, payload: Mapping[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False)
        key = self._record_key(task_id)
        if self.redis is not None:
            try:
                self.redis.setex(key, _USER_LINK_TTL, raw)
            except Exception:  # pragma: no cover - Redis failure
                log.warning("SunoService redis.setex record failed", exc_info=True)
                suno_task_store_total.labels(result="redis_error").inc()
            else:
                try:
                    self.redis.lrem(self._last_tasks_key(), 0, task_id)
                    self.redis.lpush(self._last_tasks_key(), task_id)
                    self.redis.ltrim(self._last_tasks_key(), 0, 49)
                except Exception:  # pragma: no cover
                    log.warning("SunoService redis.lpush record failed", exc_info=True)
                suno_task_store_total.labels(result="redis").inc()
                return
        expires_at = time.time() + _USER_LINK_TTL
        self._task_records_memory[key] = (expires_at, raw)
        suno_task_store_total.labels(result="memory").inc()
        if task_id in self._task_order:
            self._task_order.remove(task_id)
        self._task_order.insert(0, task_id)
        del self._task_order[50:]

    def _store_req_id(self, task_id: str, req_id: Optional[str]) -> None:
        if not task_id or not req_id:
            return
        key = self._req_key(task_id)
        req_index_key = self._req_index_key(req_id)
        if self.redis is not None:
            try:
                pipe = self.redis.pipeline()
                pipe.setex(key, _REQ_TTL, req_id)
                pipe.setex(req_index_key, _REQ_TTL, task_id)
                pipe.execute()
                return
            except Exception:  # pragma: no cover - Redis failure
                log.warning("SunoService redis.setex req-id failed", exc_info=True)
        expires_at = time.time() + _REQ_TTL
        self._req_memory[key] = (expires_at, req_id)
        self._req_index_memory[req_index_key] = (expires_at, task_id)

    def _load_mapping(self, task_id: str) -> Optional[TelegramMeta]:
        key = self._redis_key(task_id)
        raw: Optional[str] = None
        if self.redis is not None:
            try:
                value = self.redis.get(key)
                if isinstance(value, bytes):
                    raw = value.decode("utf-8")
                elif isinstance(value, str):
                    raw = value
            except Exception:  # pragma: no cover - Redis failure
                log.warning("SunoService redis.get failed", exc_info=True)
        if raw is None and key in self._memory:
            expires_at, value = self._memory[key]
            if expires_at > time.time():
                raw = value
            else:
                self._memory.pop(key, None)
        if raw is None:
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("SunoService failed to decode mapping for task %s", task_id)
            return None
        try:
            chat_id = int(data.get("chat_id"))
            msg_id = int(data.get("msg_id"))
        except (TypeError, ValueError):
            return None
        title = data.get("title")
        user_title = data.get("user_title")
        ts = data.get("ts") or datetime.now(timezone.utc).isoformat()
        req_id = data.get("req_id") or self._load_req_id(task_id)
        extras = {
            key: value
            for key, value in data.items()
            if key
            not in {
                "chat_id",
                "msg_id",
                "title",
                "ts",
                "req_id",
                "user_title",
            }
        }
        return TelegramMeta(
            chat_id=chat_id,
            msg_id=msg_id,
            title=title,
            ts=ts,
            req_id=req_id,
            user_title=user_title,
            extras=extras,
        )

    def _delete_mapping(self, task_id: str) -> None:
        key = self._redis_key(task_id)
        if self.redis is not None:
            try:
                self.redis.delete(key)
            except Exception:  # pragma: no cover - Redis failure
                log.warning("SunoService redis.delete failed", exc_info=True)
        self._memory.pop(key, None)
        req_key = self._req_key(task_id)
        req_id: Optional[str] = None
        if self.redis is not None:
            try:
                stored_req = self.redis.get(req_key)
                if isinstance(stored_req, bytes):
                    req_id = stored_req.decode("utf-8", errors="ignore")
                elif isinstance(stored_req, str):
                    req_id = stored_req
                self.redis.delete(req_key)
            except Exception:  # pragma: no cover - Redis failure
                log.warning("SunoService redis.delete req-id failed", exc_info=True)
        self._req_memory.pop(req_key, None)
        if req_id:
            index_key = self._req_index_key(req_id)
            if self.redis is not None:
                try:
                    self.redis.delete(index_key)
                except Exception:  # pragma: no cover - Redis failure
                    log.warning("SunoService redis.delete req-index failed", exc_info=True)
            self._req_index_memory.pop(index_key, None)

    def _load_req_id(self, task_id: str) -> Optional[str]:
        if not task_id:
            return None
        key = self._req_key(task_id)
        if self.redis is not None:
            try:
                value = self.redis.get(key)
            except Exception:  # pragma: no cover - Redis failure
                log.warning("SunoService redis.get req-id failed", exc_info=True)
            else:
                if isinstance(value, bytes):
                    return value.decode("utf-8", errors="ignore")
                if isinstance(value, str):
                    return value
        if key in self._req_memory:
            expires_at, value = self._req_memory[key]
            if expires_at > time.time():
                return value
            self._req_memory.pop(key, None)
        return None

    def get_request_id(self, task_id: str) -> Optional[str]:
        return self._load_req_id(task_id)

    def _load_task_id_for_req(self, req_id: str) -> Optional[str]:
        if not req_id:
            return None
        index_key = self._req_index_key(req_id)
        if self.redis is not None:
            try:
                value = self.redis.get(index_key)
            except Exception:  # pragma: no cover - Redis failure
                log.warning("SunoService redis.get req-index failed", exc_info=True)
            else:
                if isinstance(value, bytes):
                    return value.decode("utf-8", errors="ignore")
                if isinstance(value, str):
                    return value
        if index_key in self._req_index_memory:
            expires_at, value = self._req_index_memory[index_key]
            if expires_at > time.time():
                return value
            self._req_index_memory.pop(index_key, None)
        return None

    def get_task_id_by_request(self, req_id: str) -> Optional[str]:
        return self._load_task_id_for_req(req_id)

    def get_start_timestamp(self, task_id: str) -> Optional[str]:
        meta = self._load_mapping(task_id)
        if meta is None:
            return None
        return meta.ts

    def _load_user_link(self, task_id: str) -> Optional[TaskLink]:
        key = self._user_key(task_id)
        raw: Optional[str] = None
        if self.redis is not None:
            try:
                value = self.redis.get(key)
                if isinstance(value, bytes):
                    raw = value.decode("utf-8")
                elif isinstance(value, str):
                    raw = value
            except Exception:  # pragma: no cover - Redis failure
                log.warning("SunoService redis.get user-link failed", exc_info=True)
        if raw is None and key in self._user_memory:
            expires_at, value = self._user_memory[key]
            if expires_at > time.time():
                raw = value
            else:
                self._user_memory.pop(key, None)
        if raw is None:
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        try:
            user_id = int(data.get("user_id"))
        except (TypeError, ValueError):
            return None
        prompt = str(data.get("prompt") or "")
        ts = data.get("ts") or datetime.now(timezone.utc).isoformat()
        return TaskLink(user_id=user_id, prompt=prompt, ts=ts)

    def _delete_user_link(self, task_id: str) -> None:
        key = self._user_key(task_id)
        if self.redis is not None:
            try:
                self.redis.delete(key)
            except Exception:  # pragma: no cover - Redis failure
                log.warning("SunoService redis.delete user-link failed", exc_info=True)
        self._user_memory.pop(key, None)
        record_key = self._record_key(task_id)
        self._task_records_memory.pop(record_key, None)
        if task_id in self._task_order:
            self._task_order.remove(task_id)

    def _load_task_record(self, task_id: str) -> Optional[Dict[str, Any]]:
        key = self._record_key(task_id)
        raw: Optional[str] = None
        if self.redis is not None:
            try:
                value = self.redis.get(key)
                if isinstance(value, bytes):
                    raw = value.decode("utf-8")
                elif isinstance(value, str):
                    raw = value
            except Exception:
                log.warning("SunoService redis.get record failed", exc_info=True)
        if raw is None and key in self._task_records_memory:
            expires_at, value = self._task_records_memory[key]
            if expires_at > time.time():
                raw = value
            else:
                self._task_records_memory.pop(key, None)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def list_last_tasks(self, limit: int = 5) -> list[Dict[str, Any]]:
        if limit <= 0:
            return []
        task_ids: list[str] = []
        if self.redis is not None:
            try:
                values = self.redis.lrange(self._last_tasks_key(), 0, limit - 1)
                for value in values:
                    if isinstance(value, bytes):
                        task_ids.append(value.decode("utf-8"))
                    elif isinstance(value, str):
                        task_ids.append(value)
            except Exception:
                log.warning("SunoService redis.lrange record failed", exc_info=True)
        if not task_ids:
            task_ids = self._task_order[:limit]
        result: list[Dict[str, Any]] = []
        for task_id in task_ids:
            record = self._load_task_record(task_id)
            if record:
                result.append(record)
        return result

    def get_task_record(self, task_id: str) -> Optional[Dict[str, Any]]:
        return self._load_task_record(task_id)

    def resend_links(self, task_id: str) -> bool:
        record = self._load_task_record(task_id)
        if not record:
            return False
        chat_id = record.get("chat_id") or record.get("user_id")
        if not chat_id:
            return False
        if not self.telegram_token:
            log.warning("Cannot resend Suno task %s: missing TELEGRAM_TOKEN", task_id)
            return False
        reply_to = record.get("msg_id")
        try:
            reply_id = int(reply_to) if reply_to is not None else None
        except (TypeError, ValueError):
            reply_id = None
        header = f"🔁 Повторная отправка Suno | задача {task_id}"
        self._send_text(int(chat_id), header, reply_to=reply_id)
        tracks_data = record.get("tracks") or []
        if not isinstance(tracks_data, list):
            tracks_data = []
        total_tracks = len(tracks_data)
        for idx, item in enumerate(tracks_data, start=1):
            if not isinstance(item, Mapping):
                continue
            title = (
                item.get("title")
                or record.get("user_title")
                or record.get("title")
                or f"Track {idx}"
            )
            audio = item.get("source_audio_url") or item.get("audio_url")
            image = item.get("source_image_url") or item.get("image_url")
            tags = item.get("tags") if isinstance(item.get("tags"), str) else None
            duration_value = item.get("duration")
            try:
                duration = float(duration_value) if duration_value is not None else None
            except (TypeError, ValueError):
                duration = None
            take_label = f"Take {idx}" if total_tracks > 1 else None
            caption = self._build_audio_caption(
                title=title,
                tags=tags,
                duration=duration,
                take_label=take_label,
            )
            if audio:
                send_title = f"{title} ({take_label})" if take_label else title
                audio_sent = self._send_audio_url(
                    int(chat_id),
                    audio,
                    caption=caption,
                    reply_to=None,
                    title=send_title,
                    thumb=image,
                )
                if not audio_sent:
                    path = Path(audio)
                    if path.exists():
                        self._send_audio(int(chat_id), path, title=title, reply_to=None)
                    else:
                        fallback_lines = [f"🔗 Аудио ({title}): {audio}"]
                        if tags:
                            fallback_lines.append(f"🏷️ {tags}")
                        duration_label = self._format_duration_label(duration)
                        if duration_label:
                            fallback_lines.append(f"⏱️ {duration_label}")
                        self._send_text(int(chat_id), "\n".join(fallback_lines))
            if image:
                image_caption = f"🖼️ {title} (обложка)"
                if self._send_image_url(int(chat_id), image, caption=image_caption, reply_to=None):
                    continue
                img_path = Path(image)
                if img_path.exists():
                    self._send_image(int(chat_id), img_path, title=title, reply_to=None)
                else:
                    self._send_text(int(chat_id), f"🖼️ Обложка ({title}): {image}")
        return True

    # ----------------------------------------------------------------- telegram
    def _bot_url(self, method: str) -> str:
        if not self.telegram_token:
            raise RuntimeError("TELEGRAM_TOKEN is not configured")
        return f"https://api.telegram.org/bot{self.telegram_token}/{method}"

    def _send_text(self, chat_id: int, text: str, *, reply_to: Optional[int] = None) -> None:
        payload: Dict[str, Any] = {"chat_id": chat_id, "text": text}
        if reply_to:
            payload["reply_to_message_id"] = reply_to
        method = "sendMessage"
        try:
            resp = self._bot_session.post(self._bot_url(method), json=payload, timeout=20)
            if not resp.ok:
                bot_telegram_send_fail_total.labels(method=method).inc()
                log.warning("Telegram sendMessage failed | status=%s text=%s", resp.status_code, resp.text)
        except requests.RequestException:
            bot_telegram_send_fail_total.labels(method=method).inc()
            log.warning("Telegram sendMessage network error", exc_info=True)

    def _send_file(
        self,
        method: str,
        field: str,
        chat_id: int,
        path: Path,
        *,
        caption: Optional[str],
        reply_to: Optional[int],
        extra: Optional[Mapping[str, Any]] = None,
        file_name: Optional[str] = None,
    ) -> bool:
        data: Dict[str, Any] = {"chat_id": chat_id}
        if caption:
            data["caption"] = caption
        if reply_to:
            data["reply_to_message_id"] = reply_to
        if extra:
            for key, value in extra.items():
                if value is None:
                    continue
                data[key] = value
        try:
            with path.open("rb") as fh:
                upload_name = file_name or path.name
                files = {field: (upload_name, fh)}
                resp = self._bot_session.post(self._bot_url(method), data=data, files=files, timeout=120)
            if not resp.ok:
                bot_telegram_send_fail_total.labels(method=method).inc()
                log.warning("Telegram %s failed | status=%s text=%s", method, resp.status_code, resp.text)
                return False
            return True
        except FileNotFoundError:
            bot_telegram_send_fail_total.labels(method=method).inc()
            return False
        except requests.RequestException:
            log.warning("Telegram %s network error", method, exc_info=True)
            bot_telegram_send_fail_total.labels(method=method).inc()
            return False

    def _format_duration_label(self, duration: Optional[float]) -> Optional[str]:
        if duration is None:
            return None
        try:
            total_seconds = int(round(float(duration)))
        except (TypeError, ValueError):
            return None
        if total_seconds <= 0:
            return None
        minutes, seconds = divmod(total_seconds, 60)
        return f"{minutes}:{seconds:02d}"

    def _build_audio_caption(
        self,
        *,
        title: Optional[str],
        tags: Optional[str],
        duration: Optional[float],
        take_label: Optional[str],
    ) -> Optional[str]:
        base_title = (title or "Suno track").strip()
        if take_label:
            base_title = f"{base_title} ({take_label})"
        duration_seconds: Optional[int] = None
        if duration is not None:
            try:
                duration_seconds = int(round(float(duration)))
            except (TypeError, ValueError):
                duration_seconds = None
        lines: list[str] = [f"🎵 {base_title}".strip()]
        details: list[str] = []
        if duration_seconds and duration_seconds > 0:
            details.append(f"{duration_seconds} sec")
        tags_text = (tags or "").strip()
        if tags_text:
            details.append(tags_text)
        if details:
            lines.append(" • ".join(details))
        caption = "\n".join(part for part in lines if part)
        if not caption:
            return None
        if len(caption) > 1024:
            return f"{caption[:1021]}…"
        return caption

    def _send_audio(self, chat_id: int, path: Path, *, title: str, reply_to: Optional[int]) -> bool:
        caption = self._build_audio_caption(title=title, tags=None, duration=None, take_label=None)
        success = self._send_file(
            "sendAudio",
            "audio",
            chat_id,
            path,
            caption=caption,
            reply_to=reply_to,
            extra={"title": title},
        )
        if success:
            schedule_unlink(path)
        return success

    def _send_audio_url(
        self,
        chat_id: int,
        url: str,
        *,
        caption: Optional[str],
        reply_to: Optional[int],
        title: Optional[str],
        thumb: Optional[str],
    ) -> bool:
        if not url:
            return False
        payload: Dict[str, Any] = {"chat_id": chat_id, "audio": url}
        if caption:
            payload["caption"] = caption
        if reply_to:
            payload["reply_to_message_id"] = reply_to
        if title:
            payload["title"] = title
        if thumb:
            payload["thumb"] = thumb
        method = "sendAudio"
        try:
            resp = self._bot_session.post(self._bot_url(method), json=payload, timeout=60)
        except Exception:
            bot_telegram_send_fail_total.labels(method=method).inc()
            log.warning("Telegram sendAudio network error (url)", exc_info=True)
            return False
        if not resp.ok:
            bot_telegram_send_fail_total.labels(method=method).inc()
            log.warning(
                "Telegram sendAudio failed | status=%s text=%s",
                resp.status_code,
                mask_tokens(resp.text),
            )
            return False
        return True

    def _send_image(self, chat_id: int, path: Path, *, title: str, reply_to: Optional[int]) -> bool:
        caption = f"🖼️ {title} (обложка)" if title else "🖼️ Обложка"
        success = self._send_file("sendPhoto", "photo", chat_id, path, caption=caption, reply_to=reply_to)
        if success:
            schedule_unlink(path)
        return success

    def _send_image_url(
        self,
        chat_id: int,
        url: str,
        *,
        caption: Optional[str],
        reply_to: Optional[int],
    ) -> bool:
        if not url:
            return False
        payload: Dict[str, Any] = {"chat_id": chat_id, "photo": url}
        if caption:
            payload["caption"] = caption
        if reply_to:
            payload["reply_to_message_id"] = reply_to
        method = "sendPhoto"
        try:
            resp = self._bot_session.post(self._bot_url(method), json=payload, timeout=60)
        except Exception:
            bot_telegram_send_fail_total.labels(method=method).inc()
            log.warning("Telegram sendPhoto network error (url)", exc_info=True)
            return False
        if not resp.ok:
            bot_telegram_send_fail_total.labels(method=method).inc()
            log.warning(
                "Telegram sendPhoto failed | status=%s text=%s",
                resp.status_code,
                mask_tokens(resp.text),
            )
            return False
        return True

    # ------------------------------------------------------------------ helpers
    def _notify_admins(self, text: str) -> None:
        if not self.telegram_token:
            log.info("Skipping admin notify (telegram token missing)")
            return
        if not self._admin_ids:
            log.info("Skipping admin notify (no ADMIN_IDS configured)")
            return
        for admin_id in sorted(self._admin_ids):
            self._send_text(admin_id, text)

    def _stage_header(self, task: SunoTask) -> str:
        status = (task.callback_type or "unknown").lower()
        return f"🎧 Suno: этап {status} получен."

    def _base_dir(self, task_id: str) -> Path:
        return task_directory(task_id)

    def _find_local_file(self, base_dir: Path, prefix: str) -> Optional[Path]:
        if not base_dir.exists():
            return None
        candidates = sorted(base_dir.glob(f"{prefix}*"))
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None

    def _delivery_key(self, task_id: str, take_id: str) -> str:
        return f"{task_id}:{take_id}"

    def _delivery_register(self, task_id: str, take_id: str) -> bool:
        if not task_id or not take_id:
            return True
        key = self._delivery_key(task_id, take_id)
        seen = self._delivery_seen
        if key in seen:
            seen.move_to_end(key)
            return False
        seen[key] = time.time()
        while len(seen) > self._delivery_seen_limit:
            seen.popitem(last=False)
        return True

    @staticmethod
    def _log_delivery(event: str, **meta: Any) -> None:
        log.info(event, extra={"meta": meta})

    def _normalize_take_title(
        self,
        track: SunoTrack,
        *,
        meta_title: Optional[str],
        user_title: Optional[str],
    ) -> str:
        for candidate in (
            track.title,
            user_title,
            meta_title,
            "Untitled",
        ):
            if isinstance(candidate, str):
                cleaned = candidate.strip()
                if cleaned:
                    return cleaned
        return "Untitled"

    @staticmethod
    def _collect_tags(tags_value: Any, *, limit: int = 3) -> list[str]:
        if tags_value in (None, ""):
            return []
        tokens: list[str] = []
        if isinstance(tags_value, (list, tuple, set)):
            raw_values = [str(item) for item in tags_value if str(item).strip()]
        else:
            raw_values = re.split(r"[\s,;#]+", str(tags_value))
        for raw in raw_values:
            cleaned = raw.strip().lower()
            if cleaned and cleaned not in tokens:
                tokens.append(cleaned)
        if limit <= 0:
            return tokens
        return tokens[:limit]

    def _build_take_caption(
        self,
        *,
        title: str,
        take_index: int,
        duration: Optional[float],
        tags: list[str],
        preset: Optional[str] = None,
        total_takes: int = 1,
    ) -> str:
        duration_seconds: Optional[int] = None
        if duration is not None:
            try:
                duration_seconds = int(round(float(duration)))
            except (TypeError, ValueError):
                duration_seconds = None
        normalized_preset = (preset or "").strip().lower()
        if normalized_preset == AMBIENT_NATURE_PRESET_ID:
            cfg = get_preset_config(AMBIENT_NATURE_PRESET_ID) or {}
            emoji = str(cfg.get("emoji") or "🌊").strip() or "🌊"
            label = str(cfg.get("label") or "Ambient Preset").strip() or "Ambient Preset"
            if total_takes > 1:
                first_line = f"{emoji} {title} ({label} • Take {take_index})"
            else:
                first_line = f"{emoji} {title} ({label})"
            primary = str(cfg.get("caption_primary") or "cinematic ambient").strip()
            secondary_values = cfg.get("caption_secondary") or []
            if isinstance(secondary_values, (list, tuple)):
                secondary = ", ".join(
                    str(item).strip() for item in secondary_values if str(item).strip()
                )
            else:
                secondary = ""
            if not secondary:
                secondary = ", ".join(tags[1:5]) if len(tags) > 1 else ", ".join(tags[:4])
            if not primary and tags:
                primary = tags[0]
            parts: list[str] = []
            if duration_seconds and duration_seconds > 0:
                parts.append(f"{duration_seconds}s")
            if primary:
                parts.append(primary)
            if secondary:
                parts.append(secondary)
            second_line = " • ".join(part for part in parts if part)
            caption = first_line if not second_line else f"{first_line}\n{second_line}"
            if len(caption) <= 1024:
                return caption
            return caption[:1021] + "…"

        first_line = f"{title} (Take {take_index})"
        if duration_seconds and duration_seconds > 0:
            first_line = f"{first_line} • {duration_seconds}s"
        tags_text = ", ".join(tags)
        if tags_text:
            caption = f"{first_line}\n{tags_text}"
        else:
            caption = first_line
        if len(caption) <= 1024:
            return caption
        if tags_text and len(first_line) < 1024:
            remaining = max(0, 1024 - len(first_line) - 1)
            truncated_tags = (tags_text[: remaining - 1] + "…") if remaining and len(tags_text) > remaining else tags_text[:remaining]
            truncated_tags = truncated_tags.rstrip()
            if truncated_tags:
                return f"{first_line}\n{truncated_tags}"
            return first_line[:1021] + "…"
        return caption[:1021] + "…"

    @staticmethod
    def _preset_cover_url(preset: Optional[str]) -> Optional[str]:
        preset_id = (preset or "").strip().lower()
        if preset_id != AMBIENT_NATURE_PRESET_ID:
            return None
        cfg = get_preset_config(AMBIENT_NATURE_PRESET_ID) or {}
        keywords = str(cfg.get("cover_keywords") or "").strip()
        if not keywords:
            return None
        return f"https://image.pollinations.ai/prompt/{quote_plus(keywords)}"

    def _send_cover_url(
        self,
        *,
        chat_id: int,
        photo_url: str,
        caption: Optional[str],
        reply_to: Optional[int],
    ) -> tuple[bool, Optional[str]]:
        from telegram_utils import send_photo_request

        success, reason, _ = send_photo_request(
            self._bot_session,
            self._bot_url("sendPhoto"),
            chat_id=chat_id,
            photo=photo_url,
            caption=caption,
            reply_to=reply_to,
        )
        return success, reason

    def _send_audio_url_with_retry(
        self,
        *,
        chat_id: int,
        audio_url: str,
        caption: Optional[str],
        reply_to: Optional[int],
        title: str,
        thumb: Optional[str],
        base_dir: Path,
        take_id: str,
    ) -> tuple[bool, Optional[str]]:
        from telegram_utils import is_remote_file_error, send_audio_request

        last_reason: Optional[str] = None
        tags_enabled = os.getenv("AUDIO_TAGS_ENABLED", "false").strip().lower() == "true"
        embed_cover_enabled = os.getenv("AUDIO_EMBED_COVER_ENABLED", "false").strip().lower() == "true"
        default_artist = os.getenv("AUDIO_DEFAULT_ARTIST", "Best VEO3")
        transliterate_env = os.getenv("AUDIO_FILENAME_TRANSLITERATE")
        if transliterate_env is None:
            transliterate = True
        else:
            transliterate = transliterate_env.strip().lower() in {"1", "true", "yes", "on"}
        try:
            max_name = int(os.getenv("AUDIO_FILENAME_MAX", "60"))
        except (TypeError, ValueError):
            max_name = 60

        prepared_meta: Dict[str, Optional[str]] = {}
        prepared_path: Optional[Path] = None
        try:
            local_path_str, prepared_meta = prepare_audio_file_sync(
                audio_url,
                title=title,
                cover_url=thumb,
                default_artist=default_artist,
                max_name=max_name,
                tags_enabled=tags_enabled,
                embed_cover_enabled=embed_cover_enabled,
                transliterate=transliterate,
            )
        except Exception:
            log.warning(
                "suno.audio.postprocess_failed",
                extra={"meta": {"url": mask_tokens(audio_url), "take": take_id}},
                exc_info=True,
            )
        else:
            prepared_path = Path(local_path_str)

        remote_title = prepared_meta.get("title") or title

        if prepared_path and prepared_path.exists():
            try:
                audio_extra: Dict[str, Optional[str]] = {}
                if remote_title:
                    audio_extra["title"] = remote_title
                performer = prepared_meta.get("performer")
                if performer:
                    audio_extra["performer"] = performer
                file_name = prepared_meta.get("file_name") or prepared_path.name

                audio_sent = self._send_file(
                    "sendAudio",
                    "audio",
                    chat_id,
                    prepared_path,
                    caption=caption,
                    reply_to=reply_to,
                    extra=audio_extra,
                    file_name=file_name,
                )
                document_sent = self._send_file(
                    "sendDocument",
                    "document",
                    chat_id,
                    prepared_path,
                    caption=caption,
                    reply_to=reply_to,
                    extra=None,
                    file_name=file_name,
                )
                if audio_sent or document_sent:
                    return True, None
                last_reason = "upload_failed"
            finally:
                schedule_unlink(prepared_path)

        attempt = 1
        success, reason, status = send_audio_request(
            self._bot_session,
            self._bot_url("sendAudio"),
            chat_id=chat_id,
            audio=audio_url,
            caption=caption,
            reply_to=reply_to,
            title=remote_title,
            thumb=thumb,
        )
        if success:
            return True, None
        last_reason = reason or last_reason
        if is_remote_file_error(status, reason):
            self._log_delivery(
                "telegram.retry",
                kind="audio",
                attempt=attempt + 1,
                reason=last_reason or "remote_error",
            )
            attempt += 1
            success, reason, status = send_audio_request(
                self._bot_session,
                self._bot_url("sendAudio"),
                chat_id=chat_id,
                audio=audio_url,
                caption=caption,
                reply_to=reply_to,
                title=remote_title,
                thumb=thumb,
            )
            if success:
                return True, None
            last_reason = reason or last_reason
        if not self._telegram_download_fallback:
            return False, last_reason
        self._log_delivery(
            "telegram.retry",
            kind="audio",
            attempt=attempt + 1,
            reason=last_reason or "remote_error",
        )
        local_path = self._download_audio(audio_url, base_dir, take_id)
        if not local_path:
            return False, last_reason
        extra: Dict[str, Any] = {"title": remote_title} if remote_title else {}
        file_name = prepared_meta.get("file_name") or local_path.name
        try:
            audio_sent = self._send_file(
                "sendAudio",
                "audio",
                chat_id,
                local_path,
                caption=caption,
                reply_to=reply_to,
                extra=extra,
                file_name=file_name,
            )
            document_sent = self._send_file(
                "sendDocument",
                "document",
                chat_id,
                local_path,
                caption=caption,
                reply_to=reply_to,
                extra=None,
                file_name=file_name,
            )
        finally:
            schedule_unlink(local_path)
        if audio_sent or document_sent:
            return True, None
        return False, last_reason

    def _download_audio(self, url: str, base_dir: Path, take_id: str) -> Optional[Path]:
        if not url:
            return None
        try:
            response = requests.get(url, stream=True, timeout=60)
        except requests.RequestException as exc:
            log.warning(
                "suno.audio.download.failed",
                extra={"meta": {"url": mask_tokens(url), "err": str(exc)}},
            )
            return None
        if not response.ok:
            log.warning(
                "suno.audio.download.failed",
                extra={
                    "meta": {
                        "url": mask_tokens(url),
                        "status": response.status_code,
                    }
                },
            )
            return None
        suffix = Path(urlparse(url).path).suffix or ".mp3"
        base_dir.mkdir(parents=True, exist_ok=True)
        target = base_dir / f"{take_id}_dl{suffix}"
        try:
            with target.open("wb") as handle:
                for chunk in response.iter_content(65536):
                    if chunk:
                        handle.write(chunk)
        except Exception as exc:
            log.warning(
                "suno.audio.download.write_failed",
                extra={"meta": {"path": str(target), "err": str(exc)}},
            )
            with suppress(Exception):
                target.unlink(missing_ok=True)  # type: ignore[arg-type]
            return None
        return target

    # ----------------------------------------------------------------- public API
    def start_music(
        self,
        chat_id: int,
        msg_id: int,
        *,
        title: Optional[str],
        style: Optional[str],
        lyrics: Optional[str],
        model: str = "V5",
        instrumental: bool = False,
        user_id: Optional[int] = None,
        prompt: Optional[str] = None,
        req_id: Optional[str] = None,
        lang: Optional[str] = None,
        has_lyrics: bool = False,
        prepared_payload: Optional[Mapping[str, Any]] = None,
        negative_tags: Optional[Iterable[str]] = None,
        preset: Optional[str] = None,
        lyrics_source: Optional[str] = None,
        strict_enabled: bool = False,
        strict_original_lyrics: Optional[str] = None,
        strict_payload: Optional[Mapping[str, Any]] = None,
        strict_threshold: Optional[float] = None,
    ) -> SunoTask:
        prompt_text = str(
            (prompt if prompt is not None else "")
            or (style if style is not None else "")
            or (lyrics if lyrics is not None else "")
            or (title if title is not None else "")
        ).strip()
        if not prompt_text:
            prompt_text = "Untitled track"
        prompt_len = 16
        model_name = model or "V5"
        lyrics_text = lyrics if has_lyrics else None
        derived_tags: Optional[list[str]] = None
        if style:
            derived_tags = [
                part.strip().strip("#").lower()
                for part in re.split(r"[\s,]+", style)
                if part.strip()
            ]
        if prepared_payload is not None:
            final_payload = dict(prepared_payload)
        else:
            final_payload = self.client.build_payload(
                user_id=str(user_id) if user_id is not None else "0",
                title=title or prompt_text,
                prompt=prompt_text,
                instrumental=instrumental,
                has_lyrics=bool(has_lyrics),
                lyrics=lyrics_text,
                prompt_len=prompt_len,
                model=model_name,
                tags=derived_tags,
                negative_tags=negative_tags,
                preset=preset,
            )
        if lyrics_source:
            final_payload.setdefault("lyrics_source", lyrics_source)
        resolved_title = str(final_payload.get("title") or title or prompt_text)
        prompt_text = str(final_payload.get("prompt") or prompt_text)
        try:
            task_identifier = self.client.enqueue(final_payload, req_id=req_id)
        except SunoAPIError as exc:
            version = getattr(exc, "api_version", "v5") or "v5"
            suno_requests_total.labels(
                result="fail",
                reason="start_music",
                api_version=version,
                **_metric_labels("bot"),
            ).inc()
            raise
        except Exception:
            suno_requests_total.labels(
                result="fail",
                reason="start_music",
                api_version="unknown",
                **_metric_labels("bot"),
            ).inc()
            raise
        else:
            suno_requests_total.labels(
                result="ok",
                reason="start_music",
                api_version="v5",
                **_metric_labels("bot"),
            ).inc()
        task_id = str(task_identifier or "").strip()
        if not task_id:
            raise SunoAPIError("Suno did not return taskId", payload=final_payload)
        req_id = task_id if not req_id else req_id
        status_label = "queued"
        log.info(
            "SUNO[enqueue] status=%s req_id=%s task_id=%s payload=%s",
            status_label,
            req_id or task_id,
            task_id,
            _json_preview(final_payload),
        )
        task = SunoTask(task_id=task_id, callback_type="start", items=[], msg=None, code=200)
        user_title = str(title).strip() if title else None
        meta = {
            "chat_id": int(chat_id),
            "msg_id": int(msg_id),
            "title": resolved_title,
            "ts": datetime.now(timezone.utc).isoformat(),
            "req_id": req_id,
            "user_title": user_title,
            "lyrics_source": lyrics_source,
            "strict_enabled": bool(strict_enabled),
        }
        if strict_enabled:
            strict_meta: Dict[str, Any] = {
                "attempts": 0,
                "threshold": float(strict_threshold) if strict_threshold is not None else _STRICT_LYRICS_THRESHOLD,
            }
            if strict_original_lyrics:
                strict_meta["original"] = str(strict_original_lyrics)
            elif lyrics_text:
                strict_meta["original"] = str(lyrics_text)
            payload_snapshot = strict_payload or prepared_payload
            if payload_snapshot is not None:
                strict_meta["payload"] = dict(payload_snapshot)
            meta["strict_lyrics"] = strict_meta
        self._store_mapping(task.task_id, meta)
        self._store_req_id(task.task_id, task_id)
        if req_id != task_id:
            self._store_req_id(task.task_id, req_id)
        if user_id is not None:
            self._store_user_link(
                task.task_id,
                {
                    "user_id": int(user_id),
                    "prompt": prompt_text,
                    "ts": datetime.now(timezone.utc).isoformat(),
                },
            )
        record = {
            "task_id": task.task_id,
            "status": "started",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "chat_id": int(chat_id),
            "msg_id": int(msg_id),
            "user_id": int(user_id) if user_id is not None else None,
            "prompt": prompt_text,
            "title": title,
            "user_title": user_title,
            "req_id": req_id,
            "lang": str(lang).strip().lower() if lang else None,
            "has_lyrics": bool(has_lyrics),
            "preset": (str(preset).strip().lower() if preset else None),
        }
        self._save_task_record(task.task_id, record)
        log.info(
            "Suno task stored",
            extra={
                "meta": {
                    "task_id": task.task_id,
                    "chat_id": chat_id,
                    "msg_id": msg_id,
                    "user_id": user_id,
                    "req_id": req_id,
                }
            },
        )
        return task

    def _coerce_music_envelope(self, payload: Any) -> CallbackEnvelope:
        if isinstance(payload, Mapping):
            return CallbackEnvelope.model_validate(dict(payload))
        raw_payload = getattr(payload, "raw", None)
        base: Dict[str, Any]
        if isinstance(raw_payload, Mapping):
            base = dict(raw_payload)
        else:
            base = {}
        base.setdefault("code", getattr(payload, "code", None))
        base.setdefault("msg", getattr(payload, "msg", None))
        data_section = dict(base.get("data") or {})
        task_id = getattr(payload, "task_id", None)
        if task_id and "taskId" not in data_section:
            data_section["taskId"] = task_id
        callback_type = getattr(payload, "type", None) or getattr(payload, "status", None)
        if callback_type and "callbackType" not in data_section:
            data_section["callbackType"] = callback_type
        tracks_payload: list[Dict[str, Any]] = []
        for item in getattr(payload, "tracks", []) or []:
            if hasattr(item, "raw") and isinstance(item.raw, Mapping):
                track_payload = dict(item.raw)
            elif isinstance(item, Mapping):
                track_payload = dict(item)
            else:
                track_payload = {}
                audio_id = getattr(item, "audio_id", None)
                if audio_id is not None:
                    track_payload.setdefault("audioId", audio_id)
                audio_url = getattr(item, "audio_url", None)
                if audio_url:
                    track_payload.setdefault("audioUrl", audio_url)
                image_url = getattr(item, "image_url", None)
                if image_url:
                    track_payload.setdefault("imageUrl", image_url)
                video_url = getattr(item, "video_url", None)
                if video_url:
                    track_payload.setdefault("videoUrl", video_url)
            if track_payload:
                tracks_payload.append(track_payload)
        if tracks_payload:
            response = dict(data_section.get("response") or {})
            existing_tracks = response.get("tracks")
            if isinstance(existing_tracks, list):
                response["tracks"] = [*existing_tracks, *tracks_payload]
            else:
                response["tracks"] = tracks_payload
            data_section["response"] = response
        base["data"] = data_section
        return CallbackEnvelope.model_validate(base)

    def handle_music_callback(self, payload: Any) -> None:
        try:
            envelope = self._coerce_music_envelope(payload)
        except Exception:
            log.warning("Failed to coerce music callback payload", exc_info=True)
            return
        task = SunoTask.from_envelope(envelope)
        data_payload: Mapping[str, Any] = envelope.data or {}
        req_id: Optional[str] = None
        if isinstance(payload, Mapping):
            req_id = payload.get("request_id") or payload.get("requestId")
        else:
            raw_payload = getattr(payload, "raw", None)
            if isinstance(raw_payload, Mapping):
                req_id = raw_payload.get("request_id") or raw_payload.get("requestId")
        if req_id is None:
            req_id = getattr(payload, "request_id", None) or getattr(payload, "requestId", None)
        data_req_id, data_task_id = _extract_enqueue_identifiers(data_payload)
        if data_req_id and not req_id:
            req_id = data_req_id
        if data_task_id and not task.task_id:
            task = task.model_copy(update={"task_id": data_task_id})
        if not task.task_id and req_id:
            mapped = self._load_task_id_for_req(req_id)
            if mapped:
                task = task.model_copy(update={"task_id": mapped})
        self.handle_callback(task, req_id=req_id)

    def handle_callback(
        self,
        task: SunoTask,
        req_id: Optional[str] = None,
        *,
        delivery_via: str = "webhook",
    ) -> None:
        if not task.task_id:
            log.warning("Callback without task_id: %s", task)
            return
        meta = self._load_mapping(task.task_id)
        link = self._load_user_link(task.task_id)
        chat_id = meta.chat_id if meta else (link.user_id if link else None)
        if req_id is None and meta is not None:
            req_id = meta.req_id
        if req_id is None:
            req_id = self._load_req_id(task.task_id)
        if req_id and task.task_id:
            self._store_req_id(task.task_id, req_id)
        log.info(
            "Suno callback received",
            extra={
                "meta": {
                    "task_id": task.task_id,
                    "callback_type": task.callback_type,
                    "chat_id": chat_id,
                    "req_id": req_id,
                }
            },
        )
        if chat_id is None:
            log.info("No chat mapping for task %s", task.task_id)
            snippet = task.model_dump(exclude_none=True)
            prompt = link.prompt if link else "—"
            self._notify_admins(
                f"⚠️ Suno callback без получателя\nTask: {task.task_id}\nType: {task.callback_type}\nPrompt: {prompt}\nPayload: {json.dumps(snippet, ensure_ascii=False)[:500]}"
            )
            return
        if not self.telegram_token:
            log.warning("TELEGRAM_TOKEN missing; skip delivery for task %s", task.task_id)
            return
        existing_record = self._load_task_record(task.task_id) or {}
        preset_id = str(existing_record.get("preset") or "").strip().lower()
        incoming_status = (task.callback_type or "").lower()
        existing_status = str(existing_record.get("status") or "").lower()
        final_states = {"complete", "error", "failed", "success"}
        if incoming_status in final_states and self._recently_delivered(task.task_id):
            log.info(
                "Suno callback duplicate final",
                extra={
                    "meta": {
                        "task_id": task.task_id,
                        "status": incoming_status,
                        "reason": "delivered_cache",
                    }
                },
            )
            return
        if incoming_status in final_states and existing_status in final_states and existing_record.get("tracks"):
            log.info(
                "Suno callback duplicate final state",
                extra={
                    "meta": {
                        "task_id": task.task_id,
                        "status": incoming_status,
                        "existing_status": existing_status,
                    }
                },
            )
            return

        meta_user_title: Optional[str] = None

        try:
            header = self._stage_header(task)
            reply_to = meta.msg_id if meta else None
            self._send_text(chat_id, header, reply_to=reply_to)
            stored_user_title = existing_record.get("user_title")
            if meta and isinstance(meta.user_title, str) and meta.user_title.strip():
                meta_user_title = meta.user_title.strip()
            elif isinstance(stored_user_title, str) and stored_user_title.strip():
                meta_user_title = stored_user_title.strip()
            elif isinstance(existing_record.get("title"), str) and str(existing_record.get("title")).strip():
                meta_user_title = str(existing_record.get("title")).strip()

            if not task.items:
                record = dict(existing_record)
                record.update(
                    {
                        "task_id": task.task_id,
                        "status": task.callback_type,
                        "code": task.code,
                        "msg": task.msg,
                        "chat_id": chat_id,
                        "msg_id": reply_to,
                        "user_id": link.user_id if link else None,
                        "prompt": link.prompt if link else "",
                        "title": meta.title if meta else None,
                        "user_title": meta_user_title,
                        "req_id": req_id,
                        "tracks": [],
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                record.setdefault(
                    "created_at", existing_record.get("created_at") or datetime.now(timezone.utc).isoformat()
                )
                self._save_task_record(task.task_id, record)
                return
        except Exception:
            log.warning("Suno callback stage header failed", exc_info=True)

        try:
            strict_context = self._strict_context_from_meta(meta)
            strict_warning: Optional[str] = None
            strict_actual_lyrics: Optional[str] = None
            strict_similarity: Optional[float] = None
            if strict_context and incoming_status in final_states:
                strict_decision = self._process_strict_delivery(
                    task=task,
                    meta=meta,
                    link=link,
                    strict_context=strict_context,
                    req_id=req_id,
                )
                if strict_decision == "retry":
                    return
                if isinstance(strict_decision, dict):
                    strict_warning = strict_decision.get("warning")
                    strict_actual_lyrics = strict_decision.get("lyrics")
                    strict_similarity = strict_decision.get("similarity")
                    strict_context["attempts"] = strict_decision.get(
                        "attempts", strict_context.get("attempts", 0)
                    )

            base_dir = self._base_dir(task.task_id)
            track_records: list[Dict[str, Any]] = []
            durations: list[float] = []
            self._log_delivery(
                "suno.delivery.start",
                task_id=task.task_id,
                takes=len(task.items),
                via=delivery_via,
            )

            total_takes = len(task.items)
            for idx, track in enumerate(task.items, start=1):
                take_id = track.id or str(idx)
                meta_title_value = str(meta.title).strip() if meta and meta.title else None
                normalized_title = self._normalize_take_title(
                    track,
                    meta_title=meta_title_value,
                    user_title=meta_user_title,
                )
                tag_limit = 5 if preset_id == AMBIENT_NATURE_PRESET_ID else 3
                tags_list = self._collect_tags(track.tags, limit=tag_limit)
                if preset_id == AMBIENT_NATURE_PRESET_ID and not tags_list:
                    cfg = get_preset_config(AMBIENT_NATURE_PRESET_ID) or {}
                    fallback_tags: list[str] = []
                    for tag in cfg.get("tags", []):
                        text = str(tag).strip().lower()
                        if text and text not in fallback_tags:
                            fallback_tags.append(text)
                    tags_list = fallback_tags[:tag_limit]
                caption = self._build_take_caption(
                    title=normalized_title,
                    take_index=idx,
                    duration=track.duration,
                    tags=tags_list,
                    preset=preset_id,
                    total_takes=total_takes,
                )
                take_title = f"{normalized_title} (Take {idx})" if len(task.items) > 1 else normalized_title
                cover_url = track.source_image_url or track.image_url
                if not cover_url:
                    cover_url = self._preset_cover_url(preset_id)
                audio_url = track.source_audio_url or track.audio_url

                if not self._delivery_register(task.task_id, take_id):
                    self._log_delivery(
                        "suno.delivery.take.skipped",
                        task_id=task.task_id,
                        take_id=take_id,
                        reason="duplicate",
                    )
                    track_records.append(
                        {
                            "id": take_id,
                            "title": normalized_title,
                            "original_title": track.title,
                            "audio_url": track.audio_url,
                            "image_url": track.image_url,
                            "source_audio_url": track.source_audio_url,
                            "source_image_url": track.source_image_url,
                            "tags": track.tags,
                            "duration": track.duration,
                        }
                    )
                    if isinstance(track.duration, (int, float)):
                        durations.append(float(track.duration))
                    continue

                cover_sent = False
                cover_reason: Optional[str] = None
                if cover_url:
                    cover_sent, cover_reason = self._send_cover_url(
                        chat_id=chat_id,
                        photo_url=cover_url,
                        caption=take_title,
                        reply_to=None,
                    )
                    if not cover_sent:
                        image_path = self._find_local_file(base_dir, f"{take_id}_cover")
                        if image_path and self._send_image(chat_id, image_path, title=take_title, reply_to=None):
                            cover_sent = True
                            cover_reason = None
                if cover_url and not cover_sent:
                    self._log_delivery(
                        "suno.delivery.cover.failed",
                        task_id=task.task_id,
                        take_id=take_id,
                        err=mask_tokens(cover_reason or "unknown"),
                    )

                audio_sent = False
                if audio_url:
                    audio_sent, _ = self._send_audio_url_with_retry(
                        chat_id=chat_id,
                        audio_url=audio_url,
                        caption=caption,
                        reply_to=None,
                        title=take_title,
                        thumb=cover_url,
                        base_dir=base_dir,
                        take_id=take_id,
                    )
                else:
                    local_audio = self._find_local_file(base_dir, take_id)
                    if local_audio:
                        audio_sent = self._send_file(
                            "sendAudio",
                            "audio",
                            chat_id,
                            local_audio,
                            caption=caption,
                            reply_to=None,
                            extra={"title": take_title},
                        )
                        schedule_unlink(local_audio)

                if audio_sent:
                    self._log_delivery(
                        "suno.delivery.take.sent",
                        task_id=task.task_id,
                        take_id=take_id,
                        audio_url=mask_tokens(audio_url or track.audio_url or ""),
                        image_url=mask_tokens(cover_url or track.image_url or ""),
                    )
                    if not cover_sent and cover_url:
                        self._send_text(chat_id, f"🖼️ Обложка ({take_title}): {cover_url}")
                else:
                    fallback_lines = [f"⚠️ Не удалось отправить трек {take_title}."]
                    link_candidate = audio_url or track.audio_url
                    if link_candidate:
                        fallback_lines.append(f"🔗 {link_candidate}")
                    if cover_url:
                        fallback_lines.append(f"🖼️ Обложка: {cover_url}")
                    self._send_text(chat_id, "\n".join(fallback_lines))

                track_records.append(
                    {
                        "id": take_id,
                        "title": normalized_title,
                        "original_title": track.title,
                        "audio_url": track.audio_url,
                        "image_url": track.image_url,
                        "source_audio_url": track.source_audio_url,
                        "source_image_url": track.source_image_url,
                        "tags": track.tags,
                        "duration": track.duration,
                    }
                )
                if isinstance(track.duration, (int, float)):
                    durations.append(float(track.duration))

            record = dict(existing_record)
            record.update(
                {
                    "task_id": task.task_id,
                    "status": task.callback_type,
                    "code": task.code,
                    "msg": task.msg,
                    "chat_id": chat_id,
                    "msg_id": reply_to,
                    "user_id": link.user_id if link else None,
                    "prompt": link.prompt if link else "",
                    "title": meta.title if meta else None,
                    "user_title": meta_user_title,
                    "tracks": track_records,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            record.setdefault(
                "created_at", existing_record.get("created_at") or datetime.now(timezone.utc).isoformat()
            )
            if strict_context:
                record["strict_lyrics"] = {
                    "original": strict_context.get("original"),
                    "actual": strict_actual_lyrics,
                    "similarity": strict_similarity,
                    "attempts": strict_context.get("attempts"),
                    "threshold": strict_context.get("threshold", _STRICT_LYRICS_THRESHOLD),
                }
            self._save_task_record(task.task_id, record)
            if incoming_status in final_states:
                self._mark_delivered(task.task_id)
                log.info(
                    "Suno ready",
                    extra={
                        "meta": {
                            "task_id": task.task_id,
                            "takes": len(task.items),
                            "durations": durations,
                        }
                    },
                )
                if strict_warning:
                    try:
                        message_text = strict_warning
                        if strict_actual_lyrics:
                            message_text = f"{strict_warning}\n\n{strict_actual_lyrics}"
                        self._send_text(chat_id, message_text, reply_to=reply_to)
                    except Exception:
                        log.debug("Suno strict warning notify failed", exc_info=True)
                if self._should_log_once(task.task_id, task.callback_type):
                    log.info(
                        "processed | suno.callback",
                        extra={
                            "meta": {
                                "task_id": task.task_id,
                                "type": task.callback_type,
                                "code": task.code,
                                "tracks": len(task.items),
                            }
                        },
                    )
        finally:
            if (task.callback_type or "").lower() == "complete":
                self._delete_mapping(task.task_id)
                self._delete_user_link(task.task_id)


__all__ = [
    "SunoService",
    "SunoClient",
    "SunoAPIError",
    "SunoTask",
    "SunoTrack",
    "SunoError",
    "suno_generate",
    "suno_add_instrumental",
    "suno_add_vocals",
    "suno_record_info",
]
