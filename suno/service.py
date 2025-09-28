"""High level Suno service shared between the bot worker and web callback."""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional

import requests
from requests.adapters import HTTPAdapter

from metrics import bot_telegram_send_fail_total, suno_requests_total, suno_task_store_total
from suno.client import SunoClient, SunoAPIError
from suno.schemas import CallbackEnvelope, SunoTask, SunoTrack
from suno.tempfiles import cleanup_old_directories, schedule_unlink, task_directory
from settings import (
    HTTP_POOL_CONNECTIONS,
    HTTP_POOL_PER_HOST,
    REDIS_PREFIX,
    SUNO_API_BASE,
    SUNO_CALLBACK_SECRET,
    SUNO_CALLBACK_URL,
    SUNO_ENABLED,
)

try:  # pragma: no cover - optional runtime dependency
    from redis import Redis
except Exception:  # pragma: no cover - library may be unavailable
    Redis = None  # type: ignore

try:
    from redis_utils import rds as _redis_instance
except Exception:  # pragma: no cover - optional import
    _redis_instance = None

log = logging.getLogger("suno.service")

_TASK_TTL = 24 * 60 * 60
_USER_LINK_TTL = 7 * 24 * 60 * 60
_REQ_TTL = 24 * 60 * 60
_LOG_ONCE_TTL = 48 * 60 * 60
_ENV = (os.getenv("APP_ENV") or "prod").strip() or "prod"


def _metric_labels(service: str) -> dict[str, str]:
    return {"env": _ENV, "service": service}


@dataclass(slots=True)
class TelegramMeta:
    chat_id: int
    msg_id: int
    title: Optional[str]
    ts: str
    req_id: Optional[str]


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
        self._task_order: list[str] = []
        self._bot_session = requests.Session()
        adapter = HTTPAdapter(pool_connections=HTTP_POOL_CONNECTIONS, pool_maxsize=HTTP_POOL_PER_HOST)
        self._bot_session.mount("https://", adapter)
        self._bot_session.mount("http://", adapter)
        self._admin_ids = self._parse_admins(os.getenv("ADMIN_IDS"))
        self._log_once_memory: MutableMapping[str, float] = {}
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
        if self.redis is not None:
            try:
                self.redis.setex(key, _REQ_TTL, req_id)
                return
            except Exception:  # pragma: no cover - Redis failure
                log.warning("SunoService redis.setex req-id failed", exc_info=True)
        expires_at = time.time() + _REQ_TTL
        self._req_memory[key] = (expires_at, req_id)

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
        ts = data.get("ts") or datetime.now(timezone.utc).isoformat()
        req_id = data.get("req_id") or self._load_req_id(task_id)
        return TelegramMeta(chat_id=chat_id, msg_id=msg_id, title=title, ts=ts, req_id=req_id)

    def _delete_mapping(self, task_id: str) -> None:
        key = self._redis_key(task_id)
        if self.redis is not None:
            try:
                self.redis.delete(key)
            except Exception:  # pragma: no cover - Redis failure
                log.warning("SunoService redis.delete failed", exc_info=True)
        self._memory.pop(key, None)
        req_key = self._req_key(task_id)
        if self.redis is not None:
            try:
                self.redis.delete(req_key)
            except Exception:  # pragma: no cover - Redis failure
                log.warning("SunoService redis.delete req-id failed", exc_info=True)
        self._req_memory.pop(req_key, None)

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
        for idx, item in enumerate(tracks_data, start=1):
            if not isinstance(item, Mapping):
                continue
            title = item.get("title") or record.get("title") or f"Track {idx}"
            audio = item.get("audio_url")
            image = item.get("image_url")
            if audio:
                path = Path(audio)
                if path.exists():
                    self._send_audio(int(chat_id), path, title=title, reply_to=None)
                else:
                    self._send_text(int(chat_id), f"🔗 Аудио ({title}): {audio}")
            if image:
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

    def _send_file(self, method: str, field: str, chat_id: int, path: Path, *, caption: Optional[str], reply_to: Optional[int]) -> bool:
        data: Dict[str, Any] = {"chat_id": chat_id}
        if caption:
            data["caption"] = caption
        if reply_to:
            data["reply_to_message_id"] = reply_to
        try:
            with path.open("rb") as fh:
                files = {field: (path.name, fh)}
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

    def _send_audio(self, chat_id: int, path: Path, *, title: str, reply_to: Optional[int]) -> bool:
        caption = f"🎵 {title}" if title else None
        success = self._send_file("sendAudio", "audio", chat_id, path, caption=caption, reply_to=reply_to)
        if success:
            schedule_unlink(path)
        return success

    def _send_image(self, chat_id: int, path: Path, *, title: str, reply_to: Optional[int]) -> bool:
        caption = f"🖼️ {title} (обложка)" if title else "🖼️ Обложка"
        success = self._send_file("sendPhoto", "photo", chat_id, path, caption=caption, reply_to=reply_to)
        if success:
            schedule_unlink(path)
        return success

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
    ) -> SunoTask:
        prompt_text = str(
            (prompt if prompt is not None else "")
            or (lyrics if lyrics is not None else "")
            or (title if title is not None else "")
        ).strip()
        if not prompt_text:
            prompt_text = "Untitled track"
        payload = {
            "title": title,
            "style": style,
            "lyrics": lyrics,
            "model": model,
            "instrumental": instrumental,
            "prompt": prompt_text,
            "input_text": prompt_text,
        }
        try:
            result, api_version = self.client.create_music(payload, req_id=req_id)
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
                api_version=api_version,
                **_metric_labels("bot"),
            ).inc()
        task_id = str(
            result.get("task_id")
            or result.get("id")
            or (result.get("data") or {}).get("task_id")
            or ""
        )
        if not task_id:
            raise SunoAPIError("Suno did not return task_id", payload=result)
        task = SunoTask(task_id=task_id, callback_type="start", items=[], msg=result.get("msg"), code=result.get("code"))
        meta = {
            "chat_id": int(chat_id),
            "msg_id": int(msg_id),
            "title": title,
            "ts": datetime.now(timezone.utc).isoformat(),
            "req_id": req_id,
        }
        self._store_mapping(task.task_id, meta)
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
            "req_id": req_id,
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
        req_id: Optional[str] = None
        if isinstance(payload, Mapping):
            req_id = payload.get("request_id") or payload.get("requestId")
        else:
            raw_payload = getattr(payload, "raw", None)
            if isinstance(raw_payload, Mapping):
                req_id = raw_payload.get("request_id") or raw_payload.get("requestId")
        if req_id is None:
            req_id = getattr(payload, "request_id", None) or getattr(payload, "requestId", None)
        self.handle_callback(task, req_id=req_id)

    def handle_callback(self, task: SunoTask, req_id: Optional[str] = None) -> None:
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
        try:
            header = self._stage_header(task)
            reply_to = meta.msg_id if meta else None
            self._send_text(chat_id, header, reply_to=reply_to)
            if not task.items:
                existing = self._load_task_record(task.task_id) or {}
                record = dict(existing)
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
                        "req_id": req_id,
                        "tracks": [],
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                record.setdefault("created_at", existing.get("created_at") or datetime.now(timezone.utc).isoformat())
                self._save_task_record(task.task_id, record)
                return
            base_dir = self._base_dir(task.task_id)
            track_records: list[Dict[str, Any]] = []
            for idx, track in enumerate(task.items, start=1):
                track_id = track.id or str(idx)
                title = track.title or (meta.title if meta else None) or f"Track {idx}"
                audio_path = self._find_local_file(base_dir, track_id)
                if audio_path and self._send_audio(chat_id, audio_path, title=title, reply_to=None):
                    log.info("Suno audio sent | task=%s track=%s path=%s", task.task_id, track_id, audio_path)
                elif track.audio_url:
                    text = f"🔗 Аудио ({title}): {track.audio_url}"
                    self._send_text(chat_id, text)
                image_path = self._find_local_file(base_dir, f"{track_id}_cover")
                if image_path and self._send_image(chat_id, image_path, title=title, reply_to=None):
                    log.info("Suno cover sent | task=%s track=%s path=%s", task.task_id, track_id, image_path)
                elif track.image_url:
                    self._send_text(chat_id, f"🖼️ Обложка ({title}): {track.image_url}")
                track_records.append(
                    {
                        "id": track_id,
                        "title": title,
                        "audio_url": track.audio_url,
                        "image_url": track.image_url,
                    }
                )
            existing = self._load_task_record(task.task_id) or {}
            record = dict(existing)
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
                    "tracks": track_records,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            record.setdefault("created_at", existing.get("created_at") or datetime.now(timezone.utc).isoformat())
            self._save_task_record(task.task_id, record)
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


__all__ = ["SunoService", "SunoClient", "SunoAPIError", "SunoTask", "SunoTrack"]
