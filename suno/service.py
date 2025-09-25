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

from suno.client import SunoClient, SunoAPIError
from suno.schemas import SunoTask, SunoTrack

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


@dataclass(slots=True)
class TelegramMeta:
    chat_id: int
    msg_id: int
    title: Optional[str]
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
        self._bot_session = requests.Session()

    # ------------------------------------------------------------------ storage
    def _redis_key(self, task_id: str) -> str:
        return f"suno:task:{task_id}"

    def _store_mapping(self, task_id: str, payload: Mapping[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False)
        key = self._redis_key(task_id)
        if self.redis is not None:
            try:
                self.redis.setex(key, _TASK_TTL, raw)
                return
            except Exception:  # pragma: no cover - Redis failure
                log.warning("SunoService redis.setex failed", exc_info=True)
        expires_at = time.time() + _TASK_TTL
        self._memory[key] = (expires_at, raw)

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
        return TelegramMeta(chat_id=chat_id, msg_id=msg_id, title=title, ts=ts)

    def _delete_mapping(self, task_id: str) -> None:
        key = self._redis_key(task_id)
        if self.redis is not None:
            try:
                self.redis.delete(key)
            except Exception:  # pragma: no cover - Redis failure
                log.warning("SunoService redis.delete failed", exc_info=True)
        self._memory.pop(key, None)

    # ----------------------------------------------------------------- telegram
    def _bot_url(self, method: str) -> str:
        if not self.telegram_token:
            raise RuntimeError("TELEGRAM_TOKEN is not configured")
        return f"https://api.telegram.org/bot{self.telegram_token}/{method}"

    def _send_text(self, chat_id: int, text: str, *, reply_to: Optional[int] = None) -> None:
        payload: Dict[str, Any] = {"chat_id": chat_id, "text": text}
        if reply_to:
            payload["reply_to_message_id"] = reply_to
        try:
            resp = self._bot_session.post(self._bot_url("sendMessage"), json=payload, timeout=20)
            if not resp.ok:
                log.warning("Telegram sendMessage failed | status=%s text=%s", resp.status_code, resp.text)
        except requests.RequestException:
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
                log.warning("Telegram %s failed | status=%s text=%s", method, resp.status_code, resp.text)
                return False
            return True
        except FileNotFoundError:
            return False
        except requests.RequestException:
            log.warning("Telegram %s network error", method, exc_info=True)
            return False

    def _send_audio(self, chat_id: int, path: Path, *, title: str, reply_to: Optional[int]) -> bool:
        caption = f"🎵 {title}" if title else None
        return self._send_file("sendAudio", "audio", chat_id, path, caption=caption, reply_to=reply_to)

    def _send_image(self, chat_id: int, path: Path, *, title: str, reply_to: Optional[int]) -> bool:
        caption = f"🖼️ {title} (обложка)" if title else "🖼️ Обложка"
        return self._send_file("sendPhoto", "photo", chat_id, path, caption=caption, reply_to=reply_to)

    # ------------------------------------------------------------------ helpers
    def _stage_header(self, task: SunoTask) -> str:
        status = task.status or "unknown"
        return f"🎧 Suno: этап {status} получен."

    def _base_dir(self, task_id: str) -> Path:
        return Path("/tmp/suno") / task_id

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
    ) -> SunoTask:
        payload = {
            "title": title,
            "style": style,
            "lyrics": lyrics,
            "model": model,
            "instrumental": instrumental,
        }
        result = self.client.create_music(payload)
        task = SunoTask.from_payload(result)
        if not task.task_id:
            raise SunoAPIError("Suno did not return task_id", payload=result)
        meta = {
            "chat_id": int(chat_id),
            "msg_id": int(msg_id),
            "title": title,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        self._store_mapping(task.task_id, meta)
        log.info("Suno task stored | task_id=%s chat_id=%s msg_id=%s", task.task_id, chat_id, msg_id)
        return task

    def handle_callback(self, task: SunoTask) -> None:
        if not task.task_id:
            log.warning("Callback without task_id: %s", task)
            return
        meta = self._load_mapping(task.task_id)
        if not meta:
            log.info("No chat mapping for task %s", task.task_id)
            return
        if not self.telegram_token:
            log.warning("TELEGRAM_TOKEN missing; skip delivery for task %s", task.task_id)
            return
        try:
            header = self._stage_header(task)
            self._send_text(meta.chat_id, header, reply_to=meta.msg_id)
            if not task.items:
                return
            base_dir = self._base_dir(task.task_id)
            for idx, track in enumerate(task.items, start=1):
                track_id = track.id or str(idx)
                title = track.title or meta.title or f"Track {idx}"
                audio_path = self._find_local_file(base_dir, track_id)
                if audio_path and self._send_audio(meta.chat_id, audio_path, title=title, reply_to=None):
                    log.info("Suno audio sent | task=%s track=%s path=%s", task.task_id, track_id, audio_path)
                elif track.audio_url:
                    text = f"🔗 Аудио ({title}): {track.audio_url}"
                    self._send_text(meta.chat_id, text)
                image_path = self._find_local_file(base_dir, f"{track_id}_cover")
                if image_path and self._send_image(meta.chat_id, image_path, title=title, reply_to=None):
                    log.info("Suno cover sent | task=%s track=%s path=%s", task.task_id, track_id, image_path)
                elif track.image_url:
                    self._send_text(meta.chat_id, f"🖼️ Обложка ({title}): {track.image_url}")
        finally:
            if task.status.lower() == "complete":
                self._delete_mapping(task.task_id)


__all__ = ["SunoService", "SunoClient", "SunoAPIError", "SunoTask", "SunoTrack"]
