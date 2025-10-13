"""Asynchronous Codex logging handler with Telegram fallback support."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import traceback
from typing import Any, Dict, List, Optional

import aiohttp


class AsyncCodexHandler(logging.Handler):
    """A logging handler that ships structured logs to Codex asynchronously."""

    _TELEGRAM_API_TEMPLATE = "https://api.telegram.org/bot{token}/sendMessage"

    def __init__(self) -> None:
        super().__init__()
        self.enabled = os.getenv("CODEX_LOG_ENABLED", "false").lower() == "true"
        self.endpoint = os.getenv("CODEX_LOG_ENDPOINT")
        self.api_key = os.getenv("CODEX_LOG_API_KEY")
        self.app_name = os.getenv("APP_NAME", "veo3-bot")
        self.app_env = os.getenv("APP_ENV", "prod")
        self.min_level = os.getenv("CODEX_LOG_MIN_LEVEL", "INFO")
        self.batch_size = int(os.getenv("CODEX_LOG_BATCH_SIZE", "20"))
        self.flush_sec = float(os.getenv("CODEX_LOG_FLUSH_SEC", "2"))
        self.queue: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue(
            maxsize=int(os.getenv("CODEX_LOG_MAX_QUEUE", "1000"))
        )
        self.session: Optional[aiohttp.ClientSession] = None
        self._loop_task: Optional[asyncio.Task[None]] = None
        self._flush_lock = asyncio.Lock()
        self._telegram_chat_id = os.getenv("TG_LOG_CHAT_ID")
        self._telegram_token = os.getenv("TELEGRAM_TOKEN")
        self._failure_streak = 0
        self._last_fallback_ts = 0.0

        if self.enabled and self.endpoint and self.api_key:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.get_event_loop()
            self._loop_task = loop.create_task(self._loop())

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=10)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session

    async def _loop(self) -> None:
        """Background loop flushing queued records to Codex."""

        try:
            await self._ensure_session()
        except Exception as exc:  # pragma: no cover - defensive
            logging.getLogger("codex").warning("session init failed: %s", exc)
            return

        try:
            while True:
                await asyncio.sleep(self.flush_sec)
                await self.flush()
        except asyncio.CancelledError:  # pragma: no cover - graceful shutdown
            raise
        except Exception as exc:  # pragma: no cover - defensive logging
            logging.getLogger("codex").warning("flush loop crashed: %s", exc)

    async def emit_async(self, record: logging.LogRecord) -> None:
        if not self.enabled or not self.endpoint:
            return

        message = self.format(record)
        payload: Dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": message,
            "stack": traceback.format_exc() if record.exc_info else None,
            "meta": {
                "pid": os.getpid(),
                "env": self.app_env,
                "app": self.app_name,
            },
        }

        try:
            self.queue.put_nowait(payload)
        except asyncio.QueueFull:
            # Drop the message silently to avoid blocking the application.
            pass

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        if not self.enabled or not self.endpoint:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                return
        try:
            loop.create_task(self.emit_async(record))
        except RuntimeError:
            # The loop is not running (application shutdown) — drop the log silently.
            return

    async def flush(self) -> None:  # noqa: D401
        if not self.enabled or not self.endpoint:
            return
        if self.queue.empty():
            return

        async with self._flush_lock:
            if self.queue.empty():
                return

            batch: List[Dict[str, Any]] = []
            while not self.queue.empty() and len(batch) < self.batch_size:
                try:
                    batch.append(self.queue.get_nowait())
                except asyncio.QueueEmpty:
                    break

            if not batch:
                return

            session = await self._ensure_session()
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            try:
                async with session.post(self.endpoint, data=json.dumps(batch), headers=headers) as resp:
                    if resp.status >= 400:
                        raise RuntimeError(f"Codex returned {resp.status}")
            except Exception as exc:
                logging.getLogger("codex").warning("flush error: %s", exc)
                # Attempt to requeue the batch for future delivery.
                for item in batch:
                    try:
                        self.queue.put_nowait(item)
                    except asyncio.QueueFull:
                        break
                await self._handle_flush_failure(batch, str(exc))
            else:
                self._failure_streak = 0

    async def _handle_flush_failure(self, batch: List[Dict[str, Any]], reason: str) -> None:
        self._failure_streak += 1
        if (
            self._failure_streak <= 3
            or not self._telegram_chat_id
            or not self._telegram_token
        ):
            return

        now = time.time()
        if now - self._last_fallback_ts < 60:
            return

        session = await self._ensure_session()
        sample = batch[0] if batch else {}
        sample_message = sample.get("msg", "") if isinstance(sample, dict) else ""
        text_lines = [
            "⚠️ Codex fallback triggered",
            f"Reason: {reason}",
            f"Batch size: {len(batch)}",
        ]
        if sample_message:
            truncated = sample_message[:500]
            if len(sample_message) > 500:
                truncated += "…"
            text_lines.append("")
            text_lines.append(truncated)

        payload = {
            "chat_id": self._telegram_chat_id,
            "text": "\n".join(text_lines),
            "disable_web_page_preview": True,
        }

        url = self._TELEGRAM_API_TEMPLATE.format(token=self._telegram_token)
        try:
            async with session.post(url, json=payload) as resp:
                if resp.status < 400:
                    self._last_fallback_ts = now
        except Exception:
            # Ignore Telegram errors entirely — we don't want cascading failures.
            return


def attach_codex_handler() -> None:
    handler = AsyncCodexHandler()
    if not handler.enabled:
        return

    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(getattr(logging, handler.min_level.upper(), logging.INFO))

    for name in [
        "veo3-bot",
        "db.postgres",
        "asyncio",
        "aiohttp",
        "telegram",
        "redis",
        "apscheduler",
    ]:
        logging.getLogger(name).addHandler(handler)

    logging.getLogger("codex").info("Codex log handler attached")
