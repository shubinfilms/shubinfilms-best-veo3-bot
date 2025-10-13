"""Asynchronous Codex logging handler with Telegram fallback support."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
import traceback
from typing import Any, Dict, List, Optional

import aiohttp


class AsyncCodexHandler(logging.Handler):
    """A logging handler that ships structured logs to Codex asynchronously."""

    _TELEGRAM_API_TEMPLATE = "https://api.telegram.org/bot{token}/sendMessage"
    _STOP_SENTINEL: object = object()
    _FLUSH_SENTINEL: object = object()

    def __init__(self) -> None:
        super().__init__()
        self.enabled = os.getenv("CODEX_LOG_ENABLED", "false").lower() == "true"
        self.endpoint = os.getenv("CODEX_LOG_ENDPOINT")
        self.api_key = os.getenv("CODEX_LOG_API_KEY")
        self.app_name = os.getenv("APP_NAME", "veo3-bot")
        self.app_env = os.getenv("APP_ENV", "prod")
        self.min_level = os.getenv("CODEX_LOG_MIN_LEVEL", "INFO")
        self.batch_size = max(1, int(os.getenv("CODEX_LOG_BATCH_SIZE", "20")))
        self.flush_sec = max(0.5, float(os.getenv("CODEX_LOG_FLUSH_SEC", "2")))
        self.max_queue = max(1, int(os.getenv("CODEX_LOG_MAX_QUEUE", "1000")))

        self.queue: Optional[asyncio.Queue[Dict[str, Any]]] = None
        self.session: Optional[aiohttp.ClientSession] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._worker_done: Optional[asyncio.Event] = None
        self._thread: Optional[threading.Thread] = None
        self._queue_ready = threading.Event()
        self._closing = False
        self._failure_streak = 0
        self._last_fallback_ts = 0.0
        self._telegram_chat_id = os.getenv("TG_LOG_CHAT_ID")
        self._telegram_token = os.getenv("TELEGRAM_TOKEN")

        if self.enabled and self.endpoint and self.api_key:
            self._start_worker()

    # ------------------------------------------------------------------
    # Worker lifecycle
    # ------------------------------------------------------------------
    def _start_worker(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run_loop,
            name="codex-logger",
            daemon=True,
        )
        self._thread.start()
        self._queue_ready.wait(timeout=5)

    def _run_loop(self) -> None:
        try:
            asyncio.run(self._worker_main())
        except Exception as exc:  # pragma: no cover - defensive logging
            logging.getLogger("codex").warning("worker loop crashed: %s", exc)

    async def _worker_main(self) -> None:
        self._loop = asyncio.get_running_loop()
        self.queue = asyncio.Queue(maxsize=self.max_queue)
        self._worker_done = asyncio.Event()
        self._queue_ready.set()
        try:
            await self._worker()
        finally:
            if self.session and not self.session.closed:
                await self.session.close()
            self.session = None
            self.queue = None
            self._loop = None
            self._worker_done.set()
            self._queue_ready.clear()

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=10)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session

    # ------------------------------------------------------------------
    # Logging.Handler API
    # ------------------------------------------------------------------
    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        if (
            not self.enabled
            or not self.endpoint
            or not self.api_key
            or record.name == "codex"
        ):
            return
        if not self._queue_ready.is_set() or self.queue is None or self._loop is None:
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

        def _put() -> None:
            if self.queue is None or self._closing:
                return
            try:
                self.queue.put_nowait(payload)
            except asyncio.QueueFull:
                pass

        self._loop.call_soon_threadsafe(_put)

    def flush(self) -> None:  # noqa: D401
        if not self.enabled or not self._queue_ready.is_set() or not self._loop:
            return
        try:
            future = asyncio.run_coroutine_threadsafe(self._flush_async(), self._loop)
            future.result()
        except Exception as exc:
            logging.getLogger("codex").warning("flush failed: %s", exc)

    def close(self) -> None:
        try:
            self.flush()
        except Exception:
            pass
        if self.enabled and self._loop:
            try:
                future = asyncio.run_coroutine_threadsafe(self._shutdown_async(), self._loop)
                future.result()
            except Exception as exc:
                logging.getLogger("codex").warning("shutdown failed: %s", exc)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._thread = None
        super().close()

    # ------------------------------------------------------------------
    # Flush & shutdown helpers
    # ------------------------------------------------------------------
    async def _flush_async(self) -> None:
        if not self.enabled or self.queue is None or self._closing:
            return
        await self.queue.put(self._FLUSH_SENTINEL)
        await self.queue.join()

    async def _shutdown_async(self) -> None:
        if self.queue is None or self._closing:
            return
        self._closing = True
        await self.queue.put(self._STOP_SENTINEL)
        await self.queue.join()
        if self._worker_done is not None:
            await self._worker_done.wait()

    # ------------------------------------------------------------------
    # Worker implementation
    # ------------------------------------------------------------------
    async def _worker(self) -> None:
        if self.queue is None:
            return
        pending: List[Dict[str, Any]] = []
        loop = asyncio.get_running_loop()
        next_flush: Optional[float] = None

        try:
            while True:
                timeout: Optional[float] = None
                if pending:
                    timeout = max(0.0, next_flush - loop.time()) if next_flush else 0.0
                try:
                    item = await asyncio.wait_for(self.queue.get(), timeout=timeout)
                    got_item = True
                except asyncio.TimeoutError:
                    item = None
                    got_item = False

                if got_item:
                    if item is self._STOP_SENTINEL:
                        await self._flush_pending(pending)
                        self.queue.task_done()
                        break
                    if item is self._FLUSH_SENTINEL:
                        await self._flush_pending(pending)
                        self.queue.task_done()
                        next_flush = None
                        continue

                    pending.append(item)
                    if len(pending) == 1:
                        next_flush = loop.time() + self.flush_sec
                    if len(pending) >= self.batch_size:
                        await self._flush_pending(pending)
                        next_flush = None
                    continue

                if pending:
                    await self._flush_pending(pending)
                    next_flush = None
        finally:
            await self._flush_pending(pending)

    async def _flush_pending(self, pending: List[Dict[str, Any]]) -> None:
        if not pending or self.queue is None:
            return
        batch = list(pending)
        success = await self._send_batch(batch)
        if not success:
            for item in batch:
                try:
                    self.queue.put_nowait(item)
                except asyncio.QueueFull:
                    break
        for _ in batch:
            self.queue.task_done()
        pending.clear()

    async def _send_batch(self, batch: List[Dict[str, Any]]) -> bool:
        if not batch or not self.endpoint or not self.api_key:
            return True
        try:
            session = await self._ensure_session()
        except Exception as exc:  # pragma: no cover - defensive
            logging.getLogger("codex").warning("session init failed: %s", exc)
            return False

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = json.dumps(batch, ensure_ascii=False)

        try:
            async with session.post(self.endpoint, data=payload, headers=headers) as resp:
                if resp.status >= 400:
                    raise RuntimeError(f"Codex returned {resp.status}")
        except Exception as exc:
            logging.getLogger("codex").warning("flush error: %s", exc)
            await self._handle_flush_failure(batch, str(exc))
            return False

        self._failure_streak = 0
        return True

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

        try:
            session = await self._ensure_session()
        except Exception:  # pragma: no cover - defensive
            return

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
        except Exception:  # pragma: no cover - best effort fallback
            return


def attach_codex_handler() -> None:
    handler = AsyncCodexHandler()
    if not handler.enabled:
        return

    handler.setLevel(getattr(logging, handler.min_level.upper(), logging.INFO))

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
