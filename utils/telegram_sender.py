"""Asynchronous Telegram sender with global and per-chat rate limits."""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional

from telegram.error import BadRequest

log = logging.getLogger("telegram.sender")


def _read_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        log.warning("telegram.sender.invalid_env", extra={"name": name, "value": raw})
        return float(default)
    if value <= 0:
        return float(default)
    return value


@dataclass(slots=True)
class _Request:
    target_chat_id: Optional[int]
    method: Callable[..., Awaitable[Any]]
    kwargs: Dict[str, Any]
    method_name: str
    kind: str
    future: asyncio.Future[Any]
    log_context: Dict[str, Any]


class TelegramSender:
    """Serialize Telegram API calls through a rate-limited queue."""

    def __init__(self) -> None:
        self._global_rps = _read_float("TG_GLOBAL_RPS", 25.0)
        self._per_chat_rps = _read_float("TG_PER_CHAT_RPS", 1.0)
        self._queue: "asyncio.Queue[_Request]" = asyncio.Queue()
        self._worker: Optional[asyncio.Task[None]] = None
        self._stopping = asyncio.Event()
        self._global_next: float = 0.0
        self._per_chat_next: Dict[int, float] = {}
        self._stats_lock = asyncio.Lock()
        self._inflight: int = 0

    # ------------------------------------------------------------------
    # configuration helpers
    # ------------------------------------------------------------------
    def configure_from_env(self) -> None:
        self._global_rps = _read_float("TG_GLOBAL_RPS", 25.0)
        self._per_chat_rps = _read_float("TG_PER_CHAT_RPS", 1.0)

    # ------------------------------------------------------------------
    # lifecycle management
    # ------------------------------------------------------------------
    async def start(self) -> None:
        if self._worker is not None and not self._worker.done():
            return
        self._stopping.clear()
        loop = asyncio.get_running_loop()
        self._worker = loop.create_task(self._run(), name="telegram-sender")

    async def stop(self) -> None:
        self._stopping.set()
        if self._worker is None:
            return
        self._worker.cancel()
        try:
            await self._worker
        except asyncio.CancelledError:  # pragma: no cover - expected on shutdown
            pass
        self._worker = None

    # ------------------------------------------------------------------
    # queue submission
    # ------------------------------------------------------------------
    async def submit(
        self,
        target_chat_id: Optional[int],
        method: Callable[..., Awaitable[Any]],
        *,
        method_name: str,
        kind: str,
        log_context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        if "chat_id" in kwargs:
            raise TypeError("Do not pass 'chat_id' in kwargs; use positional target_chat_id")
        if self._worker is None or self._worker.done():
            await self.start()

        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        payload = _Request(
            target_chat_id=target_chat_id,
            method=method,
            kwargs=dict(kwargs),
            method_name=method_name,
            kind=kind,
            future=future,
            log_context=dict(log_context or {}),
        )
        await self._queue.put(payload)
        return await future

    # ------------------------------------------------------------------
    # diagnostics
    # ------------------------------------------------------------------
    async def snapshot(self) -> Dict[str, Any]:
        async with self._stats_lock:
            inflight = self._inflight
        return {
            "queue_depth": self._queue.qsize(),
            "inflight": inflight,
            "global_rps": self._global_rps,
            "per_chat_rps": self._per_chat_rps,
        }

    # ------------------------------------------------------------------
    # worker implementation
    # ------------------------------------------------------------------
    async def _run(self) -> None:
        try:
            while not self._stopping.is_set():
                request = await self._queue.get()
                if request is None:  # pragma: no cover - defensive
                    continue
                await self._wait_for_slot(request.target_chat_id)
                async with self._stats_lock:
                    self._inflight += 1
                try:
                    call_kwargs = dict(request.kwargs)
                    if request.target_chat_id is not None:
                        call_kwargs.setdefault("chat_id", request.target_chat_id)
                    result = await request.method(**call_kwargs)
                except BadRequest as exc:
                    lowered = str(exc).lower()
                    if "message is not modified" in lowered:
                        log.debug(
                            "telegram.sender.not_modified",
                            extra={"meta": {"method": request.method_name, **request.log_context}},
                        )
                        request.future.set_result(None)
                    elif "message to edit not found" in lowered and request.target_chat_id is not None:
                        log.info(
                            "telegram.sender.edit_missing",
                            extra={"meta": {"chat_id": request.target_chat_id, **request.log_context}},
                        )
                        request.future.set_exception(exc)
                    else:
                        request.future.set_exception(exc)
                except Exception as exc:  # pragma: no cover - passthrough
                    request.future.set_exception(exc)
                else:
                    request.future.set_result(result)
                finally:
                    async with self._stats_lock:
                        self._inflight -= 1
                    self._queue.task_done()
        except asyncio.CancelledError:  # pragma: no cover - expected on shutdown
            pass

    async def _wait_for_slot(self, chat_id: Optional[int]) -> None:
        min_delay = 0.0
        now = time.monotonic()
        if self._global_rps > 0:
            interval = 1.0 / self._global_rps
            if self._global_next > now:
                min_delay = max(min_delay, self._global_next - now)
            self._global_next = max(self._global_next, now) + interval
        if chat_id is not None and self._per_chat_rps > 0:
            interval = 1.0 / self._per_chat_rps
            next_allowed = self._per_chat_next.get(chat_id, 0.0)
            if next_allowed > now:
                min_delay = max(min_delay, next_allowed - now)
            self._per_chat_next[chat_id] = max(next_allowed, now) + interval
        if min_delay > 0:
            await asyncio.sleep(min_delay)


_SENDER = TelegramSender()


def get_sender() -> TelegramSender:
    return _SENDER


async def start_sender() -> None:
    await _SENDER.start()


async def stop_sender() -> None:
    await _SENDER.stop()


async def sender_snapshot() -> Dict[str, Any]:
    return await _SENDER.snapshot()

