"""Async logger helper that mirrors events to Redis debug channel."""

from __future__ import annotations

import json
import logging
from typing import Any, Mapping, MutableMapping, Optional

_DEFAULT_CHANNEL = "bot:debug"


class BotLogger:
    """Emit logs to standard logging and optionally publish to Redis."""

    def __init__(
        self,
        *,
        redis: Any = None,
        channel: str = _DEFAULT_CHANNEL,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._redis = redis
        self._channel = channel
        base = logger or logging.getLogger("bot.async")
        self._logger = base

    async def info(self, message: str, **fields: Any) -> None:
        """Log an info message and mirror to Redis."""

        self._emit(logging.INFO, message, fields)
        await self._publish("info", message, fields)

    async def error(self, message: str, **fields: Any) -> None:
        """Log an error message and mirror to Redis."""

        self._emit(logging.ERROR, message, fields)
        await self._publish("error", message, fields)

    def _emit(self, level: int, message: str, fields: Mapping[str, Any]) -> None:
        extra: MutableMapping[str, Any] = {"meta": {"ctx": dict(fields)}} if fields else {}
        self._logger.log(level, message, extra=extra or None)

    async def _publish(self, level: str, message: str, fields: Mapping[str, Any]) -> None:
        if self._redis is None:
            return
        publish = getattr(self._redis, "publish", None)
        if publish is None:
            return
        payload = {"level": level, "message": message, **dict(fields)}
        try:
            data = json.dumps(payload)
        except (TypeError, ValueError):
            data = json.dumps({"level": level, "message": message})
        try:
            result = publish(self._channel, data)
            if hasattr(result, "__await"):
                await result  # type: ignore[misc]
        except Exception:
            self._logger.debug("bot_logger.publish_failed", exc_info=True)


__all__ = ["BotLogger"]
