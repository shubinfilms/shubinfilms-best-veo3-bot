import logging
import time
from contextlib import suppress
from threading import Lock
from typing import Optional

from redis_utils import rds
from settings import (
    MJ_WAIT_STICKER_ID,
    PROMO_OK_STICKER_ID,
    PROMPTMASTER_WAIT_STICKER_ID,
    PURCHASE_OK_STICKER_ID,
    REDIS_PREFIX,
    SORA2_WAIT_STICKER_ID,
    SUNO_WAIT_STICKER_ID,
    VEO_WAIT_STICKER_ID,
)

_LOGGER = logging.getLogger("stickers")

_WAIT_KEY_TMPL = f"{REDIS_PREFIX}:wait-sticker:{{}}"
_WAIT_TTL_SECONDS = 6 * 60 * 60  # 6 hours

_WAIT_MEMORY: dict[int, tuple[float, int]] = {}
_WAIT_MEMORY_LOCK = Lock()

_WAIT_STICKERS = {
    "veo": VEO_WAIT_STICKER_ID,
    "sora2": SORA2_WAIT_STICKER_ID,
    "suno": SUNO_WAIT_STICKER_ID,
    "mj": MJ_WAIT_STICKER_ID,
    "promptmaster": PROMPTMASTER_WAIT_STICKER_ID,
}

_OK_STICKERS = {
    "purchase": PURCHASE_OK_STICKER_ID,
    "promo": PROMO_OK_STICKER_ID,
}


def _wait_key(chat_id: int) -> str:
    return _WAIT_KEY_TMPL.format(int(chat_id))


def _memory_set(chat_id: int, message_id: int) -> None:
    expires_at = time.time() + _WAIT_TTL_SECONDS
    with _WAIT_MEMORY_LOCK:
        _WAIT_MEMORY[int(chat_id)] = (expires_at, int(message_id))


def _memory_pop(chat_id: int) -> Optional[int]:
    now = time.time()
    with _WAIT_MEMORY_LOCK:
        entry = _WAIT_MEMORY.pop(int(chat_id), None)
    if not entry:
        return None
    expires_at, message_id = entry
    if expires_at < now:
        return None
    return message_id


def _store_wait_message_id(chat_id: int, message_id: int) -> None:
    key = _wait_key(chat_id)
    if rds:
        try:
            rds.setex(key, max(1, _WAIT_TTL_SECONDS), str(int(message_id)))
            return
        except Exception as exc:  # pragma: no cover - Redis failure fallback
            _LOGGER.warning("stickers.redis.store_failed", extra={"chat_id": chat_id, "error": str(exc)})
    _memory_set(chat_id, message_id)


def pop_wait_sticker_id(chat_id: Optional[int]) -> Optional[int]:
    if chat_id is None:
        return None
    key = _wait_key(chat_id)
    if rds:
        try:
            with rds.pipeline() as pipe:
                pipe.get(key)
                pipe.delete(key)
                stored, _ = pipe.execute()
            if stored is not None:
                try:
                    message_id = int(stored)
                except (TypeError, ValueError):
                    message_id = None
                if message_id:
                    return message_id
        except Exception as exc:  # pragma: no cover - Redis failure fallback
            _LOGGER.warning(
                "stickers.redis.pop_failed", extra={"chat_id": chat_id, "error": str(exc)}
            )
    return _memory_pop(chat_id)


def _resolve_chat_id(ctx, chat_id: Optional[int]) -> Optional[int]:
    if chat_id is not None:
        return int(chat_id)
    if ctx is None:
        return None
    candidate = getattr(ctx, "chat", None)
    if candidate is not None:
        value = getattr(candidate, "id", None)
        if value is not None:
            return int(value)
    chat_data = getattr(ctx, "chat_data", None)
    if isinstance(chat_data, dict):
        stored = chat_data.get("chat_id") or chat_data.get("_chat_id")
        if stored is not None:
            return int(stored)
    return None


async def delete_wait_sticker(ctx, *, chat_id: Optional[int] = None) -> None:
    resolved_chat_id = _resolve_chat_id(ctx, chat_id)
    if resolved_chat_id is None:
        return
    message_id = pop_wait_sticker_id(resolved_chat_id)
    if not message_id:
        return
    with suppress(Exception):
        await ctx.bot.delete_message(resolved_chat_id, int(message_id))


async def send_wait_sticker(ctx, mode: str, *, chat_id: Optional[int] = None) -> int:
    resolved_chat_id = _resolve_chat_id(ctx, chat_id)
    if resolved_chat_id is None:
        return 0
    normalized_mode = (mode or "").strip().lower()
    sticker_id = _WAIT_STICKERS.get(normalized_mode)
    if not sticker_id:
        _LOGGER.warning("stickers.wait.unknown_mode", extra={"mode": mode, "chat_id": resolved_chat_id})
        return 0
    await delete_wait_sticker(ctx, chat_id=resolved_chat_id)
    try:
        message = await ctx.bot.send_sticker(resolved_chat_id, sticker_id)
    except Exception as exc:
        _LOGGER.warning(
            "stickers.wait.send_failed", extra={"mode": normalized_mode, "chat_id": resolved_chat_id, "error": str(exc)}
        )
        return 0
    message_id = getattr(message, "message_id", None)
    if isinstance(message_id, int) and message_id > 0:
        _store_wait_message_id(resolved_chat_id, message_id)
        return message_id
    return 0


async def send_ok_sticker(ctx, kind: str, balance: int, *, chat_id: Optional[int] = None) -> None:
    resolved_chat_id = _resolve_chat_id(ctx, chat_id)
    if resolved_chat_id is None:
        return
    normalized_kind = (kind or "").strip().lower()
    sticker_id = _OK_STICKERS.get(normalized_kind)
    if sticker_id:
        try:
            await ctx.bot.send_sticker(resolved_chat_id, sticker_id)
        except Exception as exc:
            _LOGGER.warning(
                "stickers.ok.send_failed",
                extra={"kind": normalized_kind, "chat_id": resolved_chat_id, "error": str(exc)},
            )
    try:
        await ctx.bot.send_message(resolved_chat_id, f"Ваш баланс: {balance}.")
    except Exception as exc:
        _LOGGER.warning(
            "stickers.ok.message_failed",
            extra={"kind": normalized_kind, "chat_id": resolved_chat_id, "error": str(exc)},
        )
