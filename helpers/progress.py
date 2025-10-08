"""Utility helpers for sending progress updates to users."""

from __future__ import annotations

import logging
from contextlib import suppress
from typing import Any, MutableMapping, Optional

from telegram.ext import ContextTypes


logger = logging.getLogger("progress")

PROGRESS_STORAGE_KEY = "_progress_message"
_PROGRESS_CLEANUP_DELAY = 5 * 60.0


def _get_progress_state(
    ctx: ContextTypes.DEFAULT_TYPE,
) -> Optional[MutableMapping[str, Any]]:
    chat_data = getattr(ctx, "chat_data", None)
    if not isinstance(chat_data, MutableMapping):
        return None
    state = chat_data.get(PROGRESS_STORAGE_KEY)
    if not isinstance(state, MutableMapping):
        return None
    return state


async def _auto_cleanup(job_ctx: ContextTypes.DEFAULT_TYPE) -> None:
    data = getattr(job_ctx, "data", None) or {}
    chat_id = data.get("chat_id")
    message_id = data.get("message_id")
    if not chat_id or not message_id:
        return
    try:
        await job_ctx.bot.delete_message(chat_id=chat_id, message_id=message_id)
    except Exception:  # pragma: no cover - defensive cleanup
        logger.debug("progress.cleanup_fail chat_id=%s", chat_id)
    chat_data = getattr(job_ctx, "chat_data", None)
    if isinstance(chat_data, MutableMapping):
        stored = chat_data.get(PROGRESS_STORAGE_KEY)
        if isinstance(stored, MutableMapping) and stored.get("message_id") == message_id:
            chat_data.pop(PROGRESS_STORAGE_KEY, None)


def _schedule_cleanup(
    ctx: ContextTypes.DEFAULT_TYPE,
    progress: MutableMapping[str, Any],
    *,
    message_id: int,
) -> Any:
    job_queue = getattr(ctx, "job_queue", None)
    if job_queue is None:
        return None
    data = {
        "chat_id": progress.get("chat_id"),
        "message_id": message_id,
        "mode": progress.get("mode"),
        "user_id": progress.get("user_id"),
        "job_id": progress.get("job_id"),
    }
    return job_queue.run_once(_auto_cleanup, _PROGRESS_CLEANUP_DELAY, data=data)


async def send_progress_message(ctx: ContextTypes.DEFAULT_TYPE, phase: str) -> None:
    """Send or update a progress indicator message for the current chat."""

    progress = _get_progress_state(ctx)
    if not progress:
        return

    bot = getattr(ctx, "bot", None)
    if bot is None:
        return

    chat_id = progress.get("chat_id")
    if chat_id is None:
        return

    user_id = progress.get("user_id")
    mode = progress.get("mode") or "unknown"
    job_id = progress.get("job_id")
    last_phase = progress.get("phase")

    if phase == "start":
        if last_phase == "start":
            return
        reply_to = progress.get("reply_to_message_id")
        params = {
            "chat_id": chat_id,
            "text": "🎬 Отправляю задачу в обработку…",
            "disable_notification": True,
        }
        if reply_to:
            params["reply_to_message_id"] = reply_to
            params["allow_sending_without_reply"] = True
        try:
            message = await bot.send_message(**params)
        except Exception:  # pragma: no cover - defensive guard
            logger.exception("progress.start_fail user_id=%s chat_id=%s", user_id, chat_id)
            return
        message_id = getattr(message, "message_id", None)
        if message_id is not None:
            progress["message_id"] = int(message_id)
            cleanup_job = _schedule_cleanup(ctx, progress, message_id=int(message_id))
            if cleanup_job is not None:
                progress["cleanup_job"] = cleanup_job
        progress["phase"] = "start"
        logger.info(
            "progress.start user_id=%s chat_id=%s mode=%s job_id=%s",
            user_id,
            chat_id,
            mode,
            job_id,
        )
        return

    message_id = progress.get("message_id")
    if not message_id:
        return

    if phase == "render":
        if last_phase == "render":
            return
        try:
            await bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text="💫 Видео обрабатывается — оставайтесь на связи.",
            )
        except Exception:  # pragma: no cover - defensive guard
            logger.debug("progress.render_fail user_id=%s chat_id=%s", user_id, chat_id)
        else:
            progress["phase"] = "render"
            logger.info(
                "progress.render user_id=%s chat_id=%s mode=%s job_id=%s",
                user_id,
                chat_id,
                mode,
                job_id,
            )
        return

    if phase == "finish":
        cleanup_job = progress.get("cleanup_job")
        if cleanup_job is not None:
            with suppress(Exception):
                cleanup_job.schedule_removal()
        success = bool(progress.get("success"))
        try:
            if success:
                await bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=message_id,
                    text="✅ Готово!",
                )
            else:
                await bot.delete_message(chat_id=chat_id, message_id=message_id)
        except Exception:  # pragma: no cover - defensive guard
            logger.debug("progress.finish_fail user_id=%s chat_id=%s", user_id, chat_id)
        logger.info(
            "progress.finish user_id=%s chat_id=%s mode=%s job_id=%s",
            user_id,
            chat_id,
            mode,
            job_id,
        )
        chat_data = getattr(ctx, "chat_data", None)
        if isinstance(chat_data, MutableMapping):
            chat_data.pop(PROGRESS_STORAGE_KEY, None)

