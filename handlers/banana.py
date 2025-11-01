"""Banana image editing handler."""
from __future__ import annotations

import asyncio

from logging_utils import get_logger
from services.banana_client import banana_process, banana_upload_bytes
from utils.files import tg_image_bytes, validate_image
from utils.notify import admin_warn, user_error

log = get_logger("handlers.banana")

MODEL = "banana-nana"


async def handle_banana_start(message, context) -> None:
    bot = context.bot
    chat_id = message.chat_id
    try:
        data = await tg_image_bytes(message, bot)
        validate_image(data)
    except Exception as exc:
        await user_error(
            bot,
            chat_id,
            "Отправьте фото как *файл* (без сжатия) — текущее изображение не удалось прочитать.",
        )
        await admin_warn(bot, f"Banana input validate fail: {exc}")
        return

    upload_id: str | None = None
    try:
        upload = await banana_upload_bytes(bytes(data), filename="input.png")
        upload_id = upload.get("upload_id") or upload.get("id") or upload.get("temp_url")
        if not upload_id:
            raise RuntimeError(f"upload response malformed: {upload}")
        response = await banana_process(MODEL, {"upload_id": upload_id})
    except Exception:
        await asyncio.sleep(1.0)
        try:
            response = await banana_process(MODEL, {"upload_id": upload_id})
        except Exception as retry_exc:
            await user_error(
                bot,
                chat_id,
                "Сервис обработки временно недоступен. Попробуйте ещё раз.",
            )
            await admin_warn(bot, f"Banana 500: {retry_exc}")
            return

    log.debug("banana.process", extra={"chat_id": chat_id, "response": response})
    # Further response handling should happen here.
