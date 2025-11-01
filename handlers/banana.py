"""Handlers for Banana image editing workflow."""

from __future__ import annotations

from typing import Optional

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from core.balance_provider import aget_balance_snapshot
from logging_utils import get_logger
from services.banana_client import banana_process, banana_upload_bytes
from ui.renderers.banana import banana_card_kb, banana_card_text
from utils.banana_state import BananaState, clear, load, save
from utils.files import validate_image
from utils.notify import admin_warn, user_error

log = get_logger("handlers.banana")

MODEL_NAME = "banana-nana"
MAX_IMAGES = 4


async def _require_redis(context) -> object:
    redis = getattr(context, "redis", None)
    if redis is None:
        raise RuntimeError("context.redis is not configured")
    return redis


async def _get_balance_value(user_id: int) -> int:
    try:
        snapshot = await aget_balance_snapshot(int(user_id))
    except Exception:  # pragma: no cover - defensive fallback
        log.exception("banana.balance_failed", extra={"user_id": user_id})
        return 0
    return int(snapshot.value or 0)


async def _fetch_file_bytes(bot, file_id: str) -> bytes:
    tg_file = await bot.get_file(file_id)
    return await tg_file.download_as_bytearray()


async def open_banana_card(update, context) -> None:
    user_id = update.effective_user.id
    redis = await _require_redis(context)
    state = await load(redis, user_id)
    balance_value = await _get_balance_value(user_id)
    text = banana_card_text(balance_value, state)
    await context.bot.send_message(user_id, text, reply_markup=banana_card_kb(state))


async def on_banana_photo(update, context) -> None:
    message = update.effective_message
    if message is None:
        return
    user_id = update.effective_user.id
    redis = await _require_redis(context)
    state = await load(redis, user_id)
    if len(state.images) >= MAX_IMAGES:
        await user_error(
            context.bot,
            user_id,
            f"Максимум {MAX_IMAGES} фото. Удалите лишнее или начните новую генерацию.",
        )
        return
    file_id: Optional[str] = None
    if message.photo:
        file_id = message.photo[-1].file_id
    elif getattr(message, "document", None) is not None:
        document = message.document
        if document and (document.mime_type or "").startswith("image/"):
            file_id = document.file_id
    if not file_id:
        await user_error(
            context.bot,
            user_id,
            "Пришлите фото как изображение или файл без сжатия.",
        )
        return
    state.images.append(file_id)
    await save(redis, user_id, state)
    balance_value = await _get_balance_value(user_id)
    await message.reply_text(
        banana_card_text(balance_value, state), reply_markup=banana_card_kb(state)
    )


async def on_banana_text(update, context) -> None:
    message = update.effective_message
    if message is None or not message.text:
        return
    user_id = update.effective_user.id
    redis = await _require_redis(context)
    state = await load(redis, user_id)
    state.prompt = message.text.strip()
    await save(redis, user_id, state)
    balance_value = await _get_balance_value(user_id)
    await message.reply_text(
        banana_card_text(balance_value, state), reply_markup=banana_card_kb(state)
    )


async def on_banana_start(update, context) -> None:
    query = update.callback_query
    if query is None:
        return
    await query.answer()
    user_id = query.from_user.id
    redis = await _require_redis(context)
    state = await load(redis, user_id)
    if not state.ready:
        await user_error(context.bot, user_id, "Добавьте фото и промпт.")
        return

    uploads: list[str] = []
    for file_id in state.images:
        try:
            data = await _fetch_file_bytes(context.bot, file_id)
        except Exception as exc:
            await admin_warn(context.bot, f"Banana tg fetch fail: {exc}")
            continue
        try:
            validate_image(data)
            response = await banana_upload_bytes(bytes(data), filename="input.png")
        except Exception as exc:
            await admin_warn(context.bot, f"Banana upload fail: {exc}")
            continue
        upload_id = (
            response.get("upload_id")
            or response.get("id")
            or response.get("temp_url")
        )
        if upload_id:
            uploads.append(str(upload_id))

    if not uploads:
        await user_error(
            context.bot,
            user_id,
            "Не удалось загрузить изображения. Пришлите фото как файл и попробуйте снова.",
        )
        return

    payload = {"upload_ids": uploads, "prompt": state.prompt}
    try:
        result = await banana_process(MODEL_NAME, payload)
        result_bytes = result.get("result_bytes")
        if not result_bytes:
            raise RuntimeError("Banana response missing result_bytes")
        message = await context.bot.send_document(user_id, ("result.png", result_bytes))
    except Exception as exc:
        await user_error(context.bot, user_id, "Сервис занят, попробуйте ещё раз.")
        await admin_warn(context.bot, f"Banana process fail: {exc}")
        return

    state.last_result_id = str(message.message_id)
    await save(redis, user_id, state)

    keyboard = InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("🔁 Сгенерировать ещё", callback_data="banana:again")],
            [InlineKeyboardButton("🆕 Новая генерация", callback_data="banana:new")],
        ]
    )
    await context.bot.send_message(user_id, "Готово ✅", reply_markup=keyboard)


async def on_banana_again(update, context) -> None:
    query = update.callback_query
    if query is None:
        return
    await query.answer()
    await on_banana_start(update, context)


async def on_banana_new(update, context) -> None:
    query = update.callback_query
    if query is None:
        return
    await query.answer()
    user_id = query.from_user.id
    redis = await _require_redis(context)
    await clear(redis, user_id)
    state = BananaState(images=[], prompt=None)
    await save(redis, user_id, state)
    balance_value = await _get_balance_value(user_id)
    await context.bot.send_message(
        user_id,
        "Новая карточка 🆕 Отправьте фото и промпт.",
        reply_markup=banana_card_kb(state),
    )


__all__ = [
    "open_banana_card",
    "on_banana_photo",
    "on_banana_text",
    "on_banana_start",
    "on_banana_again",
    "on_banana_new",
]
