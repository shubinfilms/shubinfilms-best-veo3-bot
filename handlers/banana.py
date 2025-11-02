"""Handlers for the Banana image editing workflow."""

from __future__ import annotations

from typing import Any, Iterable, List, Optional

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from core.balance_provider import aget_balance_snapshot
from logging_utils import get_logger
from services.banana_client import banana_process, banana_upload_bytes
from ui.renderers.banana import banana_card_kb, banana_card_text
from utils.banana_state import BananaState, clear, load, save
from utils.files import validate_image
from utils.notify import admin_warn, user_error

log = get_logger("handlers.banana")

MAX_IMAGES = 4
MODEL = "banana-nana"


def _redis_client(context: ContextTypes.DEFAULT_TYPE) -> Any:
    return getattr(context, "redis", None)


async def _get_balance_value(user_id: int) -> int:
    snapshot = await aget_balance_snapshot(user_id)
    return snapshot.value or 0


async def _render_card(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    state: BananaState,
    *,
    chat_id: Optional[int] = None,
) -> None:
    target_chat = chat_id or (update.effective_chat.id if update.effective_chat else None)
    if target_chat is None:
        return
    balance = await _get_balance_value(target_chat)
    text = banana_card_text(balance, state)
    keyboard = banana_card_kb(state)
    await context.bot.send_message(target_chat, text, reply_markup=keyboard)


async def open_banana_card(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    state = await load(_redis_client(context), user_id)
    await _render_card(update, context, state, chat_id=user_id)


async def on_banana_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None or message.from_user is None:
        return
    user_id = message.from_user.id
    state = await load(_redis_client(context), user_id)
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
    elif message.document and message.document.mime_type and message.document.mime_type.startswith("image/"):
        file_id = message.document.file_id
    if not file_id:
        await user_error(context.bot, user_id, "Не удалось определить файл изображения.")
        return

    state.images.append(file_id)
    await save(_redis_client(context), user_id, state)
    balance = await _get_balance_value(user_id)
    await message.reply_text(
        banana_card_text(balance, state),
        reply_markup=banana_card_kb(state),
    )


async def on_banana_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None or message.from_user is None:
        return
    user_id = message.from_user.id
    text = (message.text or "").strip()
    if not text:
        return
    state = await load(_redis_client(context), user_id)
    state.prompt = text
    await save(_redis_client(context), user_id, state)
    balance = await _get_balance_value(user_id)
    await message.reply_text(
        banana_card_text(balance, state),
        reply_markup=banana_card_kb(state),
    )


async def _download_image_bytes(context: ContextTypes.DEFAULT_TYPE, file_id: str) -> Optional[bytes]:
    try:
        tg_file = await context.bot.get_file(file_id)
        data = await tg_file.download_as_bytearray()
        return bytes(data)
    except Exception as exc:  # pragma: no cover - network failures
        log.warning("banana.download_failed", exc_info=True, extra={"file_id": file_id})
        await admin_warn(context.bot, f"Banana tg fetch fail: {exc}")
        return None


async def _collect_upload_ids(
    context: ContextTypes.DEFAULT_TYPE, image_ids: Iterable[str]
) -> List[str]:
    uploads: List[str] = []
    for fid in image_ids:
        payload = await _download_image_bytes(context, fid)
        if not payload:
            continue
        try:
            validate_image(payload)
        except Exception as exc:
            await admin_warn(context.bot, f"Banana image invalid: {exc}")
            continue
        try:
            upload = await banana_upload_bytes(payload, filename="input.png")
        except Exception as exc:
            await admin_warn(context.bot, f"Banana upload fail: {exc}")
            continue
        upload_id = upload.get("upload_id") or upload.get("id") or upload.get("temp_url")
        if upload_id:
            uploads.append(str(upload_id))
    return uploads


async def _start_generation(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    state: BananaState,
) -> None:
    query = update.callback_query
    if query is not None:
        await query.answer()
    user_id = update.effective_user.id
    if not state.ready:
        await user_error(context.bot, user_id, "Добавьте фото и промпт.")
        return

    uploads = await _collect_upload_ids(context, state.images)
    if not uploads:
        await user_error(
            context.bot,
            user_id,
            "Не удалось загрузить изображения. Пришлите фото как файл и попробуйте снова.",
        )
        return

    payload = {"upload_ids": uploads, "prompt": state.prompt}
    try:
        response = await banana_process(MODEL, payload)
    except Exception as exc:
        await user_error(context.bot, user_id, "Сервис занят, попробуйте ещё раз.")
        await admin_warn(context.bot, f"Banana process fail: {exc}")
        return

    result_bytes = response.get("result_bytes") if isinstance(response, dict) else None
    if not result_bytes:
        await user_error(context.bot, user_id, "Banana не вернула результат. Попробуйте ещё раз.")
        await admin_warn(context.bot, f"Banana process invalid response: {response}")
        return

    doc = await context.bot.send_document(user_id, document=("result.png", result_bytes))
    state.last_result_id = str(doc.message_id)
    await save(_redis_client(context), user_id, state)

    keyboard = InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("🔁 Сгенерировать ещё", callback_data="banana:again")],
            [InlineKeyboardButton("🆕 Новая генерация", callback_data="banana:new")],
        ]
    )
    await context.bot.send_message(user_id, "Готово ✅", reply_markup=keyboard)


async def on_banana_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    state = await load(_redis_client(context), update.effective_user.id)
    await _start_generation(update, context, state=state)


async def on_banana_again(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    state = await load(_redis_client(context), update.effective_user.id)
    await _start_generation(update, context, state=state)


async def on_banana_new(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query is not None:
        await query.answer()
    user_id = update.effective_user.id
    await clear(_redis_client(context), user_id)
    state = BananaState(images=[], prompt=None)
    await save(_redis_client(context), user_id, state)
    await context.bot.send_message(
        user_id,
        "Новая карточка 🆕 Отправьте фото и промпт.",
        reply_markup=banana_card_kb(state),
    )


__all__ = [
    "MAX_IMAGES",
    "MODEL",
    "on_banana_again",
    "on_banana_new",
    "on_banana_photo",
    "on_banana_start",
    "on_banana_text",
    "open_banana_card",
]

