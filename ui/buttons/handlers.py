from __future__ import annotations

from contextlib import suppress
from typing import Optional

from telegram.error import BadRequest

from .results import UIResult, show_dialog, show_menu
from .types import ButtonContext


def _bot():  # pragma: no cover - helper for lazy import
    import bot as bot_module

    return bot_module


def _sum_handlers():  # pragma: no cover - helper for lazy import
    from handlers import sum as sum_module

    return sum_module


async def _open_menu_item(context: ButtonContext, item: str) -> UIResult:
    bot_module = _bot()
    ctx = context.app_context
    chat_data = getattr(ctx, "chat_data", None)
    fallback_key = bot_module._MENU_CHATDATA_KEYS.get(item)
    previous: Optional[int] = None
    if isinstance(chat_data, dict) and fallback_key:
        raw = chat_data.get(fallback_key)
        try:
            previous = int(raw) if raw is not None else None
        except (TypeError, ValueError):
            previous = None

    message_id = await bot_module._perform_menu_open(
        item,
        update=context.update,
        ctx=ctx,
        query=context.query,
        log_click=False,
    )

    reused = False
    if isinstance(chat_data, dict) and fallback_key:
        raw_after = chat_data.get(fallback_key)
        try:
            current = int(raw_after) if raw_after is not None else None
        except (TypeError, ValueError):
            current = None
        else:
            reused = previous is not None and previous == current

    return show_menu(context.spec.id, screen=item, message_id=message_id, reused=reused)


async def open_profile(context: ButtonContext) -> UIResult:
    return await _open_menu_item(context, "profile")


async def open_kb(context: ButtonContext) -> UIResult:
    return await _open_menu_item(context, "kb")


async def open_photo(context: ButtonContext) -> UIResult:
    return await _open_menu_item(context, "photo")


async def open_music(context: ButtonContext) -> UIResult:
    return await _open_menu_item(context, "music")


async def open_video(context: ButtonContext) -> UIResult:
    query = context.query
    if query is not None:
        with suppress(BadRequest):
            await query.answer(cache_time=0)
    return await _open_menu_item(context, "video")


async def open_dialog(context: ButtonContext) -> UIResult:
    return await _open_menu_item(context, "dialog")


async def open_help(context: ButtonContext) -> UIResult:
    bot_module = _bot()
    await bot_module.help_command_entry(context.update, context.app_context)
    return show_dialog(context.spec.id, "help", message_id=None)


async def open_sora2(context: ButtonContext) -> UIResult:
    handlers = _bot()
    await handlers.sora2_open_cb(context.update, context.app_context)
    return show_dialog(context.spec.id, "sora2", message_id=None)


async def open_sum(context: ButtonContext) -> UIResult:
    sum_module = _sum_handlers()
    message_id = await sum_module.open_from_button(context)
    return show_dialog(context.spec.id, "sum", message_id=message_id)
