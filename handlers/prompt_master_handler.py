"""Prompt-Master handlers, state helpers and prompt builder."""

from __future__ import annotations

import asyncio
import html
import logging
import re
import time
from typing import Dict, Optional, Tuple

from telegram import InlineKeyboardMarkup, Update
from telegram.constants import ChatType, ParseMode
from telegram.ext import ContextTypes

from keyboards import (
    CB_PM_PREFIX,
    prompt_master_keyboard,
    prompt_master_mode_keyboard,
    prompt_master_result_keyboard,
)
from prompt_master import Engine, PromptPayload, build_prompt

logger = logging.getLogger(__name__)

PM_STATE_KEY = "mode"
PM_ENGINE_KEY = "pm_engine"
PM_LANG_KEY = "pm_lang"
PM_PROMPTS_KEY = "pm_prompts"

PM_ENGINES = {engine.value for engine in Engine}

CYRILLIC_RE = re.compile(r"[а-яё]", re.IGNORECASE)

PM_ROOT_TEXT = {
    "ru": "🧠 <b>Prompt-Master</b>\nВыберите движок, под который нужно подготовить промпт.",
    "en": "🧠 <b>Prompt-Master</b>\nPick the engine you want a perfect prompt for.",
}

PM_ENGINE_HINTS = {
    "veo": {
        "ru": "Опишите идею ролика: сюжет, эмоции, окружение. Я соберу структурированный JSON для VEO.",
        "en": "Describe the video idea: story, emotions, surroundings. I will craft a structured JSON for VEO.",
    },
    "mj": {
        "ru": "Расскажите, какой кадр хотите получить в Midjourney. Добавьте стиль и детали.",
        "en": "Describe the Midjourney shot you need, including style and key details.",
    },
    "banana": {
        "ru": "Что нужно поправить на фото? Я соберу чек-лист для Banana и сохраню черты лица.",
        "en": "What should be fixed on the photo? I will prepare a Banana checklist keeping the real face.",
    },
    "animate": {
        "ru": "Опишите, как оживить фото: эмоции, движения камеры и лица. Лицо останется прежним.",
        "en": "Describe how the photo should come alive: facial motion and camera drift. The face will stay true.",
    },
    "suno": {
        "ru": "Опишите настроение и сюжет трека. Я подготовлю каркас промпта для Suno.",
        "en": "Describe the mood and story of the track. I'll return a neat Suno prompt skeleton.",
    },
}

ENGINE_DISPLAY = {
    "veo": {"ru": "VEO", "en": "VEO"},
    "mj": {"ru": "Midjourney", "en": "Midjourney"},
    "banana": {"ru": "Banana", "en": "Banana"},
    "animate": {"ru": "VEO Animate", "en": "VEO Animate"},
    "suno": {"ru": "Suno", "en": "Suno"},
}

_LAST_PROMPTS: Dict[Tuple[int, str], PromptPayload] = {}

START_STATUS = {"ru": "⏳ Готовлю промпт…", "en": "⏳ Crafting your prompt…"}
SLOW_STATUS = {"ru": "⚠️ Что-то долго. Ещё секунду…", "en": "⚠️ Taking longer than usual. One more second…"}
ERROR_STATUS = {
    "ru": "❌ Не удалось собрать промпт. Попробуйте ещё раз или переформулируйте задачу.",
    "en": "❌ Failed to build the prompt. Try again or rephrase the request.",
}


def detect_language(text: str) -> str:
    return "ru" if CYRILLIC_RE.search(text or "") else "en"


def _store_prompt(chat_id: int, engine: Engine, payload: PromptPayload) -> None:
    _LAST_PROMPTS[(chat_id, engine.value)] = payload


def get_pm_prompt(chat_id: int, engine: str) -> Optional[PromptPayload]:
    return _LAST_PROMPTS.get((chat_id, engine))


def clear_pm_prompts(chat_id: int) -> None:
    keys = [key for key in _LAST_PROMPTS if key[0] == chat_id]
    for key in keys:
        _LAST_PROMPTS.pop(key, None)


def _resolve_ui_lang(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    lang = context.user_data.get(PM_LANG_KEY)
    if lang in {"ru", "en"}:
        return lang
    user = update.effective_user
    if user and isinstance(user.language_code, str):
        if user.language_code.lower().startswith("ru"):
            return "ru"
    return "en"


def _header_for_engine(engine: str, lang: str) -> str:
    display = ENGINE_DISPLAY.get(engine, {}).get(lang, engine.upper())
    if lang == "ru":
        return f"<b>Готовый промпт для {html.escape(display)}</b>"
    return f"<b>Ready prompt for {html.escape(display)}</b>"


def _render_markdown(text: str) -> str:
    from html import escape

    bold_re = re.compile(r"\*\*(.+?)\*\*")

    def _inline(value: str) -> str:
        result = []
        last = 0
        for match in bold_re.finditer(value):
            result.append(escape(value[last:match.start()]))
            result.append(f"<b>{escape(match.group(1))}</b>")
            last = match.end()
        result.append(escape(value[last:]))
        return "".join(result)

    lines = text.splitlines()
    html_parts = []
    in_list = False
    in_code = False
    for line in lines:
        if line.startswith("```"):
            if in_code:
                html_parts.append("</code></pre>")
                in_code = False
            else:
                html_parts.append("<pre><code>")
                in_code = True
            continue
        if in_code:
            html_parts.append(f"{escape(line)}\n")
            continue
        stripped = line.strip()
        if stripped.startswith("- "):
            if not in_list:
                html_parts.append("<ul>")
                in_list = True
            html_parts.append(f"<li>{_inline(stripped[2:])}</li>")
            continue
        if in_list:
            html_parts.append("</ul>")
            in_list = False
        if stripped:
            html_parts.append(f"<p>{_inline(stripped)}</p>")
        else:
            html_parts.append("<br/>")
    if in_list:
        html_parts.append("</ul>")
    if in_code:
        html_parts.append("</code></pre>")
    return "".join(html_parts)


def _payload_to_html(engine: str, lang: str, payload: PromptPayload) -> str:
    body_html = _render_markdown(payload.body_markdown)
    header = _header_for_engine(engine, lang)
    return f"{header}<br/>{body_html}"


async def prompt_master_open(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    lang = _resolve_ui_lang(update, context)
    text = PM_ROOT_TEXT.get(lang, PM_ROOT_TEXT["en"])

    if message is not None:
        await message.reply_text(
            text,
            reply_markup=prompt_master_keyboard(lang),
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )
        return

    query = update.callback_query
    if query is None:
        return
    await query.answer()
    await query.edit_message_text(
        text,
        reply_markup=prompt_master_keyboard(lang),
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True,
    )


def _set_pm_state(context: ContextTypes.DEFAULT_TYPE, engine: Optional[str]) -> None:
    if engine is None:
        context.user_data.pop(PM_STATE_KEY, None)
        context.user_data.pop(PM_ENGINE_KEY, None)
        return
    context.user_data[PM_STATE_KEY] = "pm"
    context.user_data[PM_ENGINE_KEY] = engine


async def _send_engine_hint(
    query, update: Update, context: ContextTypes.DEFAULT_TYPE, engine: str
) -> None:
    lang = _resolve_ui_lang(update, context)
    hint = PM_ENGINE_HINTS.get(engine, PM_ENGINE_HINTS["veo"]).get(lang, PM_ENGINE_HINTS["veo"]["en"])
    markup = prompt_master_mode_keyboard(lang)
    if query.message is not None:
        await query.edit_message_text(
            hint,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
            reply_markup=markup,
        )
    else:
        chat = update.effective_chat
        if chat is not None:
            await context.bot.send_message(
                chat_id=chat.id,
                text=hint,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
                reply_markup=markup,
            )


async def prompt_master_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query is None or query.data is None:
        return

    data = query.data
    await query.answer()
    lang = _resolve_ui_lang(update, context)

    if data.startswith(f"{CB_PM_PREFIX}insert:"):
        parts = data.split(":", 2)
        engine = parts[2] if len(parts) > 2 else ""
        user = update.effective_user
        user_id = user.id if user else None
        logger.info("pm.insert_clicked | user_id=%s engine=%s", user_id, engine)
        return

    if data.startswith(f"{CB_PM_PREFIX}copy:"):
        parts = data.split(":", 2)
        engine = parts[2] if len(parts) > 2 else ""
        chat = query.message.chat if query.message else update.effective_chat
        chat_id = chat.id if chat else None
        prompt = get_pm_prompt(chat_id, engine) if chat_id is not None else None
        if prompt is None:
            await query.answer("Промпт не найден" if lang == "ru" else "Prompt not found", show_alert=True)
            return
        await query.answer("Готово" if lang == "ru" else "Done")
        user = update.effective_user
        user_id = user.id if user else None
        logger.info("pm.copy_clicked | user_id=%s engine=%s", user_id, engine)
        copy_text = prompt.copy_text
        if copy_text.strip().startswith("{") or copy_text.strip().startswith("["):
            text = f"<pre>{html.escape(copy_text)}</pre>"
            parse_mode = ParseMode.HTML
        else:
            text = copy_text
            parse_mode = None
        target_chat = chat_id if chat_id is not None else (update.effective_chat.id if update.effective_chat else None)
        if target_chat is not None:
            await context.bot.send_message(chat_id=target_chat, text=text, parse_mode=parse_mode)
        return

    if data in {f"{CB_PM_PREFIX}back", f"{CB_PM_PREFIX}menu"}:
        chat = query.message.chat if query.message else update.effective_chat
        if chat is not None:
            clear_pm_prompts(chat.id)
        _set_pm_state(context, None)
        await query.edit_message_text(
            PM_ROOT_TEXT.get(lang, PM_ROOT_TEXT["en"]),
            reply_markup=prompt_master_keyboard(lang),
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )
        return

    if data == f"{CB_PM_PREFIX}switch":
        _set_pm_state(context, None)
        await query.edit_message_text(
            PM_ROOT_TEXT.get(lang, PM_ROOT_TEXT["en"]),
            reply_markup=prompt_master_keyboard(lang),
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )
        return

    engine = data.removeprefix(CB_PM_PREFIX)
    if engine in PM_ENGINES:
        _set_pm_state(context, engine)
        user = update.effective_user
        user_id = user.id if user else None
        logger.info("pm.mode_set | user_id=%s engine=%s", user_id, engine)
        await _send_engine_hint(query, update, context, engine)
        return

    await query.edit_message_text(
        PM_ROOT_TEXT.get(lang, PM_ROOT_TEXT["en"]),
        reply_markup=prompt_master_keyboard(lang),
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True,
    )


async def prompt_master_handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if message is None or not isinstance(message.text, str):
        return

    if context.user_data.get(PM_STATE_KEY) != "pm":
        return

    engine = context.user_data.get(PM_ENGINE_KEY)
    if engine not in PM_ENGINES:
        return

    idea = message.text.strip()
    if not idea:
        return

    lang = detect_language(idea)
    if lang not in {"ru", "en"}:
        lang = _resolve_ui_lang(update, context)
    context.user_data[PM_LANG_KEY] = lang
    user = update.effective_user
    user_id = user.id if user else None
    logger.info("pm.start | user_id=%s engine=%s lang=%s", user_id, engine, lang)
    chat = update.effective_chat
    status_markup = prompt_master_mode_keyboard(lang)
    try:
        status_message = await message.reply_text(
            START_STATUS.get(lang, START_STATUS["en"]),
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
            reply_markup=status_markup,
        )
    except Exception:  # pragma: no cover - Telegram failure
        logger.exception("prompt_master.status_send_failed")
        return

    task = asyncio.create_task(build_prompt(Engine(engine), idea, lang))
    start_ts = time.monotonic()
    try:
        payload = await asyncio.wait_for(task, timeout=12.0)
    except asyncio.TimeoutError:
        try:
            await status_message.edit_text(
                SLOW_STATUS.get(lang, SLOW_STATUS["en"]),
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
                reply_markup=status_markup,
            )
        except Exception:
            logger.debug("prompt_master.slow_status_failed", exc_info=True)
        try:
            payload = await asyncio.wait_for(task, timeout=18.0)
        except Exception as exc:  # pragma: no cover - hard failure path
            logger.error("pm.error | user_id=%s engine=%s err=%s", user_id, engine, exc)
            try:
                await status_message.edit_text(
                    ERROR_STATUS.get(lang, ERROR_STATUS["en"]),
                    parse_mode=ParseMode.HTML,
                    disable_web_page_preview=True,
                    reply_markup=status_markup,
                )
            except Exception:
                logger.debug("prompt_master.error_status_failed", exc_info=True)
            return
    except Exception as exc:  # pragma: no cover - unexpected failure
        logger.error("pm.error | user_id=%s engine=%s err=%s", user_id, engine, exc)
        try:
            await status_message.edit_text(
                ERROR_STATUS.get(lang, ERROR_STATUS["en"]),
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
                reply_markup=status_markup,
            )
        except Exception:
            logger.debug("prompt_master.error_status_failed", exc_info=True)
        return

    duration_ms = int((time.monotonic() - start_ts) * 1000)
    logger.info(
        "pm.ready | user_id=%s engine=%s size=%s ms=%s",
        user_id,
        engine,
        len(idea),
        duration_ms,
    )

    if chat is not None:
        _store_prompt(chat.id, Engine(engine), payload)

    prompts = context.user_data.setdefault(PM_PROMPTS_KEY, {})
    prompts[engine] = payload
    context.user_data["pm_last_engine"] = engine

    if chat and chat.type == ChatType.PRIVATE:
        try:
            await message.delete()
        except Exception:
            logger.debug("prompt_master.delete_failed", exc_info=True)

    formatted = _payload_to_html(engine, lang, payload)
    markup = prompt_master_result_keyboard(engine, lang)
    try:
        await status_message.edit_text(
            formatted,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
            reply_markup=markup,
        )
    except Exception:
        logger.exception("prompt_master.reply_failed")


async def prompt_master_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    if chat is not None:
        clear_pm_prompts(chat.id)
    _set_pm_state(context, None)
    context.user_data.pop(PM_PROMPTS_KEY, None)
    context.user_data.pop(PM_LANG_KEY, None)

    lang = _resolve_ui_lang(update, context)
    text = "🧹 Prompt-Master сброшен." if lang == "ru" else "🧹 Prompt-Master reset."
    keyboard = prompt_master_keyboard(lang)

    message = update.effective_message
    if message is not None:
        await message.reply_text(
            f"{text}\n\n{PM_ROOT_TEXT.get(lang, PM_ROOT_TEXT['en'])}",
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
            reply_markup=keyboard,
        )
    else:
        target = chat.id if chat is not None else None
        if target is not None:
            await context.bot.send_message(
                target,
                f"{text}\n\n{PM_ROOT_TEXT.get(lang, PM_ROOT_TEXT['en'])}",
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
                reply_markup=keyboard,
            )


# Backwards compatibility alias
prompt_master_process = prompt_master_handle_text


__all__ = [
    "clear_pm_prompts",
    "detect_language",
    "get_pm_prompt",
    "prompt_master_callback",
    "prompt_master_handle_text",
    "prompt_master_open",
    "prompt_master_process",
    "prompt_master_reset",
]
