"""Prompt-Master handlers, state helpers and prompt builder integration."""

from __future__ import annotations

import logging
import re
from typing import Dict, Optional, Tuple

from telegram import InlineKeyboardMarkup, Message, Update
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import ContextTypes

from keyboards import (
    CB_PM_PREFIX,
    prompt_master_keyboard,
    prompt_master_mode_keyboard,
    prompt_master_result_keyboard,
)
from prompt_master import (
    Engine,
    PromptPayload,
    build_animate_prompt,
    build_banana_prompt,
    build_mj_prompt,
    build_suno_prompt,
    build_veo_prompt,
)
from utils.html_render import html_to_plain, render_pm_html, safe_lines
from utils.safe_send import safe_send, send_html_with_fallback

logger = logging.getLogger(__name__)

PM_STATE_KEY = "pm_state"

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

ENGINE_BUILDERS = {
    "veo": build_veo_prompt,
    "mj": build_mj_prompt,
    "banana": build_banana_prompt,
    "animate": build_animate_prompt,
    "suno": build_suno_prompt,
}

_ENGINE_DISPLAY = {
    "veo": {"ru": "VEO", "en": "VEO"},
    "mj": {"ru": "Midjourney", "en": "Midjourney"},
    "banana": {"ru": "Banana", "en": "Banana"},
    "animate": {"ru": "VEO Animate", "en": "VEO Animate"},
    "suno": {"ru": "Suno", "en": "Suno"},
}

PM_STATUS_TEXT = {
    "ru": "✍️ <b>Пишу промпт…</b>",
    "en": "✍️ <b>Crafting the prompt…</b>",
}

PM_ERROR_TEXT = {
    "ru": "❌ Не удалось собрать промпт. Попробуйте ещё раз.",
    "en": "❌ Failed to build the prompt. Try again.",
}

PM_WARNING_TEXT = {
    "ru": "⚠️ Не смог отправить карточку, попробуйте ещё раз.",
    "en": "⚠️ Couldn't send the card, please try again.",
}

CYRILLIC_RE = re.compile(r"[а-яё]", re.IGNORECASE)

_LAST_PROMPTS: Dict[Tuple[int, str], PromptPayload] = {}


def detect_language(text: str) -> str:
    return "ru" if CYRILLIC_RE.search(text or "") else "en"


def _ensure_state(context: ContextTypes.DEFAULT_TYPE) -> Dict[str, object]:
    state = context.user_data.get(PM_STATE_KEY)
    if not isinstance(state, dict):
        state = {
            "engine": None,
            "card_msg_id": None,
            "prompt": None,
            "autodelete": True,
            "result": None,
            "lang": None,
        }
        context.user_data[PM_STATE_KEY] = state
    state.setdefault("autodelete", True)
    return state


def _store_prompt(chat_id: int, engine: str, payload: PromptPayload) -> None:
    _LAST_PROMPTS[(chat_id, engine)] = payload


def get_pm_prompt(chat_id: int, engine: str) -> Optional[PromptPayload]:
    return _LAST_PROMPTS.get((chat_id, engine))


def clear_pm_prompts(chat_id: int) -> None:
    keys = [key for key in _LAST_PROMPTS if key[0] == chat_id]
    for key in keys:
        _LAST_PROMPTS.pop(key, None)


def _resolve_ui_lang(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    state = _ensure_state(context)
    lang = state.get("lang")
    if isinstance(lang, str) and lang in {"ru", "en"}:
        return lang
    user = update.effective_user
    if user and isinstance(user.language_code, str) and user.language_code.lower().startswith("ru"):
        return "ru"
    return "en"


def _engine_header(engine: str, lang: str) -> str:
    display = _ENGINE_DISPLAY.get(engine, {}).get(lang, engine.upper())
    if lang == "ru":
        return f"Карточка {display}"
    return f"{display} card"


def _render_card(engine: str, state: Dict[str, object], lang: str) -> Tuple[str, InlineKeyboardMarkup]:
    prompt_value = state.get("prompt") or ""
    hint = PM_ENGINE_HINTS.get(engine, PM_ENGINE_HINTS["veo"]).get(lang, PM_ENGINE_HINTS["veo"]["en"])
    if prompt_value:
        body = f"```json\n{str(prompt_value)}\n```"
    else:
        body = f"_{hint}_"
    header = _engine_header(engine, lang)
    card_markdown = "\n\n".join([f"**{header}**", body])
    text = render_pm_html(card_markdown)
    keyboard = prompt_master_mode_keyboard(lang)
    return text, keyboard


async def _upsert_card(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    engine: str,
    state: Dict[str, object],
    lang: str,
) -> None:
    text, keyboard = _render_card(engine, state, lang)
    chat = update.effective_chat
    if chat is None:
        return
    chat_id = chat.id
    message_id = state.get("card_msg_id")
    if isinstance(message_id, int):
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=text,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
                reply_markup=keyboard,
            )
            return
        except BadRequest:
            state["card_msg_id"] = None
        except Exception:
            logger.exception("prompt_master.card_edit_failed")
            state["card_msg_id"] = None
    message = await safe_send(context.bot, chat_id, text, reply_markup=keyboard)
    if message:
        state["card_msg_id"] = message.message_id


def _render_payload_html(payload: PromptPayload) -> str:
    parts: list[str] = [f"**{payload.title}**"]
    if payload.subtitle:
        parts.append(f"_{payload.subtitle}_")
    if payload.body_md:
        parts.append(payload.body_md)
    if payload.code_block:
        code_value = payload.code_block.strip()
        language = "json" if code_value.startswith("{") or code_value.startswith("[") else ""
        fence = f"```{language}\n{code_value}\n```"
        parts.append(fence)
    if payload.buttons:
        parts.append(safe_lines(payload.buttons))
    markdown = "\n\n".join(part for part in parts if part)
    return render_pm_html(markdown)


async def _edit_with_fallback(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    message_id: int,
    html_text: str,
    reply_markup: InlineKeyboardMarkup,
) -> None:
    try:
        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text=html_text,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
            reply_markup=reply_markup,
        )
    except BadRequest as exc:
        message = str(exc).lower()
        if "can't parse entities" not in message and "parse entities" not in message:
            raise
        logger.warning("pm.html_fallback", extra={"exc": repr(exc)})
        logger.info("pm.render.fallback")
        plain = html_to_plain(html_text)
        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text=plain,
            reply_markup=reply_markup,
            disable_web_page_preview=True,
        )


async def _safe_delete(message) -> None:
    if message is None:
        return
    try:
        await message.delete()
    except Exception:
        logger.debug("prompt_master.delete_failed", exc_info=True)


async def _handle_render_failure(
    context: ContextTypes.DEFAULT_TYPE,
    status_message: Optional[Message],
    lang: str,
    engine: str,
    keyboard: InlineKeyboardMarkup,
    chat_id: Optional[int],
) -> None:
    error_text = PM_ERROR_TEXT.get(lang, PM_ERROR_TEXT["en"])
    if status_message is not None:
        try:
            await context.bot.edit_message_text(
                chat_id=status_message.chat_id,
                message_id=status_message.message_id,
                text=error_text,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
                reply_markup=keyboard,
            )
        except Exception:
            logger.exception("pm.card.fail", extra={"engine": engine})
    elif chat_id is not None:
        await send_html_with_fallback(context.bot, chat_id, error_text, reply_markup=keyboard)


async def _notify_failure(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: Optional[int],
    lang: str,
) -> None:
    if chat_id is None:
        return
    warning = PM_WARNING_TEXT.get(lang, PM_WARNING_TEXT["en"])
    await send_html_with_fallback(context.bot, chat_id, warning)


async def prompt_master_open(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    state = _ensure_state(context)
    state.update({"engine": None, "prompt": None, "card_msg_id": None})
    lang = _resolve_ui_lang(update, context)
    text = PM_ROOT_TEXT.get(lang, PM_ROOT_TEXT["en"])

    message = update.effective_message
    if message is not None and message.chat:
        await safe_send(
            context.bot,
            message.chat_id,
            text,
            reply_markup=prompt_master_keyboard(lang),
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


async def _set_engine(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    engine: Optional[str],
) -> None:
    state = _ensure_state(context)
    state["engine"] = engine
    state["prompt"] = None
    state["result"] = None
    if engine:
        lang = _resolve_ui_lang(update, context)
        hint = PM_ENGINE_HINTS.get(engine, PM_ENGINE_HINTS["veo"]).get(lang, PM_ENGINE_HINTS["veo"]["en"])
        keyboard = prompt_master_mode_keyboard(lang)
        query = update.callback_query
        if query and query.message:
            await query.edit_message_text(
                hint,
                reply_markup=keyboard,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
            )
        else:
            chat = update.effective_chat
            if chat is not None:
                await safe_send(context.bot, chat.id, hint, reply_markup=keyboard)
    else:
        state["card_msg_id"] = None


async def _handle_copy(update: Update, context: ContextTypes.DEFAULT_TYPE, engine: str, lang: str) -> None:
    query = update.callback_query
    if query is None:
        return
    chat = query.message.chat if query.message else update.effective_chat
    chat_id = chat.id if chat else None
    payload = get_pm_prompt(chat_id, engine) if chat_id is not None else None
    if payload is None:
        await query.answer("Промпт не найден" if lang == "ru" else "Prompt not found", show_alert=True)
        return
    await query.answer("Готово" if lang == "ru" else "Done")
    text = payload.copy_text
    if text.strip().startswith("{") or text.strip().startswith("["):
        html_text = f"<pre><code>{html.escape(text)}</code></pre>"
    else:
        html_text = html.escape(text).replace("\n", "<br/>")
    if chat_id is not None:
        await safe_send(context.bot, chat_id, html_text)


async def _handle_insert(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    engine: str,
    lang: str,
) -> None:
    query = update.callback_query
    if query is None:
        return
    chat = query.message.chat if query.message else update.effective_chat
    chat_id = chat.id if chat else None
    if chat_id is None:
        await query.answer("Чат недоступен", show_alert=True)
        return
    payload = get_pm_prompt(chat_id, engine)
    if payload is None:
        await query.answer("Промпт не найден" if lang == "ru" else "Prompt not found", show_alert=True)
        return
    state = _ensure_state(context)
    state["engine"] = engine
    state["lang"] = lang
    state["result"] = payload
    state["prompt"] = payload.card_text
    _store_prompt(chat_id, engine, payload)
    await _upsert_card(update, context, engine=engine, state=state, lang=lang)
    await query.answer("Вставлено" if lang == "ru" else "Inserted")


async def prompt_master_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query is None or query.data is None:
        return

    data = query.data
    lang = _resolve_ui_lang(update, context)

    if data.startswith(f"{CB_PM_PREFIX}copy:"):
        engine = data.split(":", 2)[2]
        await _handle_copy(update, context, engine, lang)
        return

    if data.startswith(f"{CB_PM_PREFIX}insert:"):
        engine = data.split(":", 2)[2]
        await _handle_insert(update, context, engine, lang)
        return

    if data in {f"{CB_PM_PREFIX}back", f"{CB_PM_PREFIX}menu"}:
        state = _ensure_state(context)
        chat = query.message.chat if query.message else update.effective_chat
        chat_id = chat.id if chat else None
        if chat_id is not None:
            clear_pm_prompts(chat_id)
        await query.answer()
        state.update({"engine": None, "prompt": None, "result": None})
        await query.edit_message_text(
            PM_ROOT_TEXT.get(lang, PM_ROOT_TEXT["en"]),
            reply_markup=prompt_master_keyboard(lang),
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )
        return

    if data == f"{CB_PM_PREFIX}switch":
        await query.answer()
        await _set_engine(update, context, None)
        await query.edit_message_text(
            PM_ROOT_TEXT.get(lang, PM_ROOT_TEXT["en"]),
            reply_markup=prompt_master_keyboard(lang),
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )
        return

    engine = data.removeprefix(CB_PM_PREFIX)
    if engine in ENGINE_BUILDERS:
        await query.answer()
        state = _ensure_state(context)
        state["engine"] = engine
        state["lang"] = lang
        await _set_engine(update, context, engine)
        await _upsert_card(update, context, engine=engine, state=state, lang=lang)
        return

    await query.answer()
    await query.edit_message_text(
        PM_ROOT_TEXT.get(lang, PM_ROOT_TEXT["en"]),
        reply_markup=prompt_master_keyboard(lang),
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True,
    )


async def prompt_master_text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if message is None or not isinstance(message.text, str):
        return

    state = _ensure_state(context)
    engine = state.get("engine")
    if engine not in ENGINE_BUILDERS:
        return

    text = message.text.strip()
    if not text:
        return

    lang = detect_language(text) or _resolve_ui_lang(update, context)
    state["lang"] = lang
    state["prompt"] = text

    await _upsert_card(update, context, engine=engine, state=state, lang=lang)

    chat = update.effective_chat
    chat_id = chat.id if chat else None
    logger.info("pm.start", extra={"engine": engine})
    status = (
        await send_html_with_fallback(
            context.bot,
            chat_id,
            PM_STATUS_TEXT.get(lang, PM_STATUS_TEXT["en"]),
            reply_markup=prompt_master_mode_keyboard(lang),
        )
        if chat_id is not None
        else None
    )

    builder = ENGINE_BUILDERS.get(engine)
    payload: Optional[PromptPayload] = None
    try:
        payload = builder(text, lang) if builder else None
    except Exception as exc:
        logger.error("prompt_master.build_failed | engine=%s err=%s", engine, exc)

    if payload is None:
        if status is not None:
            try:
                await context.bot.edit_message_text(
                    chat_id=status.chat_id,
                    message_id=status.message_id,
                    text=PM_ERROR_TEXT.get(lang, PM_ERROR_TEXT["en"]),
                    parse_mode=ParseMode.HTML,
                    disable_web_page_preview=True,
                    reply_markup=prompt_master_mode_keyboard(lang),
                )
            except Exception:
                logger.debug("prompt_master.status_error_edit_failed", exc_info=True)
        return

    if chat_id is not None:
        _store_prompt(chat_id, engine, payload)
    state["result"] = payload

    try:
        result_html = _render_payload_html(payload)
        logger.info("pm.render.ok", extra={"engine": engine})
    except Exception:
        logger.exception("pm.render.error", extra={"engine": engine})
        await _handle_render_failure(
            context,
            status,
            lang,
            engine,
            prompt_master_mode_keyboard(lang),
            chat_id,
        )
        return
    markup = prompt_master_result_keyboard(engine, lang)
    if status is not None:
        try:
            await _edit_with_fallback(
                context,
                status.chat_id,
                status.message_id,
                result_html,
                markup,
            )
            logger.info("pm.card.sent", extra={"engine": engine})
            if state.get("autodelete", True):
                await _safe_delete(message)
        except Exception:
            logger.exception("pm.card.fail", extra={"engine": engine})
            await _notify_failure(context, chat_id, lang)
        else:
            if chat_id is not None:
                try:
                    message_sent = await send_html_with_fallback(
                        context.bot,
                        chat_id,
                        result_html,
                        reply_markup=markup,
                    )
                except Exception:
                    logger.exception("pm.card.fail", extra={"engine": engine})
                    await _notify_failure(context, chat_id, lang)
                    return
                if message_sent is not None:
                    logger.info("pm.card.sent", extra={"engine": engine})
                    if state.get("autodelete", True):
                        await _safe_delete(message)
                else:
                    logger.error("pm.card.fail", extra={"engine": engine})
                    await _notify_failure(context, chat_id, lang)


async def prompt_master_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    chat_id = chat.id if chat else None
    if chat_id is not None:
        clear_pm_prompts(chat_id)
    state = _ensure_state(context)
    state.update({"engine": None, "prompt": None, "result": None, "card_msg_id": None, "lang": None})
    lang = _resolve_ui_lang(update, context)
    text = "🧹 Prompt-Master сброшен." if lang == "ru" else "🧹 Prompt-Master reset."
    keyboard = prompt_master_keyboard(lang)
    if chat_id is not None:
        await safe_send(context.bot, chat_id, f"{text}\n\n{PM_ROOT_TEXT.get(lang, PM_ROOT_TEXT['en'])}", reply_markup=keyboard)


prompt_master_handle_text = prompt_master_text_handler
prompt_master_process = prompt_master_text_handler

__all__ = [
    "PM_STATE_KEY",
    "clear_pm_prompts",
    "detect_language",
    "get_pm_prompt",
    "prompt_master_callback",
    "prompt_master_handle_text",
    "prompt_master_open",
    "prompt_master_process",
    "prompt_master_reset",
]
