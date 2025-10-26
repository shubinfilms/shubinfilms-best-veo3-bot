from __future__ import annotations

import asyncio
import logging
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import Mapping, Optional

from telegram import Message
from telegram.error import BadRequest
from telegram.ext import ContextTypes

from hub_router import CallbackContext as HubCallbackContext
from kie import summary as kie_summary
from metrics import sum_open_total, sum_start_total
from redis_utils import rds as redis_client
from settings import REDIS_PREFIX
from ui.renderers.sum import (
    render_sum_card,
    render_sum_error,
    render_sum_progress,
    render_sum_prompt_card,
    render_sum_queue_notice,
    render_sum_ready_to_start,
    render_sum_result,
)
from ui.buttons.types import ButtonContext
from utils.input_state import (
    WaitInputState,
    clear_wait_states,
    set_wait,
)

log = logging.getLogger(__name__)

_LOCK_KEY_TMPL = f"{REDIS_PREFIX}:sum:lock:{{}}"
_LAST_KEY_TMPL = f"{REDIS_PREFIX}:sum:last:{{}}"
_MSG_KEY_TMPL = f"{REDIS_PREFIX}:sum:msg:{{}}"
_PAYLOAD_KEY_TMPL = f"{REDIS_PREFIX}:sum:payload:{{}}"
_RUN_KEY_TMPL = f"{REDIS_PREFIX}:sum:run:{{}}"

_LOCK_TTL_SECONDS = 5
_LAST_TTL_SECONDS = 30
_MSG_TTL_SECONDS = 24 * 60 * 60
_PAYLOAD_TTL_SECONDS = 15 * 60
_RUN_TTL_SECONDS = 60
_DEBOUNCE_WINDOW_MS = 600
_POLL_INTERVAL_SECONDS = 10
_POLL_MAX_ATTEMPTS = 12
_LAZY_CHECK_INTERVAL = 60
_LAZY_CHECK_MAX_ATTEMPTS = 10

_memory_store: dict[str, tuple[str, float]] = {}
_memory_lock = threading.Lock()


def _now_ms() -> int:
    return int(time.time() * 1000)


def _memory_get(key: str) -> Optional[str]:
    with _memory_lock:
        entry = _memory_store.get(key)
        if not entry:
            return None
        value, expires_at = entry
        if expires_at and expires_at < time.time():
            _memory_store.pop(key, None)
            return None
        return value


def _memory_setex(key: str, ttl: int, value: str) -> None:
    expires_at = time.time() + max(ttl, 1)
    with _memory_lock:
        _memory_store[key] = (value, expires_at)


def _memory_setnx(key: str, ttl: int, value: str) -> bool:
    with _memory_lock:
        entry = _memory_store.get(key)
        if entry:
            _, expires_at = entry
            if not expires_at or expires_at > time.time():
                return False
        expires_at = time.time() + max(ttl, 1)
        _memory_store[key] = (value, expires_at)
        return True


def _memory_delete(key: str) -> None:
    with _memory_lock:
        _memory_store.pop(key, None)


def _decode(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except Exception:
            return None
    return str(value)


def _setnx(key: str, ttl: int, value: str) -> bool:
    client = redis_client
    if client is not None:
        try:
            created = bool(client.setnx(key, value))
            if created:
                client.expire(key, ttl)
            return created
        except Exception:
            pass
    return _memory_setnx(key, ttl, value)


def _setex(key: str, ttl: int, value: str) -> None:
    client = redis_client
    if client is not None:
        try:
            client.setex(key, ttl, value)
            return
        except Exception:
            pass
    _memory_setex(key, ttl, value)


def _get(key: str) -> Optional[str]:
    client = redis_client
    if client is not None:
        try:
            value = client.get(key)
        except Exception:
            value = None
        decoded = _decode(value)
        if decoded is not None:
            return decoded
    return _memory_get(key)


def _delete(key: str) -> None:
    client = redis_client
    if client is not None:
        try:
            client.delete(key)
        except Exception:
            pass
    _memory_delete(key)


def _int_or_none(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass(slots=True)
class _Context:
    app_context: ContextTypes.DEFAULT_TYPE
    chat_id: int
    user_id: int
    message_id: Optional[int]

    @property
    def bot(self):  # pragma: no cover - helper
        return self.app_context.bot


async def _upsert_sum_card(ctx: _Context, payload: Mapping[str, object]) -> Optional[int]:
    message_id = ctx.message_id or _int_or_none(_get(_MSG_KEY_TMPL.format(ctx.chat_id)))
    bot = ctx.bot
    try:
        if message_id:
            await bot.edit_message_text(
                chat_id=ctx.chat_id,
                message_id=message_id,
                text=str(payload.get("text", "")),
                reply_markup=payload.get("reply_markup"),
                parse_mode=payload.get("parse_mode"),
                disable_web_page_preview=payload.get("disable_web_page_preview", True),
            )
            _setex(_MSG_KEY_TMPL.format(ctx.chat_id), MSG_TTL_SECONDS, str(message_id))
            return message_id
    except BadRequest as exc:
        if "message is not modified" in str(exc).lower():
            return message_id
        log.debug("sum.edit_failed", exc_info=True)
    except Exception:
        log.warning("sum.edit_unexpected", exc_info=True)

    try:
        sent = await bot.send_message(
            chat_id=ctx.chat_id,
            text=str(payload.get("text", "")),
            reply_markup=payload.get("reply_markup"),
            parse_mode=payload.get("parse_mode"),
            disable_web_page_preview=payload.get("disable_web_page_preview", True),
        )
    except Exception:
        log.exception("sum.send_failed")
        return message_id

    mid = getattr(sent, "message_id", None)
    if isinstance(mid, int):
        _setex(_MSG_KEY_TMPL.format(ctx.chat_id), MSG_TTL_SECONDS, str(mid))
        return mid
    return message_id


async def _open(ctx: _Context, *, source: str) -> Optional[int]:
    if not _setnx(_LOCK_KEY_TMPL.format(ctx.user_id), _LOCK_TTL_SECONDS, "1"):
        sum_open_total.labels(result="skip_inflight").inc()
        return _int_or_none(_get(_MSG_KEY_TMPL.format(ctx.chat_id)))
    try:
        now = _now_ms()
        last_raw = _get(_LAST_KEY_TMPL.format(ctx.user_id))
        try:
            last = int(last_raw) if last_raw is not None else None
        except (TypeError, ValueError):
            last = None
        if last is not None and now - last < _DEBOUNCE_WINDOW_MS:
            sum_open_total.labels(result="debounce").inc()
            return _int_or_none(_get(_MSG_KEY_TMPL.format(ctx.chat_id)))
        _setex(_LAST_KEY_TMPL.format(ctx.user_id), _LAST_TTL_SECONDS, str(now))

        clear_wait_states(ctx.user_id, reason="sum_open")

        payload = render_sum_card()
        new_ctx = _Context(
            app_context=ctx.app_context,
            chat_id=ctx.chat_id,
            user_id=ctx.user_id,
            message_id=ctx.message_id or _int_or_none(_get(_MSG_KEY_TMPL.format(ctx.chat_id))),
        )
        mid = await _upsert_sum_card(new_ctx, payload)
        sum_open_total.labels(result="ok").inc()
        return mid
    finally:
        _delete(_LOCK_KEY_TMPL.format(ctx.user_id))


async def open_from_button(context: ButtonContext) -> Optional[int]:
    chat_id = context.chat_id
    user_id = context.user_id
    if chat_id is None or user_id is None:
        return None
    message_id = None
    query = context.query
    if query is not None:
        message = getattr(query, "message", None)
        message_id = getattr(message, "message_id", None)
    ctx = _Context(
        app_context=context.app_context,
        chat_id=int(chat_id),
        user_id=int(user_id),
        message_id=_ensure_int(message_id),
    )
    return await _open(ctx, source="button")


def _ensure_int(value: object) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


async def open_via_registry(update, context: ContextTypes.DEFAULT_TYPE, payload: Optional[dict[str, object]] = None):
    chat = getattr(update, "effective_chat", None)
    user = getattr(update, "effective_user", None)
    chat_id = getattr(chat, "id", None)
    user_id = getattr(user, "id", None)
    if chat_id is None or user_id is None:
        return
    query = getattr(update, "callback_query", None)
    message = getattr(query, "message", None) if query is not None else None
    message_id = getattr(message, "message_id", None)
    ctx = _Context(
        app_context=context,
        chat_id=int(chat_id),
        user_id=int(user_id),
        message_id=_ensure_int(message_id),
    )
    await _open(ctx, source="registry")


async def open_from_callback(callback: HubCallbackContext) -> None:
    if callback.chat_id is None or callback.user_id is None:
        return
    message_id = callback.card_message_id
    if message_id is None and callback.query is not None:
        msg = getattr(callback.query, "message", None)
        message_id = getattr(msg, "message_id", None)
    ctx = _Context(
        app_context=callback.application_context,
        chat_id=int(callback.chat_id),
        user_id=int(callback.user_id),
        message_id=_ensure_int(message_id),
    )
    await _open(ctx, source="callback")


async def view_from_callback(callback: HubCallbackContext) -> None:
    payload = getattr(callback.update, "_ui_router_payload", {})
    view = None
    if isinstance(payload, Mapping):
        raw_view = payload.get("view")
        view = str(raw_view or "").strip().lower()
    if view == "prompt":
        await prepare_prompt(callback)
    else:
        await open_from_callback(callback)


async def back_from_callback(callback: HubCallbackContext) -> None:
    if callback.user_id is not None:
        clear_wait_states(int(callback.user_id), reason="sum_back")
    await open_from_callback(callback)


async def prepare_prompt(callback: HubCallbackContext) -> None:
    if callback.chat_id is None or callback.user_id is None:
        return
    query = callback.query
    if query is not None:
        with suppress(Exception):
            await query.answer()
    set_wait(
        int(callback.user_id),
        "sum_prompt",
        callback.card_message_id or 0,
        chat_id=int(callback.chat_id),
        meta={"card": callback.card_message_id or 0},
    )
    ctx = _Context(
        app_context=callback.application_context,
        chat_id=int(callback.chat_id),
        user_id=int(callback.user_id),
        message_id=_ensure_int(callback.card_message_id),
    )
    payload = render_sum_prompt_card()
    await _upsert_sum_card(ctx, payload)


def _payload_key(user_id: int) -> str:
    return _PAYLOAD_KEY_TMPL.format(int(user_id))


def _run_key(user_id: int) -> str:
    return _RUN_KEY_TMPL.format(int(user_id))


def _store_payload(user_id: int, text: str) -> None:
    _setex(_payload_key(user_id), _PAYLOAD_TTL_SECONDS, text)


def _load_payload(user_id: int) -> Optional[str]:
    return _get(_payload_key(user_id))


def _clear_payload(user_id: int) -> None:
    _delete(_payload_key(user_id))


async def handle_wait_input(
    ctx: ContextTypes.DEFAULT_TYPE,
    message: Message,
    cleaned_text: str,
    wait_state: WaitInputState,
    *,
    user_id: Optional[int],
) -> bool:
    text = cleaned_text.strip()
    if not text or len(text) < 5:
        await message.reply_text("Нужно хотя бы 5 символов для суммирования.")
        return True
    if user_id is None:
        return True
    _store_payload(int(user_id), text)
    chat_id = wait_state.chat_id
    current_mid = _int_or_none(_get(_MSG_KEY_TMPL.format(chat_id))) or wait_state.card_msg_id
    context = _Context(
        app_context=ctx,
        chat_id=int(chat_id),
        user_id=int(user_id),
        message_id=_ensure_int(current_mid),
    )
    payload = render_sum_ready_to_start(text)
    await _upsert_sum_card(context, payload)
    return True


def _acquire_run(user_id: int) -> bool:
    return _setnx(_run_key(user_id), _RUN_TTL_SECONDS, str(_now_ms()))


def _release_run(user_id: int) -> None:
    _delete(_run_key(user_id))


async def start_from_callback(callback: HubCallbackContext) -> None:
    query = callback.query
    if callback.user_id is None or callback.chat_id is None:
        if query is not None:
            with suppress(Exception):
                await query.answer()
        return
    user_id = int(callback.user_id)
    chat_id = int(callback.chat_id)
    if not _acquire_run(user_id):
        if query is not None:
            with suppress(Exception):
                await query.answer("Уже выполняю, подождите…", show_alert=False)
        sum_start_total.labels(result="skip_busy").inc()
        return
    try:
        if query is not None:
            with suppress(Exception):
                await query.answer()
        payload_text = _load_payload(user_id)
        if not payload_text:
            _release_run(user_id)
            if query is not None:
                with suppress(Exception):
                    await query.answer("Нет текста для суммирования.", show_alert=True)
            sum_start_total.labels(result="no_payload").inc()
            return
        current_mid = _int_or_none(_get(_MSG_KEY_TMPL.format(chat_id)))
        if current_mid is None:
            current_mid = callback.card_message_id
        context = _Context(
            app_context=callback.application_context,
            chat_id=chat_id,
            user_id=user_id,
            message_id=_ensure_int(current_mid),
        )
        progress_payload = render_sum_progress(step="Инициализация…", disabled=True)
        message_id = await _upsert_sum_card(context, progress_payload)
        if message_id is not None:
            context.message_id = message_id

        request = {
            "taskType": "summary",
            "input": payload_text[:10000],
            "speed": "normal",
        }
        try:
            kie_summary.validate_summary_request(request)
            task_id = kie_summary.create(request)
            if not task_id:
                raise RuntimeError("KIE: taskId is empty")
        except Exception as exc:
            _release_run(user_id)
            error_payload = render_sum_error(str(exc))
            await _upsert_sum_card(context, error_payload)
            sum_start_total.labels(result="error").inc()
            return

        for attempt in range(1, _POLL_MAX_ATTEMPTS + 1):
            status = kie_summary.status(task_id)
            if status.success and status.result:
                _release_run(user_id)
                clear_wait_states(user_id, reason="sum_complete")
                result_payload = render_sum_result(status.result)
                await _upsert_sum_card(context, result_payload)
                sum_start_total.labels(result="ok").inc()
                _clear_payload(user_id)
                return
            step_text = f"Ожидание {attempt}/{_POLL_MAX_ATTEMPTS}"
            progress_payload = render_sum_progress(step=step_text, disabled=True)
            await _upsert_sum_card(context, progress_payload)
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)

        queue_payload = render_sum_queue_notice(task_id)
        await _upsert_sum_card(context, queue_payload)
        schedule_lazy_check_summary(
            task_id,
            callback.application_context,
            chat_id=chat_id,
            message_id=context.message_id,
            user_id=user_id,
        )
        sum_start_total.labels(result="queued").inc()
    finally:
        if _get(_run_key(user_id)) is not None:
            _setex(_run_key(user_id), _RUN_TTL_SECONDS, str(_now_ms()))


def schedule_lazy_check_summary(
    task_id: str,
    ctx: ContextTypes.DEFAULT_TYPE,
    *,
    chat_id: int,
    message_id: Optional[int],
    user_id: int,
) -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        log.debug("sum.lazy_check.no_loop")
        return

    async def _check() -> None:
        for _ in range(_LAZY_CHECK_MAX_ATTEMPTS):
            await asyncio.sleep(_LAZY_CHECK_INTERVAL)
            status = kie_summary.status(task_id)
            if status.success and status.result:
                context = _Context(
                    app_context=ctx,
                    chat_id=chat_id,
                    user_id=user_id,
                    message_id=_ensure_int(message_id),
                )
                clear_wait_states(user_id, reason="sum_lazy_complete")
                payload = render_sum_result(status.result)
                await _upsert_sum_card(context, payload)
                _release_run(user_id)
                _clear_payload(user_id)
                sum_start_total.labels(result="ok").inc()
                return
        log.info("sum.lazy_check.exhausted", extra={"task": task_id, "user_id": user_id})

    loop.create_task(_check())


__all__ = [
    "open_from_button",
    "open_via_registry",
    "open_from_callback",
    "view_from_callback",
    "start_from_callback",
    "back_from_callback",
    "handle_wait_input",
    "prepare_prompt",
    "schedule_lazy_check_summary",
]
