from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import httpx
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import BadRequest
from telegram.ext import ContextTypes

from balance import ensure_tokens
from helpers.progress import PROGRESS_STORAGE_KEY, send_progress_message
from redis_utils import credit_balance, debit_try
from settings import (
    FEATURE_SORA2_ENABLED,
    KIA_SORA2_RETRY,
    KIA_SORA2_TIMEOUT,
    KIE_API_KEY,
    KIE_BASE_URL,
    SORA2_ALLOWED_AR,
    SORA2_MAX_DURATION,
    SORA2_PRICE,
    SORA2_QUEUE_KEY,
)
from utils.input_state import WaitInputState, WaitKind, clear_wait_state, set_wait_state

logger = logging.getLogger(__name__)


_SORA2_STATE_KEY = "_sora2_simple_state"
_ALLOWED_DURATIONS: Tuple[int, ...] = (3, 6, 10)
_MODEL_CHOICES: Tuple[Tuple[str, str], ...] = (
    ("⚡️ Быстрый", "sora2_fast"),
    ("🎞️ Качество", "sora2_quality"),
)


@dataclass
class _UiState:
    aspect: str = "16:9"
    duration: int = 6
    model: str = "sora2_fast"
    message_id: Optional[int] = None
    active_job_id: Optional[str] = None
    progress_state: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _Job:
    job_id: str
    user_id: int
    chat_id: int
    aspect: str
    duration: int
    model: str
    hold_amount: int
    reply_to: Optional[int]
    progress: Dict[str, Any]
    chat_state: Dict[str, Any]
    ui_state: _UiState
    ref_id: str
    queued_at: float = field(default_factory=time.time)
    cancelled: bool = False
    refunded: bool = False
    started: bool = False

    def mark_cancelled(self) -> None:
        self.cancelled = True


_QUEUE: "asyncio.Queue[_Job]" = asyncio.Queue()
_WORKERS: Dict[int, asyncio.Task[Any]] = {}
_JOBS: Dict[str, _Job] = {}


def _ensure_state(context: ContextTypes.DEFAULT_TYPE) -> _UiState:
    chat_data = getattr(context, "chat_data", None)
    if not isinstance(chat_data, dict):
        raise RuntimeError("chat_data unavailable for Sora2 state")
    state_obj = chat_data.get(_SORA2_STATE_KEY)
    if isinstance(state_obj, _UiState):
        return state_obj
    if isinstance(state_obj, dict):
        ui_state = _UiState(
            aspect=_normalize_aspect(state_obj.get("aspect")),
            duration=_normalize_duration(state_obj.get("duration")),
            model=_normalize_model(state_obj.get("model")),
            message_id=state_obj.get("message_id"),
            active_job_id=state_obj.get("active_job_id"),
            progress_state=state_obj.get("progress_state") or {},
        )
    else:
        ui_state = _UiState()
    chat_data[_SORA2_STATE_KEY] = ui_state
    return ui_state


def _normalize_aspect(value: Any) -> str:
    candidate = str(value or "16:9").strip()
    if candidate not in SORA2_ALLOWED_AR:
        return "16:9"
    return candidate


def _normalize_duration(value: Any) -> int:
    try:
        duration = int(value)
    except (TypeError, ValueError):
        duration = 6
    if duration not in _ALLOWED_DURATIONS:
        closest = min(_ALLOWED_DURATIONS, key=lambda x: abs(x - duration))
        duration = closest
    return max(1, min(int(SORA2_MAX_DURATION), duration))


def _normalize_model(value: Any) -> str:
    candidate = str(value or "sora2_fast").strip() or "sora2_fast"
    allowed = {code for _, code in _MODEL_CHOICES}
    if candidate not in allowed:
        return "sora2_fast"
    return candidate


async def _render_or_send(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    ui_state: _UiState,
) -> None:
    chat = update.effective_chat
    message = update.effective_message
    query = update.callback_query
    if query is not None:
        try:
            await query.answer()
        except BadRequest:
            pass
    if chat is None:
        return
    chat_id = chat.id

    text = _build_text()
    keyboard = _build_keyboard(ui_state)

    chat_data = getattr(context, "chat_data", None)
    if isinstance(chat_data, dict):
        chat_data[_SORA2_STATE_KEY] = ui_state

    if ui_state.message_id is not None:
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=ui_state.message_id,
                text=text,
                reply_markup=keyboard,
            )
            return
        except BadRequest as exc:
            logger.debug("sora2.edit_failed", extra={"chat_id": chat_id, "error": str(exc)})
        except Exception as exc:  # pragma: no cover - network errors
            logger.warning("sora2.edit_crash", exc_info=True, extra={"chat_id": chat_id})

    sent = await context.bot.send_message(chat_id=chat_id, text=text, reply_markup=keyboard)
    ui_state.message_id = getattr(sent, "message_id", None)


def _build_text() -> str:
    lines = [
        "🎬 Sora2",
        "Выберите параметры и запустите генерацию.",
        f"Стоимость: {SORA2_PRICE} 💎",
    ]
    return "\n".join(lines)


def _build_keyboard(ui_state: _UiState) -> InlineKeyboardMarkup:
    rows: List[List[InlineKeyboardButton]] = []
    aspect_row: List[InlineKeyboardButton] = []
    for aspect in sorted(SORA2_ALLOWED_AR):
        label = aspect
        if aspect == ui_state.aspect:
            label = f"✅ {label}"
        aspect_row.append(
            InlineKeyboardButton(label, callback_data=f"sora2:set:ar={aspect}")
        )
    rows.append(aspect_row)

    duration_row: List[InlineKeyboardButton] = []
    for duration in _ALLOWED_DURATIONS:
        label = f"{duration}с"
        if duration == ui_state.duration:
            label = f"✅ {label}"
        duration_row.append(
            InlineKeyboardButton(label, callback_data=f"sora2:set:dur={duration}")
        )
    rows.append(duration_row)

    model_row: List[InlineKeyboardButton] = []
    for label, code in _MODEL_CHOICES:
        title = label
        if code == ui_state.model:
            title = f"✅ {label}"
        model_row.append(
            InlineKeyboardButton(title, callback_data=f"sora2:set:model={code}")
        )
    rows.append(model_row)

    if ui_state.active_job_id:
        rows.append(
            [
                InlineKeyboardButton("⏳ В очереди", callback_data="noop"),
                InlineKeyboardButton("✖️ Отменить", callback_data="sora2:cancel"),
            ]
        )
        rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="back")])
    else:
        run_payload = _build_run_payload(ui_state)
        rows.append(
            [
                InlineKeyboardButton("▶️ Запустить", callback_data=run_payload),
                InlineKeyboardButton("⬅️ Назад", callback_data="back"),
            ]
        )
    return InlineKeyboardMarkup(rows)


def _build_run_payload(ui_state: _UiState) -> str:
    return (
        "sora2:run:" f"ar={ui_state.aspect};dur={ui_state.duration};model={ui_state.model}"
    )


async def sora2_open_cb(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not FEATURE_SORA2_ENABLED:
        query = update.callback_query
        if query is not None:
            await query.answer("⚠️ Режим Sora2 временно недоступен", show_alert=True)
        return
    ui_state = _ensure_state(context)
    logger.info("nav.start", extra={"kind": "sora2", "chat_id": update.effective_chat.id if update.effective_chat else None})
    await _render_or_send(update, context, ui_state=ui_state)
    logger.info("nav.finish", extra={"kind": "sora2", "chat_id": update.effective_chat.id if update.effective_chat else None})
    logger.info("sora2.open", extra={"chat_id": update.effective_chat.id if update.effective_chat else None})


async def sora2_open_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not FEATURE_SORA2_ENABLED:
        return
    ui_state = _ensure_state(context)
    logger.info("nav.start", extra={"kind": "sora2", "chat_id": update.effective_chat.id if update.effective_chat else None})
    await _render_or_send(update, context, ui_state=ui_state)
    logger.info("nav.finish", extra={"kind": "sora2", "chat_id": update.effective_chat.id if update.effective_chat else None})
    logger.info("sora2.open", extra={"chat_id": update.effective_chat.id if update.effective_chat else None, "source": "text"})


async def sora2_set_param_cb(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query is None:
        return
    if not FEATURE_SORA2_ENABLED:
        await query.answer("⚠️ Режим отключен", show_alert=True)
        return
    data = query.data or ""
    parts = data.split(":", 2)
    if len(parts) < 3:
        await query.answer()
        return
    _, _, payload = parts
    if "=" not in payload:
        await query.answer()
        return
    key, value = payload.split("=", 1)
    ui_state = _ensure_state(context)
    changed = False
    if key == "ar":
        new_value = _normalize_aspect(value)
        if new_value != ui_state.aspect:
            ui_state.aspect = new_value
            changed = True
    elif key == "dur":
        new_duration = _normalize_duration(value)
        if new_duration != ui_state.duration:
            ui_state.duration = new_duration
            changed = True
    elif key == "model":
        new_model = _normalize_model(value)
        if new_model != ui_state.model:
            ui_state.model = new_model
            changed = True
    if not changed:
        await query.answer()
        return
    logger.info(
        "sora2.param",
        extra={
            "chat_id": update.effective_chat.id if update.effective_chat else None,
            "key": key,
            "value": value,
        },
    )
    await _render_or_send(update, context, ui_state=ui_state)


def _ensure_worker(application: Any) -> None:
    app_id = id(application)
    existing = _WORKERS.get(app_id)
    if existing and not existing.done():
        return
    task = application.create_task(_queue_worker(application))
    _WORKERS[app_id] = task


async def _queue_worker(application: Any) -> None:
    while True:
        job = await _QUEUE.get()
        try:
            if job.cancelled:
                logger.info(
                    "sora2.job.skip",
                    extra={"chat_id": job.chat_id, "job_id": job.job_id, "reason": "cancelled"},
                )
                if not job.refunded:
                    _refund_hold(job)
                _finalize_job_state(job)
                continue
            job.started = True
            progress_ctx = _ProgressContext(
                application.bot,
                job.progress,
                chat_data=job.chat_state,
                job_queue=getattr(application, "job_queue", None),
            )
            await send_progress_message(progress_ctx, "render")
            try:
                result = await _call_kia(job)
            except Exception as exc:
                logger.warning(
                    "sora2.error",
                    extra={"chat_id": job.chat_id, "job_id": job.job_id, "error": str(exc)},
                )
                if not job.refunded:
                    _refund_hold(job)
                await _notify_error(application, job, "⚠️ Не удалось получить результат Sora2. 💎 Токены возвращены.")
                await send_progress_message(progress_ctx, "finish")
                _finalize_job_state(job)
                await _refresh_ui(application, job)
                continue
            if job.cancelled:
                if not job.refunded:
                    _refund_hold(job)
                await _notify_error(application, job, "Задача Sora2 отменена.")
                await send_progress_message(progress_ctx, "finish")
                _finalize_job_state(job)
                await _refresh_ui(application, job)
                continue
            await _deliver_success(application, job, result)
            job.progress["success"] = True
            await send_progress_message(progress_ctx, "finish")
            _finalize_job_state(job)
            await _refresh_ui(application, job)
        finally:
            _QUEUE.task_done()


class _ProgressContext:
    def __init__(self, bot: Any, progress: Dict[str, Any], *, chat_data: Optional[Dict[str, Any]] = None, job_queue: Any = None):
        self.bot = bot
        self.chat_data = chat_data if isinstance(chat_data, dict) else {PROGRESS_STORAGE_KEY: progress}
        if PROGRESS_STORAGE_KEY not in self.chat_data:
            self.chat_data[PROGRESS_STORAGE_KEY] = progress
        self.job_queue = job_queue


async def _notify_error(application: Any, job: _Job, text: str) -> None:
    try:
        await application.bot.send_message(chat_id=job.chat_id, text=text)
    except Exception:  # pragma: no cover - defensive logging
        logger.exception("sora2.notify_error", extra={"chat_id": job.chat_id})


def _refund_hold(job: _Job) -> None:
    if job.refunded:
        return
    try:
        new_balance = credit_balance(
            job.user_id,
            job.hold_amount,
            reason="service:refund",
            meta={"service": "SORA2", "job_id": job.job_id, "reason": "cancel"},
        )
        logger.info(
            "sora2.refund",
            extra={"chat_id": job.chat_id, "job_id": job.job_id, "balance": new_balance},
        )
    except Exception:  # pragma: no cover - ledger errors
        logger.exception("sora2.refund_failed", extra={"chat_id": job.chat_id})
    job.refunded = True


async def _call_kia(job: _Job) -> Mapping[str, Any]:
    if not KIE_API_KEY:
        raise RuntimeError("KIE_API_KEY is not configured")
    headers = {"Authorization": f"Bearer {KIE_API_KEY}"}
    payload = {
        "user_id": job.user_id,
        "ar": job.aspect,
        "duration": job.duration,
        "model": job.model,
        "ref_id": job.ref_id,
        "meta": {"queue": SORA2_QUEUE_KEY},
    }
    url = "/v2/sora2/generate"
    base_url = (KIE_BASE_URL or "").rstrip("/")
    full_url = f"{base_url}{url}" if base_url else url

    delay = 1.5
    last_exc: Optional[Exception] = None
    for attempt in range(max(1, int(KIA_SORA2_RETRY)) + 1):
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(KIA_SORA2_TIMEOUT)) as client:
                response = await client.post(full_url, json=payload, headers=headers)
        except (httpx.TimeoutException, httpx.RequestError) as exc:
            last_exc = exc
            if attempt < KIA_SORA2_RETRY:
                await asyncio.sleep(delay)
                delay = min(delay * 2, 15)
                continue
            raise
        if response.status_code >= 500:
            last_exc = RuntimeError(f"server error {response.status_code}")
            if attempt < KIA_SORA2_RETRY:
                await asyncio.sleep(delay)
                delay = min(delay * 2, 15)
                continue
            raise RuntimeError(f"Sora2 backend error {response.status_code}")
        if response.status_code >= 400:
            raise RuntimeError(f"Sora2 request failed with status {response.status_code}")
        try:
            data = response.json()
        except ValueError as exc:  # pragma: no cover - invalid json
            last_exc = exc
            if attempt < KIA_SORA2_RETRY:
                await asyncio.sleep(delay)
                delay = min(delay * 2, 15)
                continue
            raise
        if not isinstance(data, Mapping):
            raise RuntimeError("Sora2 response is not a mapping")
        return data
    if last_exc:
        raise last_exc
    raise RuntimeError("Sora2 request failed")


def _extract_media(data: Mapping[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    result = data.get("result") if isinstance(data, Mapping) else None
    if not isinstance(result, Mapping):
        result = data
    video = result.get("video") if isinstance(result, Mapping) else None
    if isinstance(video, Mapping):
        file_id = video.get("file_id") or video.get("telegram_file_id")
        url = video.get("url") or video.get("video_url")
        caption = video.get("caption") or result.get("caption")
        return str(file_id) if file_id else None, str(url) if url else None, caption
    file_id = result.get("file_id") if isinstance(result, Mapping) else None
    url = result.get("url") if isinstance(result, Mapping) else None
    caption = result.get("caption") if isinstance(result, Mapping) else None
    return str(file_id) if file_id else None, str(url) if url else None, caption


async def _deliver_success(application: Any, job: _Job, payload: Mapping[str, Any]) -> None:
    file_id, url, caption = _extract_media(payload)
    keyboard = InlineKeyboardMarkup(
        [[InlineKeyboardButton("Сгенерировать ещё", callback_data="sora2_open")]]
    )
    text_caption = caption or "🎬 Результат Sora2 готов!"
    try:
        if file_id:
            await application.bot.send_video(
                chat_id=job.chat_id,
                video=file_id,
                caption=text_caption,
                reply_markup=keyboard,
            )
        elif url:
            await application.bot.send_message(
                chat_id=job.chat_id,
                text=f"{text_caption}\n{url}",
                reply_markup=keyboard,
            )
        else:
            await application.bot.send_message(
                chat_id=job.chat_id,
                text=text_caption,
                reply_markup=keyboard,
            )
        logger.info(
            "sora2.done",
            extra={"chat_id": job.chat_id, "job_id": job.job_id},
        )
    except Exception:  # pragma: no cover - telegram errors
        logger.exception("sora2.send_failed", extra={"chat_id": job.chat_id})


def _finalize_job_state(job: _Job) -> None:
    _JOBS.pop(job.job_id, None)
    job.ui_state.active_job_id = None
    if isinstance(job.chat_state, dict):
        job.chat_state.pop(PROGRESS_STORAGE_KEY, None)
    job.ui_state.progress_state = {}
    clear_wait_state(job.user_id, reason="sora2_done")


async def _refresh_ui(application: Any, job: _Job) -> None:
    message_id = job.ui_state.message_id
    if message_id is None:
        return
    try:
        await application.bot.edit_message_text(
            chat_id=job.chat_id,
            message_id=message_id,
            text=_build_text(),
            reply_markup=_build_keyboard(job.ui_state),
        )
    except BadRequest:
        pass
    except Exception:  # pragma: no cover - telegram errors
        logger.debug("sora2.refresh_fail", exc_info=True, extra={"chat_id": job.chat_id})


async def sora2_run_cb(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query is None:
        return
    if not FEATURE_SORA2_ENABLED:
        await query.answer("⚠️ Режим отключен", show_alert=True)
        return
    user = update.effective_user
    chat = update.effective_chat
    message = query.message
    if not user or not chat or not message:
        await query.answer("⚠️ Недоступно", show_alert=True)
        return
    user_id = user.id
    chat_id = chat.id

    data = query.data or ""
    payload = data.split(":", 2)[2] if data.count(":") >= 2 else ""
    params = {}
    for part in payload.split(";"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        params[key] = value
    aspect = _normalize_aspect(params.get("ar"))
    duration = _normalize_duration(params.get("dur"))
    model = _normalize_model(params.get("model"))

    if aspect not in SORA2_ALLOWED_AR:
        await query.answer("⚠️ Неверный формат", show_alert=True)
        return
    if duration < 1 or duration > SORA2_MAX_DURATION:
        await query.answer("⚠️ Неверная длительность", show_alert=True)
        return

    ui_state = _ensure_state(context)
    ui_state.aspect = aspect
    ui_state.duration = duration
    ui_state.model = model

    if not await ensure_tokens(context, chat_id, user_id, SORA2_PRICE, reply_to=message.message_id):
        await query.answer("💎 Недостаточно токенов", show_alert=True)
        return

    ok, balance_after = debit_try(
        user_id,
        SORA2_PRICE,
        reason="service:start",
        meta={"service": "SORA2", "mode": "video", "duration": duration},
    )
    if not ok:
        await query.answer("💎 Недостаточно токенов", show_alert=True)
        return

    progress = {
        "chat_id": chat_id,
        "user_id": user_id,
        "mode": "sora2",
        "reply_to_message_id": message.message_id,
        "success": False,
        "job_id": None,
    }
    ui_state.progress_state = progress
    chat_data = getattr(context, "chat_data", None)
    if isinstance(chat_data, dict):
        chat_data[PROGRESS_STORAGE_KEY] = progress
    await send_progress_message(context, "start")

    job_id = uuid.uuid4().hex
    progress["job_id"] = job_id

    wait_state = WaitInputState(
        kind=WaitKind.SORA2,
        card_msg_id=ui_state.message_id or message.message_id,
        chat_id=chat_id,
        meta={"job_id": job_id},
    )
    set_wait_state(user_id, wait_state)

    ref_id = uuid.uuid4().hex
    chat_data_ref = chat_data if isinstance(chat_data, dict) else {}
    job = _Job(
        job_id=job_id,
        user_id=user_id,
        chat_id=chat_id,
        aspect=aspect,
        duration=duration,
        model=model,
        hold_amount=SORA2_PRICE,
        reply_to=message.message_id,
        progress=progress,
        chat_state=chat_data_ref,
        ui_state=ui_state,
        ref_id=ref_id,
    )
    _JOBS[job_id] = job
    ui_state.active_job_id = job_id
    _ensure_worker(context.application)
    await _QUEUE.put(job)
    await query.answer("🚀 Запускаю Sora2")
    logger.info(
        "sora2.run",
        extra={
            "chat_id": chat_id,
            "user_id": user_id,
            "job_id": job_id,
            "aspect": aspect,
            "duration": duration,
            "model": model,
            "balance": balance_after,
        },
    )
    await _render_or_send(update, context, ui_state=ui_state)


async def sora2_cancel_cb(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query is None:
        return
    if not FEATURE_SORA2_ENABLED:
        await query.answer("⚠️ Режим отключен", show_alert=True)
        return
    user = update.effective_user
    if not user:
        await query.answer()
        return
    chat = update.effective_chat
    if not chat:
        await query.answer()
        return
    ui_state = _ensure_state(context)
    job_id = ui_state.active_job_id
    if not job_id:
        await query.answer("Задача не найдена", show_alert=True)
        return
    job = _JOBS.get(job_id)
    if not job:
        await query.answer("Задача уже завершена", show_alert=True)
        ui_state.active_job_id = None
        clear_wait_state(user.id, reason="sora2_cancel")
        return
    job.mark_cancelled()
    if not job.started and not job.refunded:
        _refund_hold(job)
        await _notify_error(context.application, job, "Задача Sora2 отменена.")
        progress_ctx = _ProgressContext(
            context.bot,
            job.progress,
            chat_data=job.chat_state,
            job_queue=getattr(context, "job_queue", None),
        )
        await send_progress_message(progress_ctx, "finish")
        _finalize_job_state(job)
        await _refresh_ui(context.application, job)
    await query.answer("Задача остановлена")
    logger.info(
        "sora2.cancel",
        extra={"chat_id": chat.id, "job_id": job_id},
    )


__all__ = [
    "sora2_open_cb",
    "sora2_open_text",
    "sora2_set_param_cb",
    "sora2_run_cb",
    "sora2_cancel_cb",
]
