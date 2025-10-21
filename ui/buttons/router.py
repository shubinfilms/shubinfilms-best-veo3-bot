from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass
from typing import Mapping, Optional

from metrics import ui_callback_dedup_total, ui_callback_unmatched_total
from runtime_metrics import increment_ui_callback_counter
from telegram import Update
from telegram.ext import ContextTypes

from .errors import ButtonAccessDenied, ButtonFeatureDisabled, ButtonNotFound
from .guards import guard_access, guard_feature, resolve_user_tier
from .idempotency import should_process, with_idempotency
from .results import UIResult, acknowledge, show_error
from .telemetry import log_click, log_error, log_success, measure_latency
from .types import ButtonContext, ButtonId, ButtonSpec
from telegram_utils import safe_answer

log = logging.getLogger(__name__)
_ENV = (os.getenv("APP_ENV") or "prod").strip() or "prod"
_BOT_LABELS = {"env": _ENV, "service": "bot"}


@dataclass(slots=True)
class ButtonRouter:
    """Button dispatcher with access control and telemetry."""

    registry: Mapping[ButtonId, ButtonSpec]

    async def dispatch(
        self,
        button_id: ButtonId,
        *,
        update: Update,
        ctx: ContextTypes.DEFAULT_TYPE,
        request_id: Optional[str] = None,
    ) -> UIResult:
        try:
            spec = self.registry[button_id]
        except KeyError as exc:
            log.error("ui.button.unknown", extra={"button": button_id})
            raise ButtonNotFound(button_id) from exc

        query = getattr(update, "callback_query", None)
        chat = getattr(update, "effective_chat", None)
        chat_id = getattr(chat, "id", None)
        if chat_id is None and query is not None:
            message = getattr(query, "message", None)
            chat_id = getattr(getattr(message, "chat", None), "id", None)
        user = getattr(update, "effective_user", None)
        user_id = getattr(user, "id", None)

        req_id = request_id or uuid.uuid4().hex
        context = ButtonContext(
            update=update,
            app_context=ctx,
            spec=spec,
            request_id=req_id,
            query=query,
            chat_id=chat_id,
            user_id=user_id,
        )

        if query is not None:
            ack_result = await safe_answer(query, cache_time=0)
            context.mark_acknowledged(ack_result)
            try:
                setattr(update, "_ui_button_handled", True)
                setattr(query, "_ui_button_handled", True)
            except Exception:  # pragma: no cover - best effort attribute assignment
                pass

            callback_data = getattr(query, "data", None)
            if not should_process(user_id, callback_data):
                ui_callback_dedup_total.labels(action="dropped", **_BOT_LABELS).inc()
                increment_ui_callback_counter("dedup", "dropped")
                log.debug(
                    "ui.button.dedup_dropped",
                    extra={
                        "button": spec.id,
                        "user_id": user_id,
                        "callback_data": callback_data,
                    },
                )
                return acknowledge(spec.id, message="duplicate_click")
            ui_callback_dedup_total.labels(action="passed", **_BOT_LABELS).inc()
            increment_ui_callback_counter("dedup", "passed")

        user_tier = resolve_user_tier(ctx, user_id)
        log_click(spec.id, user_tier)

        try:
            guard_feature(spec, ctx)
            guard_access(spec, ctx, user_id)
        except ButtonFeatureDisabled:
            await context.answer("Скоро", show_alert=True)
            log_error(spec.id, "feature_disabled")
            return show_error(spec.id, "feature_disabled", error_kind="feature_disabled")
        except ButtonAccessDenied:
            await context.answer("Недоступно", show_alert=True)
            log_error(spec.id, "access_denied")
            return show_error(spec.id, "access_denied", error_kind="access_denied")

        handler = spec.handler
        wrapped_handler = with_idempotency(handler)

        try:
            with measure_latency(spec.id):
                result = await wrapped_handler(context)
        except Exception as exc:  # pragma: no cover - defensive
            log.exception(
                "ui.button.failed",
                extra={"button": spec.id, "ctx": context.as_dict()},
            )
            await context.answer("Что-то пошло не так", show_alert=True)
            log_error(spec.id, "exception")
            return show_error(spec.id, "exception", error_kind=exc.__class__.__name__)

        if result.kind == "error":
            error_kind = result.details.get("error_kind", "handler")
            log_error(spec.id, str(error_kind))
        else:
            log_success(spec.id)

        return result


_DEFAULT_ROUTER: Optional[ButtonRouter] = None


def _get_default_router() -> ButtonRouter:
    global _DEFAULT_ROUTER
    if _DEFAULT_ROUTER is None:
        from .registry import BUTTONS

        _DEFAULT_ROUTER = ButtonRouter(registry=BUTTONS)
    return _DEFAULT_ROUTER


async def dispatch(
    button_id: ButtonId,
    *,
    update: Update,
    ctx: ContextTypes.DEFAULT_TYPE,
    request_id: Optional[str] = None,
) -> UIResult:
    router = _get_default_router()
    return await router.dispatch(button_id, update=update, ctx=ctx, request_id=request_id)


async def handle_unmatched_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    del ctx  # context is unused but part of PTB signature

    query = getattr(update, "callback_query", None)
    if query is None:
        return

    if getattr(query, "_ui_button_handled", False) or getattr(update, "_ui_button_handled", False):
        return

    data = getattr(query, "data", None)
    if not data:
        return

    if not isinstance(data, str):
        return

    if not (
        data.startswith("menu:")
        or data.startswith("hub:open:")
        or data.startswith("kb_open")
        or data.startswith("dialog:")
    ):
        return

    user = getattr(update, "effective_user", None)
    message = getattr(query, "message", None)
    message_id = getattr(message, "message_id", None)
    user_id = getattr(user, "id", None)

    ack_result = await safe_answer(query, cache_time=0)

    log.info(
        "ui.callback.unmatched",
        extra={
            "data": data,
            "user_id": user_id,
            "message_id": message_id,
            "ack_result": ack_result,
        },
    )
    try:
        ui_callback_unmatched_total.labels(source="telegram", **_BOT_LABELS).inc()
    except Exception:  # pragma: no cover - defensive metrics guard
        log.debug("ui.callback.unmatched.metric_failed", exc_info=True)
