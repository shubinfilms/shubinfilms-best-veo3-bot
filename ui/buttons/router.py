from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Mapping, Optional

from telegram import Update
from telegram.ext import ContextTypes

from .errors import ButtonAccessDenied, ButtonFeatureDisabled, ButtonNotFound
from .guards import guard_access, guard_feature, resolve_user_tier
from .idempotency import with_idempotency
from .results import UIResult, show_error
from .telemetry import log_click, log_error, log_success, measure_latency
from .types import ButtonContext, ButtonId, ButtonSpec

log = logging.getLogger(__name__)


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
