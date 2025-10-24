from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Mapping, Optional

from metrics import (
    inc as metrics_inc,
    ui_ack_latency_ms,
    ui_callback_ack_latency_ms,
    ui_callback_ack_total,
    ui_callback_dedup_total,
    ui_callback_legacy_forwarded_total,
    ui_callback_total,
)
from runtime_metrics import increment_ui_callback_counter
from telegram import Update
from telegram.ext import ContextTypes

from .errors import ButtonAccessDenied, ButtonFeatureDisabled, ButtonNotFound
from .guards import guard_access, guard_feature, resolve_user_tier
from .idempotency import should_process, with_idempotency
from .results import UIResult, acknowledge, show_error
from .telemetry import log_click, log_error, log_success, measure_latency
from .types import ButtonContext, ButtonId, ButtonSpec
from .registry import REGISTRY
from telegram_utils import safe_answer

log = logging.getLogger(__name__)
_ENV = (os.getenv("APP_ENV") or "prod").strip() or "prod"
_BOT_LABELS = {"env": _ENV, "service": "bot"}


@dataclass(slots=True)
class NormalizedCallback:
    raw: str
    normalized: str | None
    action: str | None
    payload: dict[str, object]
    legacy: bool
    dedupe_key: str


_LEGACY_PROFILE_ALIASES: dict[str, tuple[str, str]] = {
    "profile": ("profile", "button"),
    "profile.open": ("profile", "button"),
    "open:profile": ("profile", "menu"),
    "quick:profile": ("profile", "quick"),
    "stars:profile": ("profile", "button"),
}


def _parse_params(parts: list[str]) -> dict[str, str]:
    params: dict[str, str] = {}
    for part in parts:
        chunk = (part or "").strip()
        if not chunk:
            continue
        if "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        params[key] = value
    return params


def normalize_callback_data(data: str) -> NormalizedCallback:
    stripped = (data or "").strip()
    payload: dict[str, object] = {}
    if not stripped:
        return NormalizedCallback(stripped, None, None, payload, False, stripped)

    lowered = stripped.lower()
    if stripped.startswith("btn:"):
        base, *extra = stripped.split("|")
        params = _parse_params(extra)
        src = params.pop("src", None)
        if src:
            payload["source"] = src
        for key, value in params.items():
            payload.setdefault(key, value)
        action = base.split(":", 1)[1].strip() if ":" in base else None
        return NormalizedCallback(
            raw=stripped,
            normalized=base,
            action=action,
            payload=payload,
            legacy=False,
            dedupe_key=base,
        )

    legacy_payload = _LEGACY_PROFILE_ALIASES.get(lowered)
    if legacy_payload:
        action, source = legacy_payload
        payload = {"source": source, "legacy": True}
        normalized = f"btn:{action}"
        return NormalizedCallback(
            raw=stripped,
            normalized=normalized,
            action=action,
            payload=payload,
            legacy=True,
            dedupe_key=normalized,
        )

    legacy = ":" in stripped and not stripped.startswith("btn:")
    return NormalizedCallback(stripped, None, None, payload, legacy, stripped)


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


async def dispatch_via_registry(
    action: str,
    update: Update,
    ctx: ContextTypes.DEFAULT_TYPE,
    *,
    raw_data: str,
    payload: dict[str, object] | None = None,
    legacy: bool = False,
) -> None:
    query = getattr(update, "callback_query", None)
    user = getattr(update, "effective_user", None)
    chat = getattr(update, "effective_chat", None)
    user_id = getattr(user, "id", None)
    chat_id = getattr(chat, "id", None)

    entry = REGISTRY.get(action)
    if entry is None:
        await _log_unmatched(raw_data, update, legacy=legacy)
        return

    if not should_process(user_id, raw_data):
        try:
            ui_callback_dedup_total.labels(action=action, **_BOT_LABELS).inc()
        except Exception:  # pragma: no cover - metrics guard
            log.debug("ui.callback.dedup.metric_failed", exc_info=True)
        metrics_inc("ui_callback_total", tags={"action": action, "result": "coalesced"})
        increment_ui_callback_counter("dedup", "coalesced")
        return

    handler = entry.resolve_handler()
    try:
        setattr(update, "_ui_button_handled", True)
        if query is not None:
            setattr(query, "_ui_button_handled", True)
    except Exception:  # pragma: no cover - best effort
        pass

    log.info(
        "ui.callback.matched",
        extra={
            "action": action,
            "user_id": user_id,
            "chat_id": chat_id,
            "data": raw_data,
            "legacy": legacy,
        },
    )

    try:
        await handler(update, ctx, payload=payload)
    except Exception:
        try:
            ui_callback_total.labels(action=action, result="error", **_BOT_LABELS).inc()
        except Exception:  # pragma: no cover - metrics guard
            log.debug("ui.callback.total.metric_failed", exc_info=True)
        increment_ui_callback_counter("callback", "error")
        log.exception(
            "ui.callback.failed",
            extra={"action": action, "user_id": user_id, "chat_id": chat_id},
        )
        raise

    try:
        ui_callback_total.labels(action=action, result="ok", **_BOT_LABELS).inc()
    except Exception:  # pragma: no cover - metrics guard
        log.debug("ui.callback.total.metric_failed", exc_info=True)
    increment_ui_callback_counter("callback", "ok")


async def _log_unmatched(data: str, update: Update, *, legacy: bool = False) -> None:
    user = getattr(update, "effective_user", None)
    chat = getattr(update, "effective_chat", None)
    user_id = getattr(user, "id", None)
    chat_id = getattr(chat, "id", None)

    log.info(
        "ui.callback.unmatched data=%r user_id=%s chat_id=%s legacy=%s",
        data,
        user_id,
        chat_id,
        legacy,
    )
    try:
        metrics_inc(
            "ui_callback_unmatched_total",
            tags={"source": "telegram", "legacy": "true" if legacy else "false"},
        )
    except Exception:  # pragma: no cover - metrics guard
        log.debug("ui.callback.unmatched.metric_failed", exc_info=True)
    increment_ui_callback_counter("callback", "unmatched")


async def on_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    query = getattr(update, "callback_query", None)
    if query is None:
        return

    if getattr(query, "_ui_button_handled", False) or getattr(update, "_ui_button_handled", False):
        return

    started = time.perf_counter()
    ack_result = await safe_answer(query, cache_time=0)
    latency_ms = max((time.perf_counter() - started) * 1000.0, 0.0)

    try:
        label_result = (ack_result or "ok").strip() or "ok"
        ui_callback_ack_total.labels(result=label_result, **_BOT_LABELS).inc()
        ui_callback_ack_latency_ms.labels(**_BOT_LABELS).observe(latency_ms)
        ui_ack_latency_ms.labels(source="telegram", **_BOT_LABELS).observe(latency_ms)
    except Exception:  # pragma: no cover - metrics guard
        log.debug("ui.callback.ack.metric_failed", exc_info=True)

    raw_data = getattr(query, "data", "")
    normalized = normalize_callback_data(raw_data)
    if not normalized.raw:
        return

    payload = dict(normalized.payload)
    if normalized.action and normalized.normalized:
        if normalized.legacy:
            user = getattr(update, "effective_user", None)
            chat = getattr(update, "effective_chat", None)
            user_id = getattr(user, "id", None)
            chat_id = getattr(chat, "id", None)
            log.info(
                "ui.callback.legacy_forwarded",
                extra={
                    "target": normalized.action,
                    "raw": normalized.raw,
                    "user_id": user_id,
                    "chat_id": chat_id,
                },
            )
            try:
                ui_callback_legacy_forwarded_total.labels(
                    target=normalized.action, **_BOT_LABELS
                ).inc()
            except Exception:  # pragma: no cover - metrics guard
                log.debug("ui.callback.legacy.metric_failed", exc_info=True)
        await dispatch_via_registry(
            normalized.action,
            update,
            ctx,
            raw_data=normalized.dedupe_key,
            payload=payload or None,
            legacy=normalized.legacy,
        )
        return

    await _log_unmatched(normalized.raw, update, legacy=normalized.legacy)


async def handle_unmatched_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await on_callback(update, ctx)
