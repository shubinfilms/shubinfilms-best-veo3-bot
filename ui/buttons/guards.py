from __future__ import annotations

import logging
from typing import Optional

from telegram.ext import ContextTypes

from .errors import ButtonAccessDenied, ButtonFeatureDisabled
from .types import ButtonSpec

log = logging.getLogger(__name__)


def resolve_user_tier(ctx: ContextTypes.DEFAULT_TYPE, user_id: Optional[int]) -> str:
    if user_id is not None:
        try:
            from bot import ADMIN_IDS  # local import to avoid cycles
        except Exception:  # pragma: no cover - optional
            ADMIN_IDS = set()
        if user_id in ADMIN_IDS:
            return "admin"

    user_data = getattr(ctx, "user_data", None)
    if isinstance(user_data, dict):
        explicit = user_data.get("button_user_tier") or user_data.get("user_tier")
        if isinstance(explicit, str):
            return explicit
        paid_flag = user_data.get("is_paid") or user_data.get("paid")
        if isinstance(paid_flag, bool) and paid_flag:
            return "paid"

    return "free"


def guard_access(spec: ButtonSpec, ctx: ContextTypes.DEFAULT_TYPE, user_id: Optional[int]) -> str:
    """Ensure the current user satisfies the required access level."""

    user_tier = resolve_user_tier(ctx, user_id)
    required = set(spec.access)
    if not required:
        return user_tier

    if "admin" in required and user_tier == "admin":
        return user_tier
    if "paid" in required:
        if user_tier in {"paid", "admin"}:
            return user_tier
        log.info("ui.button.access_denied", extra={"button": spec.id, "tier": user_tier})
        raise ButtonAccessDenied(spec.id)
    if "all" in required:
        return user_tier

    log.info("ui.button.access_denied", extra={"button": spec.id, "tier": user_tier})
    raise ButtonAccessDenied(spec.id)


def guard_feature(spec: ButtonSpec, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    flag = spec.feature_flag
    if not flag:
        return

    application = getattr(ctx, "application", None)
    containers = []
    if application is not None:
        bot_data = getattr(application, "bot_data", None)
        if isinstance(bot_data, dict):
            containers.append(bot_data)
    bot_data = getattr(ctx, "bot_data", None)
    if isinstance(bot_data, dict):
        containers.append(bot_data)

    for container in containers:
        features = container.get("feature_flags") or container.get("features")
        if isinstance(features, dict) and flag in features:
            value = features[flag]
            if isinstance(value, bool):
                if value:
                    return
                raise ButtonFeatureDisabled(flag)
        value = container.get(flag)
        if isinstance(value, bool):
            if value:
                return
            raise ButtonFeatureDisabled(flag)

    # default to disabled unless explicitly enabled
    raise ButtonFeatureDisabled(flag)
