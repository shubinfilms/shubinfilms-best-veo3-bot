"""Callback router with namespace-aware actions."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from collections.abc import MutableMapping
from typing import Any, Awaitable, Callable, Dict, Optional

from telegram import InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import ContextTypes

from keyboards import TEXT_TO_ACTION
from redis_utils import release_user_lock, user_lock
from state import CardInfo, StateLockTimeout, state
from utils.text_normalizer import normalize_btn_text

log = logging.getLogger(__name__)


HandlerFunc = Callable[["CallbackContext"], Awaitable[None]]


@dataclass(slots=True)
class _Route:
    handler: HandlerFunc
    module: str


@dataclass(slots=True)
class _ScheduledCall:
    func: Callable[[], Awaitable[Any]]
    description: str


class CallbackContext:
    """Context passed to callback handlers."""

    def __init__(
        self,
        *,
        update: Update,
        app_context: ContextTypes.DEFAULT_TYPE,
        namespace: str,
        action: str,
        chat_id: int,
        user_id: Optional[int],
        query,
        session: dict[str, Any],
        module: str,
        card: CardInfo,
    ) -> None:
        self.update = update
        self.application_context = app_context
        self.namespace = namespace
        self.action = action
        self.chat_id = chat_id
        self.user_id = user_id
        self.query = query
        self.session = session
        self.module = module
        self.card = card
        self._scheduled: list[_ScheduledCall] = []
        self._card_message_id: Optional[int] = card.message_id
        self._card_changed: bool = False

    # ------------------------------------------------------------------
    def defer(self, func: Callable[[], Awaitable[Any]], *, description: str) -> None:
        """Schedule an arbitrary coroutine to run after the lock is released."""

        self._scheduled.append(_ScheduledCall(func=func, description=description))

    def schedule_edit(
        self,
        *,
        text: str,
        reply_markup: Optional[InlineKeyboardMarkup],
        parse_mode: ParseMode = ParseMode.HTML,
        disable_web_page_preview: bool = True,
        message_id: Optional[int] = None,
    ) -> None:
        """Schedule a text + markup update for the current card."""

        mid = message_id if message_id is not None else self._card_message_id
        if mid is None:
            raise RuntimeError("card message id is unknown; call schedule_send first")

        async def _edit() -> None:
            try:
                await self.application_context.bot.edit_message_text(
                    chat_id=self.chat_id,
                    message_id=mid,
                    text=text,
                    reply_markup=reply_markup,
                    parse_mode=parse_mode,
                    disable_web_page_preview=disable_web_page_preview,
                )
            except BadRequest as exc:
                log.warning(
                    "router.edit_failed | chat=%s mid=%s ns=%s action=%s err=%s",
                    self.chat_id,
                    mid,
                    self.namespace,
                    self.action,
                    exc,
                )
            else:
                await state.set_card(self.chat_id, self.module, mid)

        self.defer(_edit, description=f"edit:{self.module}:{mid}")
        self._card_changed = True

    def schedule_markup(
        self,
        reply_markup: Optional[InlineKeyboardMarkup],
        *,
        message_id: Optional[int] = None,
    ) -> None:
        """Schedule an inline keyboard update for the current card."""

        mid = message_id if message_id is not None else self._card_message_id
        if mid is None:
            raise RuntimeError("card message id is unknown; call schedule_send first")

        async def _edit() -> None:
            try:
                await self.application_context.bot.edit_message_reply_markup(
                    chat_id=self.chat_id,
                    message_id=mid,
                    reply_markup=reply_markup,
                )
            except BadRequest as exc:
                log.debug(
                    "router.markup_failed | chat=%s mid=%s ns=%s action=%s err=%s",
                    self.chat_id,
                    mid,
                    self.namespace,
                    self.action,
                    exc,
                )
            else:
                await state.set_card(self.chat_id, self.module, mid)

        self.defer(_edit, description=f"markup:{self.module}:{mid}")

    def schedule_send(
        self,
        *,
        text: str,
        reply_markup: Optional[InlineKeyboardMarkup],
        parse_mode: ParseMode = ParseMode.HTML,
        disable_web_page_preview: bool = True,
    ) -> None:
        """Schedule sending a new card and track its message id."""

        async def _send() -> None:
            msg = await self.application_context.bot.send_message(
                chat_id=self.chat_id,
                text=text,
                reply_markup=reply_markup,
                parse_mode=parse_mode,
                disable_web_page_preview=disable_web_page_preview,
            )
            self._card_message_id = msg.message_id
            await state.set_card(self.chat_id, self.module, msg.message_id)

        self.defer(_send, description=f"send:{self.module}")
        self._card_changed = True

    async def execute_scheduled(self) -> None:
        for item in self._scheduled:
            await item.func()

    # ------------------------------------------------------------------
    @property
    def card_message_id(self) -> Optional[int]:
        return self._card_message_id

    @property
    def card_changed(self) -> bool:
        return self._card_changed


_ROUTES: Dict[str, Dict[str, _Route]] = {}
_FALLBACK_HANDLER: Optional[Callable[[Update, ContextTypes.DEFAULT_TYPE], Awaitable[None]]] = None


def register(namespace: str, action: str, *, module: Optional[str] = None) -> Callable[[HandlerFunc], HandlerFunc]:
    """Register a handler for ``namespace:action``."""

    ns = namespace.strip().lower()
    act = action.strip().lower()
    module_name = (module or act or ns).strip().lower()

    def decorator(func: HandlerFunc) -> HandlerFunc:
        _ROUTES.setdefault(ns, {})[act] = _Route(handler=func, module=module_name)
        return func

    return decorator


def set_fallback(handler: Callable[[Update, ContextTypes.DEFAULT_TYPE], Awaitable[None]]) -> None:
    global _FALLBACK_HANDLER
    _FALLBACK_HANDLER = handler


LEGACY_ALIASES: Dict[str, str] = {
    "nav:profile": "hub:open:profile",
    "nav:kbase": "hub:open:kb",
    "nav:photo": "hub:open:photo",
    "nav:music": "hub:open:music",
    "nav:video": "hub:open:video",
    "nav:dialog": "hub:open:dialog",
    "kb_open": "menu:kb",
    "menu:root": "menu:root",
    "banana:add_photo": "banana:add_more",
    "banana:clear": "banana:reset_all",
    "banana:templates": "banana:templates",
    "banana:restart": "banana_regenerate_fresh",
    "banana:back": "banana:back",
    "banana_back_to_card": "banana:back",
    "banana:tpl:bg_remove": "btpl_bg_remove",
    "banana:tpl:bg_studio": "btpl_bg_studio",
    "banana:tpl:outfit_black": "btpl_outfit_black",
    "banana:tpl:makeup_soft": "btpl_makeup_soft",
    "banana:tpl:desk_clean": "btpl_desk_clean",
    "music:mode_vocal": "music:vocal",
    "music:start_generation": "music:start",
    "dialog_default": "dialog:plain",
    "dialog:menu": "hub:open:dialog",
    "menu_main": "menu:root",
    "PROFILE_TRANSACTIONS": "profile:history",
    "PROFILE_PROMO": "profile:promo",
    "PROFILE_INVITE": "profile:invite",
    "PROFILE_BACK": "profile:back",
}

_TEXT_ACTION_FALLBACKS: Dict[str, str] = {
    "фото": "режим фото",
    "музыка": "режим музыки",
    "видео": "режим видео",
    "диалог": "диалог с ии",
}


def _parse_callback(data: str) -> Optional[tuple[str, str]]:
    payload = data.strip()
    if not payload:
        return None
    mapped = LEGACY_ALIASES.get(payload, payload)
    if ":" not in mapped:
        return None
    namespace, action = mapped.split(":", 1)
    return namespace.lower(), action.lower()


def resolve_text_action(text: str) -> Optional[tuple[str, str]]:
    normalized = normalize_btn_text(text)
    if not normalized:
        return None
    payload = TEXT_TO_ACTION.get(normalized)
    if not payload:
        alias_key = _TEXT_ACTION_FALLBACKS.get(normalized)
        if alias_key:
            payload = TEXT_TO_ACTION.get(alias_key)
    if not payload:
        return None
    return _parse_callback(payload)


def _log_dispatch(namespace: str, action: str, source: str) -> None:
    try:
        payload = json.dumps({"ns": namespace, "action": action, "source": source}, ensure_ascii=False)
    except Exception:
        payload = f'{{"ns": "{namespace}", "action": "{action}", "source": "{source}"}}'
    log.info("router.dispatch %s", payload)


def _log_result(ctx_obj: CallbackContext, *, source: str) -> None:
    log.debug(
        "router.result | ns=%s action=%s module=%s source=%s changed=%s mid=%s",
        ctx_obj.namespace,
        ctx_obj.action,
        ctx_obj.module,
        source,
        ctx_obj.card_changed,
        ctx_obj.card_message_id,
    )


async def _dispatch_route(
    namespace: str,
    action: str,
    *,
    update: Update,
    ctx: ContextTypes.DEFAULT_TYPE,
    source: str,
    query=None,
    apply_text_lock: bool = False,
) -> bool:
    message = getattr(query, "message", None) if query is not None else None
    if message is None:
        message = getattr(update, "effective_message", None)
    chat = getattr(message, "chat", None) if message is not None else None
    if chat is None:
        chat = getattr(update, "effective_chat", None)
    chat_id = getattr(chat, "id", None)
    if chat_id is None and message is not None:
        chat_id = getattr(message, "chat_id", None)
    if chat_id is None:
        if query is not None:
            await _safe_answer(query)
        return False

    lock_acquired = False
    chat_data_obj = getattr(ctx, "chat_data", None)
    if (
        isinstance(chat_data_obj, MutableMapping)
        and (namespace, action) in {("menu", "click"), ("kb", "open")}
    ):
        chat_data_obj["nav_event"] = True
    try:
        if apply_text_lock:
            if not user_lock(chat_id, "reply-nav", ttl=1):
                log.debug(
                    "router.debounce | chat=%s ns=%s action=%s source=%s",
                    chat_id,
                    namespace,
                    action,
                    source,
                )
                return True
            lock_acquired = True

        _log_dispatch(namespace, action, source)

        routes_for_ns = _ROUTES.get(namespace)
        route = routes_for_ns.get(action) if routes_for_ns else None
        module_name = route.module if route else namespace
        card_info = await state.get_card(chat_id, module_name)
        now = time.time()

        if not route:
            if query is not None:
                if card_info.updated_at and (now - card_info.updated_at) < 0.5:
                    await _safe_answer(query)
                    return True
                if _FALLBACK_HANDLER is not None:
                    await _FALLBACK_HANDLER(update, ctx)
                await _safe_answer(query)
                return True
            return False
        user = None
        if query is not None:
            user = getattr(query, "from_user", None)
        if user is None:
            user = getattr(update, "effective_user", None)
        if user is None and message is not None:
            user = getattr(message, "from_user", None)
        user_id = getattr(user, "id", None)

        if card_info.updated_at and (now - card_info.updated_at) < 0.5:
            if query is not None:
                await _safe_answer(query)
            return True

        if query is not None:
            await _safe_answer(query)

        t0 = time.perf_counter()

        try:
            async with state.lock(chat_id):
                session = await state.load(chat_id)
                t_state_loaded = time.perf_counter()
                ctx_obj = CallbackContext(
                    update=update,
                    app_context=ctx,
                    namespace=namespace,
                    action=action,
                    chat_id=chat_id,
                    user_id=user_id,
                    query=query,
                    session=session,
                    module=route.module,
                    card=card_info,
                )
                await route.handler(ctx_obj)
                await state.save(chat_id, session)
        except StateLockTimeout:
            log.warning(
                "router.lock_timeout | chat=%s ns=%s action=%s source=%s",
                chat_id,
                namespace,
                action,
                source,
            )
            return True
        except Exception:
            log.exception(
                "router.handler_failed | chat=%s ns=%s action=%s source=%s",
                chat_id,
                namespace,
                action,
                source,
            )
            return True

        await ctx_obj.execute_scheduled()
        _log_result(ctx_obj, source=source)

        t_done = time.perf_counter()
        t_load_ms = (t_state_loaded - t0) * 1000
        t_edit_ms = (t_done - t_state_loaded) * 1000
        log.info(
            "router.done | chat=%s ns=%s action=%s source=%s timing_ms=%.1f/%.1f/%.1f",
            chat_id,
            namespace,
            action,
            source,
            t_load_ms,
            t_edit_ms,
            (t_done - t0) * 1000,
        )
        return True
    finally:
        if apply_text_lock and lock_acquired:
            release_user_lock(chat_id, "reply-nav")


async def hub_router(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Route callback queries to registered handlers."""

    query = update.callback_query
    if not query:
        return

    data_raw = (query.data or "").strip()
    if not data_raw:
        await _safe_answer(query)
        return

    user = getattr(query, "from_user", None)
    user_id = getattr(user, "id", None)
    log.info("[CALLBACK] %s from %s", data_raw, user_id if user_id is not None else "unknown")

    chat_obj = getattr(query, "message", None)
    if chat_obj is not None:
        chat_obj = getattr(chat_obj, "chat", None)
    if chat_obj is None:
        chat_obj = getattr(update, "effective_chat", None)
    chat_id = getattr(chat_obj, "id", None)
    log.info("{\"cb\": %s, \"chat\": %s}", json.dumps(data_raw), "null" if chat_id is None else chat_id)

    parsed = _parse_callback(data_raw)
    if not parsed:
        if _FALLBACK_HANDLER is not None:
            await _FALLBACK_HANDLER(update, ctx)
        await _safe_answer(query)
        return

    namespace, action = parsed
    await _dispatch_route(
        namespace,
        action,
        update=update,
        ctx=ctx,
        source="callback",
        query=query,
    )


async def route_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> bool:
    message = getattr(update, "effective_message", None)
    if message is None:
        return False
    text = getattr(message, "text", None)
    if not isinstance(text, str):
        return False
    parsed = resolve_text_action(text)
    if not parsed:
        return False
    namespace, action = parsed
    chat_data_obj = getattr(ctx, "chat_data", None)
    if isinstance(chat_data_obj, MutableMapping) and chat_data_obj.get("nav_event"):
        return False

    nav_flag = False
    previous_nav_attr = getattr(ctx, "nav_event", False)
    if isinstance(chat_data_obj, MutableMapping):
        chat_data_obj["nav_event"] = True
        nav_flag = True
    if nav_flag:
        setattr(ctx, "nav_event", True)

    try:
        handled = await _dispatch_route(
            namespace,
            action,
            update=update,
            ctx=ctx,
            source="text",
            apply_text_lock=True,
        )
    finally:
        if nav_flag and isinstance(chat_data_obj, MutableMapping):
            chat_data_obj.pop("nav_event", None)
        if nav_flag:
            setattr(ctx, "nav_event", previous_nav_attr)
    return handled


async def _safe_answer(query) -> None:
    try:
        await query.answer()
    except Exception:
        pass


__all__ = [
    "register",
    "hub_router",
    "route_text",
    "resolve_text_action",
    "CallbackContext",
    "LEGACY_ALIASES",
    "set_fallback",
]
