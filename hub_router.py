"""Callback router with namespace-aware actions.

Each callback is expected to be formatted as ``"<namespace>:<action>"``.
Handlers register themselves through :func:`register` and receive a
``CallbackContext`` object that exposes the parsed payload together with
helpers for scheduling UI updates.

Backward compatibility is maintained via :data:`LEGACY_ALIASES`, allowing
older callback payloads to be transparently translated into the new schema.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional

from telegram import InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import ContextTypes

from state import CardInfo, StateLockTimeout, state

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
    "nav:profile": "menu:profile",
    "nav:kbase": "menu:kb",
    "nav:photo": "menu:photo",
    "nav:music": "menu:music",
    "nav:video": "menu:video",
    "nav:dialog": "menu:dialog",
    "menu:kb": "home:kb",
    "menu:photo": "home:photo",
    "menu:music": "home:music",
    "menu:video": "home:video",
    "menu:dialog": "home:dialog",
    "menu:root": "menu_main",
    "banana:add_photo": "banana:add_more",
    "banana:clear": "banana:reset_all",
    "banana:templates": "banana_templates",
    "banana:restart": "banana_regenerate_fresh",
    "banana:back": "banana_back_to_card",
    "banana:tpl:bg_remove": "btpl_bg_remove",
    "banana:tpl:bg_studio": "btpl_bg_studio",
    "banana:tpl:outfit_black": "btpl_outfit_black",
    "banana:tpl:makeup_soft": "btpl_makeup_soft",
    "banana:tpl:desk_clean": "btpl_desk_clean",
    "music:mode_vocal": "music:vocal",
    "music:start_generation": "music:start",
    "dialog_default": "dialog:plain",
    "dialog:menu": "menu:dialog",
    "menu_main": "menu:root",
    "profile:transactions": "tx:open",
    "profile:promo": "promo_open",
    "profile:invite": "ref:open",
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


async def hub_router(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Route callback queries to registered handlers."""

    query = update.callback_query
    if not query:
        return

    data_raw = (query.data or "").strip()
    if not data_raw:
        await _safe_answer(query)
        return

    parsed = _parse_callback(data_raw)
    if not parsed:
        if _FALLBACK_HANDLER is not None:
            await _FALLBACK_HANDLER(update, ctx)
        await _safe_answer(query)
        return

    namespace, action = parsed
    message = getattr(query, "message", None)
    chat_id = getattr(getattr(message, "chat", None), "id", None) or getattr(
        getattr(update, "effective_chat", None), "id", None
    )
    if chat_id is None:
        await _safe_answer(query)
        return

    routes_for_ns = _ROUTES.get(namespace)
    route = routes_for_ns.get(action) if routes_for_ns else None
    module_name = route.module if route else namespace
    card_info = await state.get_card(chat_id, module_name)
    now = time.time()

    if not route:
        if card_info.updated_at and (now - card_info.updated_at) < 0.5:
            await _safe_answer(query)
            return
        if _FALLBACK_HANDLER is not None:
            await _FALLBACK_HANDLER(update, ctx)
        await _safe_answer(query)
        return

    user = getattr(query, "from_user", None) or getattr(update, "effective_user", None)
    user_id = getattr(user, "id", None)

    if card_info.updated_at and (now - card_info.updated_at) < 0.5:
        await _safe_answer(query)
        return

    t0 = time.perf_counter()
    await _safe_answer(query)

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
            "router.lock_timeout | chat=%s ns=%s action=%s", chat_id, namespace, action
        )
        return
    except Exception:
        log.exception(
            "router.handler_failed | chat=%s ns=%s action=%s", chat_id, namespace, action
        )
        return

    await ctx_obj.execute_scheduled()
    t_done = time.perf_counter()
    t_load_ms = (t_state_loaded - t0) * 1000
    t_edit_ms = (t_done - t_state_loaded) * 1000
    log.info(
        "router.done | chat=%s ns=%s action=%s timing_ms=%.1f/%.1f/%.1f",
        chat_id,
        namespace,
        action,
        t_load_ms,
        t_edit_ms,
        (t_done - t0) * 1000,
    )


async def _safe_answer(query) -> None:
    try:
        await query.answer()
    except Exception:
        pass


__all__ = ["register", "hub_router", "CallbackContext", "LEGACY_ALIASES", "set_fallback"]
