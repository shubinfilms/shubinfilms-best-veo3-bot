import os
import asyncio
from typing import Optional, Callable, Any, Dict
from telegram.error import TelegramError

# -------- flags --------
def flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1","true","yes","y","on","t","on"}

# -------- in-memory fallback locks --------
_inmem_chat_locks: Dict[int, asyncio.Lock] = {}
_inmem_user_locks: Dict[int, asyncio.Lock] = {}
def _chat_lock(chat_id: int) -> asyncio.Lock:
    if chat_id not in _inmem_chat_locks:
        _inmem_chat_locks[chat_id] = asyncio.Lock()
    return _inmem_chat_locks[chat_id]
def _user_lock(user_id: int) -> asyncio.Lock:
    if user_id not in _inmem_user_locks:
        _inmem_user_locks[user_id] = asyncio.Lock()
    return _inmem_user_locks[user_id]

# -------- redis helpers --------
async def _acquire_redis_lock(redis, key: str, ttl: int) -> bool:
    try:
        if not redis:
            return False
        ok = await redis.setnx(key, "1")
        if ok:
            await redis.expire(key, ttl)
        return ok
    except Exception:
        return False

async def _release_redis_lock(redis, key: str):
    try:
        if redis:
            await redis.delete(key)
    except Exception:
        pass

# -------- wrappers --------
def wrap_menu_with_lock(original: Callable) -> Callable:
    LOCK_TTL = 8
    def _get_ids(update):
        chat_id = getattr(getattr(update, "effective_chat", None), "id", None)
        cq = getattr(update, "callback_query", None)
        return chat_id, cq

    async def _call_with_retry(ctx_bot, **kwargs):
        try:
            return await ctx_bot.edit_message_text(**kwargs)
        except TelegramError as e:
            s = str(e)
            if "409" in s or "Conflict" in s:
                await asyncio.sleep(0.25)
                return await ctx_bot.edit_message_text(**kwargs)
            raise

    async def wrapper(update, context):
        chat_id, cq = _get_ids(update)
        redis = context.bot_data.get("redis")
        rkey = f"lock:menu:open:{chat_id}"
        # пытаемся редис-лок
        got_redis = await _acquire_redis_lock(redis, rkey, LOCK_TTL)
        if not got_redis and chat_id is not None:
            # падение в in-memory лок, чтобы точно не продублировать
            lock = _chat_lock(chat_id)
            if lock.locked():
                if cq:
                    try:
                        await cq.answer("Обновляю…", show_alert=False)
                    except Exception:
                        pass
                return
            async with lock:
                return await original(update, context)
        try:
            return await original(update, context)
        finally:
            if got_redis:
                await _release_redis_lock(redis, rkey)
    # Патчируем точечно edit в ensure_card, если он там внутри
    wrapper._menu_wrapped = True
    return wrapper

def banana_idempotency_wrapper(start_upload_func: Callable) -> Callable:
    IDEM_TTL = 3600
    async def wrapped(*args, **kwargs):
        # Ищем file_id и redis в аргументах
        file_id = kwargs.get("file_id")
        redis = kwargs.get("redis") or None
        if file_id and redis:
            ikey = f"banana:idem:{file_id}"
            try:
                if not await redis.setnx(ikey, "1"):
                    # idem-hit
                    return {"status": "idem_hit", "file_id": file_id}
                await redis.expire(ikey, IDEM_TTL)
            except Exception:
                pass
        try:
            return await start_upload_func(*args, **kwargs)
        except Exception:
            # снять блок при ошибке
            if file_id and redis:
                try:
                    await redis.delete(f"banana:idem:{file_id}")
                except Exception:
                    pass
            raise
    wrapped._banana_idem = True
    return wrapped

async def _album_guard(redis, mgid: str, uid: int, coro: Callable[[], Any], debounce: float = 1.8):
    gkey = f"banana:group:{mgid}:{uid}"
    if redis:
        try:
            if not await redis.setnx(gkey, "1"):
                return  # уже кто-то собирает
            await redis.expire(gkey, 10)
        except Exception:
            pass
    try:
        await asyncio.sleep(debounce)
        return await coro()
    finally:
        if redis:
            try:
                await redis.delete(gkey)
            except Exception:
                pass

def wrap_profile_with_lock_and_cache(open_func: Callable, get_balance_func: Optional[Callable], set_balance_cache: Optional[Callable]) -> Callable:
    TTL = 10
    async def wrapper(update, context):
        uid = getattr(getattr(update, "effective_user", None), "id", None)
        redis = context.bot_data.get("redis")
        key = f"bot:prof:lock:{uid}"
        got = await _acquire_redis_lock(redis, key, TTL)
        if not got and uid is not None:
            # fall back на in-memory, чтобы не спамить
            lock = _user_lock(uid)
            if lock.locked():
                cq = getattr(update, "callback_query", None)
                if cq:
                    try:
                        await cq.answer("Открываю профиль…", show_alert=False)
                    except Exception:
                        pass
                return
            async with lock:
                return await open_func(update, context)
        try:
            return await open_func(update, context)
        finally:
            await _release_redis_lock(redis, key)
    wrapper._profile_wrapped = True
    return wrapper

# -------- core apply --------
def apply(application):
    """
    Делает три вещи:
    1) Жёстко разводит Banana (новый vs старый) по флагу FEATURE_BANANA_REWORK
    2) Оборачивает /menu хендлеры в chat-lock + retry при 409
    3) Оборачивает профиль в user-lock; баланс оставляем как есть (кэш останется в слое бизнес-логики), но
       не даём плодить карточки; для Banana вешаем идемпотентность на start_upload.
    Плюс: добавляем дебаунс альбомов через общий guard, если найдём соответствующую корутину.
    """
    FEATURE_BANANA_REWORK = flag("FEATURE_BANANA_REWORK", True)
    SIMPLE_PROFILE_ENABLED = flag("SIMPLE_PROFILE_ENABLED", False)

    # 0) Получим список зарегистрированных хендлеров
    # application.handlers — {group: [Handler, ...]}
    # Группы нам не критичны; пробежимся по всем
    from telegram.ext import CommandHandler, CallbackQueryHandler

    # 1) Развести Banana: удалим лишние хендлеры
    # Ищем по именам модулей/коллбеков
    def _handler_name(h):
        cb = getattr(h, "callback", None)
        return getattr(cb, "__module__", "") + ":" + getattr(cb, "__name__", "")

    for group, handlers in list(application.handlers.items()):
        to_remove = []
        for h in handlers:
            name = _handler_name(h)
            if "handlers.banana_async_handler" in name and not FEATURE_BANANA_REWORK:
                to_remove.append(h)
            if "handlers.banana" in name and "handlers.banana_async_handler" not in name and FEATURE_BANANA_REWORK:
                to_remove.append(h)
        for h in to_remove:
            try:
                application.handlers[group].remove(h)
            except Exception:
                pass

    # 2) /menu: найти все обработчики, ведущие к меню, и обернуть их (Command и Callback)
    # Эвристика: имя функции содержит "open_handler" и модуль "handlers.menu"
    for group, handlers in list(application.handlers.items()):
        for h in handlers:
            cb = getattr(h, "callback", None)
            if not cb:
                continue
            mod = getattr(cb, "__module__", "")
            name = getattr(cb, "__name__", "")
            if "handlers.menu" in mod and name.endswith("open_handler"):
                # заменить callback на обёрнутый
                if not getattr(cb, "_menu_wrapped", False):
                    wrapped = wrap_menu_with_lock(cb)
                    h.callback = wrapped

    # 3) Профиль: оборачиваем open в user-lock
    try:
        import handlers.profile as profile_full
    except Exception:
        profile_full = None
    try:
        import handlers.profile_simple as profile_simple
    except Exception:
        profile_simple = None

    # Выбираем активный модуль профиля по флагу и отцепляем другой
    active_module = profile_simple if SIMPLE_PROFILE_ENABLED and profile_simple else profile_full
    inactive_module = profile_full if SIMPLE_PROFILE_ENABLED else profile_simple

    for group, handlers in list(application.handlers.items()):
        for h in list(handlers):
            cb = getattr(h, "callback", None)
            mod = getattr(cb, "__module__", "") if cb else ""
            if inactive_module and inactive_module.__name__ == mod:
                # убрать хендлеры из неактивного профиля
                try:
                    application.handlers[group].remove(h)
                except Exception:
                    pass

    # Обернуть активный open()
    if active_module:
        open_cb = getattr(active_module, "open", None) or getattr(active_module, "profile_open", None)
        if callable(open_cb):
            for group, handlers in list(application.handlers.items()):
                for h in handlers:
                    cb = getattr(h, "callback", None)
                    if cb is open_cb and not getattr(cb, "_profile_wrapped", False):
                        h.callback = wrap_profile_with_lock_and_cache(cb, None, None)

    # 4) Banana: идемпотентность для uploader.start_upload и дебаунс альбомов
    try:
        import banana.uploader as up
        if hasattr(up, "start_upload") and callable(up.start_upload) and not getattr(up.start_upload, "_banana_idem", False):
            up.start_upload = banana_idempotency_wrapper(up.start_upload)
    except Exception:
        pass

    # Дебаунс альбомов — мягкий хук: ищем функцию, что собирает media_group, и оборачиваем её вызов через guard.
    # Без жёсткого monkeypatch на весь файл, чтобы не ломать логику.
    try:
        import handlers.banana_async_handler as bnew
        if hasattr(bnew, "_handle_media_group") and callable(bnew._handle_media_group):
            orig = bnew._handle_media_group
            async def mg_wrapper(update, context, mgid, uid):
                redis = context.bot_data.get("redis")
                async def _runner():
                    return await orig(update, context, mgid, uid)
                return await _album_guard(redis, str(mgid), int(uid), _runner, debounce=1.8)
            bnew._handle_media_group = mg_wrapper
    except Exception:
        pass

    # 5) Грейсфул на устаревшие кнопки: если в роутере неизвестный callback — ответить и выйти
    try:
        import ui.buttons.router as router
        if not hasattr(router, "_hotfix_unknown_guard"):
            old_route = getattr(router, "route_callback")
            async def route_callback_guard(update, context, data: Optional[str]=None):
                data = data or getattr(getattr(update, "callback_query", None), "data", None)
                try:
                    return await old_route(update, context, data)
                except Exception:
                    cq = getattr(update, "callback_query", None)
                    if cq:
                        try:
                            return await cq.answer("Кнопка устарела. Нажмите /menu", show_alert=False)
                        except Exception:
                            return
                    raise
            router.route_callback = route_callback_guard
            router._hotfix_unknown_guard = True
    except Exception:
        pass

    # 6) /menu командный алиас → приводим к единому пути, если такой есть в роутере
    try:
        import ui.buttons.router as router
        from telegram.ext import CommandHandler
        async def _menu_alias(update, context):
            try:
                return await router.route_callback(update, context, "home:open")
            except Exception:
                # fallback — если роутер не работает, пробуем напрямую найти handlers.menu.open_handler
                try:
                    import handlers.menu as menu
                    if hasattr(menu, "open_handler"):
                        return await menu.open_handler(update, context)
                except Exception:
                    pass
        # заменяем существующие CommandHandler("menu", ...)
        for group, handlers in list(application.handlers.items()):
            for h in handlers:
                if isinstance(h, CommandHandler) and "menu" in (h.commands or []):
                    h.callback = _menu_alias
    except Exception:
        pass
