import asyncio
import inspect
import json
import logging
import os
import time
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # pragma: no cover - optional import for type checking only
    from typing import TYPE_CHECKING
except ImportError:  # pragma: no cover - Python < 3.8 safeguard
    TYPE_CHECKING = False

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from telegram import User as TelegramUser
else:  # pragma: no cover - fallback alias to avoid runtime dependency
    TelegramUser = Any

import redis

from settings import REDIS_PREFIX

_logger = logging.getLogger("redis-utils")

_redis_url = os.getenv("REDIS_URL")
_r = redis.from_url(_redis_url) if _redis_url else None
rds = _r
_PFX = REDIS_PREFIX
_TTL = 24 * 60 * 60

_USERS_SET_KEY = f"{_PFX}:users"
_DEAD_USERS_SET_KEY = f"{_PFX}:users:dead"
_PROMO_USED_SET_KEY = f"{_PFX}:promo:used"

_REF_INVITER_KEY_TMPL = f"{_PFX}:ref:inviter_of:{{}}"
_REF_USERS_KEY_TMPL = f"{_PFX}:ref:users_of:{{}}"
_REF_EARNED_KEY_TMPL = f"{_PFX}:ref:earned:{{}}"
_REF_JOINED_AT_TMPL = f"{_PFX}:ref:joined_at:{{}}"


def _user_profile_key(user_id: int) -> str:
    return f"{_PFX}:user:{user_id}"


def _promo_member(user_id: int, code: str) -> str:
    return f"{int(user_id)}:{code.strip().upper()}"


def _ref_inviter_key(user_id: int) -> str:
    return _REF_INVITER_KEY_TMPL.format(int(user_id))


def _ref_users_key(inviter_id: int) -> str:
    return _REF_USERS_KEY_TMPL.format(int(inviter_id))


def _ref_earned_key(inviter_id: int) -> str:
    return _REF_EARNED_KEY_TMPL.format(int(inviter_id))


def _ref_joined_at_key(user_id: int) -> str:
    return _REF_JOINED_AT_TMPL.format(int(user_id))

_memory_store: Dict[str, Tuple[float, str]] = {}
_memory_lock = Lock()

_ref_lock = Lock()
_ref_inviter_memory: Dict[int, int] = {}
_ref_users_memory: Dict[int, set[int]] = {}
_ref_earned_memory: Dict[int, int] = {}
_ref_joined_memory: Dict[int, float] = {}
_user_profile_lock = Lock()
_user_profile_memory: Dict[int, Dict[str, str]] = {}

_wait_lock = Lock()
_wait_memory: Dict[str, float] = {}
_WAIT_TTL_DEFAULT = 15 * 60


async def _gather_keys(r: Any, pattern: str) -> list[str]:
    """Collect keys from ``r`` matching ``pattern`` using SCAN semantics."""

    if not pattern:
        return []

    keys: list[str] = []

    try:
        iterator = r.scan_iter(pattern)
    except AttributeError:
        iterator = None

    if iterator is not None:
        try:
            if inspect.isasyncgen(iterator):
                async for key in iterator:
                    if isinstance(key, bytes):
                        key = key.decode()
                    keys.append(str(key))
            else:
                for key in iterator:
                    if isinstance(key, bytes):
                        key = key.decode()
                    keys.append(str(key))
            return keys
        except Exception:
            log = logging.getLogger("redis-utils")
            log.warning("scan_iter_failed | pattern=%s", pattern, exc_info=True)
            return []

    try:
        if hasattr(r, "scan"):
            cursor: int | str = 0
            while True:
                if inspect.iscoroutinefunction(r.scan):
                    cursor, batch = await r.scan(cursor=cursor, match=pattern, count=64)
                else:
                    cursor, batch = await asyncio.to_thread(
                        r.scan, cursor=cursor, match=pattern, count=64
                    )
                for key in batch:
                    if isinstance(key, bytes):
                        key = key.decode()
                    keys.append(str(key))
                if not cursor:
                    break
    except Exception:
        log = logging.getLogger("redis-utils")
        log.warning("scan_failed | pattern=%s", pattern, exc_info=True)
        return []

    return keys


async def _delete_keys(r: Any, keys: Iterable[str]) -> int:
    payload = [str(key) for key in keys]
    if not payload:
        return 0

    try:
        if inspect.iscoroutinefunction(r.delete):
            deleted = await r.delete(*payload) or 0
        else:
            deleted = await asyncio.to_thread(r.delete, *payload) or 0
    except Exception:
        log = logging.getLogger("redis-utils")
        log.warning("delete_failed | keys=%s", len(payload), exc_info=True)
        return 0

    return int(deleted)


async def redis_delete_keys(r: Any, patterns: Sequence[str]) -> int:
    """Delete keys in ``r`` matching any of ``patterns`` and return total removed."""

    if not r:
        return 0

    total_deleted = 0
    for pattern in patterns:
        pattern = str(pattern or "").strip()
        if not pattern:
            continue
        keys = await _gather_keys(r, pattern)
        if not keys:
            continue
        total_deleted += await _delete_keys(r, keys)

    return total_deleted


async def clear_wait_flags(r: Any, user_id: int, prefix: str) -> int:
    """Remove all ``wait`` keys for ``user_id`` under ``prefix``.

    Works with both async and sync Redis clients. Returns number of deleted keys.
    """

    if not r:
        return 0

    try:
        uid = int(user_id)
    except (TypeError, ValueError):
        return 0

    if uid <= 0:
        return 0

    pattern = f"{prefix}:wait:{uid}:*"
    return await redis_delete_keys(r, [pattern])


async def clear_all_waits(r: Any, user_id: int, prefix: Optional[str] = None) -> int:
    """Remove all ``wait`` and ``fsm`` flags for ``user_id``.

    Parameters
    ----------
    r:
        Redis client (sync or async) used for deleting keys.
    user_id:
        Telegram user identifier. Non-positive and invalid values are ignored.
    prefix:
        Optional Redis key prefix. Defaults to :data:`settings.REDIS_PREFIX`.

    Returns
    -------
    int
        Number of keys removed from Redis.
    """

    if not r:
        return 0

    try:
        uid = int(user_id)
    except (TypeError, ValueError):
        return 0

    if uid <= 0:
        return 0

    effective_prefix = str(prefix or _PFX).strip() or _PFX
    patterns = [
        f"{effective_prefix}:wait:{uid}:*",
        f"{effective_prefix}:fsm:{uid}:*",
    ]

    deleted = await redis_delete_keys(r, patterns)
    _logger.debug(
        "cleared %s wait/fsm keys (user=%s prefix=%s)",
        deleted,
        uid,
        effective_prefix,
    )
    return deleted


async def set_wait_flag(
    r: Any,
    user_id: int,
    name: str,
    prefix: str,
    *,
    ttl_sec: int = 900,
) -> Optional[str]:
    """Set a wait flag for ``user_id`` with a bounded TTL."""

    if not r:
        return None

    try:
        uid = int(user_id)
    except (TypeError, ValueError):
        return None

    if uid <= 0:
        return None

    normalized = str(name or "").strip() or "default"
    key = f"{prefix}:wait:{uid}:{normalized}"
    payload_ttl = max(int(ttl_sec or 0), 1)

    try:
        if inspect.iscoroutinefunction(r.set):
            await r.set(key, "1", ex=payload_ttl)
        else:
            await asyncio.to_thread(r.set, key, "1", ex=payload_ttl)
    except Exception:
        log = logging.getLogger("redis-utils")
        log.warning("set_wait_flag.failed", exc_info=True)
        return None

    logging.getLogger("bot").debug("wait.set | key=%s ttl=%s", key, payload_ttl)
    return key

if not _redis_url:
    _logger.warning(
        "REDIS_URL is not configured; falling back to in-memory task-meta store"
    )


def task_key(task_id: str) -> str:
    return f"{_PFX}:task:{task_id}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _memory_set(key: str, value: str) -> None:
    expires_at = time.time() + _TTL
    with _memory_lock:
        _memory_store[key] = (expires_at, value)


def _memory_get(key: str) -> Optional[str]:
    now = time.time()
    with _memory_lock:
        entry = _memory_store.get(key)
        if not entry:
            return None
        expires_at, value = entry
        if expires_at <= now:
            _memory_store.pop(key, None)
            return None
        return value


def _memory_delete(key: str) -> None:
    with _memory_lock:
        _memory_store.pop(key, None)


def wait_key(user_id: int) -> str:
    """Return the base Redis key for wait flags for ``user_id``."""

    return f"{_PFX}:wait:{int(user_id)}"


def _wait_entry_key(user_id: int, kind: str) -> str:
    normalized = str(kind or "").strip() or "default"
    return f"{wait_key(user_id)}:{normalized}"


def wait_set(user_id: int, kind: str, *, ttl: int = _WAIT_TTL_DEFAULT) -> None:
    """Store a wait flag for ``user_id`` and ``kind``."""

    key = _wait_entry_key(user_id, kind)
    ttl = max(int(ttl or _WAIT_TTL_DEFAULT), 1)
    if _r:
        try:
            _r.setex(key, ttl, "1")
            return
        except Exception as exc:  # pragma: no cover - defensive logging
            _logger.warning("wait_set.redis_failed | user=%s kind=%s err=%s", user_id, kind, exc)
    expires_at = time.time() + ttl
    with _wait_lock:
        _wait_memory[key] = expires_at


def wait_clear(user_id: int) -> int:
    """Remove all wait flags for ``user_id``. Returns the number of deleted keys."""

    base = wait_key(user_id)
    pattern = f"{base}:*"
    removed = 0
    if _r:
        try:
            cursor = 0
            keys: List[str] = []
            while True:
                cursor, batch = _r.scan(cursor=cursor, match=pattern, count=64)
                if batch:
                    keys.extend(batch)
                if cursor == 0:
                    break
            if keys:
                removed = _r.delete(*keys) or 0
            return int(removed)
        except Exception as exc:  # pragma: no cover - defensive logging
            _logger.warning("wait_clear.redis_failed | user=%s err=%s", user_id, exc)
    with _wait_lock:
        expired_keys = [key for key in _wait_memory if key.startswith(pattern[:-1])]
        for key in expired_keys:
            _wait_memory.pop(key, None)
        removed = len(expired_keys)
    with _memory_lock:
        stale = [key for key in _memory_store if key.startswith(pattern[:-1])]
        for key in stale:
            _memory_store.pop(key, None)
        removed += len(stale)
    return removed


def clear_wait(user_id: int) -> int:
    """Remove all wait-related keys for ``user_id`` across legacy patterns."""

    try:
        uid = int(user_id)
    except (TypeError, ValueError):
        return 0

    if uid <= 0:
        return 0

    removed = wait_clear(uid)

    pattern = f"{_PFX}:wait:*:{uid}"
    if _r:
        try:
            cursor = 0
            keys: List[str] = []
            while True:
                cursor, batch = _r.scan(cursor=cursor, match=pattern, count=64)
                if batch:
                    keys.extend(batch)
                if cursor == 0:
                    break
            if keys:
                removed += _r.delete(*keys) or 0
        except Exception as exc:  # pragma: no cover - defensive logging
            _logger.warning("clear_wait.redis_failed | user=%s err=%s", uid, exc)

    suffix = f":{uid}"
    prefix = f"{_PFX}:wait:"

    with _wait_lock:
        for key in list(_wait_memory.keys()):
            if not key.startswith(prefix):
                continue
            if key.endswith(suffix):
                _wait_memory.pop(key, None)
                removed += 1

    with _memory_lock:
        for key in list(_memory_store.keys()):
            if not key.startswith(prefix):
                continue
            if key.endswith(suffix):
                _memory_store.pop(key, None)
                removed += 1

    return removed


def save_task_meta(
    task_id: str,
    chat_id: int,
    message_id: int,
    mode: str,
    aspect: str,
) -> None:
    doc: Dict[str, Any] = {
        "chat_id": int(chat_id),
        "message_id": int(message_id),
        "mode": mode,
        "aspect": aspect,
        "created_at": _now_iso(),
    }
    payload = json.dumps(doc, ensure_ascii=False)
    key = task_key(task_id)
    if _r:
        _r.setex(key, _TTL, payload)
    else:
        _memory_set(key, payload)


def load_task_meta(task_id: str) -> Optional[Dict[str, Any]]:
    key = task_key(task_id)
    raw: Optional[str]
    if _r:
        redis_raw = _r.get(key)
        raw = None if redis_raw is None else redis_raw.decode("utf-8") if isinstance(redis_raw, bytes) else str(redis_raw)
    else:
        raw = _memory_get(key)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        _logger.exception("Failed to decode task-meta for %s", task_id)
        return None


def clear_task_meta(task_id: str) -> None:
    key = task_key(task_id)
    if _r:
        _r.delete(key)
    else:
        _memory_delete(key)


def is_promo_used(user_id: int, code: str) -> bool:
    if not _r:
        raise RuntimeError("Redis client is not configured")
    member = _promo_member(user_id, code)
    return bool(_r.sismember(_PROMO_USED_SET_KEY, member))


def mark_promo_used(user_id: int, code: str) -> bool:
    if not _r:
        raise RuntimeError("Redis client is not configured")
    member = _promo_member(user_id, code)
    return bool(_r.sadd(_PROMO_USED_SET_KEY, member))


def unmark_promo_used(user_id: int, code: str) -> bool:
    if not _r:
        return False
    member = _promo_member(user_id, code)
    return bool(_r.srem(_PROMO_USED_SET_KEY, member))


def get_inviter(user_id: int) -> Optional[int]:
    if _r:
        raw = _r.get(_ref_inviter_key(user_id))
        if raw is None:
            return None
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None
    with _ref_lock:
        value = _ref_inviter_memory.get(int(user_id))
    return int(value) if value is not None else None


def set_inviter(user_id: int, inviter_id: int) -> bool:
    if int(user_id) == int(inviter_id):
        return False
    if _r:
        key = _ref_inviter_key(user_id)
        try:
            created = bool(_r.setnx(key, int(inviter_id)))
        except Exception:
            created = False
        if created:
            try:
                _r.set(_ref_joined_at_key(user_id), int(time.time()))
            except Exception:
                pass
        return created
    with _ref_lock:
        uid = int(user_id)
        if uid in _ref_inviter_memory:
            return False
        _ref_inviter_memory[uid] = int(inviter_id)
        _ref_joined_memory[uid] = time.time()
    return True


def add_ref_user(inviter_id: int, user_id: int) -> bool:
    if _r:
        try:
            return bool(_r.sadd(_ref_users_key(inviter_id), int(user_id)))
        except Exception:
            return False
    with _ref_lock:
        inv = int(inviter_id)
        users = _ref_users_memory.setdefault(inv, set())
        before = len(users)
        users.add(int(user_id))
        return len(users) != before


def incr_ref_earned(inviter_id: int, amount: int) -> int:
    amount = int(amount)
    if amount < 0:
        raise ValueError("amount must be non-negative")
    if _r:
        try:
            return int(_r.incrby(_ref_earned_key(inviter_id), amount))
        except Exception:
            return 0
    with _ref_lock:
        inv = int(inviter_id)
        total = _ref_earned_memory.get(inv, 0) + amount
        _ref_earned_memory[inv] = total
        return total


def get_ref_stats(inviter_id: int) -> Tuple[int, int]:
    if _r:
        try:
            count = int(_r.scard(_ref_users_key(inviter_id)))
        except Exception:
            count = 0
        try:
            earned_raw = _r.get(_ref_earned_key(inviter_id))
            earned = int(earned_raw) if earned_raw is not None else 0
        except Exception:
            earned = 0
        return count, earned
    with _ref_lock:
        users = _ref_users_memory.get(int(inviter_id))
        count = len(users) if users is not None else 0
        earned = _ref_earned_memory.get(int(inviter_id), 0)
        return count, int(earned)


async def add_user(redis_conn: Optional["redis.Redis"], user: "TelegramUser") -> bool:
    if user is None:
        return False
    user_id = getattr(user, "id", None)
    if not user_id:
        return False

    if not redis_conn:
        _logger.warning("add_user: Redis client is not configured; skipping user %s", user_id)
        return False

    profile = {
        "id": str(int(user_id)),
        "username": getattr(user, "username", "") or "",
        "first_name": getattr(user, "first_name", "") or "",
        "last_name": getattr(user, "last_name", "") or "",
        "is_premium": "1" if getattr(user, "is_premium", False) else "0",
        "is_bot": "1" if getattr(user, "is_bot", False) else "0",
        "language_code": getattr(user, "language_code", "") or "",
        "last_seen_ts": _now_iso(),
    }

    def _store() -> bool:
        try:
            pipe = redis_conn.pipeline()
            pipe.sadd(_USERS_SET_KEY, int(user_id))
            pipe.hset(_user_profile_key(int(user_id)), mapping=profile)
            pipe.execute()
            return True
        except Exception as exc:  # pragma: no cover - network failure path
            _logger.warning("Failed to add user %s to Redis: %s", user_id, exc)
            return False

    return await asyncio.to_thread(_store)


def set_user_preferred_language(user_id: int, language_code: str) -> bool:
    """Persist the preferred language for a user.

    Returns ``True`` if the value was stored either in Redis or the fallback
    memory store.
    """

    if not user_id:
        return False

    value = (language_code or "").strip().lower()
    if not value:
        return False

    stored = False
    if _r:
        try:
            _r.hset(_user_profile_key(int(user_id)), "preferred_language", value)
            stored = True
        except Exception as exc:  # pragma: no cover - network failure path
            _logger.warning(
                "set_user_preferred_language failed for %s: %s", user_id, exc
            )

    with _user_profile_lock:
        profile = _user_profile_memory.setdefault(int(user_id), {})
        profile["preferred_language"] = value
        stored = True

    return stored


def get_user_preferred_language(user_id: int) -> Optional[str]:
    """Return the preferred language stored for ``user_id`` if present."""

    if not user_id:
        return None

    if _r:
        try:
            raw = _r.hget(_user_profile_key(int(user_id)), "preferred_language")
        except Exception as exc:  # pragma: no cover - network failure path
            _logger.warning(
                "get_user_preferred_language failed for %s: %s", user_id, exc
            )
        else:
            if raw is None:
                pass
            elif isinstance(raw, bytes):
                return raw.decode("utf-8", "ignore") or None
            else:
                return str(raw) or None

    with _user_profile_lock:
        profile = _user_profile_memory.get(int(user_id))
        if profile:
            value = profile.get("preferred_language")
            return value if value else None

    return None


async def get_users_count(redis_conn: Optional["redis.Redis"]) -> Optional[int]:
    if not redis_conn:
        _logger.warning("get_users_count: Redis client is not configured")
        return None

    def _call() -> Optional[int]:
        try:
            return int(redis_conn.scard(_USERS_SET_KEY))
        except Exception as exc:  # pragma: no cover - network failure path
            _logger.warning("Failed to fetch users count: %s", exc)
            return None

    return await asyncio.to_thread(_call)


async def get_all_user_ids(redis_conn: Optional["redis.Redis"]) -> List[int]:
    if not redis_conn:
        _logger.warning("get_all_user_ids: Redis client is not configured")
        return []

    def _call() -> List[int]:
        try:
            raw_ids = redis_conn.smembers(_USERS_SET_KEY)
        except Exception as exc:  # pragma: no cover - network failure path
            _logger.warning("Failed to load users set: %s", exc)
            return []
        result: List[int] = []
        for item in raw_ids:
            try:
                if isinstance(item, bytes):
                    item = item.decode("utf-8")
                result.append(int(item))
            except (TypeError, ValueError):
                continue
        return result

    return await asyncio.to_thread(_call)


async def remove_user(redis_conn: Optional["redis.Redis"], user_id: int) -> None:
    if not redis_conn:
        _logger.warning("remove_user: Redis client is not configured; user_id=%s", user_id)
        return

    def _call() -> None:
        try:
            pipe = redis_conn.pipeline()
            pipe.srem(_USERS_SET_KEY, int(user_id))
            pipe.delete(_user_profile_key(int(user_id)))
            pipe.execute()
        except Exception as exc:  # pragma: no cover - network failure path
            _logger.warning("Failed to remove user %s from Redis: %s", user_id, exc)

    await asyncio.to_thread(_call)


async def mark_user_dead(redis_conn: Optional["redis.Redis"], user_id: int) -> None:
    if not redis_conn:
        _logger.warning("mark_user_dead: Redis client is not configured; user_id=%s", user_id)
        return

    def _call() -> None:
        try:
            pipe = redis_conn.pipeline()
            pipe.sadd(_DEAD_USERS_SET_KEY, int(user_id))
            pipe.srem(_USERS_SET_KEY, int(user_id))
            pipe.delete(_user_profile_key(int(user_id)))
            pipe.execute()
        except Exception as exc:  # pragma: no cover - network failure path
            _logger.warning("Failed to mark user %s as dead: %s", user_id, exc)

    await asyncio.to_thread(_call)


async def user_exists(redis_conn: Optional["redis.Redis"], user_id: int) -> bool:
    if not redis_conn:
        return False

    def _call() -> bool:
        try:
            return bool(redis_conn.exists(_user_profile_key(int(user_id))))
        except Exception as exc:  # pragma: no cover - network failure path
            _logger.warning("Failed to check user %s existence: %s", user_id, exc)
            return False

    return await asyncio.to_thread(_call)


# === Users & Balance ===
USERS_KEY = f"{_PFX}:users"


def _bal_key(uid: int) -> str:
    return f"{_PFX}:bal:{uid}"


def _ledger_key(uid: int) -> str:
    return f"{_PFX}:ledger:{uid}"


def get_ledger_entries(user_id: int, offset: int = 0, limit: int = 10) -> List[Dict[str, Any]]:
    """Return a slice of ledger entries for ``user_id``.

    Entries are returned as a list of dictionaries ordered from newest to
    oldest. The ``offset`` parameter is applied from the most recent entry and
    ``limit`` controls the number of items to fetch (defaults to 10).
    """

    if not _r:
        return []
    try:
        offset = max(int(offset), 0)
        limit = max(int(limit), 0)
    except (TypeError, ValueError):
        return []

    if limit <= 0:
        return []

    key = _ledger_key(user_id)

    try:
        total = int(_r.llen(key) or 0)
    except Exception:
        _logger.exception("Failed to read ledger length for user %s", user_id)
        return []

    if total <= 0 or offset >= total:
        return []

    end_index = total - offset - 1
    start_index = max(end_index - limit + 1, 0)

    try:
        raw_items = _r.lrange(key, start_index, end_index)
    except Exception:
        _logger.exception("Failed to read ledger entries for user %s", user_id)
        return []

    entries: List[Dict[str, Any]] = []
    for raw in raw_items:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            _logger.debug("Skipping malformed ledger entry for user %s", user_id)
            continue
        if isinstance(parsed, dict):
            entries.append(parsed)

    entries.reverse()
    return entries


def get_ledger_count(user_id: int) -> int:
    """Return the total number of ledger entries for ``user_id``."""

    if not _r:
        return 0
    try:
        return int(_r.llen(_ledger_key(user_id)) or 0)
    except Exception:
        _logger.exception("Failed to count ledger entries for user %s", user_id)
        return 0


def ensure_user(user_id: int) -> None:
    if not _r:
        return
    p = _r.pipeline()
    p.sadd(USERS_KEY, user_id)
    p.setnx(_bal_key(user_id), 0)
    p.execute()


def users_count() -> int:
    if not _r:
        return 0
    return _r.scard(USERS_KEY)


def all_user_ids() -> List[int]:
    if not _r:
        return []
    raw = _r.smembers(USERS_KEY)
    return [int(x) for x in raw] if raw else []


def get_balance(user_id: int) -> int:
    if not _r:
        return 0
    v = _r.get(_bal_key(user_id))
    return int(v) if v is not None else 0


def add_ledger(user_id: int, entry: Dict[str, Any]) -> None:
    if not _r:
        return
    ensure_user(user_id)
    _r.rpush(_ledger_key(user_id), json.dumps(entry, ensure_ascii=False))


def add_ledger_entry(user_id: int, entry: Dict[str, Any]) -> None:
    """Append a raw ``entry`` to the user's ledger history."""

    add_ledger(user_id, entry)


def credit(
    user_id: int,
    amount: int,
    reason: str,
    meta: Optional[Dict[str, Any]] = None,
    *,
    write_ledger: bool = True,
) -> int:
    if not _r:
        return 0
    ensure_user(user_id)
    amount = int(amount)
    now = int(time.time())
    bal_key = _bal_key(user_id)
    ledger_key = _ledger_key(user_id)

    while True:
        try:
            with _r.pipeline() as pipe:
                pipe.watch(bal_key)
                current = pipe.get(bal_key)
                current_balance = int(current) if current is not None else 0
                new_balance = current_balance + amount
                pipe.multi()
                pipe.set(bal_key, new_balance)
                if write_ledger:
                    entry = {
                        "ts": now,
                        "type": "credit",
                        "amount": int(amount),
                        "reason": reason,
                        "meta": meta or {},
                        "balance_after": new_balance,
                    }
                    pipe.rpush(ledger_key, json.dumps(entry, ensure_ascii=False))
                pipe.execute()
                return new_balance
        except redis.WatchError:
            continue


def debit_try(
    user_id: int,
    amount: int,
    reason: str,
    meta: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, int]:
    if not _r:
        return False, 0
    ensure_user(user_id)
    amount = int(amount)
    now = int(time.time())
    bal_key = _bal_key(user_id)
    ledger_key = _ledger_key(user_id)

    while True:
        try:
            with _r.pipeline() as pipe:
                pipe.watch(bal_key)
                current = pipe.get(bal_key)
                current_balance = int(current) if current is not None else 0
                if current_balance < amount:
                    pipe.unwatch()
                    return False, current_balance
                new_balance = current_balance - amount
                entry = {
                    "ts": now,
                    "type": "debit",
                    "amount": amount,
                    "reason": reason,
                    "meta": meta or {},
                    "balance_after": new_balance,
                }
                pipe.multi()
                pipe.set(bal_key, new_balance)
                pipe.rpush(ledger_key, json.dumps(entry, ensure_ascii=False))
                pipe.execute()
                return True, new_balance
        except redis.WatchError:
            continue


def debit_balance(user_id: int, amount: int, reason: Optional[str] = None) -> int:
    """Debit ``amount`` from ``user_id`` balance and return the new balance."""

    if amount < 0:
        raise ValueError("amount must be non-negative")

    ensure_user(user_id)
    ok, new_balance = debit_try(user_id, amount, reason or "debit")
    if not ok:
        raise RuntimeError(
            f"insufficient balance for user {user_id}: need {amount}, have {new_balance}"
        )
    return new_balance


def credit_balance(
    user_id: int,
    amount: int,
    reason: Optional[str] = None,
    *,
    meta: Optional[Dict[str, Any]] = None,
    write_ledger: bool = True,
) -> int:
    """Credit ``amount`` to ``user_id`` balance and return the new balance."""

    if amount < 0:
        raise ValueError("amount must be non-negative")

    return credit(
        user_id,
        amount,
        reason or "credit",
        meta=meta,
        write_ledger=write_ledger,
    )
