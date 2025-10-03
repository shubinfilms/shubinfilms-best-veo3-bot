import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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

_MJ_LAST_KEY_TMPL = f"{_PFX}:mj:last:{{}}"
_MJ_LOCK_KEY_TMPL = f"{_PFX}:mj:lock:{{}}"
_MJ_LOCK_TTL = 15 * 60
_MENU_LOCK_KEY_TMPL = f"{_PFX}:menu:lock:{{}}:{{}}"
_MENU_MSG_KEY_TMPL = f"{_PFX}:menu:msg:{{}}:{{}}"
_USER_LOCK_KEY_TMPL = f"{_PFX}:user:lock:{{}}:{{}}"
_SORA2_LOCK_KEY_TMPL = f"{_PFX}:lock:sora2:{{}}"
_SORA2_UNAVAILABLE_KEY = f"{_PFX}:sora2:unavailable"
_MJ_GALLERY_KEY_TMPL = f"{_PFX}:mj:gallery:{{}}:{{}}"
_MJ_GALLERY_TTL = 2 * 60 * 60


class MenuLocked(RuntimeError):
    """Raised when a menu lock is already held by another task."""

    def __init__(self, name: str, chat_id: int):
        super().__init__(f"Menu lock busy: {name} for chat {chat_id}")
        self.name = str(name)
        self.chat_id = int(chat_id)


def _menu_lock_key(name: str, chat_id: int) -> str:
    normalized = str(name or "lock").strip() or "lock"
    return _MENU_LOCK_KEY_TMPL.format(normalized, int(chat_id))


def _menu_msg_key(name: str, chat_id: int) -> str:
    normalized = str(name or "card").strip() or "card"
    return _MENU_MSG_KEY_TMPL.format(normalized, int(chat_id))


def _user_lock_key(user_id: int, key: str) -> str:
    normalized = str(key or "lock").strip() or "lock"
    return _USER_LOCK_KEY_TMPL.format(int(user_id), normalized)


def _sora2_lock_key(user_id: int) -> str:
    return _SORA2_LOCK_KEY_TMPL.format(int(user_id))

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


def _memory_set_with_ttl(key: str, value: str, ttl: int) -> None:
    expires_at = time.time() + max(ttl, 1)
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


def _memory_exists(key: str) -> bool:
    now = time.time()
    with _memory_lock:
        entry = _memory_store.get(key)
        if not entry:
            return False
        expires_at, _ = entry
        if expires_at <= now:
            _memory_store.pop(key, None)
            return False
        return True


def _memory_set_if_absent(key: str, value: str, ttl: int) -> bool:
    now = time.time()
    expires_at = now + max(ttl, 1)
    with _memory_lock:
        entry = _memory_store.get(key)
        if entry:
            expires, _ = entry
            if expires > now:
                return False
        _memory_store[key] = (expires_at, value)
        return True


def mark_sora2_unavailable(*, ttl: int = 60 * 60) -> None:
    key = _SORA2_UNAVAILABLE_KEY
    if _r:
        try:
            _r.setex(key, max(1, int(ttl)), "1")
        except Exception as exc:  # pragma: no cover - network issues
            _logger.warning("redis.setex_failed | key=%s err=%s", key, exc)
            _memory_set_with_ttl(key, "1", ttl)
    else:
        _memory_set_with_ttl(key, "1", ttl)


def clear_sora2_unavailable() -> None:
    key = _SORA2_UNAVAILABLE_KEY
    if _r:
        try:
            _r.delete(key)
        except Exception as exc:  # pragma: no cover - network issues
            _logger.warning("redis.del_failed | key=%s err=%s", key, exc)
            _memory_delete(key)
    else:
        _memory_delete(key)


def is_sora2_unavailable() -> bool:
    key = _SORA2_UNAVAILABLE_KEY
    if _r:
        try:
            return bool(_r.get(key))
        except Exception as exc:  # pragma: no cover - network issues
            _logger.warning("redis.get_failed | key=%s err=%s", key, exc)
            return _memory_exists(key)
    return _memory_exists(key)


def _mj_last_key(user_id: int) -> str:
    return _MJ_LAST_KEY_TMPL.format(int(user_id))


def _mj_lock_key(user_id: int, task_id: str, index: int) -> str:
    return _MJ_LOCK_KEY_TMPL.format(f"{int(user_id)}:{task_id}:{int(index)}")


def _mj_gallery_key(chat_id: int, message_id: int) -> str:
    return _MJ_GALLERY_KEY_TMPL.format(int(chat_id), int(message_id))


def set_last_mj_grid(
    user_id: int,
    task_id: str,
    result_urls: List[str],
    *,
    prompt: Optional[str] = None,
) -> None:
    doc = {
        "task_id": str(task_id),
        "result_urls": [str(url) for url in result_urls if isinstance(url, str)],
        "created_at": _now_iso(),
    }
    if isinstance(prompt, str):
        prompt_clean = prompt.strip()
        if prompt_clean:
            doc["prompt"] = prompt_clean
    payload = json.dumps(doc, ensure_ascii=False)
    key = _mj_last_key(user_id)
    if _r:
        _r.setex(key, _TTL, payload)
    else:
        _memory_set(key, payload)


def get_last_mj_grid(user_id: int) -> Optional[Dict[str, Any]]:
    key = _mj_last_key(user_id)
    raw: Optional[str]
    if _r:
        redis_raw = _r.get(key)
        if redis_raw is None:
            raw = None
        elif isinstance(redis_raw, bytes):
            raw = redis_raw.decode("utf-8", "ignore")
        else:
            raw = str(redis_raw)
    else:
        raw = _memory_get(key)
    if raw is None:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        _logger.warning("Failed to decode mj grid cache for user %s", user_id)
        return None
    if not isinstance(data, dict):
        return None
    task_id = data.get("task_id")
    urls = data.get("result_urls")
    if not isinstance(task_id, str) or not isinstance(urls, list):
        return None
    normalized_urls = [str(url) for url in urls if isinstance(url, str)]
    if not normalized_urls:
        return None
    prompt = data.get("prompt")
    prompt_value = prompt.strip() if isinstance(prompt, str) else None
    result: Dict[str, Any] = {"task_id": task_id, "result_urls": normalized_urls}
    if prompt_value:
        result["prompt"] = prompt_value
    return result


def set_mj_gallery(chat_id: int, message_id: int, payload: Sequence[Mapping[str, Any]]) -> None:
    key = _mj_gallery_key(chat_id, message_id)
    try:
        data = json.dumps(list(payload), ensure_ascii=False)
    except (TypeError, ValueError):
        _logger.warning("Failed to serialize mj gallery payload for chat %s", chat_id)
        return
    if _r:
        _r.setex(key, _MJ_GALLERY_TTL, data)
    else:
        _memory_set_with_ttl(key, data, _MJ_GALLERY_TTL)


def get_mj_gallery(chat_id: int, message_id: int) -> Optional[list[dict[str, Any]]]:
    key = _mj_gallery_key(chat_id, message_id)
    raw: Optional[str]
    if _r:
        redis_raw = _r.get(key)
        if redis_raw is None:
            raw = None
        elif isinstance(redis_raw, bytes):
            raw = redis_raw.decode("utf-8", "ignore")
        else:
            raw = str(redis_raw)
    else:
        raw = _memory_get(key)
    if raw is None:
        return None
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError:
        _logger.warning("Failed to decode mj gallery payload for chat %s", chat_id)
        return None
    if not isinstance(doc, list):
        return None
    result: list[dict[str, Any]] = []
    for item in doc:
        if not isinstance(item, dict):
            continue
        record = {
            "file_name": str(item.get("file_name", "")),
            "source_url": str(item.get("source_url", "")),
            "bytes_len": int(item.get("bytes_len", 0) or 0),
            "mime": str(item.get("mime", "")),
            "sent_message_id": int(item.get("sent_message_id", 0) or 0),
        }
        result.append(record)
    if not result:
        return None
    return result


def clear_mj_gallery(chat_id: int, message_id: int) -> None:
    key = _mj_gallery_key(chat_id, message_id)
    if _r:
        _r.delete(key)
    else:
        _memory_delete(key)


def clear_last_mj_grid(user_id: int) -> None:
    key = _mj_last_key(user_id)
    if _r:
        _r.delete(key)
    else:
        _memory_delete(key)


def acquire_mj_upscale_lock(user_id: int, task_id: str, index: int, ttl: int = _MJ_LOCK_TTL) -> bool:
    key = _mj_lock_key(user_id, task_id, index)
    if _r:
        try:
            return bool(_r.set(key, "1", nx=True, ex=max(ttl, 1)))
        except Exception as exc:
            _logger.warning("mj_upscale_lock.redis_error | key=%s err=%s", key, exc)
            return False
    return _memory_set_if_absent(key, "1", ttl)


def release_mj_upscale_lock(user_id: int, task_id: str, index: int) -> None:
    key = _mj_lock_key(user_id, task_id, index)
    if _r:
        try:
            _r.delete(key)
        except Exception as exc:
            _logger.warning("mj_upscale_lock.release_error | key=%s err=%s", key, exc)
    else:
        _memory_delete(key)


def acquire_menu_lock(name: str, chat_id: int, ttl: int) -> bool:
    key = _menu_lock_key(name, chat_id)
    if _r:
        try:
            return bool(_r.set(key, "1", nx=True, ex=max(ttl, 1)))
        except Exception as exc:
            _logger.warning("menu_lock.acquire_error | key=%s err=%s", key, exc)
            return False
    return _memory_set_if_absent(key, "1", ttl)


def release_menu_lock(name: str, chat_id: int) -> None:
    key = _menu_lock_key(name, chat_id)
    if _r:
        try:
            _r.delete(key)
        except Exception as exc:
            _logger.warning("menu_lock.release_error | key=%s err=%s", key, exc)
    else:
        _memory_delete(key)


@asynccontextmanager
async def with_menu_lock(name: str, chat_id: int, ttl: int = 5):
    """Async context manager that guards menu interactions for ``chat_id``."""

    acquired = acquire_menu_lock(name, chat_id, ttl)
    if not acquired:
        raise MenuLocked(name, chat_id)
    try:
        yield
    finally:
        release_menu_lock(name, chat_id)


def user_lock(user_id: Optional[int], key: str, ttl: int = 30) -> bool:
    if not user_id:
        return True
    lock_key = _user_lock_key(int(user_id), key)
    ttl_value = max(int(ttl), 1)
    if _r:
        try:
            return bool(_r.set(lock_key, "1", nx=True, ex=ttl_value))
        except Exception as exc:
            _logger.warning("user_lock.acquire_error | key=%s err=%s", lock_key, exc)
            return True
    return _memory_set_if_absent(lock_key, "1", ttl_value)


def release_user_lock(user_id: Optional[int], key: str) -> None:
    if not user_id:
        return
    lock_key = _user_lock_key(int(user_id), key)
    if _r:
        try:
            _r.delete(lock_key)
        except Exception as exc:
            _logger.warning("user_lock.release_error | key=%s err=%s", lock_key, exc)
    else:
        _memory_delete(lock_key)


def acquire_sora2_lock(user_id: Optional[int], ttl: int = 60) -> bool:
    if not user_id:
        return True
    lock_key = _sora2_lock_key(int(user_id))
    ttl_value = max(int(ttl), 1)
    if _r:
        try:
            return bool(_r.set(lock_key, "1", nx=True, ex=ttl_value))
        except Exception as exc:
            _logger.warning("sora2.lock.acquire_error | key=%s err=%s", lock_key, exc)
            return True
    return _memory_set_if_absent(lock_key, "1", ttl_value)


def release_sora2_lock(user_id: Optional[int]) -> None:
    if not user_id:
        return
    lock_key = _sora2_lock_key(int(user_id))
    if _r:
        try:
            _r.delete(lock_key)
        except Exception as exc:
            _logger.warning("sora2.lock.release_error | key=%s err=%s", lock_key, exc)
    else:
        _memory_delete(lock_key)


def save_menu_message(name: str, chat_id: int, message_id: int, ttl: int) -> None:
    payload = json.dumps(
        {
            "chat_id": int(chat_id),
            "message_id": int(message_id),
            "ts": time.time(),
        },
        ensure_ascii=False,
    )
    key = _menu_msg_key(name, chat_id)
    if _r:
        try:
            _r.setex(key, max(ttl, 1), payload)
            return
        except Exception as exc:
            _logger.warning("menu_msg.save_error | key=%s err=%s", key, exc)
    _memory_set_with_ttl(key, payload, ttl)


def get_menu_message(name: str, chat_id: int, *, max_age: Optional[int] = None) -> Optional[Tuple[int, float]]:
    key = _menu_msg_key(name, chat_id)
    raw: Optional[str]
    if _r:
        try:
            redis_raw = _r.get(key)
        except Exception as exc:
            _logger.warning("menu_msg.get_error | key=%s err=%s", key, exc)
            redis_raw = None
        if redis_raw is None:
            raw = None
        elif isinstance(redis_raw, bytes):
            raw = redis_raw.decode("utf-8", "ignore")
        else:
            raw = str(redis_raw)
    else:
        raw = _memory_get(key)
    if raw is None:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        _logger.warning("menu_msg.parse_error | key=%s", key)
        clear_menu_message(name, chat_id)
        return None
    message_id = data.get("message_id")
    timestamp = data.get("ts")
    try:
        msg_value = int(message_id)
    except (TypeError, ValueError):
        clear_menu_message(name, chat_id)
        return None
    try:
        ts_value = float(timestamp)
    except (TypeError, ValueError):
        ts_value = time.time()
    if max_age is not None and max_age > 0 and (time.time() - ts_value) > max_age:
        clear_menu_message(name, chat_id)
        return None
    return msg_value, ts_value


def clear_menu_message(name: str, chat_id: int) -> None:
    key = _menu_msg_key(name, chat_id)
    if _r:
        try:
            _r.delete(key)
        except Exception as exc:
            _logger.warning("menu_msg.clear_error | key=%s err=%s", key, exc)
    else:
        _memory_delete(key)


def cache_get(name: str) -> Optional[str]:
    key = f"{_PFX}:{name}"
    if _r:
        try:
            raw = _r.get(key)
        except Exception as exc:
            _logger.warning("cache.get.error | key=%s err=%s", key, exc)
            return None
        if raw is None:
            return None
        if isinstance(raw, bytes):
            return raw.decode("utf-8", "ignore")
        return str(raw)
    return _memory_get(key)


def cache_set(name: str, value: str, ttl: int) -> None:
    key = f"{_PFX}:{name}"
    if _r:
        try:
            _r.setex(key, max(ttl, 1), value)
            return
        except Exception as exc:
            _logger.warning("cache.set.error | key=%s err=%s", key, exc)
    _memory_set_with_ttl(key, value, ttl)


def acquire_ttl_lock(name: str, ttl: int) -> bool:
    key = f"{_PFX}:{name}"
    if _r:
        try:
            return bool(_r.set(key, "1", nx=True, ex=max(ttl, 1)))
        except Exception as exc:
            _logger.warning("lock.acquire.error | key=%s err=%s", key, exc)
    return _memory_set_if_absent(key, "1", ttl)


def release_ttl_lock(name: str) -> None:
    key = f"{_PFX}:{name}"
    if _r:
        try:
            _r.delete(key)
            return
        except Exception as exc:
            _logger.warning("lock.release.error | key=%s err=%s", key, exc)
    _memory_delete(key)


def save_task_meta(
    task_id: str,
    chat_id: int,
    message_id: int,
    mode: str,
    aspect: str,
    *,
    extra: Optional[Mapping[str, Any]] = None,
    ttl: Optional[int] = None,
) -> None:
    ttl_value = max(int(ttl) if ttl is not None else _TTL, 1)
    doc: Dict[str, Any] = {
        "chat_id": int(chat_id),
        "message_id": int(message_id),
        "mode": mode,
        "aspect": aspect,
        "created_at": _now_iso(),
        "_ttl": ttl_value,
    }
    if extra:
        for key, value in extra.items():
            if key not in {"chat_id", "message_id"}:
                doc[key] = value
    payload = json.dumps(doc, ensure_ascii=False)
    key = task_key(task_id)
    if _r:
        _r.setex(key, ttl_value, payload)
    else:
        _memory_set_with_ttl(key, payload, ttl_value)


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


def update_task_meta(
    task_id: str,
    *,
    ttl: Optional[int] = None,
    **fields: Any,
) -> Optional[Dict[str, Any]]:
    current = load_task_meta(task_id)
    if not current:
        return None
    current.update(fields)
    ttl_source = ttl if ttl is not None else current.get("_ttl")
    ttl_value = max(int(ttl_source) if ttl_source else _TTL, 1)
    current["_ttl"] = ttl_value
    payload = json.dumps(current, ensure_ascii=False)
    key = task_key(task_id)
    if _r:
        try:
            _r.setex(key, ttl_value, payload)
        except Exception as exc:
            _logger.warning("task_meta.update_error | key=%s err=%s", key, exc)
            return current
    else:
        _memory_set_with_ttl(key, payload, ttl_value)
    return current


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
_MJ_LAST_KEY_TMPL = f"{_PFX}:mj:last:{{}}"
_MJ_LOCK_KEY_TMPL = f"{_PFX}:mj:lock:{{}}"
_MJ_LOCK_TTL = 15 * 60

