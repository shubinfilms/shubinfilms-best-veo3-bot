"""Migrate legacy Redis user data into PostgreSQL tables."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar, cast

import redis.asyncio as aioredis
from redis.exceptions import ConnectionError as RedisConnectionError
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError

from db import postgres as db_postgres
from db.postgres import ENGINE

log = logging.getLogger("redis-migration")

DEFAULT_BATCH_SIZE = 200
MAX_DB_ATTEMPTS = 6
_BACKOFF_BASE = 0.5
T = TypeVar("T")

try:  # psycopg might not be present in all environments
    from psycopg.errors import OperationalError as PsycopgOperationalError
except Exception:  # pragma: no cover - optional dependency missing
    PsycopgOperationalError = None  # type: ignore

_RETRYABLE_MESSAGES = (
    "SSL connection has been closed unexpectedly",
    "server closed the connection unexpectedly",
    "connection already closed",
)


@dataclass(slots=True)
class MigrationStats:
    """Summary about a migration run."""

    users_imported: int = 0
    balances_imported: int = 0
    referrals_imported: int = 0
    skipped_entries: int = 0
    errors: List[str] = field(default_factory=list)
    redis_profiles_processed: int = 0
    redis_profiles_skipped: int = 0
    remaining_redis_keys: Optional[int] = None

    def as_lines(self) -> List[str]:
        lines = [
            f"✅ Migrated: {self.users_imported} users",
            f"💰 Balances imported: {self.balances_imported}",
            f"🤝 Referrals imported: {self.referrals_imported}",
        ]
        if self.skipped_entries:
            lines.append(f"⚠️ Skipped: {self.skipped_entries} invalid entries")
        else:
            lines.append("⚠️ Skipped: 0 invalid entries")
        lines.append(
            f"📦 Redis profiles processed: {self.redis_profiles_processed}"
        )
        lines.append(
            f"🚫 Redis profiles skipped: {self.redis_profiles_skipped}"
        )
        if self.remaining_redis_keys is not None:
            lines.append(
                f"🗂️ Remaining Redis keys: {self.remaining_redis_keys}"
            )
        return lines

    def snapshot(self) -> "MigrationStats":
        return MigrationStats(
            users_imported=self.users_imported,
            balances_imported=self.balances_imported,
            referrals_imported=self.referrals_imported,
            skipped_entries=self.skipped_entries,
            errors=list(self.errors),
            redis_profiles_processed=self.redis_profiles_processed,
            redis_profiles_skipped=self.redis_profiles_skipped,
            remaining_redis_keys=self.remaining_redis_keys,
        )


ProgressCallback = Callable[[MigrationStats, str], Awaitable[None]]


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip() or default


def safe_decode(value: Any) -> str:
    """Return a safe string representation of *value*.

    Redis sometimes stores malformed byte sequences or even non-string
    payloads. This helper normalises any value into a UTF-8 string while
    ignoring undecodable bytes. ``None`` is converted into an empty string so
    callers can treat the result uniformly.
    """

    if value is None:
        return ""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="ignore")
        except Exception:
            return str(value)
    return str(value)


def _decode(value: Any) -> Optional[str]:
    text = safe_decode(value)
    text = text.strip()
    return text or None


def _sanitize_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text_value = str(value).strip()
    return text_value or None


def _normalize_referrer(value: Any) -> Optional[int]:
    if value in (None, "", "None"):
        return None
    candidate = _parse_int(value)
    if candidate is None or candidate <= 0:
        return None
    return candidate


def _parse_int(value: Any) -> Optional[int]:
    raw = _decode(value)
    if raw is None or raw == "":
        return None
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return None


def _parse_datetime(value: Any) -> Optional[datetime]:
    raw = _decode(value)
    if not raw:
        return None
    raw = raw.strip()
    if not raw:
        return None
    for candidate in (raw, raw.replace("Z", "+00:00")):
        try:
            dt = datetime.fromisoformat(candidate)
        except ValueError:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    try:
        # Some entries store unix timestamps
        return datetime.fromtimestamp(float(raw), tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        return None


def _iter_exception_chain(exc: BaseException) -> List[BaseException]:
    chain: List[BaseException] = []
    current: Optional[BaseException] = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        chain.append(current)
        seen.add(id(current))
        current = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
    return chain


def _is_retryable_error(exc: BaseException) -> bool:
    for candidate in _iter_exception_chain(exc):
        if isinstance(candidate, (OperationalError, ConnectionError, ConnectionResetError)):
            return True
        if PsycopgOperationalError is not None and isinstance(candidate, PsycopgOperationalError):
            return True
        message = str(candidate)
        if any(token in message for token in _RETRYABLE_MESSAGES):
            return True
    return False


async def _run_with_retry(job_name: str, func: Callable[[], T]) -> T:
    delay = _BACKOFF_BASE
    for attempt in range(1, MAX_DB_ATTEMPTS + 1):
        try:
            return await asyncio.to_thread(func)
        except Exception as exc:
            if attempt >= MAX_DB_ATTEMPTS or not _is_retryable_error(exc):
                log.error(
                    "redis-migration.%s.failed | attempt=%s err=%s",
                    job_name,
                    attempt,
                    exc,
                )
                raise
            wait = min(delay, 4.0)
            log.warning(
                "redis-migration.%s.retry | attempt=%s err=%s wait=%.1fs",
                job_name,
                attempt,
                exc,
                wait,
            )
            await asyncio.sleep(wait)
            delay = min(delay * 2, 4.0)
    raise RuntimeError(f"redis-migration.{job_name} exhausted retries")


async def _execute_with_retry(engine: Engine, stmt: Any, rows: Sequence[Dict[str, Any]], label: str) -> int:
    if not rows:
        return 0

    def _run() -> int:
        with engine.begin() as conn:
            conn.execute(stmt, rows)
        return len(rows)

    return await _run_with_retry(f"{label}_batch", _run)


async def _scan_keys(conn: aioredis.Redis, pattern: str) -> AsyncIterator[str]:
    cursor: int | str = 0
    while True:
        try:
            cursor, keys = await conn.scan(cursor=cursor, match=pattern, count=500)
        except Exception as exc:
            log.warning(
                "redis-migration.scan_failed | pattern=%s cursor=%s err=%s",
                pattern,
                cursor,
                exc,
            )
            break
        for key in keys:
            decoded = _decode(key)
            if decoded:
                yield decoded
        if cursor in (0, "0", b"0"):
            break


async def _count_keys(conn: aioredis.Redis, pattern: str) -> Optional[int]:
    cursor: int | str = 0
    total = 0
    while True:
        try:
            cursor, keys = await conn.scan(cursor=cursor, match=pattern, count=1000)
        except Exception as exc:
            log.warning(
                "redis-migration.count_failed | pattern=%s cursor=%s err=%s",
                pattern,
                cursor,
                exc,
            )
            return None
        total += len(keys)
        if cursor in (0, "0", b"0"):
            break
    return total


async def _execute_users(engine: Engine, rows: Sequence[Dict[str, Any]]) -> int:
    if not rows:
        return 0

    prepared: List[Dict[str, Any]] = []
    for row in rows:
        try:
            user_id = int(row.get("id"))
        except (TypeError, ValueError):
            log.warning("redis-migration.users.skip | reason=invalid_id row=%s", row)
            continue

        joined_at = row.get("joined_at")
        if joined_at is not None and not isinstance(joined_at, datetime):
            joined_at = _parse_datetime(joined_at)
        if isinstance(joined_at, datetime):
            joined_at = joined_at.astimezone(timezone.utc)
        else:
            joined_at = None

        referral_total = row.get("referral_earned_total")
        if isinstance(referral_total, int):
            referral_total_int = max(referral_total, 0)
        else:
            parsed_total = _parse_int(referral_total)
            referral_total_int = max(parsed_total, 0) if parsed_total is not None else None

        prepared.append(
            {
                "id": user_id,
                "username": _sanitize_text(row.get("username")),
                "referrer_id": _normalize_referrer(row.get("referrer_id")),
                "joined_at": joined_at,
                "referral_earned_total": referral_total_int,
            }
        )

    if not prepared:
        return 0

    stmt = text(
        """
        INSERT INTO users (id, username, referrer_id, joined_at, referral_earned_total)
        VALUES (:id, :username, :referrer_id, COALESCE(:joined_at, NOW()), COALESCE(:referral_earned_total, 0))
        ON CONFLICT (id) DO UPDATE
           SET username = COALESCE(EXCLUDED.username, users.username),
               referrer_id = COALESCE(EXCLUDED.referrer_id, users.referrer_id),
               joined_at = CASE
                               WHEN users.joined_at IS NULL AND EXCLUDED.joined_at IS NOT NULL
                                   THEN EXCLUDED.joined_at
                               ELSE users.joined_at
                           END,
               referral_earned_total = CASE
                                           WHEN EXCLUDED.referral_earned_total IS NULL
                                               THEN users.referral_earned_total
                                           ELSE GREATEST(users.referral_earned_total, EXCLUDED.referral_earned_total)
                                       END,
               updated_at = NOW()
        """
    )
    return await _execute_with_retry(engine, stmt, prepared, "users")


async def _execute_balances(engine: Engine, rows: Sequence[Dict[str, Any]]) -> int:
    if not rows:
        return 0

    stmt = text(
        """
        INSERT INTO balances (user_id, tokens)
        VALUES (:user_id, :tokens)
        ON CONFLICT (user_id) DO UPDATE
           SET tokens = EXCLUDED.tokens,
               updated_at = NOW()
        """
    )
    return await _execute_with_retry(engine, stmt, rows, "balances")


async def _execute_referrals(engine: Engine, rows: Sequence[Dict[str, Any]]) -> int:
    if not rows:
        return 0

    stmt = text(
        """
        INSERT INTO referrals (referrer_id, referred_id, created_at)
        VALUES (:referrer_id, :referred_id, COALESCE(:created_at, NOW()))
        ON CONFLICT (referrer_id, referred_id) DO NOTHING
        """
    )
    return await _execute_with_retry(engine, stmt, rows, "referrals")


async def _gather_ref_keys(
    conn: aioredis.Redis,
    prefix: str,
    *,
    skipped: List[str],
) -> Tuple[Dict[int, int], Dict[int, datetime], Dict[int, int], List[Tuple[int, int, Optional[datetime]]]]:
    inviter_map: Dict[int, int] = {}
    joined_at_map: Dict[int, datetime] = {}
    earned_totals: Dict[int, int] = {}
    referral_pairs: List[Tuple[int, int, Optional[datetime]]] = []

    base = f"{prefix}:ref:"
    async for key in _scan_keys(conn, f"{base}*"):
        suffix = key[len(base) :]
        if suffix.startswith("inviter_of:"):
            user_part = suffix.partition(":")[2]
            user_id = _parse_int(user_part)
            if not user_id or user_id <= 0:
                skipped.append(f"invalid ref inviter key: {key}")
                continue
            try:
                inviter_raw = await conn.get(key)
            except Exception as exc:  # pragma: no cover - network failure path
                log.warning("redis-migration.get_inviter_failed | key=%s err=%s", key, exc)
                skipped.append(f"error reading {key}")
                continue
            inviter_id = _parse_int(inviter_raw)
            if not inviter_id or inviter_id <= 0 or inviter_id == user_id:
                skipped.append(f"invalid ref inviter value: {key}")
                continue
            inviter_map[user_id] = inviter_id
        elif suffix.startswith("users_of:"):
            inviter_part = suffix.partition(":")[2]
            inviter_id = _parse_int(inviter_part)
            if not inviter_id or inviter_id <= 0:
                skipped.append(f"invalid ref users key: {key}")
                continue
            try:
                members = await conn.smembers(key)
            except Exception as exc:  # pragma: no cover
                log.warning("redis-migration.smembers_failed | key=%s err=%s", key, exc)
                skipped.append(f"error reading {key}")
                continue
            for member in members:
                referred_id = _parse_int(member)
                if not referred_id or referred_id <= 0 or referred_id == inviter_id:
                    skipped.append(f"invalid ref member: {key}")
                    continue
                referral_pairs.append((inviter_id, referred_id, None))
        elif suffix.startswith("earned:"):
            inviter_part = suffix.partition(":")[2]
            inviter_id = _parse_int(inviter_part)
            if not inviter_id or inviter_id <= 0:
                skipped.append(f"invalid ref earned key: {key}")
                continue
            try:
                value_raw = await conn.get(key)
            except Exception as exc:  # pragma: no cover
                log.warning("redis-migration.get_earned_failed | key=%s err=%s", key, exc)
                skipped.append(f"error reading {key}")
                continue
            total = _parse_int(value_raw)
            if total is None or total < 0:
                skipped.append(f"invalid ref earned value: {key}")
                continue
            earned_totals[inviter_id] = max(total, earned_totals.get(inviter_id, 0))
        elif suffix.startswith("joined_at:"):
            user_part = suffix.partition(":")[2]
            user_id = _parse_int(user_part)
            if not user_id or user_id <= 0:
                skipped.append(f"invalid ref joined key: {key}")
                continue
            try:
                joined_raw = await conn.get(key)
            except Exception as exc:  # pragma: no cover
                log.warning("redis-migration.get_joined_failed | key=%s err=%s", key, exc)
                skipped.append(f"error reading {key}")
                continue
            dt = _parse_datetime(joined_raw)
            if dt is None:
                skipped.append(f"invalid ref joined value: {key}")
                continue
            joined_at_map[user_id] = dt
        else:
            # other ref keys (locks, caches) are ignored
            continue

    # Combine inviter map with referral pairs to ensure direct records
    for user_id, inviter_id in inviter_map.items():
        referral_pairs.append((inviter_id, user_id, joined_at_map.get(user_id)))

    return inviter_map, joined_at_map, earned_totals, referral_pairs


async def migrate_from_redis(
    *,
    redis_url: Optional[str] = None,
    database_url: Optional[str] = None,
    redis_prefix: Optional[str] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    progress_callback: Optional[ProgressCallback] = None,
) -> MigrationStats:
    """Perform a one-time migration from Redis into PostgreSQL."""

    redis_url = redis_url or _env("REDIS_URL")
    database_url = database_url or _env("DATABASE_URL") or _env("POSTGRES_DSN")
    redis_prefix = redis_prefix or _env("REDIS_PREFIX", "veo3:prod")

    if not redis_url:
        raise RuntimeError("REDIS_URL is required for migration")
    if not database_url:
        raise RuntimeError("DATABASE_URL (or POSTGRES_DSN) is required for migration")

    stats = MigrationStats()
    if progress_callback:
        await progress_callback(stats.snapshot(), "start")
    skipped: List[str] = []

    db_postgres.configure_engine(database_url)
    try:
        await asyncio.to_thread(db_postgres.ensure_tables)
    except Exception as exc:
        log.error("redis-migration.ensure_tables_failed | err=%s", exc, exc_info=True)
        raise RuntimeError("Failed to prepare PostgreSQL tables") from exc

    engine = cast(Engine, ENGINE) if ENGINE is not None else db_postgres.get_engine()

    conn: Optional[aioredis.Redis] = None
    retry_count = 3
    for attempt in range(1, retry_count + 1):
        candidate = aioredis.from_url(
            redis_url,
            encoding="utf-8",
            decode_responses=True,
            health_check_interval=30,
            socket_keepalive=True,
        )
        try:
            await candidate.ping()
        except RedisConnectionError as exc:
            log.warning(
                "redis-migration.connect.retry | attempt=%s err=%s",
                attempt,
                exc,
            )
            await candidate.close()
            if attempt >= retry_count:
                raise
            await asyncio.sleep(2)
            continue
        except Exception as exc:
            log.warning(
                "redis-migration.connect.error | attempt=%s err=%s",
                attempt,
                exc,
            )
            await candidate.close()
            if attempt >= retry_count:
                raise
            await asyncio.sleep(2)
            continue
        conn = candidate
        break

    if conn is None:
        raise RuntimeError("Failed to establish Redis connection")

    try:
        log.info("redis-migration.start | prefix=%s", redis_prefix)

        user_records: Dict[int, Dict[str, Any]] = {}
        user_sources: Dict[int, set[str]] = {}
        balance_rows: Dict[int, int] = {}
        stats.redis_profiles_processed = 0
        stats.redis_profiles_skipped = 0

        def _ensure_user_record(user_id: int, *, source: Optional[str] = None) -> Dict[str, Any]:
            record = user_records.setdefault(
                user_id,
                {
                    "id": user_id,
                    "username": None,
                    "referrer_id": None,
                    "joined_at": None,
                    "referral_earned_total": None,
                },
            )
            if source:
                user_sources.setdefault(user_id, set()).add(source)
            return record

        # Load user profiles
        async for key in _scan_keys(conn, f"{redis_prefix}:user:*"):
            user_part = key.rsplit(":", 1)[-1]
            user_id = _parse_int(user_part)
            if not user_id or user_id <= 0:
                skipped.append(f"invalid user key: {key}")
                stats.redis_profiles_skipped += 1
                continue
            try:
                profile = await conn.hgetall(key)
            except Exception as exc:  # pragma: no cover - network failure path
                log.warning("redis-migration.hgetall_failed | key=%s err=%s", key, exc)
                skipped.append(f"error reading {key}")
                stats.redis_profiles_skipped += 1
                continue
            if not profile:
                skipped.append(f"empty user payload: {key}")
                stats.redis_profiles_skipped += 1
                continue
            if not isinstance(profile, dict):
                skipped.append(f"invalid user payload: {key}")
                stats.redis_profiles_skipped += 1
                continue
            try:
                record = _ensure_user_record(user_id, source=key)
                username_raw = safe_decode(
                    profile.get("username") or profile.get(b"username")
                )
                username = _sanitize_text(username_raw)
                if username:
                    record["username"] = username

                referrer_raw = safe_decode(
                    profile.get("referrer_id") or profile.get(b"referrer_id")
                )
                referrer_id = _normalize_referrer(referrer_raw)
                if referrer_id:
                    record["referrer_id"] = referrer_id

                joined_raw = safe_decode(
                    profile.get("joined_at") or profile.get(b"joined_at")
                )
                joined_at = _parse_datetime(joined_raw)
                if joined_at:
                    record["joined_at"] = joined_at

                referral_total_raw = safe_decode(
                    profile.get("referral_earned_total")
                    or profile.get(b"referral_earned_total")
                )
                referral_total = _parse_int(referral_total_raw)
                if referral_total is not None and referral_total >= 0:
                    current_total = record.get("referral_earned_total")
                    current_total_int = (
                        int(current_total)
                        if isinstance(current_total, int)
                        else _parse_int(current_total)
                    )
                    if current_total_int is None or referral_total > current_total_int:
                        record["referral_earned_total"] = referral_total
            except Exception as exc:
                log.warning(
                    "redis-migration.user.decode_failed | key=%s err=%s",
                    key,
                    exc,
                )
                skipped.append(f"malformed user payload: {key}")
                stats.redis_profiles_skipped += 1
                continue

        # Load balances with both legacy and documented prefixes
        balance_patterns = [
            f"{redis_prefix}:balance:*",
            f"{redis_prefix}:bal:*",
        ]
        seen_balance_keys: set[str] = set()
        for pattern in balance_patterns:
            async for key in _scan_keys(conn, pattern):
                if key in seen_balance_keys:
                    continue
                seen_balance_keys.add(key)
                user_part = key.rsplit(":", 1)[-1]
                user_id = _parse_int(user_part)
                if not user_id or user_id <= 0:
                    skipped.append(f"invalid balance key: {key}")
                    continue
                try:
                    value = await conn.get(key)
                except Exception as exc:  # pragma: no cover
                    log.warning("redis-migration.get_balance_failed | key=%s err=%s", key, exc)
                    skipped.append(f"error reading {key}")
                    continue
                tokens = _parse_int(value)
                if tokens is None or tokens < 0:
                    skipped.append(f"invalid balance value: {key}")
                    continue
                balance_rows[user_id] = tokens
                _ensure_user_record(user_id, source=key)

        inviter_map, joined_at_map, earned_totals, referral_pairs = await _gather_ref_keys(
            conn,
            redis_prefix,
            skipped=skipped,
        )

        for user_id, inviter_id in inviter_map.items():
            record = _ensure_user_record(user_id)
            record["referrer_id"] = inviter_id
            # Ensure inviter user exists too
            _ensure_user_record(inviter_id)

        for user_id, joined in joined_at_map.items():
            record = user_records.get(user_id)
            if record:
                record["joined_at"] = joined

        for referrer_id, total in earned_totals.items():
            record = _ensure_user_record(referrer_id)
            record["referral_earned_total"] = max(total, int(record.get("referral_earned_total") or 0))

        # Prepare batches
        total_user_candidates = len(user_records)
        user_batches: List[Dict[str, Any]] = []
        for user_id, record in user_records.items():
            username_candidate = _sanitize_text(record.get("username"))
            normalized_referrer = _normalize_referrer(record.get("referrer_id"))
            if username_candidate is None and normalized_referrer is None:
                source_keys = sorted(user_sources.get(user_id, []))
                key_info = ", ".join(source_keys) if source_keys else f"user:{user_id}"
                log.warning("redis-migration.users.skip | reason=missing_identity key=%s", key_info)
                skipped.append(f"malformed user record: {key_info}")
                stats.redis_profiles_skipped += 1
                continue
            row = dict(record)
            row["username"] = username_candidate
            row["referrer_id"] = normalized_referrer
            source_keys = sorted(user_sources.get(user_id, []))
            if source_keys:
                row["_source"] = ", ".join(source_keys)
            user_batches.append(row)
            stats.redis_profiles_processed += 1
        balance_batch = [
            {"user_id": user_id, "tokens": tokens}
            for user_id, tokens in balance_rows.items()
        ]
        referral_batch = []
        seen_pairs: set[Tuple[int, int]] = set()
        for referrer_id, referred_id, created_at in referral_pairs:
            pair = (referrer_id, referred_id)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            if referrer_id <= 0 or referred_id <= 0 or referrer_id == referred_id:
                skipped.append(f"invalid referral pair: {referrer_id}->{referred_id}")
                continue
            referral_batch.append(
                {
                    "referrer_id": referrer_id,
                    "referred_id": referred_id,
                    "created_at": created_at,
                }
            )

        # Execute migrations in batches
        async def _chunked_execute(
            items: List[Dict[str, Any]],
            executor,
            stage: str,
            *,
            skipped_log: List[str],
        ) -> int:
            total = 0
            chunk_size = max(1, batch_size)
            for idx in range(0, len(items), chunk_size):
                chunk = items[idx : idx + chunk_size]
                try:
                    processed = await executor(engine, chunk)
                except Exception as exc:
                    if stage == "users":
                        log.error(
                            "redis-migration.batch_failed | stage=%s chunk=%s err=%s",
                            stage,
                            idx // chunk_size + 1,
                            exc,
                            exc_info=True,
                        )
                        recovered = 0
                        for row in chunk:
                            try:
                                single_processed = await executor(engine, [row])
                            except Exception as row_exc:  # pragma: no cover - defensive skip path
                                user_id = row.get("id")
                                error_text = str(row_exc)
                                warning_entry = (
                                    f"⚠️ user {user_id} skipped due to database error: {error_text}"
                                )
                                skipped_log.append(warning_entry)
                                log.warning(
                                    "⚠️ redis-migration.user.skip | user_id=%s err=%s row=%s",
                                    user_id,
                                    row_exc,
                                    row,
                                    exc_info=True,
                                )
                                continue
                            recovered += single_processed
                            total += single_processed
                            stats.users_imported += single_processed
                            if progress_callback:
                                await progress_callback(stats.snapshot(), stage)
                        log.info(
                            "redis-migration.batch.recovered | stage=%s chunk=%s recovered=%s cumulative=%s",
                            stage,
                            idx // chunk_size + 1,
                            recovered,
                            total,
                        )
                        continue
                    raise
                total += processed
                if stage == "users":
                    stats.users_imported += processed
                elif stage == "balances":
                    stats.balances_imported += processed
                elif stage == "referrals":
                    stats.referrals_imported += processed
                if stage == "users" and total and total % 500 == 0:
                    log.info("Progress: migrated %s users so far...", total)
                log.info(
                    "redis-migration.batch | stage=%s chunk=%s processed=%s cumulative=%s",
                    stage,
                    idx // chunk_size + 1,
                    processed,
                    total,
                )
                if progress_callback:
                    await progress_callback(stats.snapshot(), stage)
            return total

        await _chunked_execute(user_batches, _execute_users, "users", skipped_log=skipped)
        await _chunked_execute(
            balance_batch, _execute_balances, "balances", skipped_log=skipped
        )
        await _chunked_execute(
            referral_batch, _execute_referrals, "referrals", skipped_log=skipped
        )

        remaining_pattern = f"{redis_prefix}:*"
        stats.remaining_redis_keys = await _count_keys(conn, remaining_pattern)
        if stats.remaining_redis_keys is None:
            log.warning(
                "redis-migration.remaining_keys_failed | pattern=%s",
                remaining_pattern,
            )

        stats.skipped_entries = len(skipped)
        stats.errors = list(skipped)
        log.info(
            "Migration complete: %s/%s users imported successfully.",
            stats.users_imported,
            total_user_candidates,
        )

        if progress_callback:
            await progress_callback(stats.snapshot(), "summary")

        log.info(
            "redis-migration.finished | users=%s balances=%s referrals=%s skipped=%s",
            stats.users_imported,
            stats.balances_imported,
            stats.referrals_imported,
            stats.skipped_entries,
        )
        summary_lines = stats.as_lines()
        for line in summary_lines:
            level = (
                logging.WARNING
                if line.startswith("⚠️") and stats.skipped_entries
                else logging.INFO
            )
            log.log(level, "redis-migration.summary | %s", line)
        if stats.skipped_entries:
            summary_block = "\n".join(summary_lines)
            log.warning("redis-migration.summary.report\n%s", summary_block)
            for entry in skipped:
                log.warning("⚠️ redis-migration.skipped | %s", entry)
    finally:
        try:
            await conn.close()
        except Exception:  # pragma: no cover
            pass

    return stats


async def cleanup_redis(conn: "aioredis.Redis", redis_prefix: str = "veo3:prod") -> int:
    """Remove legacy Redis keys that match the given prefix."""

    pattern = f"{redis_prefix}:*"
    log.info("redis.cleanup.started | pattern=%s", pattern)

    cursor: bytes | int | str = b"0"
    total_deleted = 0

    while True:
        cursor, keys = await conn.scan(cursor=cursor, match=pattern, count=1000)
        if keys:
            await conn.delete(*keys)
            total_deleted += len(keys)
            if total_deleted and total_deleted % 5000 == 0:
                log.info("redis.cleanup.progress | deleted=%s", total_deleted)
        if cursor in (0, "0", b"0"):
            break

    log.info("redis.cleanup.completed | pattern=%s deleted=%s", pattern, total_deleted)
    return total_deleted


def _configure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")


def main() -> None:
    _configure_logging()
    try:
        stats = asyncio.run(migrate_from_redis())
    except Exception as exc:  # pragma: no cover - CLI failure path
        log.exception("redis-migration.failed | err=%s", exc)
        raise
    for line in stats.as_lines():
        print(line)
    if stats.errors:
        print("⚠️ Skipped entries:")
        for entry in stats.errors:
            print(f" - {entry}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
