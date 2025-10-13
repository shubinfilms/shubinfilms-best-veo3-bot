"""Migrate legacy Redis user data into PostgreSQL tables."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence, Tuple

import redis.asyncio as redis
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from db import postgres as db_postgres

log = logging.getLogger("redis-migration")

DEFAULT_BATCH_SIZE = 200


@dataclass(slots=True)
class MigrationStats:
    """Summary about a migration run."""

    users_imported: int = 0
    balances_imported: int = 0
    referrals_imported: int = 0
    skipped_entries: int = 0
    errors: List[str] = field(default_factory=list)

    def as_lines(self) -> List[str]:
        lines = [
            f"✅ Migrated {self.users_imported} users",
            f"✅ Migrated {self.balances_imported} balances",
            f"✅ Migrated {self.referrals_imported} referrals",
        ]
        if self.skipped_entries:
            lines.append(f"⚠️ Skipped {self.skipped_entries} malformed entries")
        return lines


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip() or default


def _decode(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.decode("utf-8", "ignore")
    return str(value)


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


async def _scan_keys(conn: redis.Redis, pattern: str) -> AsyncIterator[str]:
    cursor: int = 0
    while True:
        cursor, keys = await conn.scan(cursor=cursor, match=pattern, count=500)
        for key in keys:
            decoded = _decode(key)
            if decoded:
                yield decoded
        if cursor == 0:
            break


def _get_engine(dsn: str) -> Engine:
    return create_engine(dsn, future=True, pool_pre_ping=True)


async def _execute_users(engine: Engine, rows: Sequence[Dict[str, Any]]) -> int:
    if not rows:
        return 0

    stmt = text(
        """
        INSERT INTO users (id, username, referrer_id, joined_at, referral_earned_total)
        VALUES (:id, :username, :referrer_id, COALESCE(:joined_at, NOW()), COALESCE(:referral_earned_total, 0))
        ON CONFLICT (id) DO UPDATE
           SET username = COALESCE(EXCLUDED.username, users.username),
               referrer_id = COALESCE(users.referrer_id, EXCLUDED.referrer_id),
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

    def _run() -> int:
        with engine.begin() as conn:
            conn.execute(stmt, rows)
        return len(rows)

    return await asyncio.to_thread(_run)


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

    def _run() -> int:
        with engine.begin() as conn:
            conn.execute(stmt, rows)
        return len(rows)

    return await asyncio.to_thread(_run)


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

    def _run() -> int:
        with engine.begin() as conn:
            conn.execute(stmt, rows)
        return len(rows)

    return await asyncio.to_thread(_run)


async def _gather_ref_keys(
    conn: redis.Redis,
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
                log.warning("redis.get_inviter_failed | key=%s err=%s", key, exc)
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
                log.warning("redis.smembers_failed | key=%s err=%s", key, exc)
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
                log.warning("redis.get_earned_failed | key=%s err=%s", key, exc)
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
                log.warning("redis.get_joined_failed | key=%s err=%s", key, exc)
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
    skipped: List[str] = []

    db_postgres.configure(database_url)
    try:
        await asyncio.to_thread(db_postgres.ensure_tables)
    except Exception as exc:
        log.warning("postgres.ensure_tables_failed | err=%s", exc)

    engine = _get_engine(database_url)
    conn = redis.from_url(redis_url, encoding=None, decode_responses=False)

    try:
        log.info("redis.migration.start | prefix=%s", redis_prefix)

        user_records: Dict[int, Dict[str, Any]] = {}
        balance_rows: Dict[int, int] = {}

        # Load user profiles
        async for key in _scan_keys(conn, f"{redis_prefix}:user:*"):
            user_part = key.rsplit(":", 1)[-1]
            user_id = _parse_int(user_part)
            if not user_id or user_id <= 0:
                skipped.append(f"invalid user key: {key}")
                continue
            try:
                profile = await conn.hgetall(key)
            except Exception as exc:  # pragma: no cover - network failure path
                log.warning("redis.hgetall_failed | key=%s err=%s", key, exc)
                skipped.append(f"error reading {key}")
                continue
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
            username = _decode(profile.get(b"username")) if isinstance(profile, dict) else None
            if username:
                record["username"] = username

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
                    log.warning("redis.get_balance_failed | key=%s err=%s", key, exc)
                    skipped.append(f"error reading {key}")
                    continue
                tokens = _parse_int(value)
                if tokens is None or tokens < 0:
                    skipped.append(f"invalid balance value: {key}")
                    continue
                balance_rows[user_id] = tokens
                user_records.setdefault(
                    user_id,
                    {
                        "id": user_id,
                        "username": None,
                        "referrer_id": None,
                        "joined_at": None,
                        "referral_earned_total": None,
                    },
                )

        inviter_map, joined_at_map, earned_totals, referral_pairs = await _gather_ref_keys(
            conn,
            redis_prefix,
            skipped=skipped,
        )

        for user_id, inviter_id in inviter_map.items():
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
            record["referrer_id"] = inviter_id
            # Ensure inviter user exists too
            user_records.setdefault(
                inviter_id,
                {
                    "id": inviter_id,
                    "username": None,
                    "referrer_id": None,
                    "joined_at": None,
                    "referral_earned_total": None,
                },
            )

        for user_id, joined in joined_at_map.items():
            record = user_records.get(user_id)
            if record:
                record["joined_at"] = joined

        for referrer_id, total in earned_totals.items():
            record = user_records.setdefault(
                referrer_id,
                {
                    "id": referrer_id,
                    "username": None,
                    "referrer_id": None,
                    "joined_at": None,
                    "referral_earned_total": None,
                },
            )
            record["referral_earned_total"] = max(total, int(record.get("referral_earned_total") or 0))

        # Prepare batches
        user_batches: List[Dict[str, Any]] = list(user_records.values())
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
        async def _chunked_execute(items: List[Dict[str, Any]], executor) -> int:
            total = 0
            for idx in range(0, len(items), max(1, batch_size)):
                chunk = items[idx : idx + max(1, batch_size)]
                total += await executor(engine, chunk)
            return total

        stats.users_imported = await _chunked_execute(user_batches, _execute_users)
        stats.balances_imported = await _chunked_execute(balance_batch, _execute_balances)
        stats.referrals_imported = await _chunked_execute(referral_batch, _execute_referrals)

        stats.skipped_entries = len(skipped)
        stats.errors = skipped

        log.info(
            "redis.migration.finished | users=%s balances=%s referrals=%s skipped=%s",
            stats.users_imported,
            stats.balances_imported,
            stats.referrals_imported,
            stats.skipped_entries,
        )
        for line in stats.as_lines():
            log.info(line)
        if stats.skipped_entries:
            for entry in skipped:
                log.warning("redis.migration.skipped | %s", entry)
    finally:
        try:
            await conn.close()
        except Exception:  # pragma: no cover
            pass
        engine.dispose()

    return stats


def _configure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")


def main() -> None:
    _configure_logging()
    try:
        stats = asyncio.run(migrate_from_redis())
    except Exception as exc:  # pragma: no cover - CLI failure path
        log.exception("redis.migration.failed | err=%s", exc)
        raise
    for line in stats.as_lines():
        print(line)
    if stats.errors:
        print("⚠️ Skipped entries:")
        for entry in stats.errors:
            print(f" - {entry}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
