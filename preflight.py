#!/usr/bin/env python3
"""Environment and connectivity preflight checks for Best VEO3 Bot."""

from __future__ import annotations

import os
import json
import time
import logging
from typing import Optional

import requests
from dotenv import load_dotenv

try:
    import redis  # type: ignore
except ImportError:  # pragma: no cover - redis is required in runtime
    redis = None  # type: ignore


LOG = logging.getLogger("preflight")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

OPTIONAL_TOKEN_KEYS = ["TELEGRAM_TOKEN"]
ADMIN_FALLBACK_KEYS = ["ADMIN_ID"]
DB_ENV_KEYS = ["DATABASE_URL", "POSTGRES_DSN"]
LEDGER_BACKEND_KEY = "LEDGER_BACKEND"


def _load_env() -> None:
    """Load .env file if present."""

    load_dotenv(override=False)


class CheckError(RuntimeError):
    """Custom error with friendly output."""


def _require_env(name: str, *, allow_fallback: Optional[list[str]] = None) -> str:
    keys = [name]
    if allow_fallback:
        keys.extend(allow_fallback)
    for key in keys:
        value = os.getenv(key)
        if value is not None and value.strip():
            if key != name:
                LOG.info("Using %s from %s", name, key)
            return value.strip()
    raise CheckError(f"Environment variable {name} is required")


def _resolve_db_url() -> Optional[str]:
    for key in DB_ENV_KEYS:
        value = os.getenv(key)
        if value and value.strip():
            return value.strip()
    backend = (os.getenv(LEDGER_BACKEND_KEY) or "").strip().lower()
    if backend and backend != "memory":
        raise CheckError("DATABASE_URL required (set LEDGER_BACKEND=memory to skip Postgres)")
    os.environ.setdefault(LEDGER_BACKEND_KEY, "memory")
    return None


def _check_postgres(dsn: str) -> str:
    try:
        import psycopg  # type: ignore
    except ImportError:  # pragma: no cover - fallback for psycopg2
        psycopg = None  # type: ignore
    if psycopg is not None:
        try:
            with psycopg.connect(dsn, connect_timeout=5) as conn:  # type: ignore[attr-defined]
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
            return "connected"
        except Exception as exc:
            raise CheckError(f"Postgres connection failed: {exc}") from exc
    try:
        import psycopg2  # type: ignore
    except ImportError as exc:  # pragma: no cover - psycopg2 missing
        raise CheckError("psycopg/psycopg2 is not installed") from exc
    try:
        conn = psycopg2.connect(dsn, connect_timeout=5)  # type: ignore[attr-defined]
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.fetchone()
        cur.close()
        conn.close()
        return "connected"
    except Exception as exc:
        raise CheckError(f"Postgres connection failed: {exc}") from exc


def _check_redis(url: str) -> str:
    if redis is None:
        raise CheckError("redis package is not installed")
    try:
        client = redis.Redis.from_url(url, socket_connect_timeout=5, socket_timeout=5)  # type: ignore[attr-defined]
        pong = client.ping()
        return "ok" if pong else "no pong"
    except Exception as exc:
        raise CheckError(f"Redis connection failed: {exc}") from exc


def _send_test_message(token: str, chat_id: str) -> str:
    payload = {
        "chat_id": chat_id,
        "text": f"Preflight OK at {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "disable_notification": True,
    }
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        resp = requests.post(url, json=payload, timeout=10)
    except requests.RequestException as exc:
        raise CheckError(f"Telegram sendMessage failed: {exc}") from exc
    if resp.status_code != 200:
        try:
            data = resp.json()
        except ValueError:
            data = {"error": resp.text}
        raise CheckError(f"Telegram sendMessage error: status={resp.status_code} resp={json.dumps(data, ensure_ascii=False)}")
    try:
        data = resp.json()
    except ValueError as exc:
        raise CheckError(f"Telegram sendMessage invalid JSON: {resp.text}") from exc
    if not data.get("ok"):
        raise CheckError(f"Telegram sendMessage returned ok=false: {json.dumps(data, ensure_ascii=False)}")
    return f"message_id={data.get('result', {}).get('message_id')}"


def main() -> int:
    _load_env()

    try:
        token = _require_env("TELEGRAM_BOT_TOKEN", allow_fallback=OPTIONAL_TOKEN_KEYS)
        admin_chat = _require_env("ADMIN_CHAT_ID", allow_fallback=ADMIN_FALLBACK_KEYS)
        redis_url = _require_env("REDIS_URL")
        _require_env("OPENAI_API_KEY")
        _require_env("KIE_API_KEY")

        db_url = _resolve_db_url()

        db_status = "skipped (memory mode)"
        if db_url:
            db_status = _check_postgres(db_url)

        redis_status = _check_redis(redis_url)
        telegram_status = _send_test_message(token, admin_chat)

    except CheckError as exc:
        LOG.error("%s", exc)
        return 1

    LOG.info("Preflight succeeded: db=%s, redis=%s, telegram=%s", db_status, redis_status, telegram_status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
