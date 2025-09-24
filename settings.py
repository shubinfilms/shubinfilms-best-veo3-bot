"""Shared configuration constants for Redis and logging keys."""
import os

from dotenv import load_dotenv

load_dotenv()

_default_prefix = "veo3:prod"
_env_prefix = (os.getenv("REDIS_PREFIX") or "").strip()
REDIS_PREFIX = _env_prefix or _default_prefix

SUNO_LOG_KEY = f"{REDIS_PREFIX}:suno:logs"
