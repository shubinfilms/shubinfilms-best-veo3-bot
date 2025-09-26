"""Helpers for managing temporary Suno asset files."""
from __future__ import annotations

import logging
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Optional

from settings import TMP_CLEANUP_HOURS

log = logging.getLogger("suno.tempfiles")

BASE_DIR = Path("/tmp/suno")
_CLEAN_DELAY = 60.0


def _sanitize_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", value)
    return cleaned.strip("._") or "file"


def task_directory(task_id: Optional[str]) -> Path:
    """Return a sanitized directory path for a task and ensure it exists."""

    identifier = _sanitize_component(task_id or "task")
    target = BASE_DIR / identifier
    target.mkdir(parents=True, exist_ok=True)
    return target


def cleanup_old_directories(now: Optional[float] = None) -> None:
    """Remove directories older than the configured retention window."""

    timestamp = now if now is not None else time.time()
    cutoff = timestamp - max(1, TMP_CLEANUP_HOURS) * 3600
    if not BASE_DIR.exists():
        return
    for entry in BASE_DIR.iterdir():
        try:
            stat = entry.stat()
        except FileNotFoundError:  # pragma: no cover - race with other cleanup
            continue
        if not entry.is_dir():
            continue
        if stat.st_mtime > cutoff:
            continue
        try:
            shutil.rmtree(entry, ignore_errors=True)
        except Exception:  # pragma: no cover - defensive
            log.warning("failed to cleanup directory", extra={"meta": {"path": str(entry)}})


def schedule_unlink(path: Path, delay: float = _CLEAN_DELAY) -> None:
    """Schedule removal of a file after a delay."""

    def _remove() -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            return
        except Exception:  # pragma: no cover - defensive
            log.debug("unable to delete file", extra={"meta": {"path": str(path)}})
        parent = path.parent
        try:
            if parent != BASE_DIR and parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
        except Exception:  # pragma: no cover - defensive
            pass

    timer = threading.Timer(max(0.0, delay), _remove)
    timer.daemon = True
    timer.start()


__all__ = ["BASE_DIR", "task_directory", "cleanup_old_directories", "schedule_unlink"]
