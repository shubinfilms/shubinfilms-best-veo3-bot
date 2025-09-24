"""In-memory task store for Suno callbacks."""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Tuple

from .schemas import TaskAsset


class TaskStore(ABC):
    """Interface for callback idempotency and task tracking."""

    @abstractmethod
    def save_event(self, task_id: str, callback_type: str, payload: dict) -> bool:
        """Persist callback payload and return True if it is the first time."""

    @abstractmethod
    def is_processed(self, task_id: str, callback_type: str) -> bool:
        """Return whether callback with the same type was processed."""

    @abstractmethod
    def upsert_assets(self, task_id: str, assets: List[dict]) -> None:
        """Store asset descriptors for a task."""

    @abstractmethod
    def record_error(self, task_id: str, error: dict) -> None:
        """Persist error payload for further inspection."""

    @abstractmethod
    def get_task_summary(self, task_id: str) -> dict:
        """Return stored events/assets/errors for the task."""


class InMemoryTaskStore(TaskStore):
    """Thread-safe in-memory implementation of :class:`TaskStore`."""

    def __init__(self) -> None:
        self._events: Dict[Tuple[str, str], dict] = {}
        self._assets: Dict[str, Dict[str, TaskAsset]] = defaultdict(dict)
        self._errors: Dict[str, List[dict]] = defaultdict(list)
        self._lock = threading.RLock()

    def save_event(self, task_id: str, callback_type: str, payload: dict) -> bool:
        key = (task_id, callback_type)
        with self._lock:
            if key in self._events:
                return False
            self._events[key] = payload
            return True

    def is_processed(self, task_id: str, callback_type: str) -> bool:
        key = (task_id, callback_type)
        with self._lock:
            return key in self._events

    def upsert_assets(self, task_id: str, assets: List[dict]) -> None:
        with self._lock:
            bucket = self._assets[task_id]
            for asset in assets:
                normalized = TaskAsset(**asset)
                key = f"{normalized.asset_type}:{normalized.identifier or normalized.url}"
                bucket[key] = normalized

    def record_error(self, task_id: str, error: dict) -> None:
        with self._lock:
            self._errors[task_id].append(error)

    def get_task_summary(self, task_id: str) -> dict:
        with self._lock:
            events = {
                ctype: self._events[(task_id, ctype)]
                for (tid, ctype) in self._events
                if tid == task_id
            }
            assets = [asset.model_dump() for asset in self._assets.get(task_id, {}).values()]
            errors = list(self._errors.get(task_id, []))
        return {"events": events, "assets": assets, "errors": errors}
