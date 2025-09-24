"""Business logic for handling Suno callbacks."""
from __future__ import annotations

import logging
import os
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, List, Optional

from .downloader import download_file
from .schemas import CallbackEnvelope, TaskAsset
from .store import TaskStore

logger = logging.getLogger("suno.service")

_DEFAULT_DOWNLOAD_DIR = Path(os.getenv("SUNO_DOWNLOAD_DIR", "downloads"))


class ImmediateFuture(Future):
    """Future that is resolved immediately for synchronous executors."""

    def __init__(self, result) -> None:  # pragma: no cover - defensive
        super().__init__()
        self.set_result(result)


class SunoService:
    """Service orchestrating callback handling and downloads."""

    def __init__(
        self,
        store: TaskStore,
        download_dir: Path | str = _DEFAULT_DOWNLOAD_DIR,
        executor: Optional[ThreadPoolExecutor] = None,
        downloader=download_file,
    ) -> None:
        self.store = store
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.downloader = downloader
        self.executor = executor or ThreadPoolExecutor(max_workers=4)

    def handle_music_callback(self, payload: dict) -> None:
        self._handle_callback("music", payload)

    def handle_wav_callback(self, payload: dict) -> None:
        self._handle_callback("wav", payload)

    def handle_cover_callback(self, payload: dict) -> None:
        self._handle_callback("cover", payload)

    def handle_vocal_separation_callback(self, payload: dict) -> None:
        self._handle_callback("vocal", payload)

    def handle_mp4_callback(self, payload: dict) -> None:
        self._handle_callback("mp4", payload)

    def _handle_callback(self, task_kind: str, payload: dict) -> None:
        envelope = self._parse_payload(payload)
        if envelope is None or envelope.data is None:
            logger.warning("Received invalid callback payload for %s", task_kind)
            return
        task_id = envelope.data.task_id
        callback_type = (envelope.data.callback_type or "unknown").lower()
        if not task_id:
            logger.warning("Callback without task_id for %s", task_kind)
            return
        if self.store.is_processed(task_id, callback_type):
            logger.debug("Duplicate callback %s/%s", task_id, callback_type)
            return
        saved = self.store.save_event(task_id, callback_type, payload)
        if not saved:
            logger.debug("Callback already saved %s/%s", task_id, callback_type)
            return
        if envelope.code and envelope.code >= 400:
            self.store.record_error(task_id, {"code": envelope.code, "message": envelope.msg})
            return
        if callback_type == "error":
            error_payload = envelope.data.model_dump() if envelope.data else {}
            self.store.record_error(task_id, error_payload)
            return
        assets = self._extract_assets(task_id, envelope)
        if assets:
            self.store.upsert_assets(task_id, [asset.model_dump() for asset in assets])
            self._download_assets(task_id, assets)
        else:
            logger.debug("No assets found for callback %s/%s", task_id, callback_type)

    def _parse_payload(self, payload: dict) -> Optional[CallbackEnvelope]:
        if not isinstance(payload, dict):
            return None
        try:
            return CallbackEnvelope.model_validate(payload)
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to parse callback payload")
            return None

    def _extract_assets(self, task_id: str, envelope: CallbackEnvelope) -> List[TaskAsset]:
        data = envelope.data
        if data is None or data.response is None:
            return []
        response = data.response.model_dump()
        assets: List[TaskAsset] = []
        callback_type = (data.callback_type or "").lower()
        for asset_type, values in self._iter_candidate_assets(response):
            for idx, item in enumerate(values):
                asset = self._coerce_asset(task_id, asset_type, item, idx, callback_type)
                if asset:
                    assets.append(asset)
        return assets

    def _iter_candidate_assets(self, response: dict) -> Iterable[tuple[str, List]]:
        candidates = {
            "audio": [],
            "image": [],
            "video": [],
            "stem": [],
        }
        for key, value in response.items():
            if value is None:
                continue
            key_lower = key.lower()
            if any(token in key_lower for token in ["audio", "mp3", "wav"]):
                candidates["audio"].extend(self._normalize_to_list(value))
            elif any(token in key_lower for token in ["image", "cover"]):
                candidates["image"].extend(self._normalize_to_list(value))
            elif any(token in key_lower for token in ["video", "mp4"]):
                candidates["video"].extend(self._normalize_to_list(value))
            elif "stem" in key_lower or "vocal" in key_lower:
                candidates["stem"].extend(self._normalize_to_list(value))
        return [(kind, values) for kind, values in candidates.items() if values]

    def _normalize_to_list(self, value) -> List:
        if isinstance(value, list):
            return value
        return [value]

    def _coerce_asset(
        self,
        task_id: str,
        asset_type: str,
        item,
        index: int,
        callback_type: str,
    ) -> Optional[TaskAsset]:
        if isinstance(item, str):
            url = item
            identifier = str(index)
        elif isinstance(item, dict):
            url = item.get("url") or item.get("audioUrl") or item.get("imageUrl") or item.get("mp4Url") or item.get("fileUrl")
            identifier = (
                item.get("audioId")
                or item.get("id")
                or item.get("name")
                or item.get("stemName")
                or item.get("stemId")
                or str(index)
            )
        else:
            return None
        if not url:
            return None
        filename = f"{callback_type or asset_type}_{asset_type}_{identifier}"
        return TaskAsset(task_id=task_id, url=url, asset_type=asset_type, identifier=str(identifier), filename=filename)

    def _download_assets(self, task_id: str, assets: List[TaskAsset]) -> None:
        for asset in assets:
            relative_path = Path(task_id) / asset.filename
            self.executor.submit(self.downloader, asset.url, relative_path, self.download_dir)
