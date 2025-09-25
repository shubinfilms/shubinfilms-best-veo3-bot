"""Business logic for handling Suno callbacks."""
from __future__ import annotations

import logging
import os
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Optional, Sequence

from settings import (
    SUNO_CALLBACK_URL,
    SUNO_GEN_PATH,
    SUNO_INSTR_PATH,
    SUNO_MODEL,
    SUNO_TASK_STATUS_PATH,
    SUNO_UPLOAD_EXTEND_PATH,
)

from .client import SunoHttp
from .downloader import download_file
from .schemas import CallbackEnvelope, SunoTask, SunoTrack, TaskAsset
from .store import TaskStore

if TYPE_CHECKING:  # pragma: no cover - only used for typing
    from .callbacks import MusicCallback, Track

logger = logging.getLogger("suno.service")


def _warn_env() -> None:
    base = os.getenv("SUNO_API_BASE", "")
    gen = os.getenv("SUNO_GEN_PATH", "")
    if "suno-api" in (base or "") or "suno-api" in (gen or ""):
        logging.warning(
            "ENV contains 'suno-api' in BASE or PATH. We normalize it, but please remove the segment from ENV.",
        )


_warn_env()

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
        http: Optional[SunoHttp] = None,
    ) -> None:
        self.store = store
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.downloader = downloader
        self.executor = executor or ThreadPoolExecutor(max_workers=4)
        self._http_client: Optional[SunoHttp] = http

    def handle_music_callback(self, payload: Any) -> None:
        self._handle_callback("music", payload)

    def handle_wav_callback(self, payload: dict) -> None:
        self._handle_callback("wav", payload)

    def handle_cover_callback(self, payload: dict) -> None:
        self._handle_callback("cover", payload)

    def handle_vocal_separation_callback(self, payload: dict) -> None:
        self._handle_callback("vocal", payload)

    def handle_mp4_callback(self, payload: dict) -> None:
        self._handle_callback("mp4", payload)

    def _get_http(self) -> SunoHttp:
        if self._http_client is None:
            self._http_client = SunoHttp()
        return self._http_client

    def _handle_callback(self, task_kind: str, payload: Any) -> None:
        raw_payload: dict | None
        tracks: List["Track"] = []
        code_override: Optional[int] = None
        msg_override: Optional[str] = None
        type_override: Optional[str] = None
        task_id_override: Optional[str] = None
        if hasattr(payload, "raw") and isinstance(getattr(payload, "raw"), dict):
            raw_payload = getattr(payload, "raw")
        elif isinstance(payload, dict):
            raw_payload = payload
        else:
            raw_payload = None
        if hasattr(payload, "tracks"):
            tracks = list(getattr(payload, "tracks") or [])
        if hasattr(payload, "code"):
            code_override = getattr(payload, "code")
        if hasattr(payload, "msg"):
            msg_override = getattr(payload, "msg")
        if hasattr(payload, "type"):
            type_override = getattr(payload, "type")
        if hasattr(payload, "task_id"):
            task_id_override = getattr(payload, "task_id")

        envelope = self._parse_payload(raw_payload)
        if envelope is None or envelope.data is None:
            logger.warning("Received invalid callback payload for %s", task_kind)
            return
        task_id = task_id_override or envelope.data.task_id
        callback_type = (type_override or envelope.data.callback_type or "unknown").lower()
        if not task_id:
            logger.warning("Callback without task_id for %s", task_kind)
            return
        if self.store.is_processed(task_id, callback_type):
            logger.debug("Duplicate callback %s/%s", task_id, callback_type)
            return
        saved = self.store.save_event(task_id, callback_type, raw_payload or {})
        if not saved:
            logger.debug("Callback already saved %s/%s", task_id, callback_type)
            return
        code = code_override if code_override is not None else envelope.code or 0
        message = msg_override or envelope.msg or ""
        if code and code != 200:
            self.store.record_error(task_id, {"code": code, "message": message})
            return
        if callback_type == "error":
            error_payload = envelope.data.model_dump() if envelope.data else {}
            if message:
                error_payload["message"] = message
            if code:
                error_payload["code"] = code
            self.store.record_error(task_id, error_payload)
            return
        assets = self._deduplicate_assets(
            [
                *self._assets_from_tracks(task_id, tracks),
                *self._extract_assets(task_id, envelope),
            ]
        )
        if assets:
            new_assets = self.store.upsert_assets(task_id, [asset.model_dump() for asset in assets])
            if new_assets:
                self._download_assets(task_id, new_assets)
            else:
                logger.debug("No new assets to download for %s/%s", task_id, callback_type)
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

    def _assets_from_tracks(
        self,
        task_id: str,
        tracks: Sequence["Track"],
    ) -> List[TaskAsset]:
        assets: List[TaskAsset] = []
        for idx, track in enumerate(tracks):
            identifier = getattr(track, "audio_id", None) or str(idx)
            audio_url = getattr(track, "audio_url", None)
            image_url = getattr(track, "image_url", None)
            video_url = getattr(track, "video_url", None)
            if audio_url:
                assets.append(
                    TaskAsset(
                        task_id=task_id,
                        url=audio_url,
                        asset_type="audio",
                        identifier=str(identifier),
                        filename=f"{identifier}.mp3",
                    )
                )
            if image_url:
                assets.append(
                    TaskAsset(
                        task_id=task_id,
                        url=image_url,
                        asset_type="image",
                        identifier=f"{identifier}_image",
                        filename=f"{identifier}.jpeg",
                    )
                )
            if video_url:
                assets.append(
                    TaskAsset(
                        task_id=task_id,
                        url=video_url,
                        asset_type="video",
                        identifier=f"{identifier}_video",
                        filename=f"{identifier}.mp4",
                    )
                )
        return assets

    def _deduplicate_assets(self, assets: Iterable[TaskAsset]) -> List[TaskAsset]:
        seen = set()
        unique: List[TaskAsset] = []
        for asset in assets:
            key = (asset.asset_type, asset.identifier or asset.url)
            if key in seen:
                continue
            seen.add(key)
            unique.append(asset)
        return unique

    def _download_assets(self, task_id: str, assets: Sequence[TaskAsset]) -> None:
        for asset in assets:
            filename = asset.filename or f"{asset.asset_type}_{asset.identifier or 'asset'}"
            relative_path = Path(task_id) / filename
            self.executor.submit(self.downloader, asset.url, relative_path, self.download_dir)

    # ------------------------------------------------------------------ HTTP
    def _resolve_path(self, value: Optional[str], env_name: str) -> str:
        if not value:
            raise RuntimeError(f"{env_name} is not configured")
        if value.startswith("http"):
            return value
        return value if value.startswith("/") else "/" + value.lstrip("/")

    def _with_callback(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        callback_url = SUNO_CALLBACK_URL
        if callback_url and not any(key in payload for key in ("callback_url", "callbackUrl")):
            payload["callback_url"] = callback_url
        return payload

    def _clean_payload(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        cleaned: Dict[str, Any] = {}
        for key, value in payload.items():
            if value is None:
                continue
            cleaned[key] = value
        return cleaned

    def _normalize_task(self, payload: Optional[Mapping[str, Any]]) -> SunoTask:
        if not isinstance(payload, Mapping):
            return SunoTask(code=0, message=None, task_id=None, items=[])
        code = payload.get("code") or payload.get("status") or 0
        try:
            code_int = int(code)
        except (TypeError, ValueError):
            code_int = 0
        message = None
        for key in ("message", "msg", "detail"):
            raw_msg = payload.get(key)
            if isinstance(raw_msg, str) and raw_msg:
                message = raw_msg
                break
        data = payload.get("data") if isinstance(payload.get("data"), Mapping) else {}
        task_id = None
        if isinstance(data, Mapping):
            for key in ("taskId", "task_id", "id"):
                value = data.get(key)
                if value:
                    task_id = str(value)
                    break
            if not message:
                data_msg = data.get("message") or data.get("msg")
                if isinstance(data_msg, str):
                    message = data_msg
        if not task_id:
            for key in ("taskId", "task_id", "id"):
                value = payload.get(key)
                if value:
                    task_id = str(value)
                    break
        items_raw: Any = None
        if isinstance(data, Mapping):
            for key in ("items", "tracks", "songs"):
                maybe = data.get(key)
                if maybe:
                    items_raw = maybe
                    break
            if items_raw is None:
                response = data.get("response")
                if isinstance(response, Mapping):
                    for key in ("tracks", "items"):
                        maybe = response.get(key)
                        if maybe:
                            items_raw = maybe
                            break
        items: List[SunoTrack] = []
        if isinstance(items_raw, list):
            for entry in items_raw:
                track = self._coerce_track_entry(entry)
                if track:
                    items.append(track)
        elif isinstance(items_raw, Mapping):
            track = self._coerce_track_entry(items_raw)
            if track:
                items.append(track)
        return SunoTask(code=code_int, message=message, task_id=task_id, items=items)

    def _coerce_track_entry(self, entry: Any) -> Optional[SunoTrack]:
        if entry is None:
            return None
        if isinstance(entry, str):
            return SunoTrack(audio_url=entry)
        if not isinstance(entry, Mapping):
            return None
        payload: Dict[str, Any] = {}
        for key in ("id", "trackId", "track_id", "audioId"):
            value = entry.get(key)
            if value:
                payload["id"] = str(value)
                break
        for key in ("title", "name"):
            value = entry.get(key)
            if value:
                payload["title"] = str(value)
                break
        for key in ("audio_url", "audioUrl", "url", "fileUrl"):
            value = entry.get(key)
            if value:
                payload["audio_url"] = str(value)
                break
        for key in ("image_url", "imageUrl", "coverUrl"):
            value = entry.get(key)
            if value:
                payload["image_url"] = str(value)
                break
        for key in ("duration_ms", "durationMs", "duration"):
            value = entry.get(key)
            if value is None:
                continue
            try:
                payload["duration_ms"] = int(float(value))
            except (TypeError, ValueError):
                continue
            break
        if not payload:
            return None
        return SunoTrack.model_validate(payload)

    def create_music(
        self,
        *,
        title: Optional[str] = None,
        style: Optional[str] = None,
        lyrics: Optional[str] = None,
        model: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> SunoTask:
        payload = {
            "title": title,
            "style": style,
            "lyrics": lyrics,
            "model": model or SUNO_MODEL,
            "mode": mode,
        }
        payload = self._with_callback(self._clean_payload(payload))
        response = self._get_http().post(
            self._resolve_path(SUNO_GEN_PATH, "SUNO_GEN_PATH"),
            json=payload,
            op="create_music",
        )
        return self._normalize_task(response)

    def add_instrumental(
        self,
        *,
        task_id: Optional[str] = None,
        upload_url: Optional[str] = None,
        **extra: Any,
    ) -> SunoTask:
        if not task_id and not upload_url:
            raise ValueError("Either task_id or upload_url must be provided")
        payload = {
            "task_id": task_id,
            "upload_url": upload_url,
            **{key: value for key, value in extra.items() if value is not None},
        }
        payload = self._with_callback(self._clean_payload(payload))
        response = self._get_http().post(
            self._resolve_path(SUNO_INSTR_PATH, "SUNO_INSTR_PATH"),
            json=payload,
            op="add_instrumental",
        )
        return self._normalize_task(response)

    def extend_upload(
        self,
        file_url: str,
        *,
        start_at: int = 0,
        **extra: Any,
    ) -> SunoTask:
        if not file_url:
            raise ValueError("file_url must be provided")
        payload = {
            "file_url": file_url,
            "start_at": start_at,
            **{key: value for key, value in extra.items() if value is not None},
        }
        payload = self._with_callback(self._clean_payload(payload))
        response = self._get_http().post(
            self._resolve_path(SUNO_UPLOAD_EXTEND_PATH, "SUNO_UPLOAD_EXTEND_PATH"),
            json=payload,
            op="extend_upload",
        )
        return self._normalize_task(response)

    def get_status(self, task_id: str) -> SunoTask:
        if not task_id:
            raise ValueError("task_id must be provided")
        response = self._get_http().get(
            self._resolve_path(SUNO_TASK_STATUS_PATH, "SUNO_TASK_STATUS_PATH"),
            params={"task_id": task_id},
            op="get_status",
        )
        return self._normalize_task(response)
