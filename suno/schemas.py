"""Data-transfer objects for the Suno integration."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, List, Mapping, MutableMapping, Optional, Sequence


def _first(mapping: Mapping[str, Any], keys: Sequence[str]) -> Optional[Any]:
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return None


def _ensure_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


@dataclass(slots=True)
class SunoTrack:
    """Normalized track information returned by Suno."""

    id: str
    title: str
    audio_url: Optional[str] = None
    image_url: Optional[str] = None
    ext: Optional[str] = None

    @classmethod
    def from_payload(cls, payload: Any, *, fallback_id: str) -> Optional["SunoTrack"]:
        if payload is None:
            return None
        if isinstance(payload, str):
            return cls(id=fallback_id, title=f"Track {fallback_id}", audio_url=payload, image_url=None, ext=None)
        if not isinstance(payload, Mapping):
            return None
        data: MutableMapping[str, Any] = dict(payload)
        track_id = _first(data, ("id", "trackId", "track_id", "audioId", "songId")) or fallback_id
        title = _first(data, ("title", "name")) or f"Track {track_id}"
        audio_url = _first(
            data,
            (
                "audio_url",
                "audioUrl",
                "url",
                "audio",
                "mp3Url",
                "mp3_url",
                "fileUrl",
            ),
        )
        image_url = _first(data, ("image_url", "imageUrl", "coverUrl", "image", "cover"))
        ext = _first(data, ("ext", "extension", "audio_extension", "audioExt"))
        return cls(id=str(track_id), title=str(title), audio_url=_maybe_str(audio_url), image_url=_maybe_str(image_url), ext=_maybe_str(ext))


def _maybe_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


@dataclass(slots=True)
class SunoTask:
    """Normalized task representation used across the bot/web."""

    task_id: str
    status: str
    message: Optional[str] = None
    items: List[SunoTrack] = field(default_factory=list)

    @classmethod
    def from_payload(cls, payload: Any) -> "SunoTask":
        if not isinstance(payload, Mapping):
            return cls(task_id="", status="unknown", message=None, items=[])
        data = _ensure_mapping(payload.get("data"))
        response = _ensure_mapping(data.get("response"))

        task_id = _first(data, ("task_id", "taskId", "id")) or _first(payload, ("task_id", "taskId", "id")) or ""
        message = _first(payload, ("message", "msg", "detail", "error"))
        if not message:
            message = _first(data, ("message", "msg", "detail", "error"))

        status = _first(data, ("callbackType", "callback_type", "status", "state"))
        if not status:
            status = _first(payload, ("status", "state"))
        if not status:
            status = _first(response, ("status", "state"))
        status_text = str(status) if status is not None else "unknown"

        items_section: Iterable[Any] = []
        for source in (
            data.get("data"),
            data.get("items"),
            data.get("tracks"),
            data.get("songs"),
            response.get("data"),
            response.get("items"),
            response.get("tracks"),
        ):
            if source:
                items_section = _ensure_list(source)
                break
        else:
            fallback_items = _first(payload, ("items", "tracks"))
            if fallback_items:
                items_section = _ensure_list(fallback_items)

        tracks: List[SunoTrack] = []
        for idx, item in enumerate(items_section, start=1):
            track = SunoTrack.from_payload(item, fallback_id=str(idx))
            if track:
                tracks.append(track)

        return cls(task_id=str(task_id), status=status_text, message=_maybe_str(message), items=tracks)


__all__ = ["SunoTask", "SunoTrack"]
