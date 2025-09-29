"""Pydantic models shared between the Suno bot and callback worker."""
from __future__ import annotations

from typing import Any, Iterable, List, Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field


class EnqueueData(BaseModel):
    """Minimal payload returned when a generation request is accepted."""

    taskId: str = Field(..., alias="taskId")


class ApiEnvelope(BaseModel):
    """Common KIE API envelope used by the lightweight Suno endpoints."""

    code: int
    msg: Optional[str] = None
    data: Optional[Any] = None


class Track(BaseModel):
    """Single track description delivered via the callback."""

    audio_url: Optional[str] = None
    stream_audio_url: Optional[str] = None
    image_url: Optional[str] = None
    title: Optional[str] = None
    tags: Optional[str] = None
    duration: Optional[float] = None


class CallbackData(BaseModel):
    """Callback payload published by the KIE webhook."""

    callbackType: Literal["text", "first", "complete", "error"]
    task_id: Optional[str] = None
    data: List[Track] = Field(default_factory=list)


class SunoCallback(BaseModel):
    """Top-level callback wrapper."""

    code: int
    msg: Optional[str] = None
    data: CallbackData


class SunoTrack(BaseModel):
    """Normalized representation of a single track."""

    model_config = ConfigDict(extra="ignore")

    id: str
    title: str | None = None
    audio_url: str | None = None
    image_url: str | None = None
    tags: str | None = None
    duration: float | None = None
    source_audio_url: str | None = None
    source_image_url: str | None = None


class SunoTask(BaseModel):
    """Structured payload that the service operates on."""

    model_config = ConfigDict(extra="ignore")

    task_id: str
    callback_type: str
    items: list[SunoTrack] = Field(default_factory=list)
    msg: str | None = None
    code: int | None = None

    @classmethod
    def from_envelope(cls, envelope: "CallbackEnvelope") -> "SunoTask":
        data = envelope.data or {}
        task_id = _first(data, "task_id", "taskId", "taskID", "id") or ""
        callback_type = (
            _first(data, "callback_type", "callbackType", "status", "type") or "unknown"
        )
        raw_items = _extract_items(data)
        tracks = [_build_track(item, index) for index, item in enumerate(raw_items, start=1)]
        filtered_tracks = [track for track in tracks if track is not None]
        return cls(
            task_id=str(task_id),
            callback_type=str(callback_type or "unknown"),
            items=filtered_tracks,
            msg=envelope.msg,
            code=envelope.code,
        )


class CallbackEnvelope(BaseModel):
    """Raw structure coming from the Suno webhook."""

    model_config = ConfigDict(extra="allow")

    code: int | None = None
    msg: str | None = None
    data: dict = Field(default_factory=dict)


def _first(data: dict[str, Any], *keys: str) -> Any | None:
    for key in keys:
        if key in data and data[key] not in (None, ""):
            return data[key]
    return None


def _ensure_iterable(value: Any) -> Iterable[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return value
    return [value]


def _extract_items(data: dict[str, Any]) -> Iterable[Any]:
    for key in ("sunoData", "tracks", "items", "results", "data"):
        maybe = data.get(key)
        if maybe:
            return _ensure_iterable(maybe)
    input_section = data.get("input")
    if isinstance(input_section, Mapping):
        for key in ("tracks", "items", "results", "data"):
            maybe = input_section.get(key)
            if maybe:
                return _ensure_iterable(maybe)
    response = data.get("response")
    if isinstance(response, dict):
        for key in ("data", "items", "results", "tracks"):
            maybe = response.get(key)
            if maybe:
                return _ensure_iterable(maybe)
    return []


def _build_track(raw: Any, index: int) -> SunoTrack | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        return SunoTrack(id=str(index), title=None, audio_url=raw, image_url=None)
    if not isinstance(raw, dict):
        return None
    track_id = _first(raw, "id", "trackId", "audioId", "songId") or str(index)
    title = _first(raw, "title", "name")
    source_audio_url = _first(raw, "sourceAudioUrl", "audio_url", "audioUrl", "url", "fileUrl", "mp3Url")
    source_image_url = _first(raw, "sourceImageUrl", "image_url", "imageUrl", "coverUrl", "imgUrl")
    audio_url = _first(raw, "audio_url", "audioUrl", "url", "fileUrl", "mp3Url") or source_audio_url
    image_url = _first(raw, "image_url", "imageUrl", "coverUrl", "imgUrl") or source_image_url
    tags_value = _first(raw, "tags", "tag", "style", "styles")
    if isinstance(tags_value, list):
        tags = ", ".join(str(item) for item in tags_value if item not in (None, "")) or None
    elif isinstance(tags_value, str):
        tags = tags_value
    else:
        tags = None
    duration_value = _first(raw, "duration", "durationSec", "duration_seconds")
    duration: float | None
    if isinstance(duration_value, (int, float)):
        duration = float(duration_value)
    elif isinstance(duration_value, str):
        try:
            duration = float(duration_value)
        except ValueError:
            duration = None
    else:
        duration = None
    return SunoTrack(
        id=str(track_id),
        title=str(title) if title is not None else None,
        audio_url=str(audio_url) if audio_url else None,
        image_url=str(image_url) if image_url else None,
        tags=tags,
        duration=duration,
        source_audio_url=str(source_audio_url) if source_audio_url else None,
        source_image_url=str(source_image_url) if source_image_url else None,
    )


__all__ = [
    "CallbackEnvelope",
    "SunoTask",
    "SunoTrack",
    "EnqueueData",
    "ApiEnvelope",
    "Track",
    "CallbackData",
    "SunoCallback",
]
