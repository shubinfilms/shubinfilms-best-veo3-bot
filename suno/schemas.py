"""Pydantic models shared between the Suno bot and callback worker."""
from __future__ import annotations

from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field


class SunoTrack(BaseModel):
    """Normalized representation of a single track."""

    model_config = ConfigDict(extra="ignore")

    id: str
    title: str | None = None
    audio_url: str | None = None
    image_url: str | None = None


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
        task_id = _first(data, "task_id", "taskId", "id") or ""
        callback_type = (
            _first(data, "callback_type", "callbackType") or "unknown"
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
    for key in ("data", "items", "tracks"):
        maybe = data.get(key)
        if maybe:
            return _ensure_iterable(maybe)
    response = data.get("response")
    if isinstance(response, dict):
        for key in ("data", "items", "tracks"):
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
    audio_url = _first(raw, "audio_url", "audioUrl", "url", "fileUrl", "mp3Url")
    image_url = _first(raw, "image_url", "imageUrl", "coverUrl", "imgUrl")
    return SunoTrack(
        id=str(track_id),
        title=str(title) if title is not None else None,
        audio_url=str(audio_url) if audio_url else None,
        image_url=str(image_url) if image_url else None,
    )


__all__ = ["CallbackEnvelope", "SunoTask", "SunoTrack"]
