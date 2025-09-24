"""Flask blueprint for Suno callbacks."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Literal, Optional

from flask import Blueprint, Response, current_app, jsonify, request

from .service import SunoService
from .store import InMemoryTaskStore, TaskStore

logger = logging.getLogger("suno.callbacks")

suno_bp = Blueprint("suno", __name__)


@dataclass(slots=True)
class Track:
    """Normalized representation of a track returned in callbacks."""

    audio_id: str
    audio_url: Optional[str] = None
    image_url: Optional[str] = None
    video_url: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MusicCallback:
    """Structured payload for music callbacks."""

    task_id: str
    type: Literal["text", "first", "complete", "error"]
    code: int
    msg: str
    tracks: List[Track] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)


_KNOWN_TYPES = {"text", "first", "complete", "error"}


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _extract_url(item: Any, keys: Iterable[str]) -> Optional[str]:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in keys:
            maybe = item.get(key)
            if maybe:
                return str(maybe)
    return None


def _coerce_track(item: Any, index: int) -> Optional[Track]:
    if isinstance(item, dict):
        audio_id = str(
            item.get("audioId")
            or item.get("id")
            or item.get("name")
            or item.get("trackId")
            or index
        )
        audio_url = _extract_url(item, ("audioUrl", "url", "fileUrl"))
        image_url = _extract_url(item, ("imageUrl", "coverUrl", "imgUrl"))
        video_url = _extract_url(item, ("videoUrl", "mp4Url"))
        return Track(audio_id=audio_id, audio_url=audio_url, image_url=image_url, video_url=video_url, raw=item)
    if isinstance(item, str):
        return Track(audio_id=str(index), audio_url=item, raw={"audioUrl": item})
    return None


def _merge_additional_assets(tracks: List[Track], candidates: List[Any], kind: str) -> None:
    if not candidates:
        return
    by_id = {track.audio_id: track for track in tracks}
    unmatched = []
    for entry in candidates:
        if isinstance(entry, dict):
            audio_id = entry.get("audioId") or entry.get("id") or entry.get("name")
            url = _extract_url(entry, ("imageUrl", "coverUrl", "url", "fileUrl", "mp4Url"))
            if audio_id and url and str(audio_id) in by_id:
                track = by_id[str(audio_id)]
                if kind == "image" and not track.image_url:
                    track.image_url = url
                    continue
                if kind == "video" and not track.video_url:
                    track.video_url = url
                    continue
            if url:
                unmatched.append(url)
        else:
            url = _extract_url(entry, ("",))
            if url:
                unmatched.append(url)
    for track, url in zip([t for t in tracks if (kind == "image" and not t.image_url) or (kind == "video" and not t.video_url)], unmatched):
        if kind == "image":
            track.image_url = url
        else:
            track.video_url = url


def _parse_tracks(response: Dict[str, Any]) -> List[Track]:
    if not isinstance(response, dict):
        return []
    tracks_field = _ensure_list(response.get("tracks"))
    tracks: List[Track] = []
    if tracks_field:
        for idx, item in enumerate(tracks_field):
            track = _coerce_track(item, idx)
            if track:
                tracks.append(track)
        return tracks

    audio_candidates: List[Any] = []
    image_candidates: List[Any] = []
    video_candidates: List[Any] = []
    for key, value in response.items():
        key_lower = str(key).lower()
        values = _ensure_list(value)
        if any(token in key_lower for token in ("audio", "mp3", "wav", "song")):
            audio_candidates.extend(values)
        elif any(token in key_lower for token in ("image", "cover", "thumbnail")):
            image_candidates.extend(values)
        elif any(token in key_lower for token in ("video", "mp4")):
            video_candidates.extend(values)

    for idx, item in enumerate(audio_candidates):
        track = _coerce_track(item, idx)
        if track:
            tracks.append(track)

    _merge_additional_assets(tracks, image_candidates, "image")
    _merge_additional_assets(tracks, video_candidates, "video")
    return tracks


def parse_music_callback(payload: Dict[str, Any]) -> MusicCallback:
    if not isinstance(payload, dict):
        return MusicCallback(task_id="", type="error", code=0, msg="invalid payload", tracks=[], raw={})
    data = payload.get("data") or {}
    task_id = str(data.get("taskId") or "")
    callback_type = str(data.get("callbackType") or "text").lower()
    code = _as_int(payload.get("code"), default=0)
    msg = str(payload.get("msg") or "")
    if callback_type not in _KNOWN_TYPES:
        callback_type = "error" if code and code != 200 else "text"
    tracks = _parse_tracks(data.get("response") or {})
    return MusicCallback(task_id=task_id, type=callback_type, code=code or 0, msg=msg, tracks=tracks, raw=payload)


@lru_cache
def _default_service() -> SunoService:
    store: TaskStore = InMemoryTaskStore()
    return SunoService(store=store)


def _get_service() -> SunoService:
    service = current_app.config.get("SUNO_SERVICE") if current_app else None
    if isinstance(service, SunoService):
        return service
    return _default_service()


def _handle(service_method: str, payload: Any) -> Response:
    service = _get_service()
    handler = getattr(service, service_method)
    handler(payload)
    return jsonify({"status": "received"})


@suno_bp.route("/music", methods=["POST"])
def music_callback() -> Response:
    payload = request.get_json(silent=True, force=False) or {}
    callback = parse_music_callback(payload)
    return _handle("handle_music_callback", callback)


@suno_bp.route("/wav", methods=["POST"])
def wav_callback() -> Response:
    payload = request.get_json(silent=True, force=False) or {}
    return _handle("handle_wav_callback", payload)


@suno_bp.route("/cover", methods=["POST"])
def cover_callback() -> Response:
    payload = request.get_json(silent=True, force=False) or {}
    return _handle("handle_cover_callback", payload)


@suno_bp.route("/vocal-separation", methods=["POST"])
def vocal_separation_callback() -> Response:
    payload = request.get_json(silent=True, force=False) or {}
    return _handle("handle_vocal_separation_callback", payload)


@suno_bp.route("/mp4", methods=["POST"])
def mp4_callback() -> Response:
    payload = request.get_json(silent=True, force=False) or {}
    return _handle("handle_mp4_callback", payload)
