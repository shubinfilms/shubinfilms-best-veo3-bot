"""Flask blueprint for Suno callbacks."""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Dict

from flask import Blueprint, Response, current_app, jsonify, request

from .service import SunoService
from .store import InMemoryTaskStore, TaskStore

logger = logging.getLogger("suno.callbacks")

suno_bp = Blueprint("suno", __name__)


@lru_cache
def _default_service() -> SunoService:
    store: TaskStore = InMemoryTaskStore()
    return SunoService(store=store)


def _get_service() -> SunoService:
    service = current_app.config.get("SUNO_SERVICE") if current_app else None
    if isinstance(service, SunoService):
        return service
    return _default_service()


def _handle(service_method: str, payload: Dict[str, Any]) -> Response:
    service = _get_service()
    handler = getattr(service, service_method)
    handler(payload)
    return jsonify({"status": "received"})


@suno_bp.route("/music", methods=["POST"])
def music_callback() -> Response:
    payload = request.get_json(silent=True, force=False) or {}
    return _handle("handle_music_callback", payload)


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
