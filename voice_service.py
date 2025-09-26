"""Voice transcription service backed by OpenAI Whisper."""
from __future__ import annotations

import io
import logging
import os
import random
import time
from typing import Optional

try:  # pragma: no cover - optional runtime dependency
    import openai  # type: ignore
except Exception:  # pragma: no cover - library might be unavailable
    openai = None  # type: ignore

log = logging.getLogger("voice.service")

_OPENAI_TIMEOUT = 30
_MAX_ATTEMPTS = 3
_RETRY_CODES = {429}
_MODEL_NAME = os.getenv("VOICE_MODEL", "whisper-1")


class VoiceTranscribeError(Exception):
    """Raised when audio transcription fails."""


def _filename_from_mime(mime: Optional[str]) -> str:
    if not mime:
        return "audio.wav"
    if "wav" in mime:
        return "audio.wav"
    if "mpeg" in mime:
        return "audio.mp3"
    if "ogg" in mime or "opus" in mime:
        return "audio.ogg"
    if "m4a" in mime or "mp4" in mime:
        return "audio.m4a"
    if "webm" in mime:
        return "audio.webm"
    return "audio.bin"


def _should_retry(exc: Exception) -> bool:
    if openai is None:  # pragma: no cover - defensive
        return False
    from openai.error import APIError, RateLimitError, ServiceUnavailableError  # type: ignore

    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, ServiceUnavailableError):
        return True
    if isinstance(exc, APIError):
        status = getattr(exc, "http_status", None)
        if status is None:
            return False
        return int(status) >= 500 or int(status) in _RETRY_CODES
    status = getattr(exc, "http_status", None)
    if isinstance(status, int) and (status in _RETRY_CODES or status >= 500):
        return True
    return False


def transcribe(audio_bytes: bytes, mime: Optional[str], lang_hint: Optional[str]) -> str:
    """Return a transcript for *audio_bytes* using Whisper."""

    if openai is None or not getattr(openai, "api_key", None):
        raise VoiceTranscribeError("OpenAI client is not configured")

    filename = _filename_from_mime(mime)
    attempt = 0
    last_exc: Optional[Exception] = None
    while attempt < _MAX_ATTEMPTS:
        attempt += 1
        try:
            payload = io.BytesIO(audio_bytes)
            payload.name = filename  # type: ignore[attr-defined]
            started = time.time()
            response = openai.Audio.transcribe(  # type: ignore[union-attr]
                model=_MODEL_NAME,
                file=payload,
                language=lang_hint,
                timeout=_OPENAI_TIMEOUT,
            )
            elapsed = time.time() - started
            log.info(
                "voice.transcribe.success",
                extra={
                    "meta": {
                        "attempt": attempt,
                        "elapsed_ms": int(elapsed * 1000),
                        "lang": lang_hint,
                    }
                },
            )
            if isinstance(response, dict):
                text = str(response.get("text", ""))
            else:
                text = str(getattr(response, "text", ""))
            cleaned = text.strip()
            if not cleaned:
                raise VoiceTranscribeError("Empty transcription result")
            return cleaned
        except Exception as exc:  # pragma: no cover - flow split below
            last_exc = exc
            should_retry = attempt < _MAX_ATTEMPTS and _should_retry(exc)
            log.warning(
                "voice.transcribe.error",
                extra={
                    "meta": {
                        "attempt": attempt,
                        "will_retry": should_retry,
                        "error_type": type(exc).__name__,
                    }
                },
            )
            if not should_retry:
                break
            delay = random.uniform(1.0, 2.5)
            time.sleep(delay)
    raise VoiceTranscribeError(str(last_exc) if last_exc else "Transcription failed")


__all__ = ["transcribe", "VoiceTranscribeError"]
