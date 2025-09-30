"""Audio post-processing helpers for Suno tracks."""

from __future__ import annotations

import asyncio
import imghdr
import logging
import os
import re
import unicodedata
import uuid
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse

import aiohttp
from aiohttp import ClientError, ClientTimeout
from mutagen.id3 import APIC, ID3, ID3NoHeaderError, TIT2, TPE1

log = logging.getLogger("audio.post")

_DEFAULT_TIMEOUT = ClientTimeout(total=60)
_FILENAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")

_TRANSLIT_MAP_BASE = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "е": "e",
    "ё": "e",
    "ж": "zh",
    "з": "z",
    "и": "i",
    "й": "j",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "х": "h",
    "ц": "ts",
    "ч": "ch",
    "ш": "sh",
    "щ": "sch",
    "ъ": "",
    "ы": "y",
    "ь": "",
    "э": "e",
    "ю": "yu",
    "я": "ya",
}

_TRANSLIT_MAP: Dict[int, str] = {ord(k): v for k, v in _TRANSLIT_MAP_BASE.items()}
_TRANSLIT_MAP.update({ord(k.upper()): v.capitalize() for k, v in _TRANSLIT_MAP_BASE.items()})


async def _fetch_bytes(url: str, *, timeout: ClientTimeout = _DEFAULT_TIMEOUT) -> Tuple[bytes, Optional[str]]:
    parsed = urlparse(url)
    if parsed.scheme in {"", "file"}:
        path = Path(parsed.path)
        data = await asyncio.to_thread(path.read_bytes)
        return data, None
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.get(url) as response:
                response.raise_for_status()
                content_type = response.headers.get("Content-Type")
                data = await response.read()
                return data, content_type
        except (ClientError, asyncio.TimeoutError) as exc:
            log.warning("audio.fetch_failed", extra={"meta": {"url": url, "err": str(exc)}})
            raise


def _transliterate(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    translated = normalized.translate(_TRANSLIT_MAP)
    cleaned = "".join(ch for ch in translated if unicodedata.category(ch) != "Mn")
    return cleaned


def _normalize_filename(
    title: Optional[str], *, max_name: int, transliterate: bool
) -> str:
    base = (title or "track").strip()
    if not base:
        base = "track"
    if transliterate:
        base = _transliterate(base)
    base = unicodedata.normalize("NFKC", base)
    base = base.replace(os.sep, " ").replace("/", " ")
    base = _FILENAME_PATTERN.sub("_", base)
    base = re.sub(r"_+", "_", base)
    base = base.strip("._- ") or "track"
    if max_name > 0 and len(base) > max_name:
        base = base[:max_name].rstrip("._- ") or base[:max_name]
    return f"{base or 'track'}.mp3"


def _guess_cover_mime(data: bytes, declared: Optional[str]) -> Optional[str]:
    if declared:
        lowered = declared.split(";")[0].strip().lower()
        if lowered in {"image/jpeg", "image/jpg", "image/png"}:
            return "image/jpeg" if "jpeg" in lowered or "jpg" in lowered else "image/png"
    kind = imghdr.what(None, h=data)
    if kind == "jpeg":
        return "image/jpeg"
    if kind == "png":
        return "image/png"
    return None


def _apply_id3_tags(
    path: Path,
    *,
    title: str,
    artist: str,
    cover: Optional[Tuple[bytes, Optional[str]]],
) -> None:
    try:
        tags = ID3(path)
    except ID3NoHeaderError:
        tags = ID3()
    tags.delall("TIT2")
    tags.delall("TPE1")
    tags.add(TIT2(encoding=3, text=title))
    tags.add(TPE1(encoding=3, text=[artist]))
    if cover and cover[0]:
        mime = _guess_cover_mime(cover[0], cover[1])
        if mime:
            tags.delall("APIC")
            tags.add(
                APIC(
                    encoding=3,
                    mime=mime,
                    type=3,
                    desc="Cover",
                    data=cover[0],
                )
            )
    tags.save(path, v2_version=3)


async def prepare_audio_file(
    mp3_url: str,
    title: Optional[str],
    cover_url: Optional[str],
    *,
    default_artist: str,
    max_name: int,
    tags_enabled: bool,
    embed_cover_enabled: bool,
    transliterate: bool,
) -> Tuple[str, Dict[str, Optional[str]]]:
    file_name = _normalize_filename(title, max_name=max_name, transliterate=transliterate)
    from suno.tempfiles import task_directory

    target_dir = task_directory("audio")
    target_path = target_dir / f"{uuid.uuid4().hex}.mp3"

    mp3_bytes, _ = await _fetch_bytes(mp3_url)
    await asyncio.to_thread(target_path.write_bytes, mp3_bytes)

    meta: Dict[str, Optional[str]] = {"file_name": file_name}
    final_title = (title or "Untitled").strip() or "Untitled"
    final_artist = (default_artist or "Best VEO3").strip() or "Best VEO3"
    if tags_enabled:
        meta.update({"title": final_title, "performer": final_artist})
        cover_data: Optional[Tuple[bytes, Optional[str]]] = None
        if embed_cover_enabled and cover_url:
            try:
                cover_data = await _fetch_bytes(cover_url)
            except Exception as exc:
                log.warning(
                    "audio.cover.fetch_failed",
                    exc_info=True,
                    extra={"meta": {"url": cover_url, "err": str(exc)}},
                )
                cover_data = None
        try:
            await asyncio.to_thread(
                _apply_id3_tags,
                target_path,
                title=final_title,
                artist=final_artist,
                cover=cover_data,
            )
        except Exception as exc:
            log.warning(
                "audio.id3.write_failed",
                exc_info=True,
                extra={"meta": {"path": str(target_path), "err": str(exc)}},
            )
    return str(target_path), meta


def prepare_audio_file_sync(
    mp3_url: str,
    title: Optional[str],
    cover_url: Optional[str],
    *,
    default_artist: str,
    max_name: int,
    tags_enabled: bool,
    embed_cover_enabled: bool,
    transliterate: bool,
) -> Tuple[str, Dict[str, Optional[str]]]:
    try:
        return asyncio.run(
            prepare_audio_file(
                mp3_url,
                title,
                cover_url,
                default_artist=default_artist,
                max_name=max_name,
                tags_enabled=tags_enabled,
                embed_cover_enabled=embed_cover_enabled,
                transliterate=transliterate,
            )
        )
    except RuntimeError as exc:  # pragma: no cover - defensive
        if "asyncio.run" in str(exc):
            raise RuntimeError(
                "prepare_audio_file_sync cannot be called from a running event loop"
            ) from exc
        raise


__all__ = ["prepare_audio_file", "prepare_audio_file_sync"]
