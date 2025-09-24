"""Utilities for downloading assets from Suno callbacks."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
from requests import Response

from ._retry import RetryError, Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential


class DownloadError(RuntimeError):
    """Raised when a file cannot be downloaded."""


_DEFAULT_DOWNLOAD_DIR = Path(os.getenv("SUNO_DOWNLOAD_DIR", "downloads"))
_REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT_S", "20"))


def _sanitize_component(component: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", component)
    return safe.strip("._") or "file"


def _guess_extension(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path
    if not path:
        return ""
    suffix = Path(path).suffix
    return suffix


def resolve_destination(dest_path: str | Path, url: str, base_dir: Optional[Path] = None) -> Path:
    """Resolve destination path ensuring it is scoped to the download dir."""

    if base_dir is None:
        base_dir = _DEFAULT_DOWNLOAD_DIR
    base_dir = Path(base_dir)
    relative = Path(dest_path)
    safe_parts = [_sanitize_component(part) for part in relative.parts if part not in ("", ".")]
    safe_relative = Path(*safe_parts)
    target = base_dir.joinpath(safe_relative)
    if target.is_dir() or not target.suffix:
        extension = _guess_extension(url)
        if extension:
            target = target.with_suffix(extension)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _is_retryable(response: Response) -> bool:
    return response.status_code >= 500 or response.status_code == 429


def download_file(url: str, dest_path: str | Path, base_dir: Optional[Path] = None) -> Path:
    """Download a file with retries and return the resulting path."""

    if not url:
        raise ValueError("download url must be provided")

    destination = resolve_destination(dest_path, url, base_dir=base_dir)
    retryer = Retrying(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((requests.RequestException, DownloadError)),
        reraise=True,
    )

    def _perform_download() -> Path:
        try:
            response = requests.get(url, stream=True, timeout=_REQUEST_TIMEOUT)
        except requests.RequestException as exc:  # pragma: no cover - handled via retryer
            raise exc
        if response.status_code >= 400:
            if _is_retryable(response):
                raise DownloadError(f"temporary error {response.status_code}")
            raise DownloadError(f"failed with status {response.status_code}")
        with destination.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    fh.write(chunk)
        return destination

    try:
        return retryer(_perform_download)
    except RetryError as exc:  # pragma: no cover - defensive
        raise DownloadError(str(exc)) from exc
