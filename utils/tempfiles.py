from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path
from typing import Iterable, Union

_PathLike = Union[str, os.PathLike[str]]


def save_bytes_to_temp(data: bytes, *, suffix: str = "", prefix: str = "bot-") -> Path:
    """Persist ``data`` into a temporary file and return its path."""

    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("data must be bytes-like")
    fd, raw_path = tempfile.mkstemp(prefix=prefix, suffix=suffix)
    path = Path(raw_path)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(bytes(data))
    except Exception:
        with contextlib.suppress(Exception):
            path.unlink()
        raise
    return path


def cleanup_temp(paths: Iterable[_PathLike]) -> None:
    """Remove temporary files, ignoring errors."""

    for candidate in paths:
        if not candidate:
            continue
        path = Path(candidate)
        with contextlib.suppress(FileNotFoundError, PermissionError, OSError):
            path.unlink()


__all__ = ["save_bytes_to_temp", "cleanup_temp"]
