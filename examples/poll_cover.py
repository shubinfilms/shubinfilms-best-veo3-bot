"""Polling example for Suno cover generation."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from suno import downloader
from suno.client import SunoClient


def main(task_id: str) -> None:
    load_dotenv()
    client = SunoClient(
        base_url=os.environ["SUNO_API_BASE"],
        token=os.environ["SUNO_API_TOKEN"],
    )
    info = client.cover_record_info(task_id)
    images = info.get("data", {}).get("response", {}).get("images", [])
    for idx, url in enumerate(images):
        dest = Path(task_id) / f"cover_{idx}"
        saved = downloader.download_file(url, dest)
        print("Downloaded", saved)


if __name__ == "__main__":  # pragma: no cover - manual demo
    import sys

    if len(sys.argv) != 2:
        raise SystemExit("Usage: python poll_cover.py <task_id>")
    main(sys.argv[1])
