"""Minimal generate-music demo hitting the official Suno endpoints."""
from __future__ import annotations

import os
import time

from dotenv import load_dotenv

from suno.client import SunoClient


def main() -> None:
    load_dotenv()
    client = SunoClient(
        base_url=os.environ.get("SUNO_API_BASE"),
        token=os.environ["SUNO_API_TOKEN"],
    )
    callback_url = os.environ["SUNO_CALLBACK_PUBLIC_URL"].rstrip("/") + "/music"
    response = client.generate_music(
        prompt="Dreamy synthwave at night city lights",
        model="V4_5PLUS",
        title="Neon Drive",
        style="Synthwave",
        callBackUrl=callback_url,
        instrumental=False,
        negativeTags="heavy metal, blast beats",
    )
    task_id = response.get("data", {}).get("taskId")
    print("Task response:", response)
    if not task_id:
        print("No taskId returned")
        return
    print("taskId:", task_id)

    # Simple polling example for status debugging.
    time.sleep(1)
    status = client.record_info(task_id)
    print("record_info:", status)


if __name__ == "__main__":  # pragma: no cover - manual demo
    main()
