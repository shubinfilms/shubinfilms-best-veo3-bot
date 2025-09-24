"""Simple demo of creating an add-instrumental task."""
from __future__ import annotations

import os

from dotenv import load_dotenv

from suno.client import SunoClient


def main() -> None:
    load_dotenv()
    client = SunoClient(
        base_url=os.environ["SUNO_API_BASE"],
        token=os.environ["SUNO_API_TOKEN"],
    )
    callback_url = os.environ["SUNO_CALLBACK_PUBLIC_URL"] + "/music"
    payload = {
        "uploadUrl": os.environ.get("SUNO_SAMPLE_UPLOAD_URL", "https://example.com/sample.mp3"),
        "model": "V4_5PLUS",
        "title": "Demo Track",
        "tags": "demo",
        "callBackUrl": callback_url,
    }
    response = client.add_instrumental(payload)
    task_id = response.get("data", {}).get("taskId")
    print("Task response:", response)
    if task_id:
        print("Created task", task_id)


if __name__ == "__main__":  # pragma: no cover - manual demo
    main()
