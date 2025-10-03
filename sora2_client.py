from __future__ import annotations

import httpx

from settings import KIE_API_KEY, SORA2


def sora2_create_task(payload: dict) -> dict:
    response = httpx.post(
        SORA2["GEN_PATH"],
        headers={
            "Authorization": f"Bearer {KIE_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def sora2_get_task(task_id: str) -> dict:
    response = httpx.get(
        SORA2["STATUS_PATH"],
        headers={"Authorization": f"Bearer {KIE_API_KEY}"},
        params={"taskId": task_id},
        timeout=15,
    )
    response.raise_for_status()
    return response.json()
