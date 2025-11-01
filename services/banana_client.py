"""Thin Banana API client."""
from __future__ import annotations

import os

import aiohttp

BANANA_KEY = os.getenv("BANANA_KEY")
BANANA_UPLOAD = os.getenv("BANANA_UPLOAD", "https://api.banana.dev/v1/upload")
BANANA_PROCESS = os.getenv("BANANA_PROCESS", "https://api.banana.dev/v1/process")


async def banana_upload_bytes(data: bytes, filename: str = "image.png") -> dict:
    """Upload raw bytes to Banana and return the JSON response."""

    async with aiohttp.ClientSession() as session:
        form = aiohttp.FormData()
        form.add_field("file", data, filename=filename, content_type="application/octet-stream")
        form.add_field("api_key", BANANA_KEY)
        async with session.post(BANANA_UPLOAD, data=form) as response:
            response.raise_for_status()
            return await response.json()


async def banana_process(model: str, payload: dict) -> dict:
    """Trigger Banana model processing using an upload identifier."""

    async with aiohttp.ClientSession() as session:
        body = {"api_key": BANANA_KEY, "model": model, "input": payload}
        async with session.post(BANANA_PROCESS, json=body) as response:
            response.raise_for_status()
            return await response.json()
