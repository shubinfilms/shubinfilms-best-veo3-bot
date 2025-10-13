import asyncio
import logging
import sys
import warnings
from pathlib import Path
from typing import Any, List

import pytest
from aiohttp import web
from aiohttp.test_utils import unused_port

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.codex_logger import AsyncCodexHandler


def test_codex_handler_flush_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    asyncio.run(_exercise(monkeypatch))


async def _exercise(monkeypatch: pytest.MonkeyPatch) -> None:
    messages: List[Any] = []
    event = asyncio.Event()

    async def codex_endpoint(request: web.Request) -> web.Response:
        data = await request.json()
        if isinstance(data, list):
            messages.extend(data)
        else:
            messages.append(data)
        if not event.is_set():
            event.set()
        return web.Response(text="ok")

    app = web.Application()
    app.router.add_post("/", codex_endpoint)
    runner = web.AppRunner(app)
    await runner.setup()
    port = unused_port()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()

    endpoint = f"http://127.0.0.1:{port}/"

    monkeypatch.setenv("CODEX_LOG_ENABLED", "true")
    monkeypatch.setenv("CODEX_LOG_ENDPOINT", endpoint)
    monkeypatch.setenv("CODEX_LOG_API_KEY", "test-key")
    monkeypatch.setenv("APP_NAME", "test-app")
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("CODEX_LOG_BATCH_SIZE", "2")
    monkeypatch.setenv("CODEX_LOG_FLUSH_SEC", "0.5")
    monkeypatch.delenv("TG_LOG_CHAT_ID", raising=False)
    monkeypatch.delenv("TELEGRAM_TOKEN", raising=False)

    logger = logging.getLogger("codex-test")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    handler = AsyncCodexHandler()
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False

    try:
        logger.info("first codex message")
        logger.info("second codex message")

        await asyncio.to_thread(handler.flush)
        await asyncio.wait_for(event.wait(), timeout=5)

        with warnings.catch_warnings(record=True) as caught:
            await asyncio.to_thread(logging.shutdown)
        assert not [w for w in caught if "flush" in str(w.message)]

        assert len(messages) >= 1
        sample = messages[0]
        assert isinstance(sample, dict)
        assert sample.get("logger") == "codex-test"
    finally:
        await runner.cleanup()
        logging.basicConfig(level=logging.INFO)
