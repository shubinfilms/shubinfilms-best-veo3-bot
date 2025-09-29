from __future__ import annotations

import asyncio
import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("SUNO_API_BASE", "https://example.com")
os.environ.setdefault("SUNO_API_TOKEN", "token")
os.environ.setdefault("SUNO_CALLBACK_URL", "https://callback.example")
os.environ.setdefault("SUNO_CALLBACK_SECRET", "secret")
os.environ.setdefault("TELEGRAM_TOKEN", "dummy-token")
os.environ.setdefault("KIE_API_KEY", "test-key")
os.environ.setdefault("KIE_BASE_URL", "https://example.com")
os.environ.setdefault("DATABASE_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("LEDGER_BACKEND", "memory")
os.environ.setdefault("LOG_JSON", "false")
os.environ.setdefault("LOG_LEVEL", "WARNING")

bot_module = importlib.import_module("bot")


def test_generate_request_id_includes_user() -> None:
    rid = bot_module._generate_suno_request_id(555)
    assert rid.startswith("suno:555:"), rid
    rid2 = bot_module._generate_suno_request_id(555)
    assert rid != rid2


def test_duplicate_req_id_skips_enqueue(monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = SimpleNamespace(bot=None, user_data={})
    state_dict = bot_module.state(ctx)
    state_dict["suno_generating"] = False
    state_dict["suno_waiting_enqueue"] = False
    state_dict["suno_current_req_id"] = None
    card_meta = state_dict.get("suno_card")
    if isinstance(card_meta, dict):
        card_meta["msg_id"] = None

    monkeypatch.setattr(bot_module, "_suno_configured", lambda: True)
    monkeypatch.setattr(bot_module, "_suno_cooldown_remaining", lambda uid: 0)
    monkeypatch.setattr(bot_module, "ensure_user", lambda uid: None)
    monkeypatch.setattr(
        bot_module,
        "build_suno_generation_payload",
        lambda *_, **__: {"prompt": "demo", "instrumental": True},
    )
    monkeypatch.setattr(bot_module, "sanitize_payload_for_log", lambda payload: payload)

    seen: dict[str, str] = {}

    def fake_pending_load(req_id: str):
        seen["req_id"] = req_id
        return {"status": "enqueued", "task_id": "task-1"}

    monkeypatch.setattr(bot_module, "_suno_pending_load", fake_pending_load)

    async def fail_notify(*_args, **_kwargs):
        raise AssertionError("should not notify while job pending")

    monkeypatch.setattr(bot_module, "_suno_notify", fail_notify)

    def fail_start_music(*_args, **_kwargs):
        raise AssertionError("duplicate launch should not enqueue")

    monkeypatch.setattr(bot_module.SUNO_SERVICE, "start_music", fail_start_music)

    async def scenario() -> None:
        params = {"instrumental": True, "title": "", "style": "", "lyrics": "", "has_lyrics": False}
        await bot_module._launch_suno_generation(
            123,
            ctx,
            params=params,
            user_id=777,
            reply_to=None,
            trigger="test",
        )

    asyncio.run(scenario())

    assert "req_id" in seen
    assert seen["req_id"].startswith("suno:777:")
