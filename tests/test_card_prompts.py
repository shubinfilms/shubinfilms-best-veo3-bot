from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

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

import bot as bot_module


def test_veo_card_prompt_starts_empty() -> None:
    ctx = SimpleNamespace(bot=None, user_data={})
    state = bot_module.state(ctx)
    veo_text = bot_module.veo_card_text(state)
    assert "<code>—</code>" not in veo_text
    assert "<code> </code>" in veo_text


def test_mj_card_prompt_starts_empty() -> None:
    ctx = SimpleNamespace(bot=None, user_data={})
    state = bot_module.state(ctx)
    mj_text = bot_module._mj_prompt_card_text("16:9", state.get("last_prompt"))
    assert "Промпт: <i>—</i>" not in mj_text
    assert "Промпт: <i> </i>" in mj_text
