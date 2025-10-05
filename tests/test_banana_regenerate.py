import asyncio
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def bot_module(monkeypatch):
    monkeypatch.setenv("TELEGRAM_TOKEN", "dummy-token")
    monkeypatch.setenv("SUNO_API_BASE", "https://example.com")
    monkeypatch.setenv("SUNO_API_TOKEN", "test-token")
    monkeypatch.setenv("LEDGER_BACKEND", "memory")
    monkeypatch.setenv("DATABASE_URL", "postgres://test")
    module = importlib.import_module("bot")
    return importlib.reload(module)


def _build_ctx():
    return SimpleNamespace(bot=None, user_data={})


def _build_update(data: str, chat_id: int = 888):
    answers: list[dict[str, object]] = []

    async def _answer(text: str | None = None, **kwargs):
        payload: dict[str, object] = {}
        if text is not None:
            payload["text"] = text
        payload.update(kwargs)
        answers.append(payload)

    message = SimpleNamespace(chat_id=chat_id)
    query = SimpleNamespace(data=data, message=message, answer=_answer)
    update = SimpleNamespace(
        callback_query=query,
        effective_chat=SimpleNamespace(id=chat_id),
        effective_user=SimpleNamespace(id=101),
        _answers=answers,
    )
    return update


def test_banana_regenerate_clears_state(monkeypatch, bot_module):
    ctx = _build_ctx()
    update = _build_update("banana_regenerate_fresh")
    state_dict: dict[str, object] = {
        "banana_images": [{"url": "https://example.com/a.jpg"}],
        "last_prompt": "make it dramatic",
        "_last_text_banana": "cached",
        "banana_balance": 12,
    }

    async def fake_ensure_user_record(_update):
        return None

    calls: list[tuple[int, dict[str, object]]] = []

    async def fake_show_banana_card(chat_id, _ctx, *, force_new=False):
        calls.append((chat_id, {"force_new": force_new}))

    monkeypatch.setattr(bot_module, "ensure_user_record", fake_ensure_user_record)
    monkeypatch.setattr(bot_module, "state", lambda _ctx: state_dict)
    monkeypatch.setattr(bot_module, "show_banana_card", fake_show_banana_card)

    asyncio.run(bot_module.on_callback(update, ctx))

    assert state_dict["banana_images"] == []
    assert state_dict.get("last_prompt") is None
    assert not state_dict.get("_last_text_banana")
    assert calls and calls[-1][0] == update.effective_chat.id
    assert calls[-1][1]["force_new"] is True
    assert update._answers and update._answers[-1]["text"] == "Новая карточка Banana ✨"
