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
    monkeypatch.setenv("TELEGRAM_TOKEN", "test-token")
    monkeypatch.setenv("SUNO_API_BASE", "https://example.com")
    monkeypatch.setenv("SUNO_API_TOKEN", "dummy-token")
    monkeypatch.setenv("LEDGER_BACKEND", "memory")
    monkeypatch.setenv("DATABASE_URL", "postgres://test")
    module = importlib.import_module("bot")
    return importlib.reload(module)


def _make_ctx():
    return SimpleNamespace(bot=SimpleNamespace(), user_data={}, application=None)


def test_toggle_to_user_requests_lyrics(monkeypatch, bot_module):
    ctx = _make_ctx()
    suno_state = bot_module.SunoState(mode="lyrics")
    bot_module.save_suno_state(ctx, suno_state)
    state_dict = bot_module.state(ctx)
    state_dict["suno_flow"] = "lyrics"

    refresh_calls = []
    notify_calls = []
    wait_calls = []

    async def _fake_refresh(ctx_param, chat_id, state, price, state_key="last_ui_msg_id_suno", force_new=False):
        refresh_calls.append((chat_id, price))

    async def _fake_sync(ctx_param, chat_id, state, *, suno_state, ready, generating, waiting_enqueue):
        return None

    def _fake_activate_wait_state(*, user_id, chat_id, card_msg_id, kind, meta=None):
        wait_calls.append({"user_id": user_id, "chat_id": chat_id, "kind": kind, "meta": meta})

    async def _fake_notify(ctx_param, chat_id, text, **kwargs):
        notify_calls.append({"chat_id": chat_id, "text": text})
        return SimpleNamespace(message_id=999)

    monkeypatch.setattr(bot_module, "refresh_suno_card", _fake_refresh)
    monkeypatch.setattr(bot_module, "sync_suno_start_message", _fake_sync)
    monkeypatch.setattr(bot_module, "_activate_wait_state", _fake_activate_wait_state)
    monkeypatch.setattr(bot_module, "_suno_notify", _fake_notify)

    class _Query:
        def __init__(self):
            self.data = "suno:card:lyrics_source:toggle"
            self.message = SimpleNamespace(chat_id=123, chat=SimpleNamespace(id=123))
            self.from_user = SimpleNamespace(id=456)

        async def answer(self, *args, **kwargs):
            return None

    update = SimpleNamespace(
        callback_query=_Query(),
        effective_chat=SimpleNamespace(id=123),
        effective_user=SimpleNamespace(id=456),
    )

    asyncio.run(bot_module.on_callback(update, ctx))

    stored = ctx.user_data.get("suno_state")
    assert isinstance(stored, dict)
    assert stored.get("lyrics_source") == bot_module.LyricsSource.USER.value
    saved_state = bot_module.load_suno_state(ctx)
    assert saved_state.lyrics_source == bot_module.LyricsSource.USER
    assert state_dict["suno_waiting_state"] == bot_module.WAIT_SUNO_LYRICS
    assert not state_dict["suno_lyrics_confirmed"]
    assert refresh_calls
    assert wait_calls and wait_calls[0]["kind"] == bot_module.WaitKind.SUNO_LYRICS
    expected_limit = str(bot_module._SUNO_LYRICS_MAXLEN)
    assert notify_calls and expected_limit in notify_calls[0]["text"]


def test_toggle_back_to_ai_clears_wait(monkeypatch, bot_module):
    ctx = _make_ctx()
    suno_state = bot_module.SunoState(mode="lyrics")
    bot_module.set_suno_lyrics(suno_state, "hello")
    bot_module.save_suno_state(ctx, suno_state)
    state_dict = bot_module.state(ctx)
    state_dict["suno_flow"] = "lyrics"
    state_dict["suno_waiting_state"] = bot_module.WAIT_SUNO_LYRICS

    notify_calls = []
    cleared: list[int] = []

    async def _fake_refresh(ctx_param, chat_id, state, price, state_key="last_ui_msg_id_suno", force_new=False):
        return None

    async def _fake_sync(ctx_param, chat_id, state, *, suno_state, ready, generating, waiting_enqueue):
        return None

    def _fake_clear_wait_state(user_id: int, *, reason: str = "manual") -> None:
        cleared.append(user_id)

    async def _fake_notify(ctx_param, chat_id, text, **kwargs):
        notify_calls.append(text)
        return None

    monkeypatch.setattr(bot_module, "refresh_suno_card", _fake_refresh)
    monkeypatch.setattr(bot_module, "sync_suno_start_message", _fake_sync)
    monkeypatch.setattr(bot_module, "clear_wait_state", _fake_clear_wait_state)
    monkeypatch.setattr(bot_module, "_suno_notify", _fake_notify)

    class _Query:
        def __init__(self):
            self.data = "suno:card:lyrics_source:toggle"
            self.message = SimpleNamespace(chat_id=99, chat=SimpleNamespace(id=99))
            self.from_user = SimpleNamespace(id=77)

        async def answer(self, *args, **kwargs):
            return None

    update = SimpleNamespace(
        callback_query=_Query(),
        effective_chat=SimpleNamespace(id=99),
        effective_user=SimpleNamespace(id=77),
    )

    asyncio.run(bot_module.on_callback(update, ctx))

    stored = ctx.user_data.get("suno_state")
    assert isinstance(stored, dict)
    assert stored.get("lyrics_source") == bot_module.LyricsSource.AI.value
    saved_state = bot_module.load_suno_state(ctx)
    assert saved_state.lyrics_source == bot_module.LyricsSource.AI
    assert state_dict["suno_waiting_state"] == bot_module.IDLE_SUNO
    assert cleared == [77]
    assert notify_calls and "✨" in notify_calls[0]


def test_user_lyrics_length_validation(monkeypatch, bot_module):
    ctx = _make_ctx()
    suno_state = bot_module.SunoState(mode="lyrics")
    bot_module.save_suno_state(ctx, suno_state)
    state_dict = bot_module.state(ctx)
    state_dict["suno_flow"] = "lyrics"
    state_dict["suno_waiting_state"] = bot_module.WAIT_SUNO_LYRICS

    class _Message:
        def __init__(self):
            self.replies: list[str] = []

        async def reply_text(self, text):
            self.replies.append(text)
            return SimpleNamespace(message_id=len(self.replies))

        text = ""

    long_text = "a" * (bot_module.LYRICS_MAX_LENGTH + 5)
    message = _Message()
    message.text = long_text

    result = asyncio.run(
        bot_module._handle_suno_waiting_input(
            ctx,
            chat_id=1,
            message=message,
            state_dict=state_dict,
            waiting_field=bot_module.WAIT_SUNO_LYRICS,
            user_id=10,
        )
    )

    assert result is True
    assert any("слишком длинный" in reply for reply in message.replies)
    saved_state = bot_module.load_suno_state(ctx)
    assert not saved_state.lyrics
