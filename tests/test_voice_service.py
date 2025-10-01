import asyncio
import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import voice_service
from voice_service import VoiceTranscribeError, transcribe


class _FakeRateLimitError(Exception):
    pass


class _FakeAPIError(Exception):
    def __init__(self, status: int) -> None:
        super().__init__("api error")
        self.http_status = status


class _FakeServiceUnavailableError(Exception):
    pass


def _setup_openai(monkeypatch, transcribe_fn):
    fake_openai = types.ModuleType("openai")
    fake_error_module = types.ModuleType("openai.error")
    fake_error_module.APIError = _FakeAPIError
    fake_error_module.RateLimitError = _FakeRateLimitError
    fake_error_module.ServiceUnavailableError = _FakeServiceUnavailableError
    fake_audio = SimpleNamespace(transcribe=transcribe_fn)
    fake_openai.Audio = fake_audio
    fake_openai.api_key = "sk-test"
    fake_openai.error = fake_error_module
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setitem(sys.modules, "openai.error", fake_error_module)
    monkeypatch.setattr(voice_service, "openai", fake_openai)


def test_transcribe_success(monkeypatch):
    captured = {}

    def fake_transcribe(**kwargs):
        captured.update(kwargs)
        return {"text": "  hello world  "}

    _setup_openai(monkeypatch, fake_transcribe)
    result = transcribe(b"data", "audio/wav", "en")
    assert result == "hello world"
    assert captured["model"] == voice_service._MODEL_NAME
    assert captured["language"] == "en"


def test_transcribe_retries_on_rate_limit(monkeypatch):
    attempts = []

    def fake_transcribe(**kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise _FakeRateLimitError("429")
        return {"text": "done"}

    _setup_openai(monkeypatch, fake_transcribe)
    monkeypatch.setattr(voice_service.time, "sleep", lambda _: None)
    result = transcribe(b"voice", "audio/mp3", "ru")
    assert result == "done"
    assert len(attempts) == 2


def test_transcribe_retries_on_server_error(monkeypatch):
    attempts = []

    def fake_transcribe(**kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise _FakeAPIError(500)
        return {"text": "ok"}

    _setup_openai(monkeypatch, fake_transcribe)
    monkeypatch.setattr(voice_service.time, "sleep", lambda _: None)
    result = transcribe(b"voice", "audio/mp3", "ru")
    assert result == "ok"
    assert len(attempts) == 2


@pytest.fixture
def bot_module(monkeypatch):
    monkeypatch.setenv("TELEGRAM_TOKEN", "test-token")
    monkeypatch.setenv("SUNO_API_BASE", "https://example.com")
    monkeypatch.setenv("SUNO_API_TOKEN", "dummy-token")
    monkeypatch.setenv("LEDGER_BACKEND", "memory")
    monkeypatch.setenv("DATABASE_URL", "postgres://test")
    module = importlib.import_module("bot")
    module = importlib.reload(module)
    return module


def test_handle_voice_success(monkeypatch, bot_module):
    edits: list[str] = []
    sends: list[str] = []
    downloads: list[str] = []
    ffmpeg_calls: list[list[str]] = []
    transcribe_args: list[tuple[bytes, str, str]] = []

    async def fake_ensure_user_record(update):
        return None

    async def fake_safe_send_placeholder(bot, chat_id, text):
        edits.append(text)
        return SimpleNamespace(message_id=42)

    async def fake_safe_edit(bot, chat_id, message_id, text):
        edits.append(text)

    async def fake_safe_send_text(bot, chat_id, text, **kwargs):
        sends.append(text)

    async def fake_run_ffmpeg(data, args):
        ffmpeg_calls.append(args)
        return b"converted"

    async def fake_download(url):
        downloads.append(url)
        return b"voice-bytes"

    def fake_voice_transcribe(audio_bytes, mime, lang):
        transcribe_args.append((audio_bytes, mime or "", lang or ""))
        return "а" * 3100

    dummy_history: list[dict[str, str]] = []

    monkeypatch.setattr(bot_module, "ensure_user_record", fake_ensure_user_record)
    monkeypatch.setattr(bot_module, "safe_send_placeholder", fake_safe_send_placeholder)
    monkeypatch.setattr(bot_module, "safe_edit_markdown_v2", fake_safe_edit)
    monkeypatch.setattr(bot_module, "safe_send_text", fake_safe_send_text)
    monkeypatch.setattr(bot_module, "run_ffmpeg", fake_run_ffmpeg)
    monkeypatch.setattr(bot_module, "_download_telegram_file", fake_download)
    monkeypatch.setattr(bot_module, "voice_transcribe", fake_voice_transcribe)
    monkeypatch.setattr(bot_module, "rate_limit_hit", lambda user_id: False)
    monkeypatch.setattr(bot_module, "is_mode_on", lambda user_id: True)
    monkeypatch.setattr(bot_module, "_mode_get", lambda chat_id: bot_module.MODE_CHAT)
    monkeypatch.setattr(bot_module, "load_ctx", lambda user_id: list(dummy_history))
    monkeypatch.setattr(bot_module, "estimate_tokens", lambda text: len(text) // 4 or 1)
    monkeypatch.setattr(bot_module, "append_ctx", lambda user_id, role, content: dummy_history.append({"role": role, "content": content}))
    monkeypatch.setattr(bot_module, "build_messages", lambda sys_prompt, history, text, lang: ["sys", text])
    monkeypatch.setattr(bot_module, "call_llm", lambda messages: "Ответ")

    chat_context_tokens_mock = MagicMock()
    monkeypatch.setattr(bot_module, "chat_context_tokens", chat_context_tokens_mock)

    chat_messages_total_mock = MagicMock()
    chat_messages_handle = MagicMock()
    chat_messages_total_mock.labels.return_value = chat_messages_handle
    monkeypatch.setattr(bot_module, "chat_messages_total", chat_messages_total_mock)

    chat_latency_ms_mock = MagicMock()
    monkeypatch.setattr(bot_module, "chat_latency_ms", chat_latency_ms_mock)

    chat_voice_total_mock = MagicMock()
    chat_voice_total_handle = MagicMock()
    chat_voice_total_mock.labels.return_value = chat_voice_total_handle
    monkeypatch.setattr(bot_module, "chat_voice_total", chat_voice_total_mock)

    chat_voice_latency_mock = MagicMock()
    chat_voice_latency_handle = MagicMock()
    chat_voice_latency_mock.labels.return_value = chat_voice_latency_handle
    monkeypatch.setattr(bot_module, "chat_voice_latency_ms", chat_voice_latency_mock)

    chat_transcribe_latency_mock = MagicMock()
    chat_transcribe_latency_handle = MagicMock()
    chat_transcribe_latency_mock.labels.return_value = chat_transcribe_latency_handle
    monkeypatch.setattr(bot_module, "chat_transcribe_latency_ms", chat_transcribe_latency_mock)

    bot_logger = MagicMock()

    class DummyBot:
        async def send_chat_action(self, chat_id, action):
            return None

        async def get_file(self, file_id):
            return SimpleNamespace(file_path="voice.oga", mime_type="audio/ogg")

    ctx = SimpleNamespace(
        bot=DummyBot(),
        application=SimpleNamespace(logger=bot_logger),
    )

    update = SimpleNamespace(
        message=SimpleNamespace(
            chat_id=100,
            voice=SimpleNamespace(file_id="file1", file_size=10_000, duration=30, mime_type="audio/ogg"),
            audio=None,
            caption=None,
        ),
        effective_user=SimpleNamespace(id=55, language_code="ru", first_name="Имя", last_name=None),
        effective_chat=SimpleNamespace(id=100),
    )

    asyncio.run(bot_module.handle_voice(update, ctx))

    assert downloads == [bot_module.tg_direct_file_url("test-token", "voice.oga")]
    assert ffmpeg_calls == [["-i", "pipe:0", "-ac", "1", "-ar", "16000", "-f", "wav", "pipe:1"]]
    assert transcribe_args == [(b"converted", "audio/wav", "ru")]
    assert "Думаю над ответом…" in edits[1]
    assert "…\\." in edits[1]
    assert edits[-1] == bot_module.md2_escape("Ответ")
    chat_voice_total_mock.labels.assert_any_call(outcome="ok", **bot_module._VOICE_METRIC_LABELS)
    chat_voice_total_handle.inc.assert_called()
    chat_voice_latency_mock.labels.assert_any_call(**bot_module._VOICE_METRIC_LABELS)
    chat_voice_latency_handle.observe.assert_called()
    chat_transcribe_latency_mock.labels.assert_any_call(**bot_module._VOICE_METRIC_LABELS)
    chat_transcribe_latency_handle.observe.assert_called()
    assert sends == []


def test_handle_voice_transcribe_error(monkeypatch, bot_module):
    edits: list[str] = []

    async def fake_ensure_user_record(update):
        return None

    async def fake_safe_send_placeholder(bot, chat_id, text):
        return SimpleNamespace(message_id=99)

    async def fake_safe_edit(bot, chat_id, message_id, text):
        edits.append(text)

    async def fake_run_ffmpeg(data, args):
        return b"converted"

    async def fake_download(url):
        return b"voice"

    def fake_voice_transcribe(audio_bytes, mime, lang):
        raise VoiceTranscribeError("boom")

    monkeypatch.setattr(bot_module, "ensure_user_record", fake_ensure_user_record)
    monkeypatch.setattr(bot_module, "safe_send_placeholder", fake_safe_send_placeholder)
    monkeypatch.setattr(bot_module, "safe_edit_markdown_v2", fake_safe_edit)
    monkeypatch.setattr(bot_module, "safe_send_text", MagicMock())
    monkeypatch.setattr(bot_module, "run_ffmpeg", fake_run_ffmpeg)
    monkeypatch.setattr(bot_module, "_download_telegram_file", fake_download)
    monkeypatch.setattr(bot_module, "voice_transcribe", fake_voice_transcribe)

    chat_voice_total_mock = MagicMock()
    chat_voice_total_handle = MagicMock()
    chat_voice_total_mock.labels.return_value = chat_voice_total_handle
    monkeypatch.setattr(bot_module, "chat_voice_total", chat_voice_total_mock)

    chat_voice_latency_mock = MagicMock()
    chat_voice_latency_handle = MagicMock()
    chat_voice_latency_mock.labels.return_value = chat_voice_latency_handle
    monkeypatch.setattr(bot_module, "chat_voice_latency_ms", chat_voice_latency_mock)

    chat_transcribe_latency_mock = MagicMock()
    chat_transcribe_latency_handle = MagicMock()
    chat_transcribe_latency_mock.labels.return_value = chat_transcribe_latency_handle
    monkeypatch.setattr(bot_module, "chat_transcribe_latency_ms", chat_transcribe_latency_mock)

    class DummyBot:
        async def send_chat_action(self, chat_id, action):
            return None

        async def get_file(self, file_id):
            return SimpleNamespace(file_path="voice.oga", mime_type="audio/ogg")

    ctx = SimpleNamespace(
        bot=DummyBot(),
        application=SimpleNamespace(logger=MagicMock()),
    )

    update = SimpleNamespace(
        message=SimpleNamespace(
            chat_id=200,
            voice=SimpleNamespace(file_id="file1", file_size=5000, duration=60, mime_type="audio/ogg"),
            audio=None,
            caption=None,
        ),
        effective_user=SimpleNamespace(id=12, language_code="en", first_name="User", last_name=None),
        effective_chat=SimpleNamespace(id=200),
    )

    asyncio.run(bot_module.handle_voice(update, ctx))

    assert edits[0] == bot_module.md2_escape(bot_module.VOICE_TRANSCRIBE_ERROR_TEXT)
    chat_voice_total_mock.labels.assert_any_call(outcome="error", **bot_module._VOICE_METRIC_LABELS)
    chat_voice_total_handle.inc.assert_called()
    chat_voice_latency_mock.labels.assert_any_call(**bot_module._VOICE_METRIC_LABELS)
    chat_voice_latency_handle.observe.assert_called()
    chat_transcribe_latency_mock.labels.assert_any_call(**bot_module._VOICE_METRIC_LABELS)
    chat_transcribe_latency_handle.observe.assert_called()
