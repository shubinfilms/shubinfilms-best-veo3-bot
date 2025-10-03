import importlib
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def bot_module(monkeypatch):
    monkeypatch.setenv("TELEGRAM_TOKEN", "dummy-token")
    monkeypatch.setenv("SUNO_API_BASE", "https://example.com")
    monkeypatch.setenv("SUNO_API_TOKEN", "token")
    monkeypatch.setenv("LEDGER_BACKEND", "memory")
    monkeypatch.setenv("DATABASE_URL", "postgres://test")
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://bot.example")
    module = importlib.import_module("bot")
    return importlib.reload(module)


def test_sora2_text_to_video_payload(bot_module):
    payload = bot_module._build_sora2_payload("sora2_ttv", "Hello world", [])

    assert payload["model"] == "sora-2-text-to-video"
    assert payload["callBackUrl"] == "https://bot.example/sora2-callback"
    assert payload["input"]["prompt"] == "Hello world"
    assert payload["input"]["duration"] == 10
    assert payload["input"]["aspect_ratio"] == "16:9"
    assert payload["input"]["quality"] == "standard"
    assert payload["input"]["audio"] is True
    assert "image_urls" not in payload["input"]


def test_sora2_image_to_video_payload(bot_module):
    image_urls = ["https://example.com/one.png", "https://example.com/two.png"]
    payload = bot_module._build_sora2_payload("sora2_itv", "Prompt", image_urls)

    assert payload["model"] == "sora-2-image-to-video"
    assert payload["input"]["prompt"] == "Prompt"
    assert payload["input"]["image_urls"] == image_urls
