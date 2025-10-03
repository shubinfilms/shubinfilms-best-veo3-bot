import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
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
    module = importlib.reload(module)
    module.mj_log.disabled = True
    return module


def test_mj_document_naming(monkeypatch, bot_module):
    responses = {
        "https://cdn.example/a.png": [
            (b"a" * 2048, "image/png"),
        ],
        "https://cdn.example/b.jpeg": [
            (b"b" * 2048, "image/jpeg"),
        ],
        "https://cdn.example/c": [
            (b"c" * 2048, "image/jpeg"),
        ],
        "https://cdn.example/d.jpeg": [
            (b"", "image/jpeg"),
            (b"d" * 4096, "image/jpeg"),
        ],
    }
    counters = {key: 0 for key in responses}

    def fake_get(url, timeout=None, headers=None):
        base = url.split("?", 1)[0]
        content, content_type = responses[base][counters[base]]
        counters[base] = min(counters[base] + 1, len(responses[base]) - 1)
        return SimpleNamespace(status_code=200, content=content, headers={"Content-Type": content_type})

    monkeypatch.setattr(bot_module.requests, "get", fake_get)

    data1 = bot_module._download_mj_image_bytes("https://cdn.example/a.png", 1)
    data2 = bot_module._download_mj_image_bytes("https://cdn.example/b.jpeg", 2)
    data3 = bot_module._download_mj_image_bytes("https://cdn.example/c", 3)
    data4 = bot_module._download_mj_image_bytes("https://cdn.example/d.jpeg", 4)

    assert data1[1].endswith(".png")
    assert data2[1].endswith(".jpeg")
    assert data3[1].endswith(".jpeg")
    assert data3[2] == "image/jpeg"
    assert len(data4[0]) > 1024
