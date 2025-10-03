import importlib
import sys
from pathlib import Path

import httpx

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import settings
import sora2_client


def test_sora2_client_uses_configured_timeout(monkeypatch):
    monkeypatch.setenv("SORA2_TIMEOUT_CONNECT", "15")
    monkeypatch.setenv("SORA2_TIMEOUT_READ", "25")
    monkeypatch.setenv("SORA2_TIMEOUT_WRITE", "35")
    monkeypatch.setenv("SORA2_TIMEOUT_POOL", "45")
    monkeypatch.setenv("SORA2_API_KEY", "key")

    importlib.reload(settings)
    importlib.reload(sora2_client)

    captured: dict[str, httpx.Timeout | None] = {"timeout": None}

    class DummyResponse:
        status_code = 200
        text = "{}"

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {}

        @property
        def is_success(self) -> bool:
            return True

    class DummyClient:
        def __init__(self, *args, **kwargs):
            captured["timeout"] = kwargs.get("timeout")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, headers=None, json=None):
            return DummyResponse()

    monkeypatch.setattr(httpx, "Client", DummyClient)

    result = sora2_client._perform_request("https://example.invalid", {"ping": "pong"})

    assert result == {}

    timeout = captured["timeout"]
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.connect == 15
    assert timeout.read == 25
    assert timeout.write == 35
    assert timeout.pool == 45

    monkeypatch.delenv("SORA2_TIMEOUT_CONNECT", raising=False)
    monkeypatch.delenv("SORA2_TIMEOUT_READ", raising=False)
    monkeypatch.delenv("SORA2_TIMEOUT_WRITE", raising=False)
    monkeypatch.delenv("SORA2_TIMEOUT_POOL", raising=False)

    importlib.reload(settings)
    importlib.reload(sora2_client)
