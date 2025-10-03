import importlib

import httpx

import sora2_client


def _clear_timeout_env(monkeypatch):
    monkeypatch.delenv("SORA2_TIMEOUT", raising=False)
    monkeypatch.delenv("SORA2_TIMEOUT_CONNECT", raising=False)
    monkeypatch.delenv("SORA2_TIMEOUT_READ", raising=False)
    monkeypatch.delenv("SORA2_TIMEOUT_WRITE", raising=False)
    monkeypatch.delenv("SORA2_TIMEOUT_POOL", raising=False)


def test_timeout_total_from_single_env(monkeypatch):
    _clear_timeout_env(monkeypatch)
    monkeypatch.setenv("SORA2_TIMEOUT", "12.5")
    importlib.reload(sora2_client)

    timeout = sora2_client._timeout()

    assert isinstance(timeout, httpx.Timeout)
    assert timeout.connect == 12.5
    assert timeout.read == 12.5
    assert timeout.write == 12.5
    assert timeout.pool == 12.5


def test_timeout_quartet(monkeypatch):
    _clear_timeout_env(monkeypatch)
    monkeypatch.setenv("SORA2_TIMEOUT_CONNECT", "1.5")
    monkeypatch.setenv("SORA2_TIMEOUT_READ", "2.5")
    monkeypatch.setenv("SORA2_TIMEOUT_WRITE", "3.5")
    monkeypatch.setenv("SORA2_TIMEOUT_POOL", "4.5")
    importlib.reload(sora2_client)

    timeout = sora2_client._timeout()

    assert timeout.connect == 1.5
    assert timeout.read == 2.5
    assert timeout.write == 3.5
    assert timeout.pool == 4.5
