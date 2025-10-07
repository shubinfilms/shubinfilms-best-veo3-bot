import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os

os.environ.setdefault("TELEGRAM_TOKEN", "test-token")
os.environ.setdefault("REDIS_URL", "memory://")
os.environ.setdefault("KIE_API_KEY", "test-key")
os.environ.setdefault("LEDGER_BACKEND", "memory")

import bot as bot_module
from core.balance_provider import BalanceSnapshot


def _make_context() -> SimpleNamespace:
    return SimpleNamespace(user_data={}, application=SimpleNamespace(logger=None))


def test_balance_memoization_between_cards(monkeypatch):
    ctx = _make_context()
    uid = 777

    snapshots = [
        BalanceSnapshot(value=495, display="495"),
        BalanceSnapshot(value=777, display="777"),
    ]

    def fake_get_balance_snapshot(user_id: int):
        assert user_id == uid
        return snapshots.pop(0)

    monkeypatch.setattr(bot_module, "get_balance_snapshot", fake_get_balance_snapshot)

    welcome = bot_module.render_welcome_for(uid, ctx)
    assert "495" in welcome

    cached = bot_module._resolve_balance_snapshot(ctx, uid, prefer_cached=True)
    assert cached.display == "495"


def test_balance_profile_uses_cached_snapshot(monkeypatch):
    ctx = _make_context()
    uid = 555

    snapshots = [
        BalanceSnapshot(value=495, display="495"),
        BalanceSnapshot(value=123, display="123"),
    ]

    def fake_get_balance_snapshot(user_id: int):
        assert user_id == uid
        return snapshots.pop(0)

    monkeypatch.setattr(bot_module, "get_balance_snapshot", fake_get_balance_snapshot)

    bot_module.render_welcome_for(uid, ctx)

    profile_snapshot = bot_module._resolve_balance_snapshot(ctx, uid, prefer_cached=True)
    profile_text = bot_module._profile_balance_text(profile_snapshot)
    assert "495" in profile_text
    assert "123" not in profile_text


def test_balance_error_shows_placeholder(monkeypatch):
    ctx = _make_context()
    uid = 888

    fallback = BalanceSnapshot(value=None, display="—", warning="⚠️ Сервер недоступен")

    monkeypatch.setattr(bot_module, "get_balance_snapshot", lambda user_id: fallback)

    welcome = bot_module.render_welcome_for(uid, ctx, balance=fallback)
    assert "—" in welcome
    assert "сервер недоступен" in welcome.lower()
    assert "0" not in welcome

    profile_text = bot_module._profile_balance_text(fallback)
    assert "—" in profile_text
    assert "сервер недоступен" in profile_text.lower()
