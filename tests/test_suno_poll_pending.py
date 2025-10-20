import os
import sys
from pathlib import Path

import requests

os.environ.setdefault("DATABASE_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("LEDGER_BACKEND", "memory")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bot  # noqa: E402


def test_poll_404_is_pending(monkeypatch):
    response = requests.Response()
    response.status_code = 404
    response._content = b'{"code": 404, "msg": "Not ready"}'

    def fake_get(*args, **kwargs):
        return response

    monkeypatch.setattr(bot.SUNO_SERVICE._api_session, "get", fake_get, raising=False)

    result = bot.SUNO_SERVICE.poll_record_info_once("task-404")
    assert result.state == "pending"
    assert result.status_code == 404
