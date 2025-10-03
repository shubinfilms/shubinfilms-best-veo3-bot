import importlib
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def redis_module(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    module = importlib.import_module("redis_utils")
    return importlib.reload(module)


def test_user_lock_blocks_duplicates(redis_module):
    assert redis_module.user_lock(123, "video_start") is True
    assert redis_module.user_lock(123, "video_start") is False

    redis_module.release_user_lock(123, "video_start")
    assert redis_module.user_lock(123, "video_start") is True
