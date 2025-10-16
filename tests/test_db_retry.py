import logging
import sys
from pathlib import Path

import psycopg

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.db.retry import with_db_retries


def test_with_db_retries_success_after_retry(caplog):
    attempts = {"value": 0}

    def flaky() -> str:
        attempts["value"] += 1
        if attempts["value"] == 1:
            raise psycopg.OperationalError("SSL connection has been closed unexpectedly")
        return "ok"

    logger = logging.getLogger("test.db_retry")
    caplog.set_level(logging.INFO, logger="test.db_retry")

    result = with_db_retries(
        flaky,
        attempts=3,
        backoff=0.0,
        logger=logger,
        context={"op": "test"},
    )

    assert result == "ok"
    messages = [record.getMessage() for record in caplog.records if record.name == "test.db_retry"]
    assert "DB_RETRY" in messages
    assert "DB_RETRY_OK" in messages
    assert attempts["value"] == 2
