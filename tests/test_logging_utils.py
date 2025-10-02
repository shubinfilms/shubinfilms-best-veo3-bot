import logging
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from logging_utils import build_log_extra, get_logger


def test_build_log_extra_wraps_fields() -> None:
    payload = build_log_extra({"name": "x"}, user_id=1)
    assert set(payload.keys()) == {"extra"}
    extra = payload["extra"]
    assert "name" not in extra
    assert extra["ctx_name"] == "x"
    assert extra["ctx_user_id"] == 1


def test_logging_extra_name_not_crash(caplog) -> None:
    logger = get_logger("veo3-bot-test")
    with caplog.at_level(logging.DEBUG, logger="veo3-bot-test"):
        logger.debug("check", **build_log_extra({"name": "menu", "user": 123}))

    target = next((record for record in caplog.records if record.message == "check"), None)
    assert target is not None
    ctx = getattr(target, "ctx")
    assert ctx["ctx_name"] == "menu"
    assert ctx["user"] == 123
