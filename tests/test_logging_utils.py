import logging
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from logging_utils import get_logger


def test_logging_extra_name_not_crash(caplog) -> None:
    logger = get_logger("veo3-bot-test")
    with caplog.at_level(logging.DEBUG, logger="veo3-bot-test"):
        logger.debug("check", extra={"name": "menu", "user": 123})

    target = next((record for record in caplog.records if record.message == "check"), None)
    assert target is not None
    assert getattr(target, "extra_name", None) == "menu"
    assert getattr(target, "user", None) == 123
    assert hasattr(target, "cmd") and getattr(target, "cmd") is None
