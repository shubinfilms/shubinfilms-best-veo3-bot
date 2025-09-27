"""Smoke tests for Prompt-Master PTB handlers."""

import sys
from pathlib import Path

from telegram.ext import AIORateLimiter, ApplicationBuilder

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from handlers import prompt_master_conv


def test_prompt_master_conversation_flags() -> None:
    """Conversation should enforce per-chat flow without per-message duplication."""

    assert prompt_master_conv.per_chat is True
    assert prompt_master_conv.per_user is True
    assert prompt_master_conv.per_message is False


def test_prompt_master_conversation_registration() -> None:
    """Conversation handler should be compatible with PTB 21 application builder."""

    application = (
        ApplicationBuilder()
        .token("123:ABC")
        .rate_limiter(AIORateLimiter())
        .build()
    )
    application.add_handler(prompt_master_conv)
