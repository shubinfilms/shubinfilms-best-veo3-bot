"""Smoke tests for Prompt-Master PTB handlers."""

import sys
from pathlib import Path

from telegram.ext import AIORateLimiter, ApplicationBuilder

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from handlers import (
    PM_SELECT,
    PM_WAIT_IDEA,
    PROMPT_MASTER_BACK,
    PROMPT_MASTER_OPEN,
    prompt_master_conv,
)


def test_prompt_master_conversation_flags() -> None:
    """Conversation should expose expected states and settings."""

    assert prompt_master_conv.allow_reentry is True
    assert prompt_master_conv.conversation_timeout == 600
    assert PM_SELECT in prompt_master_conv.states
    assert PM_WAIT_IDEA in prompt_master_conv.states


def test_prompt_master_conversation_registration() -> None:
    """Conversation handler should be compatible with PTB 21 application builder."""

    application = (
        ApplicationBuilder()
        .token("123:ABC")
        .rate_limiter(AIORateLimiter())
        .build()
    )
    application.add_handler(prompt_master_conv)

    patterns = []
    for handler in prompt_master_conv.states[PM_SELECT]:
        pattern = getattr(handler, "pattern", None)
        if pattern is None:
            continue
        if hasattr(pattern, "pattern"):
            patterns.append(pattern.pattern)
        else:
            patterns.append(str(pattern))
    assert any(pattern == f"^{PROMPT_MASTER_OPEN}$" for pattern in patterns)
    assert any(pattern == f"^{PROMPT_MASTER_BACK}$" for pattern in patterns)
