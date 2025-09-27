"""Public handler shortcuts."""

from .faq_handler import configure_faq, faq_callback, faq_command
from .prompt_master_handler import (
    prompt_master_callback,
    prompt_master_open,
    prompt_master_process,
)

__all__ = [
    "configure_faq",
    "faq_callback",
    "faq_command",
    "prompt_master_callback",
    "prompt_master_open",
    "prompt_master_process",
]
