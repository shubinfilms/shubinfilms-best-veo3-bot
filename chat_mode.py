"""Helpers for managing the chat mode flag in Redis."""

from __future__ import annotations

from chat_service import is_mode_on as _is_mode_on
from chat_service import set_mode as _set_mode


def turn_on(user_id: int) -> None:
    """Mark the conversational chat mode as enabled for the user."""

    _set_mode(user_id, True)


def is_on(user_id: int) -> bool:
    """Return whether conversational chat mode is enabled for the user."""

    return _is_mode_on(user_id)


def turn_off(user_id: int) -> None:
    """Disable conversational chat mode for the user."""

    _set_mode(user_id, False)


__all__ = ["turn_on", "turn_off", "is_on"]
