"""Unified UI button infrastructure."""

from .registry import BUTTONS, ButtonId, ButtonSpec
from .results import UIResult
from .router import ButtonRouter, dispatch
from .types import ButtonContext, ButtonHandler

__all__ = [
    "BUTTONS",
    "ButtonId",
    "ButtonSpec",
    "ButtonRouter",
    "ButtonContext",
    "ButtonHandler",
    "UIResult",
    "dispatch",
]
