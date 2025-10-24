"""Unified UI button infrastructure."""

from .registry import BUTTONS, REGISTRY, ButtonId, ButtonSpec, btn_data
from .results import UIResult
from .router import ButtonRouter, dispatch
from .types import ButtonAction, ButtonContext, ButtonHandler

__all__ = [
    "BUTTONS",
    "REGISTRY",
    "ButtonId",
    "ButtonAction",
    "ButtonSpec",
    "ButtonRouter",
    "ButtonContext",
    "ButtonHandler",
    "btn_data",
    "UIResult",
    "dispatch",
]
