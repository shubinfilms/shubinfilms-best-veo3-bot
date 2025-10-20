from __future__ import annotations


class ButtonError(RuntimeError):
    """Base class for button router errors."""


class ButtonNotFound(ButtonError):
    pass


class ButtonAccessDenied(ButtonError):
    pass


class ButtonFeatureDisabled(ButtonError):
    pass
