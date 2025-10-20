from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional


ResultKind = Literal["menu", "dialog", "error", "noop", "ack"]


@dataclass(slots=True)
class UIResult:
    """Declarative description of what happened after a button click."""

    kind: ResultKind
    button_id: str
    screen: Optional[str] = None
    message: Optional[str] = None
    details: dict[str, Any] = field(default_factory=dict)
    changed: bool = True


def show_menu(button_id: str, screen: str, *, message_id: Optional[int], reused: bool) -> UIResult:
    return UIResult(
        kind="menu",
        button_id=button_id,
        screen=screen,
        details={"message_id": message_id, "reused": reused},
        changed=not reused,
    )


def show_dialog(button_id: str, screen: str, *, message_id: Optional[int]) -> UIResult:
    return UIResult(
        kind="dialog",
        button_id=button_id,
        screen=screen,
        details={"message_id": message_id},
    )


def show_error(button_id: str, message: str, *, error_kind: str) -> UIResult:
    return UIResult(
        kind="error",
        button_id=button_id,
        message=message,
        details={"error_kind": error_kind},
        changed=False,
    )


def noop(button_id: str) -> UIResult:
    return UIResult(kind="noop", button_id=button_id, changed=False)


def acknowledge(button_id: str, *, message: Optional[str] = None) -> UIResult:
    return UIResult(kind="ack", button_id=button_id, message=message, changed=False)
