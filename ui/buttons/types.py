from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Protocol, Sequence

from telegram import CallbackQuery, Update
from telegram.ext import ContextTypes

from .results import UIResult

ButtonId = Literal[
    "profile",
    "music",
    "photo",
    "video",
    "sora2",
    "kb",
    "help",
    "dialog",
]

AccessLevel = Literal["all", "admin", "paid"]


class ButtonHandler(Protocol):
    """Callable protocol for button handlers."""

    async def __call__(self, context: "ButtonContext") -> UIResult:  # pragma: no cover - protocol
        ...


@dataclass(slots=True)
class ButtonSpec:
    """Metadata that describes how a button behaves."""

    id: ButtonId
    title_i18n_key: str
    access: Sequence[AccessLevel]
    handler: ButtonHandler
    open_telemetry_event: str
    feature_flag: Optional[str] = None


@dataclass(slots=True)
class ButtonContext:
    """Execution context available to button handlers."""

    update: Update
    app_context: ContextTypes.DEFAULT_TYPE
    spec: ButtonSpec
    request_id: str
    query: Optional[CallbackQuery]
    chat_id: Optional[int]
    user_id: Optional[int]

    def as_dict(self) -> dict[str, object]:
        """Return a structured representation for logging/debugging."""

        return {
            "button_id": self.spec.id,
            "chat_id": self.chat_id,
            "user_id": self.user_id,
            "request_id": self.request_id,
        }

    async def answer(self, text: str, *, show_alert: bool = False) -> None:
        """Answer the callback query if available."""

        if self.query is None:
            return
        try:
            await self.query.answer(text=text, show_alert=show_alert)
        except Exception:  # pragma: no cover - network errors are ignored
            pass


ButtonFactory = Callable[[ButtonSpec], ButtonHandler]
