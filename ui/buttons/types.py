from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Protocol, Sequence

import inspect
from importlib import import_module

from telegram import CallbackQuery, Update
from telegram.ext import ContextTypes

from .results import UIResult
from telegram_utils import safe_answer

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
    ack_result: Optional[str] = None
    acknowledged: bool = False

    def as_dict(self) -> dict[str, object]:
        """Return a structured representation for logging/debugging."""

        return {
            "button_id": self.spec.id,
            "chat_id": self.chat_id,
            "user_id": self.user_id,
            "request_id": self.request_id,
        }

    def mark_acknowledged(self, result: str) -> None:
        self.ack_result = result
        if result != "error":
            self.acknowledged = True

    async def answer(self, text: str, *, show_alert: bool = False) -> None:
        """Answer the callback query if available."""

        if self.query is None:
            return
        result = await safe_answer(self.query, text=text, show_alert=show_alert)
        self.mark_acknowledged(result)


ButtonFactory = Callable[[ButtonSpec], ButtonHandler]


class ActionHandler(Protocol):
    async def __call__(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        payload: Optional[dict[str, object]] = None,
    ) -> None:  # pragma: no cover - protocol
        ...


@dataclass(slots=True)
class ButtonAction:
    action_id: str
    handler: ActionHandler | str
    feature_flag: Optional[str] = None
    acl: Optional[str] = None

    def resolve_handler(self) -> ActionHandler:
        if callable(self.handler):
            return self.handler
        module_path, _, attr = self.handler.partition(":")
        if not module_path or not attr:
            raise ValueError(f"invalid handler path: {self.handler!r}")
        module = import_module(module_path)
        candidate = getattr(module, attr)
        if not callable(candidate):
            raise TypeError(f"handler {self.handler!r} is not callable")

        signature = inspect.signature(candidate)
        accepts_payload = any(
            param.name == "payload"
            and param.kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
            for param in signature.parameters.values()
        ) or any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())

        async def _call(
            update: Update,
            context: ContextTypes.DEFAULT_TYPE,
            payload: Optional[dict[str, object]] = None,
        ) -> None:
            if accepts_payload:
                result = candidate(update, context, payload=payload)
            else:
                result = candidate(update, context)
            if inspect.isawaitable(result):
                await result

        return _call
