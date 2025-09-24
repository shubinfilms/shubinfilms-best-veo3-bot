"""Pydantic schemas for Suno callbacks and responses."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class CallbackResponse(BaseModel):
    """Arbitrary payload returned inside the callback envelope."""

    model_config = {"extra": "allow"}


class CallbackData(BaseModel):
    """Data field of callback payload."""

    task_id: Optional[str] = Field(default=None, alias="taskId")
    callback_type: Optional[str] = Field(default=None, alias="callbackType")
    status: Optional[str] = None
    response: Optional[CallbackResponse] = None

    model_config = {"populate_by_name": True, "extra": "allow"}


class CallbackEnvelope(BaseModel):
    """Top-level callback payload."""

    code: Optional[int] = 200
    msg: Optional[str] = None
    data: Optional[CallbackData] = None

    model_config = {"extra": "allow"}


class TaskAsset(BaseModel):
    """Normalized representation of downloadable asset."""

    task_id: str
    url: str
    asset_type: str
    identifier: Optional[str] = None
    filename: Optional[str] = None

    model_config = {"populate_by_name": True, "extra": "allow"}


class CallbackEvent(BaseModel):
    """Normalized callback event that service operates with."""

    task_id: str
    callback_type: str
    payload: dict[str, Any]
    assets: list[TaskAsset] = Field(default_factory=list)
    status: Optional[str] = None
    code: Optional[int] = None
    message: Optional[str] = None

    model_config = {"populate_by_name": True, "extra": "allow"}
