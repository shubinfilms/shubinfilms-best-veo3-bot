from __future__ import annotations

import os
from typing import Any, Dict

from pydantic import BaseModel


class SettingsConfigDict(dict):
    """Lightweight stand-in for :class:`pydantic_settings.SettingsConfigDict`."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - simple wrapper
        super().__init__(*args, **kwargs)


class BaseSettings(BaseModel):
    """Minimal fallback implementation used in tests."""

    model_config: SettingsConfigDict = SettingsConfigDict()

    def __init__(self, **data: Any) -> None:  # pragma: no cover - simple env loader
        env_values: Dict[str, Any] = {}
        for field_name in self.model_fields:
            env_key = field_name.upper()
            if env_key in os.environ:
                env_values[field_name] = os.environ[env_key]
        env_values.update(data)
        super().__init__(**env_values)

    def model_dump(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:  # pragma: no cover - passthrough
        return super().model_dump(*args, **kwargs)
