from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple


@dataclass(frozen=True)
class SunoModeConfig:
    key: str
    emoji: str
    title: str
    button_label: str
    required_fields: Tuple[str, ...]
    optional_fields: Tuple[str, ...] = ()
    default_tags: Tuple[str, ...] = ()


_MODE_CONFIGS: Dict[str, SunoModeConfig] = {
    "instrumental": SunoModeConfig(
        key="instrumental",
        emoji="🎼",
        title="Instrumental",
        button_label="🎼 Instrumental",
        required_fields=("title", "style"),
        default_tags=("ambient", "cinematic pads", "soft drums"),
    ),
    "lyrics": SunoModeConfig(
        key="lyrics",
        emoji="🎤",
        title="Vocal (with lyrics)",
        button_label="🎤 Vocal",
        required_fields=("title", "style", "lyrics"),
        default_tags=("pop", "ballad", "modern mix"),
    ),
    "cover": SunoModeConfig(
        key="cover",
        emoji="🎛️",
        title="Cover",
        button_label="🎛️ Cover",
        required_fields=("title", "reference"),
        optional_fields=("style",),
        default_tags=("cover", "remake", "modern mix"),
    ),
}


FIELD_LABELS: Dict[str, str] = {
    "title": "Title",
    "style": "Style / tags",
    "lyrics": "Lyrics",
    "reference": "Reference",
}


FIELD_ICONS: Dict[str, str] = {
    "title": "🏷️",
    "style": "🎛️",
    "lyrics": "📝",
    "reference": "🎧",
}


FIELD_PROMPTS: Dict[str, str] = {
    "title": "Enter a short track title.",
    "style": "Describe style/tags (e.g., ‘ambient, soft drums’).",
    "lyrics": "Paste lyrics (multi-line).",
    "reference": "Send an audio file or URL to the reference track.",
}


def get_mode_config(key: str) -> SunoModeConfig:
    return _MODE_CONFIGS.get(key, _MODE_CONFIGS["instrumental"])


def iter_mode_configs() -> Iterable[SunoModeConfig]:
    return _MODE_CONFIGS.values()


def default_style_text(key: str) -> str:
    config = get_mode_config(key)
    if not config.default_tags:
        return ""
    return ", ".join(config.default_tags)


__all__ = [
    "FIELD_ICONS",
    "FIELD_LABELS",
    "FIELD_PROMPTS",
    "SunoModeConfig",
    "default_style_text",
    "get_mode_config",
    "iter_mode_configs",
]
