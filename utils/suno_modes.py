from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple


from texts import t


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
        title=t("suno.mode.instrumental"),
        button_label=f"🎼 {t('suno.mode.instrumental')}",
        required_fields=("title", "style"),
        default_tags=("ambient", "cinematic pads", "soft drums"),
    ),
    "lyrics": SunoModeConfig(
        key="lyrics",
        emoji="🎤",
        title=t("suno.mode.vocal"),
        button_label=f"🎤 {t('suno.mode.vocal')}",
        required_fields=("title", "style", "lyrics"),
        default_tags=("pop", "ballad", "modern mix"),
    ),
    "cover": SunoModeConfig(
        key="cover",
        emoji="🎚️",
        title=t("suno.mode.cover"),
        button_label=f"🎚️ {t('suno.mode.cover')}",
        required_fields=("title", "reference"),
        optional_fields=("style",),
        default_tags=("cover", "remake", "modern mix"),
    ),
}


FIELD_LABELS: Dict[str, str] = {
    "title": t("suno.field.title"),
    "style": t("suno.field.style"),
    "lyrics": t("suno.field.lyrics"),
    "reference": t("suno.field.source"),
}


FIELD_ICONS: Dict[str, str] = {
    "title": "✏️",
    "style": "🎛️",
    "lyrics": "📝",
    "reference": "🎧",
}


FIELD_PROMPTS: Dict[str, str] = {
    "title": t("suno.prompt.step.title", index=1, total=1),
    "style": t("suno.prompt.step.style", index=1, total=1),
    "lyrics": t("suno.prompt.step.lyrics", index=1, total=1),
    "reference": t("suno.prompt.step.source", index=1, total=1),
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
