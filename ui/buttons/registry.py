from __future__ import annotations

from typing import Dict

from .handlers import (
    open_dialog,
    open_help,
    open_kb,
    open_music,
    open_photo,
    open_profile,
    open_sora2,
    open_video,
)
from .types import ButtonId, ButtonSpec


BUTTONS: Dict[ButtonId, ButtonSpec] = {
    "profile": ButtonSpec(
        id="profile",
        title_i18n_key="button.profile",
        access=("all",),
        handler=open_profile,
        open_telemetry_event="ui.button.open",
    ),
    "kb": ButtonSpec(
        id="kb",
        title_i18n_key="button.kb",
        access=("all",),
        handler=open_kb,
        open_telemetry_event="ui.button.open",
    ),
    "photo": ButtonSpec(
        id="photo",
        title_i18n_key="button.photo",
        access=("all",),
        handler=open_photo,
        open_telemetry_event="ui.button.open",
    ),
    "music": ButtonSpec(
        id="music",
        title_i18n_key="button.music",
        access=("paid",),
        handler=open_music,
        open_telemetry_event="ui.button.open",
    ),
    "video": ButtonSpec(
        id="video",
        title_i18n_key="button.video",
        access=("all",),
        handler=open_video,
        open_telemetry_event="ui.button.open",
    ),
    "dialog": ButtonSpec(
        id="dialog",
        title_i18n_key="button.dialog",
        access=("all",),
        handler=open_dialog,
        open_telemetry_event="ui.button.open",
    ),
    "help": ButtonSpec(
        id="help",
        title_i18n_key="button.help",
        access=("all",),
        handler=open_help,
        open_telemetry_event="ui.button.open",
    ),
    "sora2": ButtonSpec(
        id="sora2",
        title_i18n_key="button.sora2",
        access=("paid",),
        handler=open_sora2,
        open_telemetry_event="ui.button.open",
        feature_flag="FEATURE_SORA2_ENABLED",
    ),
}
