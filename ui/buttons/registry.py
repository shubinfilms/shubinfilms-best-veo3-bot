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
    open_sum,
    open_video,
)
from .types import ButtonAction, ButtonId, ButtonSpec


def btn_data(action: str) -> str:
    normalized = (action or "").strip().lower()
    if not normalized:
        raise ValueError("action must be provided")
    return f"btn:{normalized}"


REGISTRY: Dict[str, ButtonAction] = {
    "profile": ButtonAction(
        action_id="profile",
        handler="handlers.profile:open",
        feature_flag=None,
        acl=None,
    ),
    "sum": ButtonAction(
        action_id="sum",
        handler="handlers.sum:open_via_registry",
        feature_flag=None,
        acl=None,
    ),
}


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
    "sum": ButtonSpec(
        id="sum",
        title_i18n_key="button.sum",
        access=("all",),
        handler=open_sum,
        open_telemetry_event="ui.button.open",
    ),
}


def _home_button_callbacks() -> Dict[str, str]:
    from keyboards import (
        HOME_CB_DIALOG,
        HOME_CB_KB,
        HOME_CB_MUSIC,
        HOME_CB_PHOTO,
        HOME_CB_PROFILE,
        HOME_CB_VIDEO,
    )

    return {
        "profile": HOME_CB_PROFILE,
        "kb": HOME_CB_KB,
        "photo": HOME_CB_PHOTO,
        "music": HOME_CB_MUSIC,
        "video": HOME_CB_VIDEO,
        "dialog": HOME_CB_DIALOG,
    }


def iter_home_button_mappings() -> Dict[str, str]:
    """Return the mapping of registry ids to home keyboard payloads."""

    return dict(_home_button_callbacks())


def registry_keyboard_inconsistencies() -> Dict[str, str]:
    """Return buttons whose keyboard payload does not align with the registry."""

    inconsistencies: Dict[str, str] = {}
    for button_id, callback in _home_button_callbacks().items():
        spec = BUTTONS.get(button_id)
        if spec is None:
            inconsistencies[button_id] = "<missing-spec>"
            continue
        payload = str(callback)
        if not payload:
            inconsistencies[button_id] = "<empty>"
            continue
        if button_id not in payload:
            inconsistencies[button_id] = payload
    return inconsistencies


def assert_registry_matches_keyboard() -> None:
    mismatches = registry_keyboard_inconsistencies()
    if mismatches:
        raise AssertionError(
            "Keyboard callback payloads do not match registry ids: "
            + ", ".join(f"{key} -> {value}" for key, value in sorted(mismatches.items()))
        )
