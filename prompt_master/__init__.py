"""Prompt-Master package exports."""

from importlib import util
from pathlib import Path

from .generator import (
    Engine,
    PromptPayload,
    build_animate_prompt,
    build_banana_prompt,
    build_mj_prompt,
    build_prompt,
    build_suno_prompt,
    build_veo_prompt,
)

_LEGACY_PATH = Path(__file__).resolve().parent.parent / "prompt_master.py"
_LEGACY_SPEC = util.spec_from_file_location("_prompt_master_legacy", _LEGACY_PATH)
if _LEGACY_SPEC and _LEGACY_SPEC.loader:  # pragma: no cover - import safety
    _legacy = util.module_from_spec(_LEGACY_SPEC)
    _LEGACY_SPEC.loader.exec_module(_legacy)
    legacy_build_video_prompt = _legacy.build_video_prompt
    legacy_build_banana_json = _legacy.build_banana_json
    legacy_build_mj_json = _legacy.build_mj_json
    legacy_build_animate_prompt = _legacy.build_animate_prompt
    legacy_build_suno_prompt = _legacy.build_suno_prompt
else:  # pragma: no cover - unexpected environment
    legacy_build_video_prompt = legacy_build_banana_json = None
    legacy_build_mj_json = legacy_build_animate_prompt = None
    legacy_build_suno_prompt = None

__all__ = [
    "Engine",
    "PromptPayload",
    "build_prompt",
    "build_veo_prompt",
    "build_mj_prompt",
    "build_banana_prompt",
    "build_animate_prompt",
    "build_suno_prompt",
    "legacy_build_video_prompt",
    "legacy_build_banana_json",
    "legacy_build_mj_json",
    "legacy_build_animate_prompt",
    "legacy_build_suno_prompt",
]
