import json
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from prompt_master import build_veo_prompt


def _extract_timeslot(value: str) -> tuple[float, float]:
    match = re.match(r"([0-9.]+)–([0-9.]+)s", value)
    assert match is not None
    return float(match.group(1)), float(match.group(2))


def test_veo_generator_smoke_structure() -> None:
    payload = build_veo_prompt("Сумеречный город и герой в поиске", "ru")
    data = json.loads(payload["copy_text"])
    assert data["idea"].startswith("Сумеречный")
    assert data["scene_description"]["setting"]
    assert len(data["timeline"]) == 4
    assert data["timeline"][0]["t"] == "0–2s"
    assert "ambience" in data["audio"] and "sfx" in data["audio"]
    assert "--ar" not in payload["copy_text"]
    assert "lip-sync required" not in data["notes"]


def test_veo_voiceover_inline_preserved() -> None:
    user_text = (
        "Два героя спорят у окна во время грозы. Озвучка:\n"
        "А: Я не собирался скрывать правду.\n"
        "Б: Тогда почему молчал?\n"
        "А: Хотел тебя защитить."
    )
    payload = build_veo_prompt(user_text, "ru")
    data = json.loads(payload["copy_text"])
    voiceover = data["audio"].get("voiceover")
    assert voiceover is not None and len(voiceover) == 3
    assert [line["line"] for line in voiceover] == [
        "Я не собирался скрывать правду.",
        "Тогда почему молчал?",
        "Хотел тебя защитить.",
    ]
    assert "lip-sync required" in data["notes"]
    assert payload["raw_payload"]["voiceover_origin"] == "provided"
    for slot in voiceover:
        start, end = _extract_timeslot(slot["time"])
        assert 0.0 <= start < end <= 8.0


def test_veo_generates_voiceover_when_requested() -> None:
    user_text = "В классе учитель и ученик находят решение сложной задачи. Напишите короткий диалог."
    payload = build_veo_prompt(user_text, "ru")
    data = json.loads(payload["copy_text"])
    voiceover = data["audio"].get("voiceover")
    assert voiceover is not None and 3 <= len(voiceover) <= 5
    assert "lip-sync required" in data["notes"]
    assert payload["raw_payload"]["voiceover_origin"] == "generated"
    lengths = [len(line["line"]) for line in voiceover]
    assert max(lengths) < 120
    for slot in voiceover:
        start, end = _extract_timeslot(slot["time"])
        assert 0.0 <= start < end <= 8.0


def test_veo_features_influence_text() -> None:
    payload = build_veo_prompt("Два героя спорят у окна ночью во время грозы", "ru")
    data = json.loads(payload["copy_text"])
    assert "молни" in data["lighting"].lower()
    assert "дожд" in data["audio"]["ambience"].lower()
    assert any(keyword in data["details"].lower() for keyword in ("гроз", "дожд"))


def test_veo_generator_english_locale() -> None:
    payload = build_veo_prompt("Two detectives argue under neon rain", "en")
    data = json.loads(payload["copy_text"])
    assert "Video ≈ 8 seconds" in payload["body_html"]
    assert data["notes"].startswith("8s total")
    assert "neon" in data["details"].lower() or "rain" in data["audio"]["ambience"].lower()
    assert payload["raw_payload"]["voiceover_origin"] == "none"

