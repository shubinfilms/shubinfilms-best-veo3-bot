import asyncio
import base64
import importlib
from pathlib import Path
import sys

from mutagen.id3 import ID3

sys.path.append(str(Path(__file__).resolve().parents[1]))
audio_post = importlib.import_module("utils.audio_post")

MP3_BYTES = b"ID3\x03\x00\x00\x00\x00\x00\x00" + b"\x00" * 16
PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)


def test_audio_tags_with_cover_ok(monkeypatch, tmp_path):
    async def fake_fetch(url: str, *, timeout=audio_post._DEFAULT_TIMEOUT):  # type: ignore[attr-defined]
        if url.endswith("audio.mp3"):
            return MP3_BYTES, "audio/mpeg"
        if url.endswith("cover.png"):
            return PNG_BYTES, "image/png"
        raise AssertionError("unexpected url")

    def fake_task_directory(_: str) -> Path:
        target = tmp_path / "audio"
        target.mkdir(parents=True, exist_ok=True)
        return target

    async def _run() -> None:
        monkeypatch.setattr(audio_post, "_fetch_bytes", fake_fetch)
        monkeypatch.setattr("suno.tempfiles.task_directory", fake_task_directory)

        local_path, meta = await audio_post.prepare_audio_file(
            "https://example.com/audio.mp3",
            title="Neon City",
            cover_url="https://example.com/cover.png",
            default_artist="Best VEO3",
            max_name=40,
            tags_enabled=True,
            embed_cover_enabled=True,
            transliterate=False,
        )

        id3 = ID3(local_path)
        assert id3.getall("TIT2")[0].text[0] == "Neon City"
        assert id3.getall("TPE1")[0].text[0] == "Best VEO3"
        apic_frames = id3.getall("APIC")
        assert apic_frames and apic_frames[0].mime == "image/png"
        assert meta["file_name"] == "Neon_City.mp3"
        Path(local_path).unlink(missing_ok=True)

    asyncio.run(_run())
