import asyncio
import importlib
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
audio_post = importlib.import_module("utils.audio_post")

MP3_BYTES = b"ID3\x03\x00\x00\x00\x00\x00\x00" + b"\x00" * 16

def test_audio_filename_transliterate(monkeypatch, tmp_path):
    async def fake_fetch(url: str, *, timeout=audio_post._DEFAULT_TIMEOUT):  # type: ignore[attr-defined]
        assert url == "https://example.com/audio.mp3"
        return MP3_BYTES, "audio/mpeg"

    def fake_task_directory(_: str) -> Path:
        target = tmp_path / "audio"
        target.mkdir(parents=True, exist_ok=True)
        return target

    async def _run() -> None:
        monkeypatch.setattr(audio_post, "_fetch_bytes", fake_fetch)
        monkeypatch.setattr("suno.tempfiles.task_directory", fake_task_directory)

        local_path, meta = await audio_post.prepare_audio_file(
            "https://example.com/audio.mp3",
            title="Путешествие по неону",
            cover_url=None,
            default_artist="Best VEO3",
            max_name=10,
            tags_enabled=False,
            embed_cover_enabled=False,
            transliterate=True,
        )

        assert meta["file_name"] == "Puteshestv.mp3"
        assert Path(local_path).exists()
        Path(local_path).unlink(missing_ok=True)

    asyncio.run(_run())
