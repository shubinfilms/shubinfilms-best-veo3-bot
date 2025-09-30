import asyncio
import importlib
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
audio_post = importlib.import_module("utils.audio_post")

MP3_BYTES = b"ID3\x03\x00\x00\x00\x00\x00\x00" + b"\x00" * 16


def test_audio_tags_fallback_on_error(monkeypatch, tmp_path):
    async def fake_fetch(url: str, *, timeout=audio_post._DEFAULT_TIMEOUT):  # type: ignore[attr-defined]
        return MP3_BYTES, "audio/mpeg"

    def fake_task_directory(_: str) -> Path:
        target = tmp_path / "audio"
        target.mkdir(parents=True, exist_ok=True)
        return target

    def boom(*_, **__):
        raise RuntimeError("id3 failed")

    async def _run() -> None:
        monkeypatch.setattr(audio_post, "_fetch_bytes", fake_fetch)
        monkeypatch.setattr("suno.tempfiles.task_directory", fake_task_directory)
        monkeypatch.setattr(audio_post, "_apply_id3_tags", boom)

        local_path, meta = await audio_post.prepare_audio_file(
            "https://example.com/audio.mp3",
            title="Fallback",
            cover_url=None,
            default_artist="Artist",
            max_name=40,
            tags_enabled=True,
            embed_cover_enabled=False,
            transliterate=False,
        )

        assert Path(local_path).exists()
        assert meta["title"] == "Fallback"
        Path(local_path).unlink(missing_ok=True)

    asyncio.run(_run())
