from pathlib import Path

from suno.service import SunoService


def test_send_audio_uses_prepared_file(monkeypatch, tmp_path):
    monkeypatch.setenv("AUDIO_TAGS_ENABLED", "true")
    monkeypatch.setenv("AUDIO_DEFAULT_ARTIST", "Tester")
    monkeypatch.setenv("AUDIO_EMBED_COVER_ENABLED", "true")
    monkeypatch.setenv("AUDIO_FILENAME_TRANSLITERATE", "true")
    monkeypatch.setenv("AUDIO_FILENAME_MAX", "32")
    monkeypatch.setenv("TELEGRAM_TOKEN", "dummy")

    prepared_path = tmp_path / "prepared.mp3"
    prepared_path.write_bytes(b"mp3")

    service = SunoService()

    def fake_prepare(*_, **__):
        return str(prepared_path), {
            "file_name": "Prepared.mp3",
            "title": "Song",
            "performer": "Tester",
        }

    calls: list[dict[str, object]] = []

    def fake_send_file(
        method: str,
        field: str,
        chat_id: int,
        local_path: Path,
        *,
        caption,
        reply_to,
        extra,
        file_name,
    ) -> bool:
        calls.append(
            {
                "method": method,
                "field": field,
                "chat_id": chat_id,
                "path": local_path,
                "file_name": file_name,
                "extra": extra,
                "caption": caption,
            }
        )
        return True

    def fail_send_audio_request(*args, **kwargs):  # pragma: no cover - defensive
        raise AssertionError("send_audio_request should not be called")

    monkeypatch.setattr("utils.audio_post.prepare_audio_file_sync", fake_prepare)
    monkeypatch.setattr("suno.service.prepare_audio_file_sync", fake_prepare)
    monkeypatch.setattr(service, "_send_file", fake_send_file)
    monkeypatch.setattr("suno.service.schedule_unlink", lambda path: None)
    monkeypatch.setattr("telegram_utils.send_audio_request", fail_send_audio_request)

    success, reason = service._send_audio_url_with_retry(
        chat_id=123,
        audio_url="https://example.com/audio.mp3",
        caption="Caption",
        reply_to=None,
        title="Title",
        thumb="https://example.com/cover.png",
        base_dir=tmp_path,
        take_id="take1",
    )

    assert success is True
    assert reason is None
    assert len(calls) == 2
    first_call, second_call = calls
    assert first_call["method"] == "sendAudio"
    assert first_call["file_name"] == "Prepared.mp3"
    assert first_call["extra"] == {"title": "Song", "performer": "Tester"}
    assert first_call["path"] == prepared_path
    assert second_call["method"] == "sendDocument"
    assert second_call["file_name"] == "Prepared.mp3"
    assert second_call["extra"] is None
