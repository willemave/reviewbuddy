import pytest

from app.services.youtube_transcriber import (
    YouTubeError,
    select_youtube_videos,
    transcribe_audio,
)


def test_select_youtube_videos_limits_and_dedupes() -> None:
    entries = [
        {"id": "abc123", "title": "A"},
        {"id": "abc123", "title": "A duplicate"},
        {"url": "https://youtu.be/def456", "title": "B"},
        {"url": "https://www.youtube.com/watch?v=ghi789", "title": "C"},
    ]

    videos = select_youtube_videos(entries, max_videos=2)
    assert len(videos) == 2
    assert videos[0].url.startswith("https://")


def test_select_youtube_videos_ignores_non_youtube() -> None:
    entries = [{"url": "https://example.com/not-youtube", "title": "Nope"}]
    videos = select_youtube_videos(entries, max_videos=3)
    assert videos == []


def test_transcribe_audio_rejects_empty_file(tmp_path) -> None:
    audio_path = tmp_path / "empty.mp3"
    audio_path.write_bytes(b"")

    with pytest.raises(YouTubeError) as excinfo:
        transcribe_audio(audio_path, model_name="base")
    assert "empty" in str(excinfo.value).lower()


def test_transcribe_audio_rejects_empty_samples(tmp_path, monkeypatch) -> None:
    audio_path = tmp_path / "audio.mp3"
    audio_path.write_bytes(b"data")

    monkeypatch.setattr(
        "app.services.youtube_transcriber._load_audio_samples", lambda _: []
    )
    monkeypatch.setattr(
        "app.services.youtube_transcriber._load_whisper_model", lambda _: object()
    )

    with pytest.raises(YouTubeError) as excinfo:
        transcribe_audio(audio_path, model_name="base")
    assert "no samples" in str(excinfo.value).lower()
