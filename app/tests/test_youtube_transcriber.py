import pytest

from app.services.youtube_transcriber import (
    YouTubeError,
    YouTubeVideo,
    _parse_json3_captions,
    extract_youtube_transcript,
    select_youtube_videos,
    transcribe_audio,
    transcribe_youtube_videos,
    transcribe_youtube_videos_with_timeout,
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

    monkeypatch.setattr("app.services.youtube_transcriber._load_audio_samples", lambda _: [])
    monkeypatch.setattr(
        "app.services.youtube_transcriber._load_whisper_model",
        lambda *_args, **_kwargs: object(),
    )

    with pytest.raises(YouTubeError) as excinfo:
        transcribe_audio(audio_path, model_name="base")
    assert "no samples" in str(excinfo.value).lower()


def test_parse_json3_captions_flattens_and_dedupes(tmp_path) -> None:
    caption_path = tmp_path / "captions.json3"
    caption_path.write_text(
        """
        {
          "events": [
            {"segs": [{"utf8": "first"}, {"utf8": " line"}]},
            {"segs": [{"utf8": "first line"}]},
            {"segs": [{"utf8": "second\\nline"}]},
            {"segs": [{"utf8": " "}]}
          ]
        }
        """,
        encoding="utf-8",
    )

    transcript = _parse_json3_captions(caption_path)

    assert transcript == "first line\nsecond line"


def test_extract_youtube_transcript_prefers_captions(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.youtube_transcriber.download_captions",
        lambda *_args, **_kwargs: ("abc123", "Video Title", "Caption transcript"),
    )

    def _fail(*_args, **_kwargs):
        raise AssertionError("audio fallback should not run")

    monkeypatch.setattr("app.services.youtube_transcriber.download_audio", _fail)

    transcript_id, title, transcript = extract_youtube_transcript(
        "https://www.youtube.com/watch?v=abc123",
        tmp_path / "audio",
        tmp_path / "transcripts",
        "base",
    )

    assert transcript_id == "abc123"
    assert title == "Video Title"
    assert transcript == "Caption transcript"


def test_extract_youtube_transcript_falls_back_to_audio(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.youtube_transcriber.download_captions",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(YouTubeError("no captions")),
    )
    monkeypatch.setattr(
        "app.services.youtube_transcriber.download_audio",
        lambda *_args, **_kwargs: (tmp_path / "audio" / "abc123.mp3", "Video Title"),
    )
    monkeypatch.setattr(
        "app.services.youtube_transcriber.transcribe_audio",
        lambda *_args, **_kwargs: "Whisper transcript",
    )

    transcript_id, title, transcript = extract_youtube_transcript(
        "https://www.youtube.com/watch?v=abc123",
        tmp_path / "audio",
        tmp_path / "transcripts",
        "base",
    )

    assert transcript_id == "abc123"
    assert title == "Video Title"
    assert transcript == "Whisper transcript"


def test_transcribe_youtube_videos_skips_failed_video(tmp_path, monkeypatch) -> None:
    videos = [
        YouTubeVideo(url="https://www.youtube.com/watch?v=bad111", title="Bad"),
        YouTubeVideo(url="https://www.youtube.com/watch?v=good222", title="Good"),
    ]
    monkeypatch.setattr("app.services.youtube_transcriber.search_youtube", lambda *_args, **_kwargs: videos)

    def _extract(video_url: str, *_args, **_kwargs):
        if "bad111" in video_url:
            raise YouTubeError("blocked")
        return "good222", "Recovered", "Transcript text"

    monkeypatch.setattr("app.services.youtube_transcriber.extract_youtube_transcript", _extract)

    transcripts = transcribe_youtube_videos(
        prompt="coffee",
        max_videos=2,
        audio_dir=tmp_path / "audio",
        transcript_dir=tmp_path / "transcripts",
        model_name="base",
    )

    assert len(transcripts) == 1
    assert transcripts[0].url.endswith("good222")
    assert transcripts[0].transcript == "Transcript text"
    assert (tmp_path / "transcripts" / "good222.txt").exists()


def test_transcribe_youtube_videos_with_timeout_terminates_worker(
    tmp_path,
    monkeypatch,
) -> None:
    class FakeQueue:
        def close(self) -> None:
            return

        def get_nowait(self):  # noqa: ANN201
            raise AssertionError("timed out worker should not return a payload")

    class FakeProcess:
        def __init__(self) -> None:
            self.started = False
            self.terminated = False
            self.killed = False

        def start(self) -> None:
            self.started = True

        def join(self, timeout=None) -> None:  # noqa: ANN001
            return

        def is_alive(self) -> bool:
            return not self.terminated and not self.killed

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.killed = True

    fake_process = FakeProcess()

    class FakeContext:
        def Queue(self, maxsize=1):  # noqa: N802, ANN001, ANN201
            return FakeQueue()

        def Process(self, target, args):  # noqa: N802, ANN001, ANN201
            del target, args
            return fake_process

    monkeypatch.setattr(
        "app.services.youtube_transcriber.multiprocessing.get_context",
        lambda method: FakeContext(),
    )

    with pytest.raises(TimeoutError, match="timed out"):
        transcribe_youtube_videos_with_timeout(
            prompt="coffee",
            max_videos=2,
            audio_dir=tmp_path / "audio",
            transcript_dir=tmp_path / "transcripts",
            model_name="base",
            timeout_seconds=5,
        )

    assert fake_process.started is True
    assert fake_process.terminated is True
