from app.services.youtube_transcriber import select_youtube_videos


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
