"""YouTube ingestion and Whisper transcription helpers."""

from __future__ import annotations

import re
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

import youtube_dl
from pydantic import BaseModel


class YouTubeError(RuntimeError):
    """Raised when YouTube ingestion fails."""


class YouTubeVideo(BaseModel):
    """YouTube video metadata."""

    url: str
    title: str | None = None


class YouTubeTranscript(BaseModel):
    """YouTube transcript with metadata."""

    url: str
    title: str | None = None
    transcript: str


YOUTUBE_URL_RE = re.compile(r"(https?://)?(www\.)?(youtube\.com|youtu\.be)/")
YOUTUBE_QUERY_TEMPLATES = (
    "{prompt} review",
    "{prompt} comparison",
    "{prompt} buyer guide",
)


def is_youtube_url(url: str) -> bool:
    """Check if a URL is a YouTube URL.

    Args:
        url: URL string.

    Returns:
        True if the URL is YouTube.
    """

    return bool(YOUTUBE_URL_RE.search(url))


def _normalize_youtube_url(value: str) -> str:
    if value.startswith("http"):
        return value
    return f"https://www.youtube.com/watch?v={value}"


def _dedupe_videos(videos: Iterable[YouTubeVideo]) -> list[YouTubeVideo]:
    seen: set[str] = set()
    unique: list[YouTubeVideo] = []
    for video in videos:
        if video.url in seen:
            continue
        seen.add(video.url)
        unique.append(video)
    return unique


def select_youtube_videos(entries: Iterable[dict], max_videos: int) -> list[YouTubeVideo]:
    """Select unique YouTube videos from youtube-dl search entries.

    Args:
        entries: Iterable of youtube-dl entries.
        max_videos: Maximum videos to return.

    Returns:
        List of YouTubeVideo records.
    """

    if max_videos <= 0:
        return []

    videos: list[YouTubeVideo] = []
    for entry in entries:
        url = entry.get("webpage_url") or entry.get("url") or entry.get("id")
        if not url:
            continue
        url = _normalize_youtube_url(url)
        if not is_youtube_url(url):
            continue
        videos.append(YouTubeVideo(url=url, title=entry.get("title")))
        if len(videos) >= max_videos:
            break

    return _dedupe_videos(videos)[:max_videos]


def search_youtube(prompt: str, max_videos: int) -> list[YouTubeVideo]:
    """Search YouTube via youtube-dl.

    Args:
        prompt: User prompt.
        max_videos: Maximum number of videos.

    Returns:
        List of YouTubeVideo results.
    """

    if max_videos <= 0:
        return []

    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "extract_flat": True,
        "default_search": f"ytsearch{max_videos}",
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(prompt, download=False)

    entries = info.get("entries", []) if isinstance(info, dict) else []
    return select_youtube_videos(entries, max_videos)


def download_audio(video_url: str, output_dir: Path) -> tuple[Path, str | None]:
    """Download YouTube audio locally.

    Args:
        video_url: YouTube URL.
        output_dir: Output directory for audio files.

    Returns:
        Tuple of (audio path, title).
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
        "quiet": True,
        "noplaylist": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)

    if not isinstance(info, dict):
        raise YouTubeError("Failed to download YouTube audio")

    audio_path = output_dir / f"{info.get('id')}.mp3"
    if not audio_path.exists():
        raise YouTubeError("Audio file not found after download")
    return audio_path, info.get("title")


@lru_cache
def _load_whisper_model(model_name: str):
    try:
        import whisper
    except ImportError as exc:
        raise YouTubeError("Whisper is not installed") from exc

    return whisper.load_model(model_name)


def transcribe_audio(audio_path: Path, model_name: str) -> str:
    """Transcribe audio locally using Whisper.

    Args:
        audio_path: Audio file path.
        model_name: Whisper model name.

    Returns:
        Transcript text.
    """

    model = _load_whisper_model(model_name)
    result = model.transcribe(str(audio_path))
    text = result.get("text", "")
    return text.strip()


def transcribe_youtube_videos(
    prompt: str,
    max_videos: int,
    audio_dir: Path,
    transcript_dir: Path,
    model_name: str,
) -> list[YouTubeTranscript]:
    """Search, download, and transcribe YouTube videos.

    Args:
        prompt: User prompt.
        max_videos: Maximum number of videos to transcribe.
        output_dir: Base output directory.
        model_name: Whisper model name.

    Returns:
        List of YouTubeTranscript results.
    """

    if max_videos <= 0:
        return []

    videos: list[YouTubeVideo] = []
    for template in YOUTUBE_QUERY_TEMPLATES:
        remaining = max_videos - len(videos)
        if remaining <= 0:
            break
        query = template.format(prompt=prompt)
        videos.extend(search_youtube(query, max_videos=remaining))

    videos = _dedupe_videos(videos)[:max_videos]
    if not videos:
        return []

    transcripts: list[YouTubeTranscript] = []
    transcript_dir.mkdir(parents=True, exist_ok=True)

    for video in videos[:max_videos]:
        audio_path, title = download_audio(video.url, audio_dir)
        transcript_text = transcribe_audio(audio_path, model_name=model_name)
        transcript_path = transcript_dir / f"{audio_path.stem}.txt"
        transcript_path.write_text(transcript_text, encoding="utf-8")
        transcripts.append(
            YouTubeTranscript(
                url=video.url,
                title=video.title or title,
                transcript=transcript_text,
            )
        )

    return transcripts
