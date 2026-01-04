"""YouTube ingestion and Whisper transcription helpers."""

from __future__ import annotations

import re
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

import yt_dlp
from pydantic import BaseModel

from app.core.settings import get_settings


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
    """Select unique YouTube videos from yt-dlp search entries.

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
    """Search YouTube via yt-dlp.

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
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": True,
        "default_search": f"ytsearch{max_videos}",
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(prompt, download=False)
    except Exception as exc:
        raise YouTubeError(f"YouTube search failed: {exc}") from exc

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
        "no_warnings": True,
        "noplaylist": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
    except Exception as exc:
        raise YouTubeError(f"YouTube download failed: {exc}") from exc

    if not isinstance(info, dict):
        raise YouTubeError("Failed to download YouTube audio")

    audio_path = output_dir / f"{info.get('id')}.mp3"
    if not audio_path.exists():
        raise YouTubeError("Audio file not found after download")
    return audio_path, info.get("title")


def _resolve_whisper_device(device_setting: str) -> str:
    try:
        import torch
    except ImportError as exc:
        raise YouTubeError("Torch is required for Whisper") from exc

    if device_setting != "auto":
        return device_setting

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        # MPS can be unstable with Whisper; default to CPU.
        return "cpu"
    return "cpu"


@lru_cache
def _load_whisper_model(model_name: str, device: str):
    try:
        import whisper
    except ImportError as exc:
        raise YouTubeError("Whisper is not installed") from exc

    return whisper.load_model(model_name, device=device)


def _load_audio_samples(audio_path: Path):
    try:
        import whisper
    except ImportError as exc:
        raise YouTubeError("Whisper is not installed") from exc

    try:
        return whisper.load_audio(str(audio_path))
    except Exception as exc:
        raise YouTubeError(f"Failed to decode audio: {exc}") from exc


def transcribe_audio(audio_path: Path, model_name: str) -> str:
    """Transcribe audio locally using Whisper.

    Args:
        audio_path: Audio file path.
        model_name: Whisper model name.

    Returns:
        Transcript text.
    """

    if not audio_path.exists():
        raise YouTubeError("Audio file missing for Whisper")
    if audio_path.stat().st_size == 0:
        raise YouTubeError("Audio file is empty")

    audio = _load_audio_samples(audio_path)
    if not hasattr(audio, "__len__") or len(audio) == 0:
        raise YouTubeError("Audio decode returned no samples")

    settings = get_settings()
    device = _resolve_whisper_device(settings.whisper_device)
    model = _load_whisper_model(model_name, device)
    try:
        result = model.transcribe(
            str(audio_path),
            fp16=device != "cpu",
            language=None,
            task="transcribe",
            verbose=False,
        )
    except RuntimeError as exc:
        message = str(exc).lower()
        if "cannot reshape tensor of 0 elements" in message:
            raise YouTubeError("Whisper failed on empty audio") from exc
        if ("mps" in message or "sparse" in message or "_sparse_coo_tensor" in message) and (
            device != "cpu"
        ):
            cpu_model = _load_whisper_model(model_name, "cpu")
            result = cpu_model.transcribe(
                str(audio_path),
                fp16=False,
                language=None,
                task="transcribe",
                verbose=False,
            )
        else:
            raise YouTubeError(f"Whisper failed: {exc}") from exc

    text = result.get("text", "") if isinstance(result, dict) else ""
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
