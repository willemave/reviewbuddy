"""YouTube ingestion and Whisper transcription helpers."""

from __future__ import annotations

import json
import logging
import multiprocessing
import queue
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
YOUTUBE_CAPTION_LANGS = ("en", "en-US", "en-GB")
YOUTUBE_CAPTION_EXTENSIONS = (".json3", ".vtt", ".ttml", ".srv3")

logger = logging.getLogger(__name__)


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
    seen: set[str] = set()
    for entry in entries:
        url = entry.get("webpage_url") or entry.get("url") or entry.get("id")
        if not url:
            continue
        url = _normalize_youtube_url(url)
        if not is_youtube_url(url):
            continue
        if url in seen:
            continue
        seen.add(url)
        videos.append(YouTubeVideo(url=url, title=entry.get("title")))
        if len(videos) >= max_videos:
            break

    return videos[:max_videos]


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


def _captions_dir(base_dir: Path) -> Path:
    return base_dir / "_captions"


def _find_caption_file(output_dir: Path, video_id: str) -> Path | None:
    for extension in YOUTUBE_CAPTION_EXTENSIONS:
        matches = sorted(output_dir.glob(f"{video_id}*.{extension.lstrip('.')}"))
        if matches:
            return matches[0]
    return None


def _parse_json3_captions(caption_path: Path) -> str:
    payload = json.loads(caption_path.read_text(encoding="utf-8"))
    lines: list[str] = []
    for event in payload.get("events", []):
        if not isinstance(event, dict):
            continue
        segments = event.get("segs")
        if not isinstance(segments, list):
            continue
        line = "".join(
            segment.get("utf8", "")
            for segment in segments
            if isinstance(segment, dict) and isinstance(segment.get("utf8"), str)
        )
        line = re.sub(r"\s+", " ", line.replace("\n", " ")).strip()
        if not line:
            continue
        if lines and lines[-1] == line:
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _parse_vtt_captions(caption_path: Path) -> str:
    lines: list[str] = []
    for raw_line in caption_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line == "WEBVTT" or "-->" in line or line.isdigit():
            continue
        if line.startswith("NOTE"):
            continue
        line = re.sub(r"<[^>]+>", "", line)
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue
        if lines and lines[-1] == line:
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _parse_xml_captions(caption_path: Path) -> str:
    text = caption_path.read_text(encoding="utf-8")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _parse_caption_text(caption_path: Path) -> str:
    suffix = caption_path.suffix.lower()
    if suffix == ".json3":
        return _parse_json3_captions(caption_path)
    if suffix == ".vtt":
        return _parse_vtt_captions(caption_path)
    return _parse_xml_captions(caption_path)


def download_captions(video_url: str, output_dir: Path) -> tuple[str, str | None, str]:
    """Download YouTube subtitles or auto-captions and return transcript text.

    Args:
        video_url: YouTube URL.
        output_dir: Output directory for caption files.

    Returns:
        Tuple of (video id, title, transcript text).
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": list(YOUTUBE_CAPTION_LANGS),
        "subtitlesformat": "json3/vtt/best",
        "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
    except Exception as exc:
        raise YouTubeError(f"YouTube caption download failed: {exc}") from exc

    if not isinstance(info, dict):
        raise YouTubeError("Failed to download YouTube captions")

    video_id = info.get("id")
    if not isinstance(video_id, str) or not video_id:
        raise YouTubeError("YouTube caption download returned no video id")

    caption_path = _find_caption_file(output_dir, video_id)
    if caption_path is None:
        raise YouTubeError("No YouTube captions were available")

    transcript_text = _parse_caption_text(caption_path)
    if not transcript_text:
        raise YouTubeError("Downloaded YouTube captions were empty")

    return video_id, info.get("title"), transcript_text


def extract_youtube_transcript(
    video_url: str,
    audio_dir: Path,
    transcript_dir: Path,
    model_name: str,
) -> tuple[str, str | None, str]:
    """Extract a YouTube transcript using captions first, then Whisper.

    Args:
        video_url: YouTube URL.
        audio_dir: Output directory for audio downloads.
        transcript_dir: Output directory for transcript artifacts.
        model_name: Whisper model name for audio fallback.

    Returns:
        Tuple of (transcript id, title, transcript text).
    """

    try:
        return download_captions(video_url, _captions_dir(transcript_dir))
    except YouTubeError as exc:
        logger.info("Falling back to audio transcription for %s: %s", video_url, exc)

    audio_path, title = download_audio(video_url, audio_dir)
    transcript_text = transcribe_audio(audio_path, model_name=model_name)
    return audio_path.stem, title, transcript_text


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
        try:
            transcript_id, title, transcript_text = extract_youtube_transcript(
                video.url,
                audio_dir,
                transcript_dir,
                model_name,
            )
        except YouTubeError as exc:
            logger.warning("Skipping YouTube video %s: %s", video.url, exc)
            continue

        transcript_path = transcript_dir / f"{transcript_id}.txt"
        transcript_path.write_text(transcript_text, encoding="utf-8")
        transcripts.append(
            YouTubeTranscript(
                url=video.url,
                title=video.title or title,
                transcript=transcript_text,
            )
        )

    return transcripts


def _transcribe_youtube_videos_worker(
    prompt: str,
    max_videos: int,
    audio_dir: Path,
    transcript_dir: Path,
    model_name: str,
    result_queue,
) -> None:
    """Run YouTube transcription work inside a child process."""

    try:
        transcripts = transcribe_youtube_videos(
            prompt=prompt,
            max_videos=max_videos,
            audio_dir=audio_dir,
            transcript_dir=transcript_dir,
            model_name=model_name,
        )
    except Exception as exc:  # noqa: BLE001
        result_queue.put(
            {
                "status": "error",
                "error": str(exc),
            }
        )
        return

    result_queue.put(
        {
            "status": "ok",
            "transcripts": [item.model_dump() for item in transcripts],
        }
    )


def transcribe_youtube_videos_with_timeout(
    prompt: str,
    max_videos: int,
    audio_dir: Path,
    transcript_dir: Path,
    model_name: str,
    timeout_seconds: int | float | None,
) -> list[YouTubeTranscript]:
    """Run YouTube transcription in a killable child process."""

    if timeout_seconds is None or timeout_seconds <= 0:
        return transcribe_youtube_videos(
            prompt=prompt,
            max_videos=max_videos,
            audio_dir=audio_dir,
            transcript_dir=transcript_dir,
            model_name=model_name,
        )

    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_transcribe_youtube_videos_worker,
        args=(
            prompt,
            max_videos,
            audio_dir,
            transcript_dir,
            model_name,
            result_queue,
        ),
    )

    try:
        process.start()
        process.join(timeout_seconds)
        if process.is_alive():
            process.terminate()
            process.join()
            if process.is_alive():
                process.kill()
                process.join()
            raise TimeoutError(
                f"YouTube ingestion timed out after {timeout_seconds} seconds"
            )

        try:
            payload = result_queue.get_nowait()
        except queue.Empty as exc:
            raise YouTubeError(
                "YouTube transcription worker exited without returning results"
            ) from exc

        if not isinstance(payload, dict):
            raise YouTubeError("YouTube transcription worker returned an invalid payload")

        if payload.get("status") == "error":
            raise YouTubeError(str(payload.get("error") or "Unknown YouTube worker error"))

        if payload.get("status") != "ok":
            raise YouTubeError("YouTube transcription worker returned an unknown status")

        transcripts = payload.get("transcripts", [])
        if not isinstance(transcripts, list):
            raise YouTubeError("YouTube transcription worker returned invalid transcripts")
        return [YouTubeTranscript.model_validate(item) for item in transcripts]
    finally:
        if hasattr(result_queue, "close"):
            result_queue.close()
