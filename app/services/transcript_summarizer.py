"""YouTube transcript summarization helpers."""

from __future__ import annotations

import asyncio
import logging

from pydantic_ai import Agent
from pydantic_ai.usage import RunUsage

from app.core.settings import get_settings
from app.services.youtube_transcriber import YouTubeTranscript

logger = logging.getLogger(__name__)
settings = get_settings()

SUMMARY_SYSTEM_PROMPT = (
    "You summarize transcripts for research synthesis. Keep only high-signal content and "
    "avoid filler. Return concise markdown with sections: Highlights, Quantitative Details, "
    "Caveats, and Who It Fits. Use only information present in the transcript."
)


class TranscriptSummaryError(RuntimeError):
    """Raised when transcript summarization cannot be performed."""


def _clip_text(value: str, max_chars: int) -> str:
    text = value.strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


def _build_flash_agent(model_name: str, api_key: str) -> Agent:
    try:
        from pydantic_ai.models.google import GoogleModel
        from pydantic_ai.providers.google import GoogleProvider
    except ImportError as exc:
        raise TranscriptSummaryError("Google model provider dependencies are not installed") from exc

    provider = GoogleProvider(api_key=api_key)
    model = GoogleModel(model_name, provider=provider)
    return Agent(model=model, output_type=str, system_prompt=SUMMARY_SYSTEM_PROMPT)


async def summarize_transcript(
    transcript: str,
    title: str | None,
    url: str,
    model_name: str,
    max_chars: int,
) -> tuple[str, RunUsage | None]:
    """Summarize a transcript with Gemini Flash and return compact markdown.

    Args:
        transcript: Raw transcript text.
        title: Optional video title.
        url: Source URL.
        model_name: Flash model name.
        max_chars: Maximum output size.

    Returns:
        Tuple of (summary_text, usage).
    """

    if not transcript.strip():
        return "", None

    api_key = settings.google_api_key
    if not api_key:
        logger.warning("GOOGLE_API_KEY not configured; clipping transcript for %s", url)
        return _clip_text(transcript, max_chars), None

    try:
        agent = _build_flash_agent(model_name, api_key)
        prompt = (
            "Summarize this YouTube transcript for research usage.\n"
            "Focus on concrete claims, quantitative details, caveats, and recommendation fit.\n\n"
            f"Title: {title or '(untitled)'}\n"
            f"URL: {url}\n\n"
            f"Transcript:\n{transcript}"
        )
        result = await agent.run(prompt)
        summary = _clip_text(str(result.output), max_chars)
        if not summary:
            summary = _clip_text(transcript, max_chars)
        return summary, result.usage()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Transcript summarization failed for %s (%s)", url, exc)
        return _clip_text(transcript, max_chars), None


async def summarize_youtube_transcripts(
    transcripts: list[YouTubeTranscript],
    model_name: str,
    max_chars: int,
    concurrency: int,
) -> tuple[list[YouTubeTranscript], list[RunUsage]]:
    """Summarize all transcripts with bounded concurrency.

    Args:
        transcripts: Transcript records to summarize.
        model_name: Flash model name.
        max_chars: Maximum chars per summary.
        concurrency: Max concurrent summarization calls.

    Returns:
        Tuple of (updated transcripts, usage records).
    """

    if not transcripts:
        return [], []

    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def _summarize_one(transcript: YouTubeTranscript) -> tuple[YouTubeTranscript, RunUsage | None]:
        async with semaphore:
            summary, usage = await summarize_transcript(
                transcript=transcript.transcript,
                title=transcript.title,
                url=transcript.url,
                model_name=model_name,
                max_chars=max_chars,
            )
            updated = transcript.model_copy(update={"transcript": summary})
            return updated, usage

    results = await asyncio.gather(*(_summarize_one(item) for item in transcripts))
    updated_transcripts = [item[0] for item in results]
    usages = [usage for _, usage in results if usage is not None]
    return updated_transcripts, usages
