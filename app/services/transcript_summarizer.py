"""YouTube transcript summarization helpers."""

from __future__ import annotations

import asyncio
import logging

from app.services.codex_exec import CodexUsage, run_codex_prompt
from app.services.youtube_transcriber import YouTubeTranscript

logger = logging.getLogger(__name__)

SUMMARY_SYSTEM_PROMPT = (
    "You summarize transcripts for research synthesis. Keep only high-signal content and "
    "avoid filler. Return concise markdown with sections: Highlights, Quantitative Details, "
    "Caveats, and Who It Fits. Use only information present in the transcript."
)


def _clip_text(value: str, max_chars: int) -> str:
    text = value.strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


async def summarize_transcript(
    transcript: str,
    title: str | None,
    url: str,
    model_name: str,
    max_chars: int,
) -> tuple[str, CodexUsage | None]:
    """Summarize a transcript and return compact markdown."""

    if not transcript.strip():
        return "", None

    try:
        response = await run_codex_prompt(
            (
                f"{SUMMARY_SYSTEM_PROMPT}\n\n"
                "Summarize this YouTube transcript for research usage.\n"
                "Focus on concrete claims, quantitative details, caveats, and recommendation fit.\n\n"
                f"Title: {title or '(untitled)'}\n"
                f"URL: {url}\n\n"
                f"Transcript:\n{transcript}"
            ),
            model_name=model_name,
        )
        summary = _clip_text(response.message, max_chars)
        if not summary:
            summary = _clip_text(transcript, max_chars)
        return summary, response.usage
    except Exception as exc:  # noqa: BLE001
        logger.warning("Transcript summarization failed for %s (%s)", url, exc)
        return _clip_text(transcript, max_chars), None


async def summarize_youtube_transcripts(
    transcripts: list[YouTubeTranscript],
    model_name: str,
    max_chars: int,
    concurrency: int,
) -> tuple[list[YouTubeTranscript], list[CodexUsage]]:
    """Summarize all transcripts with bounded concurrency."""

    if not transcripts:
        return [], []

    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def _summarize_one(
        transcript: YouTubeTranscript,
    ) -> tuple[YouTubeTranscript, CodexUsage | None]:
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
