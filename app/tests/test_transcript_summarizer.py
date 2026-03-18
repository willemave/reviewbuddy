import pytest

from app.services.transcript_summarizer import _clip_text, summarize_youtube_transcripts
from app.services.youtube_transcriber import YouTubeTranscript


def test_clip_text_respects_limit() -> None:
    assert _clip_text("abcdefgh", 5) == "abcde"
    assert _clip_text("abc", 10) == "abc"


@pytest.mark.asyncio
async def test_summarize_youtube_transcripts_empty() -> None:
    transcripts, usages = await summarize_youtube_transcripts(
        transcripts=[],
        model_name="gpt-5.4",
        max_chars=200,
        concurrency=2,
    )
    assert transcripts == []
    assert usages == []


@pytest.mark.asyncio
async def test_summarize_youtube_transcripts_fallback(monkeypatch) -> None:
    async def fail_run(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("boom")

    monkeypatch.setattr("app.services.transcript_summarizer.run_codex_prompt", fail_run)
    items = [
        YouTubeTranscript(
            url="https://youtube.com/watch?v=abc",
            title="Video",
            transcript="x" * 50,
        )
    ]
    transcripts, usages = await summarize_youtube_transcripts(
        transcripts=items,
        model_name="gpt-5.4",
        max_chars=20,
        concurrency=1,
    )
    assert len(transcripts) == 1
    assert transcripts[0].transcript == "x" * 20
    assert usages == []
