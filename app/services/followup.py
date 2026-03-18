"""Follow-up memory loading and question answering."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from pydantic import RootModel

from app.constants import FOLLOWUP_MEMORY_FILENAME, URL_STATUS_FETCHED, YOUTUBE_TRANSCRIPTS_FILENAME
from app.core.settings import get_settings
from app.models.review import FollowupMemory, FollowupSourceCard
from app.services.codex_exec import run_codex_prompt
from app.services.storage import list_run_urls
from app.services.youtube_transcriber import YouTubeTranscript
from app.workflows.review import (
    _build_source_card,
    _format_source_cards,
    _pack_source_cards,
    _score_source_card,
    _tokenize_score_terms,
)

settings = get_settings()

FOLLOWUP_ANSWER_SYSTEM_PROMPT = (
    "You answer follow-up questions for an existing ReviewBuddy research run. "
    "Use only the stored synthesis and stored source cards that are provided. "
    "Do not browse, search, or invent facts. If the evidence is incomplete, say so clearly. "
    "Cite source URLs explicitly in the answer."
)

VIDEO_EVIDENCE_LANE_NAME = "Video Evidence"
VIDEO_EVIDENCE_LANE_GOAL = "Supplementary evidence from YouTube transcripts"
STORED_EVIDENCE_LANE_GOAL = "Persisted evidence from a prior run"


def followup_memory_path(run_dir: Path) -> Path:
    """Return the persisted follow-up memory path.

    Args:
        run_dir: Concrete run directory.

    Returns:
        Path to the follow-up memory JSON file.
    """

    return run_dir / FOLLOWUP_MEMORY_FILENAME


async def load_followup_memory(
    run_id: str,
    run_dir: Path,
    prompt: str,
    synthesis_markdown: str,
) -> FollowupMemory:
    """Load persisted follow-up memory for a run.

    Args:
        run_id: Run identifier.
        run_dir: Concrete run directory.
        prompt: Original user prompt.
        synthesis_markdown: Saved synthesis markdown.

    Returns:
        Loaded or reconstructed follow-up memory.
    """

    memory_file = followup_memory_path(run_dir)
    if memory_file.exists():
        return FollowupMemory.model_validate_json(memory_file.read_text(encoding="utf-8"))
    return await _rebuild_followup_memory(
        run_id=run_id,
        run_dir=run_dir,
        prompt=prompt,
        synthesis_markdown=synthesis_markdown,
    )


def build_followup_answer_prompt(
    memory: FollowupMemory,
    question: str,
    source_cards: list[FollowupSourceCard],
) -> str:
    """Build the follow-up answer prompt.

    Args:
        memory: Persisted run memory.
        question: User follow-up question.
        source_cards: Relevant source cards for this question.

    Returns:
        Prompt text for the answering model.
    """

    cards_markdown = _format_source_cards(source_cards) if source_cards else "(none)"
    return _build_followup_answer_prompt_from_markdown(memory, question, cards_markdown)


def _build_followup_answer_prompt_from_markdown(
    memory: FollowupMemory,
    question: str,
    source_cards_markdown: str,
) -> str:
    return (
        f"{FOLLOWUP_ANSWER_SYSTEM_PROMPT}\n\n"
        f"Original prompt:\n{memory.prompt}\n\n"
        f"Current follow-up question:\n{question}\n\n"
        f"Saved synthesis:\n{memory.synthesis_markdown}\n\n"
        f"Relevant source cards:\n{source_cards_markdown}\n\n"
        "Answer the follow-up directly. Include a short 'Sources' section listing the URLs you used. "
        "If the saved material does not support the answer, say what is missing."
    )


def rank_followup_source_cards(
    memory: FollowupMemory,
    question: str,
) -> list[FollowupSourceCard]:
    """Rank stored source cards for a follow-up question.

    Args:
        memory: Persisted run memory.
        question: User follow-up question.

    Returns:
        Ranked source cards with question-specific scores.
    """

    ranked: list[FollowupSourceCard] = []
    question_tokens = _tokenize_score_terms(question)
    for card in memory.source_cards:
        question_score = _score_source_card(
            prompt=question,
            lane_name=card.lane_name,
            lane_goal=card.lane_goal,
            url=card.url,
            title=card.title,
            source_query=card.source_query,
            distilled_text=card.distilled_text,
            source_kind=card.source_kind,
        )
        card_text_tokens = _tokenize_score_terms(
            card.title,
            card.source_query,
            card.url,
            card.distilled_text,
            card.lane_name,
            card.lane_goal,
        )
        overlap_boost = min(
            24, sum(1 for token in question_tokens if token in card_text_tokens) * 8
        )
        combined_score = min(
            100,
            round((question_score * 0.85) + (card.relevance_score * 0.15)) + overlap_boost,
        )
        ranked.append(card.model_copy(update={"relevance_score": combined_score}))
    return sorted(
        ranked,
        key=lambda card: (card.relevance_score, card.value_density, len(card.distilled_text)),
        reverse=True,
    )


async def answer_followup_question(
    memory: FollowupMemory,
    question: str,
    *,
    model_name: str | None = None,
) -> str:
    """Answer a follow-up question from stored run memory.

    Args:
        memory: Persisted run memory.
        question: User follow-up question.
        model_name: Optional model override.

    Returns:
        Markdown answer string.
    """

    ranked_cards = rank_followup_source_cards(memory, question)
    packed_cards = _pack_source_cards(
        ranked_cards,
        prompt_builder=lambda source_cards_markdown: _build_followup_answer_prompt_from_markdown(
            memory,
            question,
            source_cards_markdown,
        ),
        max_target_tokens=settings.synthesis_final_target_tokens,
        max_hard_tokens=settings.synthesis_final_hard_max_tokens,
        max_sources=settings.synthesis_final_max_sources,
    )
    prompt = build_followup_answer_prompt(memory, question, packed_cards)
    response = await run_codex_prompt(
        prompt,
        model_name=model_name or settings.synthesizer_model,
    )
    return response.message.strip()


async def _rebuild_followup_memory(
    run_id: str,
    run_dir: Path,
    prompt: str,
    synthesis_markdown: str,
) -> FollowupMemory:
    url_records = await list_run_urls(settings.database_path, run_id, status=URL_STATUS_FETCHED)
    source_cards: list[FollowupSourceCard] = []

    for record in url_records:
        if record.markdown_path is None or not record.markdown_path.exists():
            continue
        lane_name, source_query = _split_source_query(record.source_query)
        raw = record.markdown_path.read_text(encoding="utf-8", errors="ignore")
        card = _build_source_card(
            prompt=prompt,
            lane_name=lane_name,
            lane_goal=STORED_EVIDENCE_LANE_GOAL,
            url=record.url,
            title=record.title,
            raw=raw,
            source_query=source_query,
        )
        if card is None:
            continue
        source_cards.append(FollowupSourceCard.model_validate(asdict(card)))

    source_cards.extend(_load_transcript_source_cards(run_dir, prompt))
    return FollowupMemory(
        run_id=run_id,
        prompt=prompt,
        synthesis_markdown=synthesis_markdown,
        source_cards=source_cards,
    )


def _load_transcript_source_cards(run_dir: Path, prompt: str) -> list[FollowupSourceCard]:
    transcript_path = run_dir / YOUTUBE_TRANSCRIPTS_FILENAME
    if not transcript_path.exists():
        return []
    transcripts = [
        YouTubeTranscript.model_validate(item)
        for item in FollowupTranscriptList.model_validate_json(
            transcript_path.read_text(encoding="utf-8")
        ).root
    ]

    cards: list[FollowupSourceCard] = []
    for transcript in transcripts:
        card = _build_source_card(
            prompt=prompt,
            lane_name=VIDEO_EVIDENCE_LANE_NAME,
            lane_goal=VIDEO_EVIDENCE_LANE_GOAL,
            url=transcript.url,
            title=transcript.title,
            raw=transcript.transcript,
            source_query=None,
        )
        if card is None:
            continue
        cards.append(FollowupSourceCard.model_validate(asdict(card)))
    return cards


def _split_source_query(value: str | None) -> tuple[str, str | None]:
    if not value:
        return "Stored Evidence", None
    lane_name, separator, source_query = value.partition(": ")
    if not separator:
        return "Stored Evidence", value
    return lane_name.strip() or "Stored Evidence", source_query.strip() or None


class FollowupTranscriptList(RootModel[list[YouTubeTranscript]]):
    """Stored transcript list for follow-up reconstruction."""
