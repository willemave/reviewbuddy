"""Synthesizer backed by Codex exec."""

from app.agents.base import AgentDeps, LaneSynthesis, ReviewSynthesis
from app.core.settings import get_settings
from app.services.codex_exec import run_codex_prompt
from app.services.usage_tracker import UsageTracker

settings = get_settings()

LANE_SYNTHESIZER_SYSTEM_PROMPT = (
    "You synthesize one research lane into a dense evidence summary. Use only the "
    "provided source cards. Preserve caveats, conflicts, and source URLs."
)

FINAL_SYNTHESIZER_SYSTEM_PROMPT = (
    "You synthesize review research into a concise recommendation. Use only the "
    "provided research summary material and evidence appendix. Keep it tight and pragmatic. "
    "Always include source URLs."
)


def build_lane_synthesis_prompt(
    prompt: str,
    lane_name: str,
    lane_goal: str,
    source_cards_markdown: str,
) -> str:
    """Build the leaf lane synthesis prompt."""

    return (
        f"{LANE_SYNTHESIZER_SYSTEM_PROMPT}\n\n"
        "You are given distilled source cards for a single lane. Produce a dense lane "
        "summary, the strongest findings, top supporting sources, and any gaps or conflicts.\n\n"
        f"User prompt: {prompt}\n"
        f"Lane: {lane_name}\n"
        f"Goal: {lane_goal}\n\n"
        f"Source cards:\n{source_cards_markdown}"
    )


def build_merge_synthesis_prompt(
    prompt: str,
    node_name: str,
    child_summaries_markdown: str,
    supporting_evidence_markdown: str,
) -> str:
    """Build the merge-node synthesis prompt."""

    evidence_block = supporting_evidence_markdown or "(none)"
    return (
        f"{LANE_SYNTHESIZER_SYSTEM_PROMPT}\n\n"
        "You are given multiple child summaries from the same research tree plus supporting "
        "evidence cards. Merge them into one denser intermediate summary. Deduplicate repeated "
        "claims, preserve the strongest caveats and conflicts, and keep source URLs.\n\n"
        f"User prompt: {prompt}\n"
        f"Merge node: {node_name}\n\n"
        f"Child summaries:\n{child_summaries_markdown}\n\n"
        f"Supporting evidence:\n{evidence_block}"
    )


def build_final_synthesis_prompt(
    prompt: str,
    merged_summary_markdown: str,
    evidence_appendix_markdown: str,
) -> str:
    """Build the final user-facing synthesis prompt."""

    appendix_block = evidence_appendix_markdown or "(none)"
    return (
        f"{FINAL_SYNTHESIZER_SYSTEM_PROMPT}\n\n"
        "You are given compact research summary material plus a supporting evidence appendix. "
        "The summary material may be a merged synthesis or a short set of lane summaries. "
        "Produce a tight synthesis with recommendation and cite sources explicitly by URL.\n\n"
        f"Prompt: {prompt}\n\n"
        f"Merged summary:\n{merged_summary_markdown}\n\n"
        f"Evidence appendix:\n{appendix_block}"
    )


async def synthesize_lane(
    prompt: str,
    lane_name: str,
    lane_goal: str,
    source_cards_markdown: str,
    deps: AgentDeps,
    usage_tracker: UsageTracker | None = None,
    model_name: str | None = None,
) -> LaneSynthesis:
    """Synthesize a single lane into a dense summary."""

    del deps
    result, response = await run_codex_prompt(
        build_lane_synthesis_prompt(
            prompt=prompt,
            lane_name=lane_name,
            lane_goal=lane_goal,
            source_cards_markdown=source_cards_markdown,
        ),
        model_name=model_name or settings.synthesizer_model,
        output_type=LaneSynthesis,
    )
    if usage_tracker is not None:
        await usage_tracker.add(response.usage, model_name=model_name or settings.synthesizer_model)
    return result


async def synthesize_merge_node(
    prompt: str,
    node_name: str,
    child_summaries_markdown: str,
    supporting_evidence_markdown: str,
    deps: AgentDeps,
    usage_tracker: UsageTracker | None = None,
    model_name: str | None = None,
) -> LaneSynthesis:
    """Merge multiple child summaries into one intermediate summary."""

    del deps
    result, response = await run_codex_prompt(
        build_merge_synthesis_prompt(
            prompt=prompt,
            node_name=node_name,
            child_summaries_markdown=child_summaries_markdown,
            supporting_evidence_markdown=supporting_evidence_markdown,
        ),
        model_name=model_name or settings.synthesizer_model,
        output_type=LaneSynthesis,
    )
    if usage_tracker is not None:
        await usage_tracker.add(response.usage, model_name=model_name or settings.synthesizer_model)
    return result


async def synthesize_review(
    prompt: str,
    merged_summary_markdown: str,
    evidence_appendix_markdown: str,
    deps: AgentDeps,
    usage_tracker: UsageTracker | None = None,
    model_name: str | None = None,
) -> ReviewSynthesis:
    """Synthesize review findings into a concise recommendation."""

    del deps
    result, response = await run_codex_prompt(
        build_final_synthesis_prompt(
            prompt=prompt,
            merged_summary_markdown=merged_summary_markdown,
            evidence_appendix_markdown=evidence_appendix_markdown,
        ),
        model_name=model_name or settings.synthesizer_model,
        output_type=ReviewSynthesis,
    )
    if usage_tracker is not None:
        await usage_tracker.add(response.usage, model_name=model_name or settings.synthesizer_model)
    return result
