"""Synthesizer agent for review research."""

import time

from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

from app.agents.base import AgentDeps, ReviewSynthesis
from app.core.settings import get_settings
from app.services.usage_tracker import UsageTracker

settings = get_settings()

synthesizer_agent = Agent(
    model=settings.synthesizer_model,
    output_type=ReviewSynthesis,
    deps_type=AgentDeps,
    model_settings=ModelSettings(temperature=0.4, tool_choice="auto"),
    system_prompt=(
        "You synthesize review research into a concise recommendation. Use only the "
        "provided sources. Keep it tight and pragmatic. Always include source URLs."
    ),
)


async def synthesize_review(
    prompt: str,
    source_markdown: str,
    deps: AgentDeps,
    usage_tracker: UsageTracker | None = None,
    model_name: str | None = None,
) -> ReviewSynthesis:
    """Synthesize review findings into a concise recommendation.

    Args:
        prompt: User prompt to answer.
        source_markdown: Aggregated markdown notes with URLs.
        deps: Agent dependencies.
        usage_tracker: Optional usage tracker.
        model_name: Optional model override for the synthesizer agent.

    Returns:
        ReviewSynthesis output.
    """

    start = time.perf_counter()
    agent = synthesizer_agent if model_name is None else synthesizer_agent.clone(model=model_name)
    result = await agent.run(
        (
            "You are given markdown summaries of sources. Produce a tight synthesis with "
            "recommendation and cite sources explicitly by URL.\n\n"
            f"Prompt: {prompt}\n\n"
            f"Sources:\n{source_markdown}"
        ),
        deps=deps,
    )
    if usage_tracker is not None:
        await usage_tracker.add(result.usage(), model_name=model_name or settings.synthesizer_model)
    _ = time.perf_counter() - start
    return result.output
