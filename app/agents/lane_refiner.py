"""Lane refinement agent for follow-up queries."""

import time

from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

from app.agents.base import AgentDeps, LaneRefinement
from app.core.settings import get_settings
from app.services.usage_tracker import UsageTracker

settings = get_settings()

lane_refiner_agent = Agent(
    model=settings.refiner_model,
    output_type=LaneRefinement,
    deps_type=AgentDeps,
    model_settings=ModelSettings(temperature=settings.agent_temperature, tool_choice="auto"),
    system_prompt=(
        "You are refining search queries for a specific research lane. Use the evidence "
        "snippets to propose 1-3 new, high-signal queries that expand coverage. Focus on "
        "forums, comparative reviews, blogs, and real user feedback. Avoid ecommerce "
        "storefronts and avoid site: filters. Do not repeat existing queries."
    ),
)


async def refine_lane_queries(
    prompt: str,
    lane_name: str,
    lane_goal: str,
    evidence_snippets: str,
    deps: AgentDeps,
    usage_tracker: UsageTracker | None = None,
    model_name: str | None = None,
) -> LaneRefinement:
    """Generate follow-up queries for a lane.

    Args:
        prompt: User prompt.
        lane_name: Lane name.
        lane_goal: Lane goal.
        evidence_snippets: Extracted evidence snippets.
        deps: Agent dependencies.
        usage_tracker: Optional usage tracker.
        model_name: Optional model override for the refiner agent.

    Returns:
        LaneRefinement containing follow-up queries.
    """

    start = time.perf_counter()
    agent = lane_refiner_agent if model_name is None else lane_refiner_agent.clone(model=model_name)
    result = await agent.run(
        (
            "Generate 1-3 new queries for this lane based on evidence.\n\n"
            f"Prompt: {prompt}\n"
            f"Lane: {lane_name}\n"
            f"Goal: {lane_goal}\n\n"
            f"Evidence:\n{evidence_snippets}"
        ),
        deps=deps,
    )
    if usage_tracker is not None:
        await usage_tracker.add(result.usage(), model_name=model_name or settings.refiner_model)
    _ = time.perf_counter() - start
    return result.output
