"""Lane planner agent for review research."""

import time

from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

from app.agents.base import AgentDeps, LanePlan
from app.core.settings import get_settings
from app.services.usage_tracker import UsageTracker

settings = get_settings()

lane_planner_agent = Agent(
    model=settings.default_model,
    output_type=LanePlan,
    deps_type=AgentDeps,
    model_settings=ModelSettings(temperature=settings.agent_temperature, tool_choice="auto"),
    system_prompt=(
        "You are a research planner. Break the user's prompt into 4-8 independent "
        "research lanes that can run in parallel. Each lane must include a short name, "
        "a clear goal, and 2-6 seed search queries. Lanes should cover diverse sources "
        "(forums, expert reviews, user experiences, comparisons, pricing/value, "
        "reliability/complaints). If the topic is not a physical product (e.g., education, "
        "books, services), adapt lanes accordingly (e.g., curriculum outcomes, author "
        "credibility, service quality, alternatives). Avoid ecommerce/storefront pages; "
        "bias toward forums, blogs, discussions, and hands-on evaluations. Prefer query "
        "phrasing like 'forum', 'discussion', 'blog', or 'user review' over site: filters."
    ),
)


async def plan_lanes(
    prompt: str,
    deps: AgentDeps,
    usage_tracker: UsageTracker | None = None,
    model_name: str | None = None,
) -> LanePlan:
    """Plan research lanes for the given prompt.

    Args:
        prompt: User prompt.
        deps: Agent dependencies.
        usage_tracker: Optional usage tracker.
        model_name: Optional model override for the planner agent.

    Returns:
        LanePlan containing lane specs.
    """

    start = time.perf_counter()
    agent = lane_planner_agent if model_name is None else lane_planner_agent.clone(model=model_name)
    result = await agent.run(
        (
            "Design research lanes for this request. Provide 2-6 seed queries per lane.\n\n"
            f"Prompt: {prompt}"
        ),
        deps=deps,
    )
    if usage_tracker is not None:
        await usage_tracker.add(result.usage())
    _ = time.perf_counter() - start
    return result.output
