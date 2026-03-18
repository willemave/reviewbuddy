"""Lane planner backed by Codex exec."""

from app.agents.base import AgentDeps, LanePlan
from app.core.settings import get_settings
from app.services.codex_exec import run_codex_prompt
from app.services.usage_tracker import UsageTracker

settings = get_settings()

LANE_PLANNER_SYSTEM_PROMPT = (
    "You are a research planner. Break the user's prompt into 4-8 independent "
    "research lanes that can run in parallel. Each lane must include a short name, "
    "a clear goal, and 2-6 seed search queries. Lanes should cover diverse sources "
    "(forums, expert reviews, user experiences, comparisons, pricing/value, "
    "reliability/complaints). If the topic is not a physical product (e.g., education, "
    "books, services), adapt lanes accordingly (e.g., curriculum outcomes, author "
    "credibility, service quality, alternatives). Avoid ecommerce/storefront pages; "
    "bias toward forums, blogs, discussions, and hands-on evaluations. Prefer query "
    "phrasing like 'forum', 'discussion', 'blog', or 'user review' over site: filters."
)


async def plan_lanes(
    prompt: str,
    deps: AgentDeps,
    usage_tracker: UsageTracker | None = None,
    model_name: str | None = None,
) -> LanePlan:
    """Plan research lanes for the given prompt."""

    del deps
    result, response = await run_codex_prompt(
        (
            f"{LANE_PLANNER_SYSTEM_PROMPT}\n\n"
            "Design research lanes for this request. Provide 2-6 seed queries per lane.\n\n"
            f"Prompt: {prompt}"
        ),
        model_name=model_name or settings.planner_model,
        output_type=LanePlan,
    )
    if usage_tracker is not None:
        await usage_tracker.add(response.usage, model_name=model_name or settings.planner_model)
    return result
