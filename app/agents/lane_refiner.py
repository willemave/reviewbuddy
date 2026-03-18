"""Lane refinement backed by Codex exec."""

from app.agents.base import AgentDeps, LaneRefinement
from app.core.settings import get_settings
from app.services.codex_exec import run_codex_prompt
from app.services.usage_tracker import UsageTracker

settings = get_settings()

LANE_REFINER_SYSTEM_PROMPT = (
    "You are refining search queries for a specific research lane. Use the evidence "
    "snippets to propose 3-8 new, high-signal queries that expand coverage. Focus on "
    "forums, comparative reviews, blogs, and real user feedback. Avoid ecommerce "
    "storefronts and avoid site: filters. Do not repeat existing queries."
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
    """Generate follow-up queries for a lane."""

    del deps
    result, response = await run_codex_prompt(
        (
            f"{LANE_REFINER_SYSTEM_PROMPT}\n\n"
            "Generate 3-8 new queries for this lane based on evidence.\n\n"
            f"Prompt: {prompt}\n"
            f"Lane: {lane_name}\n"
            f"Goal: {lane_goal}\n\n"
            f"Evidence:\n{evidence_snippets}"
        ),
        model_name=model_name or settings.refiner_model,
        output_type=LaneRefinement,
    )
    if usage_tracker is not None:
        await usage_tracker.add(response.usage, model_name=model_name or settings.refiner_model)
    return result
