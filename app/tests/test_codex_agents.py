import pytest

from app.agents.base import AgentDeps, LanePlan, LaneRefinement, LaneSynthesis, ReviewSynthesis
from app.agents.lane_planner import plan_lanes
from app.agents.lane_refiner import refine_lane_queries
from app.agents.synthesizer import synthesize_lane, synthesize_merge_node, synthesize_review
from app.services.codex_exec import CodexResponse, CodexUsage


@pytest.mark.asyncio
async def test_plan_lanes_uses_codex_runner(monkeypatch) -> None:
    expected = LanePlan.model_validate(
        {
            "lanes": [
                {
                    "name": "Forums",
                    "goal": "Collect forum opinions",
                    "seed_queries": [
                        {"query": "a forum", "rationale": "signal"},
                        {"query": "b discussion", "rationale": "signal"},
                    ],
                    "url_budget": 2,
                },
                {
                    "name": "Reviews",
                    "goal": "Collect reviews",
                    "seed_queries": [
                        {"query": "c review", "rationale": "signal"},
                        {"query": "d hands on", "rationale": "signal"},
                    ],
                    "url_budget": 2,
                },
                {
                    "name": "Complaints",
                    "goal": "Collect complaints",
                    "seed_queries": [
                        {"query": "e problems", "rationale": "signal"},
                        {"query": "f reliability", "rationale": "signal"},
                    ],
                    "url_budget": 2,
                },
            ]
        }
    )

    async def fake_run(*args, **kwargs):  # noqa: ANN002, ANN003
        return expected, CodexResponse(message="", usage=CodexUsage())

    monkeypatch.setattr("app.agents.lane_planner.run_codex_prompt", fake_run)

    result = await plan_lanes("best espresso grinder", AgentDeps(session_id="s", job_id="j"))

    assert result == expected


@pytest.mark.asyncio
async def test_refine_lane_queries_uses_codex_runner(monkeypatch) -> None:
    expected = LaneRefinement.model_validate(
        {
            "queries": [
                {"query": "grinder forum retention", "rationale": "dig into complaints"},
            ]
        }
    )

    async def fake_run(*args, **kwargs):  # noqa: ANN002, ANN003
        return expected, CodexResponse(message="", usage=CodexUsage())

    monkeypatch.setattr("app.agents.lane_refiner.run_codex_prompt", fake_run)

    result = await refine_lane_queries(
        prompt="best espresso grinder",
        lane_name="Forums",
        lane_goal="Collect complaints",
        evidence_snippets="retention is a common complaint",
        deps=AgentDeps(session_id="s", job_id="j"),
    )

    assert result == expected


@pytest.mark.asyncio
async def test_synthesize_lane_uses_codex_runner(monkeypatch) -> None:
    expected = LaneSynthesis.model_validate(
        {
            "summary": "Forums prefer model A.",
            "key_findings": ["quiet", "easy workflow"],
            "sources": [{"url": "https://example.com", "title": "Review", "notes": "high signal"}],
            "gaps": [],
        }
    )

    async def fake_run(*args, **kwargs):  # noqa: ANN002, ANN003
        return expected, CodexResponse(message="", usage=CodexUsage())

    monkeypatch.setattr("app.agents.synthesizer.run_codex_prompt", fake_run)

    result = await synthesize_lane(
        prompt="best espresso grinder",
        lane_name="Forums",
        lane_goal="Collect user-reported experience",
        source_cards_markdown="https://example.com",
        deps=AgentDeps(session_id="s", job_id="j"),
    )

    assert result == expected


@pytest.mark.asyncio
async def test_synthesize_review_uses_codex_runner(monkeypatch) -> None:
    expected = ReviewSynthesis.model_validate(
        {
            "summary": "Buy model A.",
            "key_findings": ["quiet", "reliable"],
            "recommendation": "Choose model A.",
            "sources": [{"url": "https://example.com", "title": "Review", "notes": "high signal"}],
            "gaps": [],
        }
    )

    async def fake_run(*args, **kwargs):  # noqa: ANN002, ANN003
        return expected, CodexResponse(message="", usage=CodexUsage())

    monkeypatch.setattr("app.agents.synthesizer.run_codex_prompt", fake_run)

    result = await synthesize_review(
        prompt="best espresso grinder",
        merged_summary_markdown="https://example.com",
        evidence_appendix_markdown="",
        deps=AgentDeps(session_id="s", job_id="j"),
    )

    assert result == expected


@pytest.mark.asyncio
async def test_synthesize_merge_node_uses_codex_runner(monkeypatch) -> None:
    expected = LaneSynthesis.model_validate(
        {
            "summary": "Merged summary.",
            "key_findings": ["quiet", "consistent"],
            "sources": [{"url": "https://example.com", "title": "Review", "notes": "high signal"}],
            "gaps": [],
        }
    )

    async def fake_run(*args, **kwargs):  # noqa: ANN002, ANN003
        return expected, CodexResponse(message="", usage=CodexUsage())

    monkeypatch.setattr("app.agents.synthesizer.run_codex_prompt", fake_run)

    result = await synthesize_merge_node(
        prompt="best espresso grinder",
        node_name="Merge L1 G1",
        child_summaries_markdown="summary a\n\nsummary b",
        supporting_evidence_markdown="https://example.com",
        deps=AgentDeps(session_id="s", job_id="j"),
    )

    assert result == expected
