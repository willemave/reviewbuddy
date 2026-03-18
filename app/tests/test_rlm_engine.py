import pytest

from app.agents.base import AgentDeps
from app.models.rlm import ContextDocument, RlmRunRequest
from app.services.codex_exec import CodexResponse, CodexUsage
from app.services.rlm_engine import run_rlm
from app.services.usage_tracker import UsageTracker


@pytest.mark.asyncio
async def test_run_rlm_completes_after_repl_subquery(monkeypatch) -> None:
    responses = iter(
        [
            (
                "```repl\nprint(llm_query('What does the snippet say?'))\n```",
                CodexResponse(message="", usage=CodexUsage(input_tokens=5, output_tokens=2)),
            ),
            (
                "FINAL(done)",
                CodexResponse(message="", usage=CodexUsage(input_tokens=3, output_tokens=1)),
            ),
        ]
    )

    async def fake_root(prompt, deps, model_name=None):  # noqa: ANN001
        del prompt, deps, model_name
        return next(responses)

    def fake_subquery(prompt, deps, model_name=None):  # noqa: ANN001
        del prompt, deps, model_name
        return (
            "The snippet says model A is quieter.",
            CodexResponse(message="", usage=CodexUsage(input_tokens=2, output_tokens=1)),
        )

    monkeypatch.setattr("app.services.rlm_engine.run_rlm_root_prompt", fake_root)
    monkeypatch.setattr("app.services.rlm_engine.run_rlm_subquery_prompt", fake_subquery)

    tracker = UsageTracker()
    result = await run_rlm(
        RlmRunRequest(
            run_id="run-1",
            prompt="Which one is quieter?",
            context_docs=[ContextDocument(content="Model A is quieter.", char_len=18)],
            deps=AgentDeps(session_id="s", job_id="j"),
            usage_tracker=tracker,
            max_iterations=2,
        )
    )

    snapshot = await tracker.snapshot()

    assert result.completed is True
    assert result.output == "done"
    assert snapshot.requests == 3
    assert snapshot.total_tokens == 14
