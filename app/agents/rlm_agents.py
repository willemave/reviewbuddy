"""Codex-backed helpers for RLM-style REPL runs."""

from app.agents.base import AgentDeps
from app.core.settings import get_settings
from app.services.codex_exec import CodexResponse, run_codex_prompt, run_codex_prompt_sync

settings = get_settings()

RLM_SYSTEM_PROMPT = (
    "You are operating inside an interactive REPL environment. The full context is stored "
    "in a variable named `context`. Use code blocks tagged with ```repl``` to inspect and "
    "analyze context. You can use `print()` to view outputs, and call `llm_query(prompt)` "
    "or `llm_query_batched(prompts)` to run sub-LLM calls on selected slices. Use cheap "
    "programmatic filters (regex, keyword search, slicing) before calling llm_query. "
    "When you are finished, respond with FINAL(<your answer>) or FINAL_VAR(<variable>)."
)

RLM_SUBQUERY_SYSTEM_PROMPT = (
    "Answer the question using only the provided snippet. If the snippet does not "
    "contain relevant information, say so clearly and briefly."
)


async def run_rlm_root_prompt(
    prompt: str,
    deps: AgentDeps,
    model_name: str | None = None,
) -> tuple[str, CodexResponse]:
    """Run the root RLM prompt through Codex."""

    del deps
    response = await run_codex_prompt(
        f"{RLM_SYSTEM_PROMPT}\n\n{prompt}",
        model_name=model_name or settings.rlm_root_model,
    )
    return response.message, response


def run_rlm_subquery_prompt(
    prompt: str,
    deps: AgentDeps,
    model_name: str | None = None,
) -> tuple[str, CodexResponse]:
    """Run a synchronous subquery prompt through Codex."""

    del deps
    response = run_codex_prompt_sync(
        f"{RLM_SUBQUERY_SYSTEM_PROMPT}\n\n{prompt}",
        model_name=model_name or settings.rlm_subquery_model,
    )
    return response.message, response
