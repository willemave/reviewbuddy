"""Agents for RLM-style REPL runs."""

from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

from app.agents.base import AgentDeps
from app.core.settings import get_settings

settings = get_settings()

RLM_SYSTEM_PROMPT = (
    "You are operating inside an interactive REPL environment. The full context is stored "
    "in a variable named `context`. Use code blocks tagged with ```repl``` to inspect and "
    "analyze context. You can use `print()` to view outputs, and call `llm_query(prompt)` "
    "or `llm_query_batched(prompts)` to run sub-LLM calls on selected slices. Use cheap "
    "programmatic filters (regex, keyword search, slicing) before calling llm_query. "
    "When you are finished, respond with FINAL(<your answer>) or FINAL_VAR(<variable>)."
)

rlm_root_agent = Agent(
    model=settings.rlm_root_model,
    output_type=str,
    deps_type=AgentDeps,
    model_settings=ModelSettings(temperature=0.2, tool_choice="auto"),
    system_prompt=RLM_SYSTEM_PROMPT,
)

rlm_subquery_agent = Agent(
    model=settings.rlm_subquery_model,
    output_type=str,
    deps_type=AgentDeps,
    model_settings=ModelSettings(temperature=0.2, tool_choice="auto"),
    system_prompt=(
        "Answer the question using only the provided snippet. If the snippet does not "
        "contain relevant information, say so clearly and briefly."
    ),
)
