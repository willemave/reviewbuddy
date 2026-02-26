"""Pydantic models for RLM-style runs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.agents.base import AgentDeps
from app.services.usage_tracker import UsageTracker


class ContextDocument(BaseModel):
    """Single context document exposed to the REPL."""

    lane_name: str | None = None
    lane_goal: str | None = None
    url: str | None = None
    title: str | None = None
    kind: Literal["web", "youtube"] = "web"
    content: str
    char_len: int = Field(ge=0)


class RlmRunRequest(BaseModel):
    """Request to run an RLM-style loop."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_id: str
    prompt: str
    context_docs: list[ContextDocument]
    deps: AgentDeps
    usage_tracker: UsageTracker | None = None
    root_model: str | None = None
    subquery_model: str | None = None
    max_iterations: int | None = None


class RlmRunResult(BaseModel):
    """Result from an RLM-style loop."""

    output: str
    iterations: int
    completed: bool
    completed_at: datetime


class RlmRefineRequest(BaseModel):
    """Request to refine lane queries using RLM."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_id: str
    prompt: str
    lane_name: str
    lane_goal: str
    context_docs: list[ContextDocument]
    deps: AgentDeps
    usage_tracker: UsageTracker | None = None
    root_model: str | None = None
    subquery_model: str | None = None
    max_iterations: int | None = None


class DspySynthesisRequest(BaseModel):
    """Request to synthesize a review using DSPy RLM."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt: str
    context_docs: list[ContextDocument]
    max_iterations: int | None = None
    max_llm_calls: int | None = None
    run_dir: Path | None = None
