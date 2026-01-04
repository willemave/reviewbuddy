"""Pydantic models for the review workflow."""

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field


class ReviewRunRequest(BaseModel):
    """Input request for a review run."""

    prompt: str
    max_urls: int
    max_agents: int
    headful: bool
    navigation_timeout_ms: int
    output_dir: Path
    planner_model: str | None = None
    sub_agent_model: str | None = None


class ReviewRunConfig(BaseModel):
    """Runtime configuration for a review run."""

    max_urls: int = Field(ge=1, le=1000)
    max_agents: int = Field(ge=1, le=25)
    headful: bool
    navigation_timeout_ms: int = Field(ge=1000, le=120000)
    output_dir: Path
    planner_model: str | None = None
    sub_agent_model: str | None = None


class ReviewRunStats(BaseModel):
    """Basic run statistics."""

    total_urls: int
    fetched: int
    failed: int


class ReviewRunResult(BaseModel):
    """Final output of a review run."""

    run_id: str
    prompt: str
    created_at: datetime
    stats: ReviewRunStats
    synthesis_markdown: str


class ExaSearchResult(BaseModel):
    """Single Exa search result."""

    url: str
    title: str | None = None
    score: float | None = None
    published_date: str | None = None


class ExaSearchResponse(BaseModel):
    """Parsed Exa search response."""

    results: list[ExaSearchResult]


class UrlTask(BaseModel):
    """URL task with source metadata."""

    url: str
    title: str | None = None
    source_query: str
    lane_name: str


class LaneResult(BaseModel):
    """Output from a single research lane."""

    lane_name: str
    goal: str
    url_tasks: list[UrlTask]


class UrlRecord(BaseModel):
    """Stored URL record for a run."""

    run_id: str
    url: str
    title: str | None
    source_query: str
    status: str
    html_path: Path | None = None
    markdown_path: Path | None = None
    error: str | None = None


class RunRecord(BaseModel):
    """Stored run record."""

    run_id: str
    prompt: str
    created_at: datetime
    status: str
    max_urls: int
    max_agents: int
    headful: bool
    output_dir: Path
