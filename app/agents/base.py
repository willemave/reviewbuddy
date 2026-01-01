"""Shared agent models for ReviewBuddy."""

from pydantic import BaseModel, Field


class AgentDeps(BaseModel):
    """Dependencies passed to all agents."""

    session_id: str
    job_id: str


class SearchQuery(BaseModel):
    """Single search query with rationale."""

    query: str = Field(description="Search query to send to Exa")
    rationale: str = Field(description="Why this query is useful")


class SearchQuerySet(BaseModel):
    """Collection of search queries."""

    queries: list[SearchQuery] = Field(min_length=3, max_length=8)


class LaneSpec(BaseModel):
    """Specification for a research lane."""

    name: str = Field(description="Short lane name, e.g. 'Community Reviews'")
    goal: str = Field(description="What this lane should discover")
    seed_queries: list[SearchQuery] = Field(
        min_length=2,
        max_length=6,
        description="Initial queries to run for this lane",
    )
    url_budget: int | None = Field(
        default=None,
        description="Optional URL budget for this lane",
        ge=1,
    )


class LanePlan(BaseModel):
    """Plan containing multiple research lanes."""

    lanes: list[LaneSpec] = Field(min_length=3, max_length=10)


class LaneRefinement(BaseModel):
    """Additional queries based on lane findings."""

    queries: list[SearchQuery] = Field(min_length=1, max_length=3)


class SourceDigest(BaseModel):
    """Digest of a single source after crawling."""

    url: str
    title: str | None = None
    notes: str = Field(description="Short notes about why this source matters")


class ReviewSynthesis(BaseModel):
    """Structured synthesis output."""

    summary: str = Field(description="Concise synthesis of findings")
    key_findings: list[str] = Field(description="Bulleted key findings")
    recommendation: str = Field(description="Actionable recommendation")
    sources: list[SourceDigest] = Field(description="Top sources used")
    gaps: list[str] = Field(default_factory=list, description="Open questions or gaps")
