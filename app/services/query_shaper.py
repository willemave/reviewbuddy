"""Query shaping helpers to bias toward non-ecommerce sources."""

from pydantic import BaseModel, Field

QUERY_SHAPING_KEYWORDS = (
    "forum",
    "reddit",
    "discussion",
    "blog",
    "user review",
    "hands on",
    "hands-on",
)


class QueryShapeRequest(BaseModel):
    """Request payload for shaping a query."""

    query: str
    suffix: str = Field(default="")
    enabled: bool = True


class QueryShapeResult(BaseModel):
    """Result of shaping a query."""

    query: str
    applied: bool


def shape_query(request: QueryShapeRequest) -> QueryShapeResult:
    """Shape a query to bias toward forums, blogs, and discussions.

    Args:
        request: Query shaping request.

    Returns:
        QueryShapeResult with the shaped query and applied flag.
    """

    if not request.enabled or not request.suffix:
        return QueryShapeResult(query=request.query, applied=False)

    lowered = request.query.lower()
    if any(keyword in lowered for keyword in QUERY_SHAPING_KEYWORDS):
        return QueryShapeResult(query=request.query, applied=False)

    shaped = f"{request.query} {request.suffix}".strip()
    return QueryShapeResult(query=shaped, applied=True)
