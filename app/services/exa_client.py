"""Exa Search API client."""

from collections.abc import Iterable

import httpx
from pydantic import ValidationError

from app.models.review import ExaSearchResponse, ExaSearchResult


class ExaError(RuntimeError):
    """Raised when Exa API calls fail."""


def _parse_results(items: Iterable[dict]) -> list[ExaSearchResult]:
    results: list[ExaSearchResult] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            results.append(ExaSearchResult(**item))
        except ValidationError:
            url = item.get("url")
            if url:
                results.append(
                    ExaSearchResult(
                        url=url,
                        title=item.get("title"),
                        score=item.get("score"),
                        published_date=item.get("publishedDate"),
                    )
                )
    return results


async def search_exa(
    query: str,
    api_key: str,
    num_results: int,
    search_type: str,
    user_location: str,
    client: httpx.AsyncClient | None = None,
) -> ExaSearchResponse:
    """Call Exa Search API.

    Args:
        query: Search query.
        api_key: Exa API key.
        num_results: Number of results to request.
        search_type: Exa search type (auto, neural, keyword, fast).
        user_location: Two-letter ISO country code.
        client: Optional HTTP client.

    Returns:
        ExaSearchResponse with parsed results.
    """

    if not api_key:
        raise ExaError("EXA_API_KEY is not configured")

    payload = {
        "query": query,
        "type": search_type,
        "numResults": num_results,
        "userLocation": user_location,
    }

    close_client = False
    if client is None:
        client = httpx.AsyncClient(timeout=30)
        close_client = True

    try:
        response = await client.post(
            "https://api.exa.ai/search",
            headers={"x-api-key": api_key},
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPError as exc:
        raise ExaError(f"Exa search failed: {exc}") from exc
    finally:
        if close_client:
            await client.aclose()

    results = _parse_results(data.get("results", []))
    return ExaSearchResponse(results=results)
