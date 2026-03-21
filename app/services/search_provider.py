"""Pluggable search providers for ReviewBuddy."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

import httpx
from pydantic import ValidationError

from app.core.settings import Settings
from app.models.review import SearchResponse, SearchResult


class SearchProviderError(RuntimeError):
    """Raised when a search provider request fails."""


class SearchProvider(Protocol):
    """Contract for pluggable search providers."""

    provider_name: str

    async def search(
        self,
        query: str,
        num_results: int,
        client: httpx.AsyncClient | None = None,
    ) -> SearchResponse:
        """Execute a search query and return normalized results."""


def build_search_provider(settings: Settings) -> SearchProvider:
    """Build the configured search provider."""

    provider_name = settings.get_effective_search_provider()

    if provider_name == "exa":
        return ExaSearchProvider(
            api_key=settings.exa_api_key,
            search_type=settings.exa_search_type,
            user_location=settings.exa_user_location,
        )
    if provider_name == "tavily":
        return TavilySearchProvider(
            api_key=settings.tavily_api_key,
            search_depth=settings.tavily_search_depth,
            topic=settings.tavily_topic,
            auto_parameters=settings.tavily_auto_parameters,
            max_results=settings.tavily_max_results,
        )
    return FirecrawlSearchProvider(
        api_key=settings.firecrawl_api_key,
        country=settings.firecrawl_country,
        location=settings.firecrawl_location,
    )


def _parse_result(item: dict, **overrides: object) -> SearchResult | None:
    payload = {**item, **overrides}
    try:
        return SearchResult(**payload)
    except ValidationError:
        url = payload.get("url")
        if not isinstance(url, str) or not url:
            return None
        return SearchResult(
            url=url,
            title=_string_or_none(payload.get("title")),
            score=_float_or_none(payload.get("score")),
            published_date=_string_or_none(payload.get("published_date")),
            content_markdown=_string_or_none(payload.get("content_markdown")),
            content_html=_string_or_none(payload.get("content_html")),
        )


def _string_or_none(value: object) -> str | None:
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    return None


def _float_or_none(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


@dataclass(frozen=True)
class ExaSearchProvider:
    """Exa-backed search provider."""

    api_key: str
    search_type: str
    user_location: str
    provider_name: str = "exa"

    async def search(
        self,
        query: str,
        num_results: int,
        client: httpx.AsyncClient | None = None,
    ) -> SearchResponse:
        """Search Exa and include extracted markdown text when available."""

        if not self.api_key:
            raise SearchProviderError("EXA_API_KEY is not configured")

        payload = {
            "query": query,
            "type": self.search_type,
            "numResults": min(num_results, 100),
            "userLocation": self.user_location,
            "contents": {"text": True},
        }
        data = await _post_json(
            url="https://api.exa.ai/search",
            headers={"x-api-key": self.api_key},
            payload=payload,
            client=client,
            error_label="Exa search",
        )
        results = _parse_exa_results(data.get("results", []))
        return SearchResponse(results=results)


def _parse_exa_results(items: Iterable[object]) -> list[SearchResult]:
    results: list[SearchResult] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        result = _parse_result(
            item,
            published_date=item.get("publishedDate"),
            content_markdown=item.get("text"),
        )
        if result is not None:
            results.append(result)
    return results


@dataclass(frozen=True)
class TavilySearchProvider:
    """Tavily-backed search provider."""

    api_key: str
    search_depth: str
    topic: str
    auto_parameters: bool
    max_results: int
    provider_name: str = "tavily"

    async def search(
        self,
        query: str,
        num_results: int,
        client: httpx.AsyncClient | None = None,
    ) -> SearchResponse:
        """Search Tavily and return normalized results with raw content."""

        if not self.api_key:
            raise SearchProviderError("TAVILY_API_KEY is not configured")

        payload = {
            "query": query,
            "search_depth": self.search_depth,
            "topic": self.topic,
            "auto_parameters": self.auto_parameters,
            "max_results": min(num_results, self.max_results),
            "include_answer": False,
            "include_raw_content": True,
        }
        data = await _post_json(
            url="https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {self.api_key}"},
            payload=payload,
            client=client,
            error_label="Tavily search",
        )
        results = _parse_tavily_results(data.get("results", []))
        return SearchResponse(results=results)


def _parse_tavily_results(items: Iterable[object]) -> list[SearchResult]:
    results: list[SearchResult] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        result = _parse_result(
            item,
            content_markdown=item.get("raw_content") or item.get("content"),
        )
        if result is not None:
            results.append(result)
    return results


@dataclass(frozen=True)
class FirecrawlSearchProvider:
    """Firecrawl-backed search provider."""

    api_key: str
    country: str
    location: str | None
    provider_name: str = "firecrawl"

    async def search(
        self,
        query: str,
        num_results: int,
        client: httpx.AsyncClient | None = None,
    ) -> SearchResponse:
        """Search Firecrawl and request markdown content for each result."""

        if not self.api_key:
            raise SearchProviderError("FIRECRAWL_API_KEY is not configured")

        payload = {
            "query": query,
            "limit": min(num_results, 100),
            "country": self.country,
            "sources": ["web"],
            "scrapeOptions": {"formats": ["markdown"]},
        }
        if self.location:
            payload["location"] = self.location

        data = await _post_json(
            url="https://api.firecrawl.dev/v2/search",
            headers={"Authorization": f"Bearer {self.api_key}"},
            payload=payload,
            client=client,
            error_label="Firecrawl search",
        )
        results = _parse_firecrawl_results(data)
        return SearchResponse(results=results)


def _parse_firecrawl_results(payload: dict) -> list[SearchResult]:
    data = payload.get("data")
    items = data.get("web", []) if isinstance(data, dict) else payload.get("web", [])

    results: list[SearchResult] = []
    if not isinstance(items, list):
        return results

    for item in items:
        if not isinstance(item, dict):
            continue
        result = _parse_result(
            item,
            score=item.get("score") or item.get("position"),
            content_markdown=item.get("markdown"),
            content_html=item.get("html"),
        )
        if result is not None:
            results.append(result)
    return results


async def _post_json(
    url: str,
    headers: dict[str, str],
    payload: dict[str, object],
    error_label: str,
    client: httpx.AsyncClient | None = None,
) -> dict:
    close_client = False
    if client is None:
        client = httpx.AsyncClient(timeout=30)
        close_client = True

    try:
        response = await client.post(
            url,
            headers={
                "Content-Type": "application/json",
                **headers,
            },
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPError as exc:
        raise SearchProviderError(f"{error_label} failed: {exc}") from exc
    finally:
        if close_client:
            await client.aclose()

    if not isinstance(data, dict):
        raise SearchProviderError(f"{error_label} returned an invalid response payload")
    return data
