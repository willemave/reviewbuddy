from decimal import Decimal

import httpx
import pytest

from app.core.settings import Settings
from app.services.search_provider import (
    ExaSearchProvider,
    FirecrawlSearchProvider,
    SearchProviderError,
    TavilySearchProvider,
    build_search_provider,
)


def test_build_search_provider_returns_configured_provider() -> None:
    settings = Settings(
        search_provider="firecrawl",
        firecrawl_api_key="fc-key",
    )

    provider = build_search_provider(settings)

    assert isinstance(provider, FirecrawlSearchProvider)
    assert provider.provider_name == "firecrawl"


def test_build_search_provider_auto_selects_available_provider(monkeypatch) -> None:
    monkeypatch.delenv("SEARCH_PROVIDER", raising=False)

    settings = Settings(
        exa_api_key="",
        tavily_api_key="tvly-key",
        firecrawl_api_key="",
    )

    provider = build_search_provider(settings)

    assert isinstance(provider, TavilySearchProvider)
    assert provider.provider_name == "tavily"


@pytest.mark.asyncio
async def test_exa_provider_parses_markdown_content() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("x-api-key") == "exa-key"
        assert request.url == httpx.URL("https://api.exa.ai/search")
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "url": "https://example.com/exa",
                        "title": "Exa Result",
                        "score": 0.9,
                        "text": "# Exa markdown",
                        "publishedDate": "2026-03-15",
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        provider = ExaSearchProvider(
            api_key="exa-key",
            search_type="auto",
            user_location="US",
        )
        response = await provider.search("quiet dishwasher", 5, client=client)

    assert response.results[0].content_markdown == "# Exa markdown"
    assert response.results[0].published_date == "2026-03-15"
    assert response.usage is not None
    assert response.usage.provider_name == "exa"
    assert response.usage.requested_results == 5
    assert response.usage.returned_results == 1
    assert response.usage.cost_amount_usd == Decimal("0.007")


@pytest.mark.asyncio
async def test_tavily_provider_parses_raw_content() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("Authorization") == "Bearer tvly-key"
        assert request.url == httpx.URL("https://api.tavily.com/search")
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "url": "https://example.com/tavily",
                        "title": "Tavily Result",
                        "score": 0.8,
                        "raw_content": "Tavily body",
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        provider = TavilySearchProvider(
            api_key="tvly-key",
            search_depth="basic",
            topic="general",
            auto_parameters=False,
            max_results=20,
        )
        response = await provider.search("quiet dishwasher", 5, client=client)

    assert response.results[0].content_markdown == "Tavily body"
    assert response.usage is not None
    assert response.usage.provider_name == "tavily"
    assert response.usage.requested_results == 5
    assert response.usage.returned_results == 1
    assert response.usage.credit_amount == Decimal("1")
    assert response.usage.cost_amount_usd == Decimal("0.008")


@pytest.mark.asyncio
async def test_firecrawl_provider_parses_markdown_content() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("Authorization") == "Bearer fc-key"
        assert request.url == httpx.URL("https://api.firecrawl.dev/v2/search")
        return httpx.Response(
            200,
            json={
                "success": True,
                "data": {
                    "web": [
                        {
                            "url": "https://example.com/firecrawl",
                            "title": "Firecrawl Result",
                            "description": "desc",
                            "markdown": "# Firecrawl markdown",
                            "position": 1,
                        }
                    ]
                },
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        provider = FirecrawlSearchProvider(
            api_key="fc-key",
            country="US",
            location="United States",
        )
        response = await provider.search("quiet dishwasher", 5, client=client)

    assert response.results[0].content_markdown == "# Firecrawl markdown"
    assert response.results[0].score == 1.0
    assert response.usage is not None
    assert response.usage.provider_name == "firecrawl"
    assert response.usage.requested_results == 5
    assert response.usage.returned_results == 1
    assert response.usage.credit_amount == Decimal("3")
    assert response.usage.cost_amount_usd is None


@pytest.mark.asyncio
async def test_provider_requires_api_key() -> None:
    provider = TavilySearchProvider(
        api_key="",
        search_depth="basic",
        topic="general",
        auto_parameters=False,
        max_results=20,
    )

    with pytest.raises(SearchProviderError):
        await provider.search("quiet dishwasher", 5)
