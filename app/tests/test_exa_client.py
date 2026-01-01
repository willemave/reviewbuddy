import httpx
import pytest

from app.services.exa_client import ExaError, search_exa


@pytest.mark.asyncio
async def test_search_exa_parses_results():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("x-api-key") == "test-key"
        return httpx.Response(
            200,
            json={"results": [{"url": "https://example.com", "title": "Example", "score": 0.9}]},
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        response = await search_exa(
            query="test",
            api_key="test-key",
            num_results=5,
            search_type="auto",
            user_location="US",
            client=client,
        )

    assert response.results
    assert response.results[0].url == "https://example.com"
    assert response.results[0].title == "Example"


@pytest.mark.asyncio
async def test_search_exa_requires_key():
    with pytest.raises(ExaError):
        await search_exa(
            query="test",
            api_key="",
            num_results=5,
            search_type="auto",
            user_location="US",
        )
