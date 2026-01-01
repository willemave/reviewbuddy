"""HTML to markdown conversion via Crawl4AI."""

from pathlib import Path

from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig


class MarkdownError(RuntimeError):
    """Raised when markdown conversion fails."""


async def html_file_to_markdown(html_path: Path) -> str:
    """Convert a local HTML file into markdown.

    Args:
        html_path: Path to HTML file.

    Returns:
        Markdown string.
    """

    if not html_path.exists():
        raise MarkdownError(f"HTML file missing: {html_path}")

    url = html_path.resolve().as_uri()
    config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url, config=config)

    if not result.success:
        raise MarkdownError(f"Crawl4AI failed: {result.error}")

    return result.markdown or ""
