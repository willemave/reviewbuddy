"""HTML to markdown conversion via Crawl4AI."""

from pathlib import Path

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig


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
    browser_config = BrowserConfig(headless=True, verbose=False)
    config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, verbose=False, log_console=False)

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(url=url, config=config)

    if not result.success:
        raise MarkdownError(f"Crawl4AI failed: {result.error}")

    return result.markdown or ""
