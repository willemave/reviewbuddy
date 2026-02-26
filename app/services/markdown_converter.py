"""HTML to markdown conversion via Crawl4AI."""

from pathlib import Path

from crawl4ai import (
    AsyncWebCrawler,
    BM25ContentFilter,
    BrowserConfig,
    CacheMode,
    CrawlerRunConfig,
    DefaultMarkdownGenerator,
    PruningContentFilter,
)

from app.core.settings import get_settings

settings = get_settings()


class MarkdownError(RuntimeError):
    """Raised when markdown conversion fails."""


def _build_markdown_generator(user_query: str | None) -> DefaultMarkdownGenerator:
    query = (user_query or "").strip()
    if query:
        content_filter = BM25ContentFilter(
            user_query=query,
            bm25_threshold=settings.markdown_bm25_threshold,
        )
    else:
        content_filter = PruningContentFilter(
            threshold=settings.markdown_pruning_threshold,
        )
    return DefaultMarkdownGenerator(content_filter=content_filter, content_source="cleaned_html")


def _select_markdown_text(markdown_result) -> str:
    fit_markdown = (getattr(markdown_result, "fit_markdown", "") or "").strip()
    if fit_markdown:
        return fit_markdown

    markdown_with_citations = (
        getattr(markdown_result, "markdown_with_citations", "") or ""
    ).strip()
    if markdown_with_citations:
        return markdown_with_citations

    return (getattr(markdown_result, "raw_markdown", "") or "").strip()


async def html_file_to_markdown(
    html_path: Path,
    user_query: str | None = None,
) -> str:
    """Convert a local HTML file into markdown.

    Args:
        html_path: Path to HTML file.
        user_query: Optional query hint for relevance filtering.

    Returns:
        Markdown string.
    """

    if not html_path.exists():
        raise MarkdownError(f"HTML file missing: {html_path}")

    url = html_path.resolve().as_uri()
    browser_config = BrowserConfig(headless=True, verbose=False)
    config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        verbose=False,
        log_console=False,
        word_count_threshold=settings.markdown_word_count_threshold,
        remove_forms=True,
        excluded_tags=["script", "style", "noscript", "nav", "footer", "aside", "form"],
        markdown_generator=_build_markdown_generator(user_query),
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(url=url, config=config)

    if not result.success:
        error = getattr(result, "error_message", None) or getattr(result, "error", "unknown error")
        raise MarkdownError(f"Crawl4AI failed: {error}")

    markdown_result = result.markdown
    if markdown_result is None:
        return ""
    return _select_markdown_text(markdown_result)
