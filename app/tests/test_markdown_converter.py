from app.services.markdown_converter import (
    _build_markdown_generator,
    _select_markdown_text,
)


class _FakeMarkdownResult:
    def __init__(
        self,
        fit_markdown: str = "",
        markdown_with_citations: str = "",
        raw_markdown: str = "",
    ) -> None:
        self.fit_markdown = fit_markdown
        self.markdown_with_citations = markdown_with_citations
        self.raw_markdown = raw_markdown


def test_select_markdown_text_prefers_fit_markdown() -> None:
    result = _FakeMarkdownResult(
        fit_markdown="fit text",
        markdown_with_citations="citation text",
        raw_markdown="raw text",
    )
    assert _select_markdown_text(result) == "fit text"


def test_select_markdown_text_falls_back_to_markdown_with_citations() -> None:
    result = _FakeMarkdownResult(
        fit_markdown="",
        markdown_with_citations="citation text",
        raw_markdown="raw text",
    )
    assert _select_markdown_text(result) == "citation text"


def test_select_markdown_text_falls_back_to_raw_markdown() -> None:
    result = _FakeMarkdownResult(
        fit_markdown="",
        markdown_with_citations="",
        raw_markdown="raw text",
    )
    assert _select_markdown_text(result) == "raw text"


def test_build_markdown_generator_uses_bm25_for_query() -> None:
    generator = _build_markdown_generator("quiet dishwasher forum review")
    assert generator.content_filter.__class__.__name__ == "BM25ContentFilter"


def test_build_markdown_generator_uses_pruning_without_query() -> None:
    generator = _build_markdown_generator("")
    assert generator.content_filter.__class__.__name__ == "PruningContentFilter"
