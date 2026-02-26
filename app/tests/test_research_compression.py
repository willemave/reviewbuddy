from app.workflows.review import (
    CandidateUrl,
    _distill_source_text,
    _rank_candidate_urls,
)


def test_rank_candidate_urls_prefers_domain_diversity() -> None:
    candidates = [
        CandidateUrl(
            url="https://site-a.com/review-1",
            title="Review A1",
            source_query="query",
            lane_name="Lane",
            score=0.99,
            domain="site-a.com",
            title_key="review a1",
        ),
        CandidateUrl(
            url="https://site-a.com/review-2",
            title="Review A2",
            source_query="query",
            lane_name="Lane",
            score=0.95,
            domain="site-a.com",
            title_key="review a2",
        ),
        CandidateUrl(
            url="https://site-b.com/review-1",
            title="Review B1",
            source_query="query",
            lane_name="Lane",
            score=0.70,
            domain="site-b.com",
            title_key="review b1",
        ),
    ]

    ranked = _rank_candidate_urls(candidates, budget=2)

    assert len(ranked) == 2
    assert ranked[0].url == "https://site-a.com/review-1"
    assert ranked[1].url == "https://site-b.com/review-1"


def test_distill_source_text_extracts_signal_and_caveats() -> None:
    raw = """
    Cookie policy and privacy policy details.
    The unit measured 38 dB in our tests and uses 420 kWh per year.
    However, multiple owners reported reliability issues after year two.
    Reviewers recommend this model for quiet apartments.
    """

    distilled = _distill_source_text(raw, max_chars=2000)

    assert "Cookie policy" not in distilled
    assert "Quantitative Signals" in distilled
    assert "38 dB" in distilled
    assert "Caveats" in distilled
    assert "reliability issues" in distilled
