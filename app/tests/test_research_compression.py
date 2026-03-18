from app.agents.base import LaneSynthesis
from app.workflows.review import (
    CandidateUrl,
    LaneSummaryPacket,
    SourceCard,
    _build_final_synthesis_input,
    _distill_source_text,
    _group_summary_packets_for_merge,
    _pack_source_cards,
    _rank_candidate_urls,
    _score_source_card,
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


def test_score_source_card_prefers_prompt_and_lane_overlap() -> None:
    high = _score_source_card(
        prompt="best quiet espresso grinder",
        lane_name="Forum consensus",
        lane_goal="Find owner reports on quiet grinders",
        url="https://example.com/grinder-review",
        title="Quiet grinder owner review",
        source_query="quiet espresso grinder forum",
        distilled_text="### Highlights\n- Owners say this grinder is quiet and consistent.",
        source_kind="web",
    )
    low = _score_source_card(
        prompt="best quiet espresso grinder",
        lane_name="Forum consensus",
        lane_goal="Find owner reports on quiet grinders",
        url="https://example.com/unrelated",
        title="Coffee mugs overview",
        source_query="best mugs",
        distilled_text="### Highlights\n- This article is about ceramic mugs.",
        source_kind="web",
    )

    assert high > low


def test_pack_source_cards_prefers_density_after_top_seed(monkeypatch) -> None:
    monkeypatch.setattr("app.workflows.review._estimate_prompt_tokens", lambda text: len(text))
    top = SourceCard(
        lane_name="Lane",
        lane_goal="Goal",
        url="https://example.com/top",
        title="Top",
        source_query="query",
        source_kind="web",
        distilled_text="A" * 600,
        relevance_score=90,
    )
    dense = SourceCard(
        lane_name="Lane",
        lane_goal="Goal",
        url="https://example.com/dense",
        title="Dense",
        source_query="query",
        source_kind="web",
        distilled_text="B" * 180,
        relevance_score=70,
    )
    bulky = SourceCard(
        lane_name="Lane",
        lane_goal="Goal",
        url="https://example.com/bulky",
        title="Bulky",
        source_query="query",
        source_kind="web",
        distilled_text="C" * 1400,
        relevance_score=72,
    )

    packed = _pack_source_cards(
        [top, dense, bulky],
        prompt_builder=lambda cards_markdown: cards_markdown,
        max_target_tokens=1200,
        max_hard_tokens=2000,
        max_sources=3,
    )

    assert packed[0].url == "https://example.com/top"
    assert any(card.url == "https://example.com/dense" for card in packed)
    assert all(card.url != "https://example.com/bulky" for card in packed)


def test_group_summary_packets_for_merge_packs_multiple_children(monkeypatch) -> None:
    lane_a = SourceCard(
        lane_name="Lane A",
        lane_goal="Goal A",
        url="https://example.com/a",
        title="A",
        source_query="query a",
        source_kind="web",
        distilled_text="A" * 300,
        relevance_score=95,
    )
    lane_b = SourceCard(
        lane_name="Lane B",
        lane_goal="Goal B",
        url="https://example.com/b",
        title="B",
        source_query="query b",
        source_kind="web",
        distilled_text="B" * 300,
        relevance_score=80,
    )
    lane_c = SourceCard(
        lane_name="Lane C",
        lane_goal="Goal C",
        url="https://example.com/c",
        title="C",
        source_query="query c",
        source_kind="web",
        distilled_text="C" * 300,
        relevance_score=75,
    )

    summary_packets = [
        LaneSummaryPacket(
            lane_name="Lane A",
            lane_goal="Goal A",
            synthesis=LaneSynthesis.model_validate(
                {
                    "summary": "A",
                    "key_findings": ["A"],
                    "sources": [{"url": lane_a.url, "title": "A", "notes": "A"}],
                    "gaps": [],
                }
            ),
            cards=[lane_a],
        ),
        LaneSummaryPacket(
            lane_name="Lane B",
            lane_goal="Goal B",
            synthesis=LaneSynthesis.model_validate(
                {
                    "summary": "B",
                    "key_findings": ["B"],
                    "sources": [{"url": lane_b.url, "title": "B", "notes": "B"}],
                    "gaps": [],
                }
            ),
            cards=[lane_b],
        ),
        LaneSummaryPacket(
            lane_name="Lane C",
            lane_goal="Goal C",
            synthesis=LaneSynthesis.model_validate(
                {
                    "summary": "C",
                    "key_findings": ["C"],
                    "sources": [{"url": lane_c.url, "title": "C", "notes": "C"}],
                    "gaps": [],
                }
            ),
            cards=[lane_c],
        ),
    ]

    monkeypatch.setattr("app.workflows.review._estimate_prompt_tokens", lambda _: 100)
    groups = _group_summary_packets_for_merge(
        prompt="best espresso grinder",
        summary_packets=summary_packets,
        level=1,
    )

    assert len(groups) == 1
    assert len(groups[0]) == 3


def test_group_summary_packets_splits_oversized_group(monkeypatch) -> None:
    def fake_estimate(text: str) -> int:
        return 200000 if "Lane A" in text and "Lane B" in text else 100

    monkeypatch.setattr("app.workflows.review._estimate_prompt_tokens", fake_estimate)
    monkeypatch.setattr("app.workflows.review.settings.synthesis_merge_target_tokens", 180000)

    summary_packets = [
        LaneSummaryPacket(
            lane_name=f"Lane {label}",
            lane_goal=f"Goal {label}",
            synthesis=LaneSynthesis.model_validate(
                {
                    "summary": label,
                    "key_findings": [label],
                    "sources": [
                        {"url": f"https://example.com/{label}", "title": label, "notes": label}
                    ],
                    "gaps": [],
                }
            ),
            cards=[
                SourceCard(
                    lane_name=f"Lane {label}",
                    lane_goal=f"Goal {label}",
                    url=f"https://example.com/{label}",
                    title=label,
                    source_query=f"query {label}",
                    source_kind="web",
                    distilled_text=label * 50,
                    relevance_score=80,
                )
            ],
        )
        for label in ("A", "B", "C")
    ]

    groups = _group_summary_packets_for_merge(
        prompt="best espresso grinder",
        summary_packets=summary_packets,
        level=1,
    )

    assert len(groups) == 2
    assert len(groups[0]) == 1
    assert len(groups[1]) == 2


def test_build_final_synthesis_input_respects_hard_cap(monkeypatch) -> None:
    monkeypatch.setattr("app.workflows.review._estimate_prompt_tokens", lambda text: len(text))
    root_packet = LaneSummaryPacket(
        lane_name="Root",
        lane_goal="Goal",
        synthesis=LaneSynthesis.model_validate(
            {
                "summary": "A" * 500,
                "key_findings": ["B" * 200],
                "sources": [{"url": "https://example.com/a", "title": "A", "notes": "A"}],
                "gaps": [],
            }
        ),
        cards=[
            SourceCard(
                lane_name="Root",
                lane_goal="Goal",
                url=f"https://example.com/{idx}",
                title=f"Source {idx}",
                source_query="query",
                source_kind="web",
                distilled_text="C" * 5000,
                relevance_score=90 - idx,
            )
            for idx in range(6)
        ],
    )

    merged_summary, appendix, estimated = _build_final_synthesis_input(
        prompt="best espresso grinder",
        summary_packets=[root_packet],
    )

    assert merged_summary
    assert estimated <= 200000
    assert appendix is not None


def test_build_final_synthesis_input_accepts_multiple_leaf_summaries(monkeypatch) -> None:
    monkeypatch.setattr("app.workflows.review._estimate_prompt_tokens", lambda text: len(text))
    packets = [
        LaneSummaryPacket(
            lane_name=f"Lane {idx}",
            lane_goal=f"Goal {idx}",
            synthesis=LaneSynthesis.model_validate(
                {
                    "summary": f"Summary {idx}",
                    "key_findings": [f"Finding {idx}"],
                    "sources": [
                        {
                            "url": f"https://example.com/{idx}",
                            "title": f"Source {idx}",
                            "notes": f"Notes {idx}",
                        }
                    ],
                    "gaps": [],
                }
            ),
            cards=[
                SourceCard(
                    lane_name=f"Lane {idx}",
                    lane_goal=f"Goal {idx}",
                    url=f"https://example.com/{idx}",
                    title=f"Source {idx}",
                    source_query=f"query {idx}",
                    source_kind="web",
                    distilled_text="A" * 400,
                    relevance_score=90 - idx,
                )
            ],
        )
        for idx in range(2)
    ]

    merged_summary, appendix, estimated = _build_final_synthesis_input(
        prompt="best espresso grinder",
        summary_packets=packets,
    )

    assert "## Lane 0" in merged_summary
    assert "## Lane 1" in merged_summary
    assert appendix
    assert estimated <= 200000
