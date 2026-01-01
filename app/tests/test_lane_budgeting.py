from app.agents.base import LaneSpec, SearchQuery
from app.workflows.review import _allocate_lane_budgets


def test_allocate_lane_budgets_even_split():
    lanes = [
        LaneSpec(
            name=f"Lane {idx}",
            goal="Test",
            seed_queries=[
                SearchQuery(query="q1", rationale="r"),
                SearchQuery(query="q2", rationale="r"),
            ],
        )
        for idx in range(3)
    ]

    allocated = _allocate_lane_budgets(lanes, max_urls=10)
    budgets = [lane.url_budget for lane in allocated]

    assert sum(budgets) == 10
    assert min(budgets) >= 3
    assert max(budgets) <= 4


def test_allocate_lane_budgets_respects_minimum():
    lanes = [
        LaneSpec(
            name=f"Lane {idx}",
            goal="Test",
            seed_queries=[
                SearchQuery(query="q1", rationale="r"),
                SearchQuery(query="q2", rationale="r"),
            ],
            url_budget=1,
        )
        for idx in range(5)
    ]

    allocated = _allocate_lane_budgets(lanes, max_urls=3)
    budgets = [lane.url_budget for lane in allocated]

    assert sum(budgets) == 3
    assert all(budget >= 1 for budget in budgets)
