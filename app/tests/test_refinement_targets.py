from app.workflows.review import _refinement_targets


def test_refinement_targets_zero_budget():
    assert _refinement_targets(0) == []


def test_refinement_targets_one_budget():
    assert _refinement_targets(1) == [1]


def test_refinement_targets_two_budget():
    assert _refinement_targets(2) == [1, 2]


def test_refinement_targets_three_budget():
    assert _refinement_targets(3) == [1, 2, 3]


def test_refinement_targets_four_budget():
    assert _refinement_targets(4) == [1, 2, 4]


def test_refinement_targets_five_budget():
    assert _refinement_targets(5) == [2, 3, 5]


def test_refinement_targets_eight_budget():
    assert _refinement_targets(8) == [2, 4, 8]
