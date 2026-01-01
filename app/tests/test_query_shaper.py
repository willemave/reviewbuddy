from app.services.query_shaper import QueryShapeRequest, shape_query


def test_shape_query_applies_suffix() -> None:
    request = QueryShapeRequest(query="best dishwasher", suffix="forum")
    result = shape_query(request)
    assert result.applied is True
    assert result.query.endswith("forum")


def test_shape_query_skips_when_keyword_present() -> None:
    request = QueryShapeRequest(query="best dishwasher forum", suffix="forum")
    result = shape_query(request)
    assert result.applied is False
    assert result.query == request.query


def test_shape_query_disabled() -> None:
    request = QueryShapeRequest(query="best dishwasher", suffix="forum", enabled=False)
    result = shape_query(request)
    assert result.applied is False
    assert result.query == request.query
