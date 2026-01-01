from app.services.playwright_fetcher import should_retry_headful


def test_should_retry_headful_true() -> None:
    assert should_retry_headful(Exception("Access Denied 403"))


def test_should_retry_headful_false() -> None:
    assert should_retry_headful(Exception("Timeout fetching")) is False
