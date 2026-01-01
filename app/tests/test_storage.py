from pathlib import Path

from app.services.storage import build_run_paths, url_to_filename


def test_build_run_paths(tmp_path: Path) -> None:
    paths = build_run_paths(tmp_path, "run123")
    assert paths["run"].exists()
    assert paths["html"].exists()
    assert paths["markdown"].exists()


def test_url_to_filename_stable() -> None:
    url = "https://example.com/page"
    assert url_to_filename(url, ".html") == url_to_filename(url, ".html")
    assert url_to_filename(url, ".html").endswith(".html")
