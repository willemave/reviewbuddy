from pathlib import Path

from app.models.homebrew import TapExportRequest
from app.services.homebrew_tap import (
    build_short_tap_name,
    build_source_tarball_url,
    detect_github_remote,
    export_tap_repository,
    render_formula,
)


def build_request(tmp_path: Path) -> TapExportRequest:
    return TapExportRequest(
        output_dir=tmp_path / "homebrew-reviewbuddy",
        github_owner="willemave",
        source_repo="reviewbuddy",
        tap_repo="homebrew-reviewbuddy",
        version="0.1.0",
        app_description="AI-powered review research assistant with parallel crawling and synthesis",
    )


def test_render_formula_includes_expected_tap_metadata(tmp_path: Path) -> None:
    formula = render_formula(build_request(tmp_path))

    assert 'homepage "https://github.com/willemave/reviewbuddy"' in formula
    assert (
        'url "https://github.com/willemave/reviewbuddy/archive/refs/tags/v0.1.0.tar.gz"' in formula
    )
    assert 'depends_on "ffmpeg"' in formula
    assert 'pkgshare.install "skills"' in formula
    assert 'shell_output("#{bin}/reviewbuddy commands --agent")' in formula


def test_export_tap_repository_writes_expected_files(tmp_path: Path) -> None:
    result = export_tap_repository(build_request(tmp_path))

    expected = {
        tmp_path / "homebrew-reviewbuddy" / "README.md",
        tmp_path / "homebrew-reviewbuddy" / "Formula" / "reviewbuddy.rb",
        tmp_path / "homebrew-reviewbuddy" / ".github" / "workflows" / "validate.yml",
        tmp_path / "homebrew-reviewbuddy" / "skills" / "reviewbuddy-tap-maintainer" / "SKILL.md",
        tmp_path
        / "homebrew-reviewbuddy"
        / "skills"
        / "reviewbuddy-tap-maintainer"
        / "references"
        / "publishing.md",
    }

    assert set(result.files) == expected
    assert all(path.exists() for path in expected)


def test_build_short_tap_name_removes_homebrew_prefix(tmp_path: Path) -> None:
    short_tap = build_short_tap_name(build_request(tmp_path))

    assert short_tap == "willemave/reviewbuddy"


def test_build_source_tarball_url_uses_versioned_tag(tmp_path: Path) -> None:
    url = build_source_tarball_url(build_request(tmp_path))

    assert url.endswith("/archive/refs/tags/v0.1.0.tar.gz")


def test_detect_github_remote_returns_none_for_missing_origin(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    assert detect_github_remote(repo_root) is None
