from datetime import UTC, datetime

from typer.testing import CliRunner

from app import cli
from app.cli_doctor import DoctorCheck
from app.cli_help import build_command_reference
from app.services.setup_runtime import SetupAction, SetupResult

runner = CliRunner()


def test_build_command_reference_includes_primary_commands() -> None:
    reference = build_command_reference()

    assert 'reviewbuddy run "<prompt>" [--stats]' in reference
    assert "reviewbuddy ask" in reference
    assert "reviewbuddy setup" in reference
    assert "reviewbuddy doctor" in reference
    assert "reviewbuddy tap export" in reference
    assert "docs/agent-cli-reference.md" in reference


def test_build_command_reference_agent_mode_is_machine_friendly() -> None:
    reference = build_command_reference(agent=True)

    assert "## ask" in reference
    assert "Purpose:" in reference
    assert "Usage: `reviewbuddy ask <run_id>" in reference


def test_commands_command_prints_agent_reference() -> None:
    result = runner.invoke(cli.app, ["commands", "--agent"])

    assert result.exit_code == 0
    assert "ReviewBuddy CLI For Agents" in result.stdout
    assert "Usage: `reviewbuddy ask <run_id>" in result.stdout
    assert "## setup" in result.stdout
    assert "## tap export" in result.stdout


def test_interactive_commands_are_not_registered() -> None:
    interactive_result = runner.invoke(cli.app, ["interactive"])
    resume_result = runner.invoke(cli.app, ["resume", "abc123"])

    assert interactive_result.exit_code == 2
    assert resume_result.exit_code == 2


def test_doctor_command_returns_nonzero_on_failures(monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "run_doctor_checks",
        lambda _settings: [DoctorCheck(name="local agent harness", ok=False, detail="missing")],
    )

    result = runner.invoke(cli.app, ["doctor"])

    assert result.exit_code == 1
    assert "[FAIL] local agent harness" in result.stdout


def test_setup_command_prints_setup_and_doctor_reports(monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "run_setup",
        lambda _settings, install_playwright: SetupResult(
            actions=[SetupAction(name="search config", ok=True, detail="configured")],
            doctor_checks=[DoctorCheck(name="local agent harness", ok=True, detail="codex")],
        ),
    )

    result = runner.invoke(cli.app, ["setup", "--skip-playwright"])

    assert result.exit_code == 0
    assert "ReviewBuddy Setup" in result.stdout
    assert "ReviewBuddy Doctor" in result.stdout
    assert "[OK] search config" in result.stdout


def test_run_command_only_prints_stats_when_requested(monkeypatch) -> None:
    monkeypatch.setattr(cli, "setup_logging", lambda _level: None)

    async def fake_run_review(_request, _deps, reporter=None):  # noqa: ANN001, ANN202
        return cli.ReviewRunResult(
            run_id="run-123",
            prompt="Best office chair",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            stats=cli.ReviewRunStats(total_urls=4, fetched=3, failed=1),
            synthesis_markdown="Saved synthesis text",
        )

    monkeypatch.setattr(
        cli,
        "run_review",
        fake_run_review,
    )

    default_result = runner.invoke(cli.app, ["run", "Best office chair"])
    stats_result = runner.invoke(cli.app, ["run", "Best office chair", "--stats"])

    assert default_result.exit_code == 0
    assert "Run ID: run-123" in default_result.stdout
    assert "Fetched 3/4 URLs" not in default_result.stdout
    assert "Saved synthesis text" in default_result.stdout

    assert stats_result.exit_code == 0
    assert "Fetched 3/4 URLs (1 failed)" in stats_result.stdout


def test_tap_export_command_writes_tap_repo(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        cli, "detect_github_remote", lambda _repo_root: ("willemave", "reviewbuddy")
    )

    output_dir = tmp_path / "homebrew-reviewbuddy"
    result = runner.invoke(cli.app, ["tap", "export", "--output-dir", str(output_dir)])

    assert result.exit_code == 0
    assert (output_dir / "Formula" / "reviewbuddy.rb").exists()
    assert (output_dir / "skills" / "reviewbuddy-tap-maintainer" / "SKILL.md").exists()
