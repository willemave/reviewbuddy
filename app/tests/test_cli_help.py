from typer.testing import CliRunner

from app import cli
from app.cli_doctor import DoctorCheck
from app.cli_help import build_command_reference

runner = CliRunner()


def test_build_command_reference_includes_primary_commands() -> None:
    reference = build_command_reference()

    assert "reviewbuddy run" in reference
    assert "reviewbuddy ask" in reference
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


def test_tap_export_command_writes_tap_repo(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        cli, "detect_github_remote", lambda _repo_root: ("willemave", "reviewbuddy")
    )

    output_dir = tmp_path / "homebrew-reviewbuddy"
    result = runner.invoke(cli.app, ["tap", "export", "--output-dir", str(output_dir)])

    assert result.exit_code == 0
    assert (output_dir / "Formula" / "reviewbuddy.rb").exists()
    assert (output_dir / "skills" / "reviewbuddy-tap-maintainer" / "SKILL.md").exists()
