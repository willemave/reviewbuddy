from typer.testing import CliRunner

from app import cli
from app.cli_doctor import DoctorCheck
from app.cli_help import build_command_reference

runner = CliRunner()


def test_build_command_reference_includes_primary_commands() -> None:
    reference = build_command_reference()

    assert "reviewbuddy run" in reference
    assert "reviewbuddy interactive" in reference
    assert "reviewbuddy ask" in reference
    assert "reviewbuddy resume" in reference
    assert "reviewbuddy doctor" in reference
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


def test_doctor_command_returns_nonzero_on_failures(monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "run_doctor_checks",
        lambda _settings: [DoctorCheck(name="codex", ok=False, detail="missing")],
    )

    result = runner.invoke(cli.app, ["doctor"])

    assert result.exit_code == 1
    assert "[FAIL] codex" in result.stdout
