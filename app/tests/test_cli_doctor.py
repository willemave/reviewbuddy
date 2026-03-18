from pathlib import Path

from app.cli_doctor import (
    DoctorCheck,
    format_doctor_report,
    has_doctor_failures,
    run_doctor_checks,
)
from app.core.settings import Settings


def test_format_doctor_report_includes_statuses() -> None:
    report = format_doctor_report(
        [
            DoctorCheck(
                name="local agent harness",
                ok=True,
                detail="codex: /usr/local/bin/codex",
            ),
            DoctorCheck(name="OPENAI_API_KEY", ok=False, detail="missing"),
        ]
    )

    assert "[OK] local agent harness" in report
    assert "[FAIL] OPENAI_API_KEY" in report
    assert "Failures: 1" in report


def test_has_doctor_failures_detects_failure() -> None:
    assert has_doctor_failures([DoctorCheck(name="x", ok=False, detail="missing")]) is True
    assert has_doctor_failures([DoctorCheck(name="x", ok=True, detail="set")]) is False


def test_run_doctor_checks_uses_selected_provider(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily")
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    monkeypatch.setattr("app.cli_doctor.shutil.which", lambda binary: f"/usr/bin/{binary}")

    settings = Settings(
        search_provider="tavily",
        storage_path=tmp_path / "storage",
        database_path=tmp_path / "db" / "reviewbuddy.db",
    )

    checks = run_doctor_checks(settings)

    tavily_checks = [check for check in checks if check.name == "tavily provider"]
    assert len(tavily_checks) == 1
    assert tavily_checks[0].ok is True
