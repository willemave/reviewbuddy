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
            DoctorCheck(
                name="search provider",
                ok=False,
                detail="no provider API key configured",
            ),
        ]
    )

    assert "[OK] local agent harness" in report
    assert "[FAIL] search provider" in report
    assert "Failures: 1" in report


def test_has_doctor_failures_detects_failure() -> None:
    assert has_doctor_failures([DoctorCheck(name="x", ok=False, detail="missing")]) is True
    assert has_doctor_failures([DoctorCheck(name="x", ok=True, detail="set")]) is False


def test_run_doctor_checks_uses_selected_provider(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("SEARCH_PROVIDER", raising=False)
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily")
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    monkeypatch.setattr("app.cli_doctor.shutil.which", lambda binary: f"/usr/bin/{binary}")
    monkeypatch.setattr("app.cli_doctor.Path.home", lambda: tmp_path)

    settings = Settings(
        exa_api_key="",
        tavily_api_key="test-tavily",
        firecrawl_api_key="",
        storage_path=tmp_path / "storage",
        database_path=tmp_path / "db" / "reviewbuddy.db",
    )

    checks = run_doctor_checks(settings)

    tavily_checks = [check for check in checks if check.name == "tavily provider"]
    assert len(tavily_checks) == 1
    assert tavily_checks[0].ok is True
    assert "auto-selected provider" in tavily_checks[0].detail
    assert any(check.name == "agent host" and check.ok for check in checks)
    assert all(check.name != "OPENAI_API_KEY" for check in checks)


def test_run_doctor_checks_detects_openclaw_install(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("SEARCH_PROVIDER", raising=False)
    monkeypatch.setenv("EXA_API_KEY", "test-exa")
    monkeypatch.setattr("app.cli_doctor.shutil.which", lambda binary: f"/usr/bin/{binary}")
    monkeypatch.setattr("app.cli_doctor.Path.home", lambda: tmp_path)
    (tmp_path / ".openclaw").mkdir()

    settings = Settings(
        exa_api_key="test-exa",
        tavily_api_key="",
        firecrawl_api_key="",
        storage_path=tmp_path / "storage",
        database_path=tmp_path / "db" / "reviewbuddy.db",
    )

    checks = run_doctor_checks(settings)

    host_checks = [check for check in checks if check.name == "agent host"]
    assert len(host_checks) == 1
    assert host_checks[0].ok is True
    assert "openclaw" in host_checks[0].detail
