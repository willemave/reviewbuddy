from pathlib import Path
from types import SimpleNamespace

from app.cli_doctor import (
    DoctorCheck,
    _check_playwright_browser,
    format_doctor_report,
    has_doctor_failures,
    run_doctor_checks,
)
from app.core.settings import Settings


def _patch_playwright_check(monkeypatch, *, ok: bool = True, detail: str = "Chromium launch OK") -> None:
    monkeypatch.setattr(
        "app.cli_doctor._check_playwright_browser",
        lambda: DoctorCheck(name="playwright browsers", ok=ok, detail=detail),
    )


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


def test_run_doctor_checks_includes_signup_urls_when_no_provider_configured(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("SEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    monkeypatch.setattr("app.cli_doctor.shutil.which", lambda binary: f"/usr/bin/{binary}")
    monkeypatch.setattr("app.cli_doctor.Path.home", lambda: tmp_path)
    _patch_playwright_check(monkeypatch)
    monkeypatch.setattr(
        "app.cli_doctor.detect_local_agent_harness",
        lambda _settings: ("codex", "/usr/bin/codex"),
    )
    monkeypatch.setattr(
        "app.cli_doctor.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="Logged in using ChatGPT\n",
            stderr="",
        ),
    )

    settings = Settings(
        exa_api_key="",
        tavily_api_key="",
        firecrawl_api_key="",
        storage_path=tmp_path / "storage",
        database_path=tmp_path / "db" / "researchbuddy.db",
    )

    checks = run_doctor_checks(settings)

    provider_checks = [check for check in checks if check.name == "search provider"]
    assert len(provider_checks) == 1
    assert provider_checks[0].ok is False
    assert "https://dashboard.exa.ai/api-keys" in provider_checks[0].detail
    assert "https://www.firecrawl.dev/app/api-keys" in provider_checks[0].detail


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
    _patch_playwright_check(monkeypatch)
    monkeypatch.setattr(
        "app.cli_doctor.detect_local_agent_harness",
        lambda _settings: ("codex", "/usr/bin/codex"),
    )
    monkeypatch.setattr(
        "app.cli_doctor.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="Logged in using ChatGPT\n",
            stderr="",
        ),
    )

    settings = Settings(
        exa_api_key="",
        tavily_api_key="test-tavily",
        firecrawl_api_key="",
        storage_path=tmp_path / "storage",
        database_path=tmp_path / "db" / "researchbuddy.db",
    )

    checks = run_doctor_checks(settings)

    tavily_checks = [check for check in checks if check.name == "tavily provider"]
    assert len(tavily_checks) == 1
    assert tavily_checks[0].ok is True
    assert "auto-selected provider" in tavily_checks[0].detail
    codex_checks = [check for check in checks if check.name == "codex auth"]
    assert len(codex_checks) == 1
    assert codex_checks[0].ok is True
    assert any(check.name == "agent host" and check.ok for check in checks)
    assert all(check.name != "OPENAI_API_KEY" for check in checks)


def test_run_doctor_checks_detects_openclaw_install(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("SEARCH_PROVIDER", raising=False)
    monkeypatch.setenv("EXA_API_KEY", "test-exa")
    monkeypatch.setattr("app.cli_doctor.shutil.which", lambda binary: f"/usr/bin/{binary}")
    monkeypatch.setattr("app.cli_doctor.Path.home", lambda: tmp_path)
    _patch_playwright_check(monkeypatch)
    monkeypatch.setattr(
        "app.cli_doctor.detect_local_agent_harness",
        lambda _settings: ("codex", "/usr/bin/codex"),
    )
    monkeypatch.setattr(
        "app.cli_doctor.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="Logged in using ChatGPT\n",
            stderr="",
        ),
    )
    (tmp_path / ".openclaw").mkdir()

    settings = Settings(
        exa_api_key="test-exa",
        tavily_api_key="",
        firecrawl_api_key="",
        storage_path=tmp_path / "storage",
        database_path=tmp_path / "db" / "researchbuddy.db",
    )

    checks = run_doctor_checks(settings)

    host_checks = [check for check in checks if check.name == "agent host"]
    assert len(host_checks) == 1
    assert host_checks[0].ok is True
    assert "openclaw" in host_checks[0].detail


def test_run_doctor_checks_includes_firecrawl_signup_url_when_selected_provider_missing(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SEARCH_PROVIDER", "firecrawl")
    monkeypatch.setenv("EXA_API_KEY", "test-exa")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    monkeypatch.setattr("app.cli_doctor.shutil.which", lambda binary: f"/usr/bin/{binary}")
    monkeypatch.setattr("app.cli_doctor.Path.home", lambda: tmp_path)
    _patch_playwright_check(monkeypatch)
    monkeypatch.setattr(
        "app.cli_doctor.detect_local_agent_harness",
        lambda _settings: ("codex", "/usr/bin/codex"),
    )
    monkeypatch.setattr(
        "app.cli_doctor.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="Logged in using ChatGPT\n",
            stderr="",
        ),
    )

    settings = Settings(
        exa_api_key="test-exa",
        tavily_api_key="",
        firecrawl_api_key="",
        search_provider="firecrawl",
        storage_path=tmp_path / "storage",
        database_path=tmp_path / "db" / "researchbuddy.db",
    )

    checks = run_doctor_checks(settings)

    firecrawl_checks = [check for check in checks if check.name == "firecrawl provider"]
    assert len(firecrawl_checks) == 1
    assert firecrawl_checks[0].ok is False
    assert "https://www.firecrawl.dev/app/api-keys" in firecrawl_checks[0].detail


def test_run_doctor_checks_fails_when_codex_is_not_authenticated(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("EXA_API_KEY", "test-exa")
    monkeypatch.setattr("app.cli_doctor.shutil.which", lambda binary: f"/usr/bin/{binary}")
    monkeypatch.setattr("app.cli_doctor.Path.home", lambda: tmp_path)
    _patch_playwright_check(monkeypatch)
    monkeypatch.setattr(
        "app.cli_doctor.detect_local_agent_harness",
        lambda _settings: ("codex", "/usr/bin/codex"),
    )
    monkeypatch.setattr(
        "app.cli_doctor.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stdout="Not logged in\n",
            stderr="",
        ),
    )

    settings = Settings(
        exa_api_key="test-exa",
        tavily_api_key="",
        firecrawl_api_key="",
        storage_path=tmp_path / "storage",
        database_path=tmp_path / "db" / "researchbuddy.db",
    )

    checks = run_doctor_checks(settings)

    codex_checks = [check for check in checks if check.name == "codex auth"]
    assert len(codex_checks) == 1
    assert codex_checks[0].ok is False
    assert "run `codex login`" in codex_checks[0].detail


def test_run_doctor_checks_accepts_logged_in_status_from_stderr(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("EXA_API_KEY", "test-exa")
    monkeypatch.setattr("app.cli_doctor.shutil.which", lambda binary: f"/usr/bin/{binary}")
    monkeypatch.setattr("app.cli_doctor.Path.home", lambda: tmp_path)
    _patch_playwright_check(monkeypatch)
    monkeypatch.setattr(
        "app.cli_doctor.detect_local_agent_harness",
        lambda _settings: ("codex", "/usr/bin/codex"),
    )
    monkeypatch.setattr(
        "app.cli_doctor.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="",
            stderr="Logged in using ChatGPT\n",
        ),
    )

    settings = Settings(
        exa_api_key="test-exa",
        tavily_api_key="",
        firecrawl_api_key="",
        storage_path=tmp_path / "storage",
        database_path=tmp_path / "db" / "researchbuddy.db",
    )

    checks = run_doctor_checks(settings)

    codex_checks = [check for check in checks if check.name == "codex auth"]
    assert len(codex_checks) == 1
    assert codex_checks[0].ok is True
    assert codex_checks[0].detail == "Logged in using ChatGPT"


def test_run_doctor_checks_uses_user_codex_home_for_auth_probe(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("EXA_API_KEY", "test-exa")
    monkeypatch.setenv("CODEX_HOME", "/tmp/openclaw-isolated-codex-home")
    monkeypatch.setattr("app.cli_doctor.shutil.which", lambda binary: f"/usr/bin/{binary}")
    monkeypatch.setattr("app.cli_doctor.Path.home", lambda: tmp_path)
    monkeypatch.setattr("app.services.codex_exec.Path.home", lambda: tmp_path)
    _patch_playwright_check(monkeypatch)
    monkeypatch.setattr(
        "app.cli_doctor.detect_local_agent_harness",
        lambda _settings: ("codex", "/usr/bin/codex"),
    )

    def fake_run(*args, **kwargs):  # noqa: ANN001, ARG001
        assert kwargs["env"]["CODEX_HOME"] == str(tmp_path / ".codex")
        return SimpleNamespace(returncode=0, stdout="Logged in using ChatGPT\n", stderr="")

    monkeypatch.setattr("app.cli_doctor.subprocess.run", fake_run)

    settings = Settings(
        exa_api_key="test-exa",
        tavily_api_key="",
        firecrawl_api_key="",
        storage_path=tmp_path / "storage",
        database_path=tmp_path / "db" / "researchbuddy.db",
    )

    checks = run_doctor_checks(settings)

    codex_checks = [check for check in checks if check.name == "codex auth"]
    assert len(codex_checks) == 1
    assert codex_checks[0].ok is True


def test_check_playwright_browser_reports_failure_when_launch_fails(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.cli_doctor.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="Executable doesn't exist at /tmp/chromium\n",
        ),
    )

    check = _check_playwright_browser()

    assert check.name == "playwright browsers"
    assert check.ok is False
    assert "researchbuddy doctor --fix" in check.detail


def test_check_playwright_browser_summarizes_traceback_action(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.cli_doctor.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(  # noqa: ARG005
            returncode=1,
            stdout="",
            stderr=(
                "Traceback (most recent call last):\n"
                '  File "<string>", line 3, in <module>\n'
                "playwright._impl._errors.Error: BrowserType.launch: Executable doesn't exist\n"
                "Please run the following command to download new browsers:\n"
                "    playwright install\n"
            ),
        ),
    )

    check = _check_playwright_browser()

    assert check.ok is False
    assert "Traceback" not in check.detail
    assert "Executable doesn't exist" in check.detail
    assert "playwright install" in check.detail


def test_check_playwright_browser_reports_success(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.cli_doctor.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="Chromium launch OK\n",
            stderr="",
        ),
    )

    check = _check_playwright_browser()

    assert check.name == "playwright browsers"
    assert check.ok is True
    assert check.detail == "Chromium launch OK"
