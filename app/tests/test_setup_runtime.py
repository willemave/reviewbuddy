import sqlite3
from pathlib import Path
from types import SimpleNamespace

from app.core.settings import Settings
from app.services.setup_runtime import _install_playwright, run_setup


class FakeSettingsLoader:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def __call__(self) -> Settings:
        return self._settings

    def cache_clear(self) -> None:
        return None


def test_run_setup_uses_detected_provider_without_copying_credentials(
    monkeypatch,
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    (workspace_root / "pyproject.toml").write_text("[project]\nname='researchbuddy'\n", encoding="utf-8")
    (workspace_root / ".env.example").write_text("EXA_API_KEY=\n", encoding="utf-8")

    settings = Settings(
        exa_api_key="openclaw-exa",
        storage_path=workspace_root / "data" / "storage",
        database_path=workspace_root / "data" / "researchbuddy.db",
    )
    monkeypatch.setattr(
        "app.services.setup_runtime.get_settings",
        FakeSettingsLoader(settings),
    )
    monkeypatch.setattr(
        "app.services.setup_runtime.run_doctor_checks",
        lambda _settings: [],
    )

    result = run_setup(settings, cwd=workspace_root, install_playwright=False)

    assert result.actions[0].ok is True
    assert result.actions[0].detail == (
        "EXA_API_KEY available from environment or shared agent config; no credentials copied"
    )
    assert not (workspace_root / ".env").exists()
    assert (workspace_root / "data" / "storage").is_dir()
    assert (workspace_root / "data" / "researchbuddy.db").exists()
    with sqlite3.connect(workspace_root / "data" / "researchbuddy.db") as conn:
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'runs'")
        assert cursor.fetchone() == ("runs",)


def test_run_setup_fails_when_no_provider_key_is_available(
    monkeypatch,
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    (workspace_root / "pyproject.toml").write_text("[project]\nname='researchbuddy'\n", encoding="utf-8")
    (workspace_root / ".env.example").write_text("EXA_API_KEY=\n", encoding="utf-8")

    settings = Settings(
        exa_api_key="",
        tavily_api_key="",
        firecrawl_api_key="",
        storage_path=workspace_root / "data" / "storage",
        database_path=workspace_root / "data" / "researchbuddy.db",
    )
    monkeypatch.setattr(
        "app.services.setup_runtime.get_settings",
        FakeSettingsLoader(settings),
    )
    monkeypatch.setattr(
        "app.services.setup_runtime.run_doctor_checks",
        lambda _settings: [],
    )

    result = run_setup(settings, cwd=workspace_root, install_playwright=False)

    assert result.actions[0].ok is False
    assert "no configured provider key" in result.actions[0].detail


def test_install_playwright_uses_current_python_without_workspace(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(*args, **kwargs):  # noqa: ANN001, ANN202
        captured["args"] = args
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="installed\n", stderr="")

    monkeypatch.setattr("app.services.setup_runtime.subprocess.run", fake_run)
    monkeypatch.setattr("app.services.setup_runtime.sys.executable", "/tmp/python")

    action = _install_playwright(None)

    assert action.ok is True
    assert action.detail == "installed"
    assert captured["args"][0] == ["/tmp/python", "-m", "playwright", "install", "chromium"]
    assert captured["kwargs"]["cwd"] is None


def test_install_playwright_reports_failure_summary(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.setup_runtime.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(  # noqa: ARG005
            returncode=1,
            stdout="",
            stderr="one\ntwo\nthree\nfour\n",
        ),
    )

    action = _install_playwright(None)

    assert action.ok is False
    assert action.detail == "one | two | three"
