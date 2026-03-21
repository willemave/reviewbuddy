from pathlib import Path

from app.core.settings import Settings
from app.services.setup_runtime import run_setup


class FakeSettingsLoader:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def __call__(self) -> Settings:
        return self._settings

    def cache_clear(self) -> None:
        return None


def test_run_setup_persists_detected_provider_to_local_env(
    monkeypatch,
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    (workspace_root / "pyproject.toml").write_text("[project]\nname='reviewbuddy'\n", encoding="utf-8")
    (workspace_root / ".env.example").write_text("EXA_API_KEY=\n", encoding="utf-8")

    settings = Settings(
        exa_api_key="openclaw-exa",
        storage_path=workspace_root / "data" / "storage",
        database_path=workspace_root / "data" / "reviewbuddy.db",
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
    env_lines = (workspace_root / ".env").read_text(encoding="utf-8").splitlines()
    assert env_lines == [
        "SEARCH_PROVIDER=exa",
        "EXA_API_KEY=openclaw-exa",
        "EXA_SEARCH_TYPE=auto",
    ]
    assert (workspace_root / "data" / "storage").is_dir()
    assert (workspace_root / "data" / "reviewbuddy.db").exists()


def test_run_setup_fails_when_no_provider_key_is_available(
    monkeypatch,
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    (workspace_root / "pyproject.toml").write_text("[project]\nname='reviewbuddy'\n", encoding="utf-8")
    (workspace_root / ".env.example").write_text("EXA_API_KEY=\n", encoding="utf-8")

    settings = Settings(
        exa_api_key="",
        tavily_api_key="",
        firecrawl_api_key="",
        storage_path=workspace_root / "data" / "storage",
        database_path=workspace_root / "data" / "reviewbuddy.db",
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
