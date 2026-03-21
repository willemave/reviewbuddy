"""Workspace setup helpers for local ReviewBuddy development."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

from app.cli_doctor import DoctorCheck, run_doctor_checks
from app.core.settings import Settings, get_settings


@dataclass(frozen=True)
class SetupAction:
    """One setup action result."""

    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class SetupResult:
    """Setup actions plus post-setup doctor checks."""

    actions: list[SetupAction]
    doctor_checks: list[DoctorCheck]


def run_setup(
    settings: Settings,
    *,
    cwd: Path | None = None,
    install_playwright: bool = True,
) -> SetupResult:
    """Run local workspace setup for ReviewBuddy.

    Args:
        settings: Loaded application settings.
        cwd: Starting directory for workspace discovery.
        install_playwright: Whether to install Playwright browsers.

    Returns:
        Setup action results and a post-setup doctor report.
    """

    workspace_root = resolve_workspace_root(cwd or Path.cwd())
    actions = [
        _persist_search_config(workspace_root, settings),
        _prepare_storage(settings.storage_path),
        _prepare_database(settings.database_path),
    ]
    if install_playwright:
        actions.append(_install_playwright(workspace_root))
    get_settings.cache_clear()
    doctor_checks = run_doctor_checks(get_settings())
    return SetupResult(actions=actions, doctor_checks=doctor_checks)


def resolve_workspace_root(start_path: Path) -> Path | None:
    """Find the nearest ReviewBuddy workspace root from a starting path."""

    current = start_path.resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / ".env.example").exists():
            return candidate
    return None


def format_setup_report(actions: list[SetupAction]) -> str:
    """Render setup action results for terminal output."""

    lines = ["# ReviewBuddy Setup", ""]
    for action in actions:
        status = "OK" if action.ok else "FAIL"
        lines.append(f"- [{status}] {action.name}: {action.detail}")
    failures = sum(1 for action in actions if not action.ok)
    lines.extend(["", f"Failures: {failures}"])
    return "\n".join(lines)


def has_setup_failures(actions: list[SetupAction]) -> bool:
    """Return True when any setup action failed."""

    return any(not action.ok for action in actions)


def _persist_search_config(workspace_root: Path | None, settings: Settings) -> SetupAction:
    provider = settings.get_effective_search_provider()
    key_name = settings.get_search_provider_key_name(provider)
    key_value = settings.get_search_provider_api_key(provider)

    if not key_value:
        return SetupAction(
            name="search config",
            ok=False,
            detail="no configured provider key available to persist",
        )
    if workspace_root is None:
        return SetupAction(
            name="search config",
            ok=True,
            detail=f"{key_name} available but no local workspace root was found",
        )

    env_path = workspace_root / ".env"
    existing_values = {
        key: value
        for key, value in dotenv_values(env_path).items()
        if isinstance(value, str)
    }
    additions: list[str] = []
    if not env_path.exists():
        env_path.write_text("", encoding="utf-8")
        additions.append("created .env")

    _append_env_line(env_path, existing_values, "SEARCH_PROVIDER", provider, additions)
    _append_env_line(env_path, existing_values, key_name, key_value, additions)
    if provider == "exa":
        _append_env_line(
            env_path,
            existing_values,
            "EXA_SEARCH_TYPE",
            settings.exa_search_type,
            additions,
        )

    if not additions:
        return SetupAction(name="search config", ok=True, detail=f"{env_path} already configured")
    return SetupAction(name="search config", ok=True, detail=f"{env_path} ({', '.join(additions)})")


def _append_env_line(
    env_path: Path,
    existing_values: dict[str, str],
    key: str,
    value: str,
    additions: list[str],
) -> None:
    if not value.strip() or key in existing_values:
        return

    existing_text = env_path.read_text(encoding="utf-8")
    with env_path.open("a", encoding="utf-8") as file_handle:
        if existing_text and not existing_text.endswith("\n"):
            file_handle.write("\n")
        file_handle.write(f"{key}={value.strip()}\n")
    existing_values[key] = value.strip()
    additions.append(f"set {key}")


def _prepare_storage(path: Path) -> SetupAction:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return SetupAction(name="storage path", ok=False, detail=f"{path} ({exc})")
    return SetupAction(name="storage path", ok=True, detail=str(path))


def _prepare_database(path: Path) -> SetupAction:
    parent = path.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
    except OSError as exc:
        return SetupAction(name="database path", ok=False, detail=f"{path} ({exc})")
    return SetupAction(name="database path", ok=True, detail=str(path))


def _install_playwright(workspace_root: Path | None) -> SetupAction:
    if workspace_root is None:
        return SetupAction(
            name="playwright browsers",
            ok=True,
            detail="skipped because no local workspace root was found",
        )
    uv_path = shutil.which("uv")
    if uv_path is None:
        return SetupAction(name="playwright browsers", ok=False, detail="uv not found in PATH")

    try:
        completed = subprocess.run(
            [uv_path, "run", "playwright", "install"],
            check=True,
            capture_output=True,
            cwd=workspace_root,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        error_text = (exc.stderr or exc.stdout or "").strip() or "install failed"
        return SetupAction(name="playwright browsers", ok=False, detail=error_text)

    output = (completed.stdout or completed.stderr or "").strip()
    detail = output.splitlines()[-1] if output else "installed"
    return SetupAction(name="playwright browsers", ok=True, detail=detail)
