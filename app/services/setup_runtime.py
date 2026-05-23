"""Workspace setup helpers for local ResearchBuddy development."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from app.cli_doctor import DoctorCheck, run_doctor_checks
from app.core.settings import Settings, get_settings
from app.services.storage import init_db


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
    """Run local workspace setup for ResearchBuddy.

    Args:
        settings: Loaded application settings.
        cwd: Starting directory for workspace discovery.
        install_playwright: Whether to install Playwright browsers.

    Returns:
        Setup action results and a post-setup doctor report.
    """

    workspace_root = resolve_workspace_root(cwd or Path.cwd())
    actions = [
        _check_search_config(settings),
        _prepare_storage(settings.storage_path),
        _prepare_database(settings.database_path),
    ]
    if install_playwright:
        actions.append(_install_playwright(workspace_root))
    get_settings.cache_clear()
    doctor_checks = run_doctor_checks(get_settings())
    return SetupResult(actions=actions, doctor_checks=doctor_checks)


def resolve_workspace_root(start_path: Path) -> Path | None:
    """Find the nearest ResearchBuddy workspace root from a starting path."""

    current = start_path.resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / ".env.example").exists():
            return candidate
    return None


def format_setup_report(actions: list[SetupAction]) -> str:
    """Render setup action results for terminal output."""

    lines = ["# ResearchBuddy Setup", ""]
    for action in actions:
        status = "OK" if action.ok else "FAIL"
        lines.append(f"- [{status}] {action.name}: {action.detail}")
    failures = sum(1 for action in actions if not action.ok)
    lines.extend(["", f"Failures: {failures}"])
    return "\n".join(lines)


def has_setup_failures(actions: list[SetupAction]) -> bool:
    """Return True when any setup action failed."""

    return any(not action.ok for action in actions)


def _check_search_config(settings: Settings) -> SetupAction:
    provider = settings.get_effective_search_provider()
    key_name = settings.get_search_provider_key_name(provider)
    key_value = settings.get_search_provider_api_key(provider)

    if not key_value:
        return SetupAction(
            name="search config",
            ok=False,
            detail=(
                "no configured provider key available "
                "(set local .env, process env, ~/.hermes/.env, ~/.openclaw/.env, or ~/.openclaw/openclaw.json)"
            ),
        )
    return SetupAction(
        name="search config",
        ok=True,
        detail=f"{key_name} available from environment or shared agent config; no credentials copied",
    )


def _prepare_storage(path: Path) -> SetupAction:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return SetupAction(name="storage path", ok=False, detail=f"{path} ({exc})")
    return SetupAction(name="storage path", ok=True, detail=str(path))


def _prepare_database(path: Path) -> SetupAction:
    try:
        asyncio.run(init_db(path))
    except OSError as exc:
        return SetupAction(name="database path", ok=False, detail=f"{path} ({exc})")
    return SetupAction(name="database path", ok=True, detail=str(path))


def _install_playwright(workspace_root: Path | None) -> SetupAction:
    command = [sys.executable, "-m", "playwright", "install", "chromium"]
    cwd = workspace_root

    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            cwd=cwd,
            text=True,
        )
    except OSError as exc:
        return SetupAction(name="playwright browsers", ok=False, detail=f"install failed ({exc})")

    output = (completed.stdout or completed.stderr or "").strip()
    if completed.returncode != 0:
        detail = _summarize_command_output(completed.stdout, completed.stderr) or "install failed"
        return SetupAction(name="playwright browsers", ok=False, detail=detail)
    detail = output.splitlines()[-1] if output else "installed"
    return SetupAction(name="playwright browsers", ok=True, detail=detail)


def _summarize_command_output(stdout: str, stderr: str) -> str:
    lines = [line.strip() for line in (*stdout.splitlines(), *stderr.splitlines()) if line.strip()]
    return " | ".join(lines[:3])
