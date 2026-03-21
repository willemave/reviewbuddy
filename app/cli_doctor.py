"""Environment checks for CLI and bot deployments."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from app.core.settings import Settings
from app.services.codex_exec import detect_local_agent_harness


@dataclass(frozen=True)
class DoctorCheck:
    """Single environment readiness check."""

    name: str
    ok: bool
    detail: str


def run_doctor_checks(settings: Settings) -> list[DoctorCheck]:
    """Run runtime checks needed for ReviewBuddy.

    Args:
        settings: Loaded application settings.

    Returns:
        Ordered check results.
    """

    checks = [
        _check_search_provider(settings),
        _check_agent_hosts(),
        _check_local_agent_harness(settings),
        _check_binary("uv", "uv"),
        _check_binary("ffmpeg", "ffmpeg"),
        _check_storage_path(settings.storage_path),
        _check_database_parent(settings.database_path),
    ]
    return checks


def format_doctor_report(checks: list[DoctorCheck]) -> str:
    """Render check results for terminal output.

    Args:
        checks: Ordered check results.

    Returns:
        Multi-line report string.
    """

    lines = ["# ReviewBuddy Doctor", ""]
    for check in checks:
        status = "OK" if check.ok else "FAIL"
        lines.append(f"- [{status}] {check.name}: {check.detail}")
    failures = sum(1 for check in checks if not check.ok)
    lines.extend(["", f"Failures: {failures}"])
    return "\n".join(lines)


def has_doctor_failures(checks: list[DoctorCheck]) -> bool:
    """Return True when any doctor check failed."""

    return any(not check.ok for check in checks)


def _check_search_provider(settings: Settings) -> DoctorCheck:
    detected_provider = settings.detect_search_provider_from_keys()
    provider = settings.get_effective_search_provider()
    key_name = settings.get_search_provider_key_name(provider)
    value = settings.get_search_provider_api_key(provider)
    if value:
        if "search_provider" in settings.model_fields_set:
            detail = f"{key_name} set (explicit provider)"
        elif detected_provider == provider:
            detail = f"{key_name} set (auto-selected provider)"
        else:
            detail = f"{key_name} set"
        return DoctorCheck(name=f"{provider} provider", ok=True, detail=detail)
    if detected_provider is None:
        return DoctorCheck(
            name="search provider",
            ok=False,
            detail="no provider API key configured (set EXA_API_KEY, TAVILY_API_KEY, or FIRECRAWL_API_KEY)",
        )
    return DoctorCheck(name=f"{provider} provider", ok=False, detail=f"{key_name} missing")


def _check_agent_hosts() -> DoctorCheck:
    home_dir = Path.home()
    openclaw_dir = home_dir / ".openclaw"
    hermes_dir = home_dir / ".hermes"

    detected = []
    if openclaw_dir.exists():
        detected.append(f"openclaw: {openclaw_dir}")
    if hermes_dir.exists():
        detected.append(f"hermes: {hermes_dir}")
    if detected:
        return DoctorCheck(name="agent host", ok=True, detail=", ".join(detected))
    return DoctorCheck(name="agent host", ok=True, detail="not detected (optional)")


def _check_binary(name: str, binary: str) -> DoctorCheck:
    path = shutil.which(binary)
    if path:
        return DoctorCheck(name=name, ok=True, detail=path)
    return DoctorCheck(name=name, ok=False, detail=f"{binary} not found in PATH")


def _check_local_agent_harness(settings: Settings) -> DoctorCheck:
    resolved = detect_local_agent_harness(settings)
    if resolved is None:
        candidates = ", ".join(settings.agent_exec_candidates)
        return DoctorCheck(
            name="local agent harness",
            ok=False,
            detail=f"none found (checked: {candidates})",
        )
    harness_name, executable = resolved
    return DoctorCheck(
        name="local agent harness",
        ok=True,
        detail=f"{harness_name}: {executable}",
    )


def _check_storage_path(path: Path) -> DoctorCheck:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return DoctorCheck(name="storage path", ok=False, detail=f"{path} ({exc})")
    return DoctorCheck(name="storage path", ok=True, detail=str(path))


def _check_database_parent(path: Path) -> DoctorCheck:
    parent = path.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return DoctorCheck(name="database path", ok=False, detail=f"{parent} ({exc})")
    return DoctorCheck(name="database path", ok=True, detail=str(path))
