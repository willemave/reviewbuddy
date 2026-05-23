"""Environment checks for CLI and bot deployments."""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from app.core.settings import Settings
from app.services.codex_exec import detect_local_agent_harness, local_agent_env

DOCTOR_CODEX_AUTH_TIMEOUT_SECONDS = 10
DOCTOR_PLAYWRIGHT_TIMEOUT_SECONDS = 20
PLAYWRIGHT_BROWSER_CHECK_SCRIPT = (
    "from playwright.sync_api import sync_playwright\n"
    "with sync_playwright() as playwright:\n"
    "    browser = playwright.chromium.launch(headless=True)\n"
    "    browser.close()\n"
    "print('Chromium launch OK')\n"
)

SEARCH_PROVIDER_SIGNUP_URLS = {
    "exa": "https://dashboard.exa.ai/api-keys",
    "firecrawl": "https://www.firecrawl.dev/app/api-keys",
}


@dataclass(frozen=True)
class DoctorCheck:
    """Single environment readiness check."""

    name: str
    ok: bool
    detail: str


def run_doctor_checks(settings: Settings) -> list[DoctorCheck]:
    """Run runtime checks needed for ResearchBuddy.

    Args:
        settings: Loaded application settings.

    Returns:
        Ordered check results.
    """

    checks = [
        _check_search_provider(settings),
        _check_agent_hosts(),
        *_check_local_agent_checks(settings),
        _check_binary("uv", "uv"),
        _check_binary("ffmpeg", "ffmpeg"),
        _check_playwright_browser(),
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

    lines = ["# ResearchBuddy Doctor", ""]
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
            detail=(
                "no provider API key configured "
                "(set EXA_API_KEY, TAVILY_API_KEY, or FIRECRAWL_API_KEY; "
                "Exa signup: https://dashboard.exa.ai/api-keys; "
                "Firecrawl signup: https://www.firecrawl.dev/app/api-keys)"
            ),
        )
    signup_url = SEARCH_PROVIDER_SIGNUP_URLS.get(provider)
    if signup_url:
        return DoctorCheck(
            name=f"{provider} provider",
            ok=False,
            detail=f"{key_name} missing (get one at {signup_url})",
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


def _check_local_agent_checks(settings: Settings) -> list[DoctorCheck]:
    resolved = detect_local_agent_harness(settings)
    if resolved is None:
        candidates = ", ".join(settings.agent_exec_candidates)
        return [
            DoctorCheck(
                name="local agent harness",
                ok=False,
                detail=f"none found (checked: {candidates})",
            )
        ]
    harness_name, executable = resolved
    checks = [
        DoctorCheck(
            name="local agent harness",
            ok=True,
            detail=f"{harness_name}: {executable}",
        )
    ]
    if harness_name == "codex":
        checks.append(_check_codex_auth(executable))
    return checks


def _check_codex_auth(executable: str) -> DoctorCheck:
    try:
        completed = subprocess.run(
            [executable, "login", "status"],
            check=False,
            capture_output=True,
            text=True,
            timeout=DOCTOR_CODEX_AUTH_TIMEOUT_SECONDS,
            env=local_agent_env("codex"),
        )
    except OSError as exc:
        return DoctorCheck(name="codex auth", ok=False, detail=f"status check failed ({exc})")
    except subprocess.TimeoutExpired:
        return DoctorCheck(
            name="codex auth",
            ok=False,
            detail=f"`codex login status` timed out after {DOCTOR_CODEX_AUTH_TIMEOUT_SECONDS}s",
        )

    summary = _summarize_command_output(completed.stdout, completed.stderr)
    combined_output = "\n".join([completed.stdout, completed.stderr]).lower()
    if "not logged in" not in combined_output and "logged in" in combined_output:
        return DoctorCheck(name="codex auth", ok=True, detail=summary or "logged in")

    if completed.returncode != 0:
        detail = summary or f"`codex login status` exited with code {completed.returncode}"
        return DoctorCheck(name="codex auth", ok=False, detail=f"{detail}; run `codex login`")

    detail = summary or "not authenticated"
    return DoctorCheck(name="codex auth", ok=False, detail=f"{detail}; run `codex login`")


def _check_playwright_browser() -> DoctorCheck:
    try:
        completed = subprocess.run(
            [sys.executable, "-c", PLAYWRIGHT_BROWSER_CHECK_SCRIPT],
            check=False,
            capture_output=True,
            text=True,
            timeout=DOCTOR_PLAYWRIGHT_TIMEOUT_SECONDS,
        )
    except OSError as exc:
        return DoctorCheck(
            name="playwright browsers",
            ok=False,
            detail=f"browser check failed ({exc}); run `researchbuddy doctor --fix`",
        )
    except subprocess.TimeoutExpired:
        return DoctorCheck(
            name="playwright browsers",
            ok=False,
            detail=(
                "Chromium launch check timed out "
                f"after {DOCTOR_PLAYWRIGHT_TIMEOUT_SECONDS}s; run `researchbuddy doctor --fix`"
            ),
        )

    summary = _summarize_command_output(completed.stdout, completed.stderr)
    if completed.returncode == 0:
        return DoctorCheck(
            name="playwright browsers",
            ok=True,
            detail=summary or "Chromium launch OK",
        )

    detail = summary or "Chromium launch failed"
    return DoctorCheck(
        name="playwright browsers",
        ok=False,
        detail=f"{detail}; run `researchbuddy doctor --fix`",
    )


def _summarize_command_output(stdout: str, stderr: str) -> str:
    lines = [_clean_command_line(line) for line in (*stdout.splitlines(), *stderr.splitlines())]
    lines = [line for line in lines if line]
    actionable_lines = [line for line in lines if _is_actionable_command_line(line)]
    if actionable_lines:
        return " | ".join(actionable_lines[:3])

    filtered_lines = [line for line in lines if not _is_traceback_noise_line(line)]
    return " | ".join((filtered_lines or lines)[:3])


def _clean_command_line(line: str) -> str:
    return line.strip().strip("║").strip()


def _is_actionable_command_line(line: str) -> bool:
    lowered = line.lower()
    return (
        "error:" in lowered
        or "executable doesn't exist" in lowered
        or "please run" in lowered
        or "playwright install" in lowered
        or "not logged in" in lowered
        or "logged in" in lowered
    )


def _is_traceback_noise_line(line: str) -> bool:
    if line.startswith(("Traceback ", 'File "', "raise ", "return ", "await ", "self.")):
        return True
    if set(line) <= {"^", " ", "-"}:
        return True
    return line.startswith(("╔", "╚", "═"))


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
