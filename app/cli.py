"""CLI entrypoint for ReviewBuddy."""

import asyncio
import subprocess
from dataclasses import dataclass
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from app.agents.base import AgentDeps
from app.cli_doctor import format_doctor_report, has_doctor_failures, run_doctor_checks
from app.cli_help import build_command_reference
from app.constants import APP_VERSION
from app.core.logging import setup_logging
from app.core.settings import get_settings
from app.models.homebrew import TapExportRequest
from app.models.review import (
    FollowupMemory,
    ReviewRunConfig,
    ReviewRunRequest,
    ReviewRunResult,
    ReviewRunStats,
)
from app.services.followup import answer_followup_question, load_followup_memory
from app.services.homebrew_tap import detect_github_remote, export_tap_repository
from app.services.setup_runtime import format_setup_report, has_setup_failures, run_setup
from app.services.storage import fetch_run, fetch_run_stats, resolve_run_dir
from app.workflows.review import run_review

app = typer.Typer(add_completion=False)
tap_app = typer.Typer(help="Generate Homebrew tap assets.")
app.add_typer(tap_app, name="tap")
console = Console()
settings = get_settings()

PROMPT_ARGUMENT = typer.Argument(..., help="Research question or product to investigate")
RUN_ID_ARGUMENT = typer.Argument(..., help="Saved run identifier")
QUESTION_ARGUMENT = typer.Argument(..., help="Follow-up question to answer from saved memory")
MAX_URLS_OPTION = typer.Option(None, help="Maximum total URLs to process")
MAX_AGENTS_OPTION = typer.Option(None, help="Maximum parallel agents")
HEADFUL_OPTION = typer.Option(True, help="Allow headful fallback if headless is blocked")
TIMEOUT_OPTION = typer.Option(None, help="Navigation timeout in ms")
OUTPUT_DIR_OPTION = typer.Option(None, help="Base output directory")
PLANNER_MODEL_OPTION = typer.Option(None, help="Override model for the lane planner agent")
SUB_AGENT_MODEL_OPTION = typer.Option(None, help="Override model for sub-agents")
TAP_OUTPUT_OPTION = typer.Option(None, help="Output directory for the generated Homebrew tap repo")
STATS_OPTION = typer.Option(False, "--stats", help="Print run statistics before the synthesis")
INSTALL_PLAYWRIGHT_OPTION = typer.Option(
    True,
    "--install-playwright/--skip-playwright",
    help="Install Playwright browsers during setup",
)


@dataclass
class FollowupSessionState:
    """Cached memory for local follow-up answers."""

    run_id: str
    run_dir: Path
    prompt: str
    synthesis_markdown: str
    memory: FollowupMemory | None = None


async def _ensure_followup_memory(state: FollowupSessionState) -> FollowupMemory:
    """Load cached follow-up memory for a run when needed."""

    if state.memory is None:
        state.memory = await load_followup_memory(
            run_id=state.run_id,
            run_dir=state.run_dir,
            prompt=state.prompt,
            synthesis_markdown=state.synthesis_markdown,
        )
    return state.memory


async def _load_followup_state_for_run(
    run_id: str,
    output_dir: Path | None = None,
) -> tuple[ReviewRunResult, FollowupSessionState]:
    """Load a saved run and construct follow-up session state.

    Args:
        run_id: Saved run identifier.
        output_dir: Optional base output directory override.

    Returns:
        Tuple of the saved report and follow-up state.

    Raises:
        typer.Exit: If the run or saved synthesis cannot be found.
    """

    run_record = await fetch_run(settings.database_path, run_id)
    if run_record is None:
        console.print(f"Run not found: {run_id}")
        raise typer.Exit(code=1)

    run_dir = resolve_run_dir(run_record.output_dir, run_id, output_dir)
    synthesis_path = run_dir / "synthesis.md"
    if not synthesis_path.exists():
        console.print(f"Synthesis not found: {synthesis_path}")
        raise typer.Exit(code=1)

    synthesis_text = synthesis_path.read_text(encoding="utf-8", errors="ignore")
    total, fetched, failed = await fetch_run_stats(settings.database_path, run_id)
    report = ReviewRunResult(
        run_id=run_id,
        prompt=run_record.prompt,
        created_at=run_record.created_at,
        stats=ReviewRunStats(total_urls=total, fetched=fetched, failed=failed),
        synthesis_markdown=synthesis_text,
    )
    followup_state = FollowupSessionState(
        run_id=run_id,
        run_dir=run_dir,
        prompt=run_record.prompt,
        synthesis_markdown=synthesis_text,
    )
    return report, followup_state


@app.command()
def run(
    prompt: str = PROMPT_ARGUMENT,
    max_urls: int = MAX_URLS_OPTION,
    max_agents: int = MAX_AGENTS_OPTION,
    headful: bool = HEADFUL_OPTION,
    timeout_ms: int = TIMEOUT_OPTION,
    output_dir: Path = OUTPUT_DIR_OPTION,
    planner_model: str | None = PLANNER_MODEL_OPTION,
    sub_agent_model: str | None = SUB_AGENT_MODEL_OPTION,
    stats: bool = STATS_OPTION,
) -> None:
    """Run a review research workflow."""

    setup_logging(settings.log_level)

    config = ReviewRunConfig(
        max_urls=max_urls or settings.max_urls,
        max_agents=max_agents or settings.max_agents,
        headful=headful,
        navigation_timeout_ms=timeout_ms or settings.navigation_timeout_ms,
        output_dir=output_dir or settings.storage_path,
        planner_model=planner_model,
        sub_agent_model=sub_agent_model,
    )
    request = ReviewRunRequest(prompt=prompt, **config.model_dump())

    deps = AgentDeps(session_id="cli", job_id="cli")

    result = asyncio.run(run_review(request, deps, reporter=None))

    console.print(Panel.fit("ReviewBuddy complete", style="green"))
    console.print(f"Run ID: {result.run_id}")
    if stats:
        console.print(
            f"Fetched {result.stats.fetched}/{result.stats.total_urls} URLs"
            f" ({result.stats.failed} failed)"
        )
    console.print()
    console.print(result.synthesis_markdown)


@app.command()
def commands(
    agent: bool = typer.Option(
        False,
        "--agent",
        help="Render a flatter command reference for agents and scripts",
    ),
) -> None:
    """Print a compact command reference."""

    console.print(build_command_reference(agent=agent))


@app.command()
def doctor() -> None:
    """Check whether the local environment is ready to run ReviewBuddy."""

    checks = run_doctor_checks(settings)
    console.print(format_doctor_report(checks))
    if has_doctor_failures(checks):
        raise typer.Exit(code=1)


@app.command()
def setup(
    install_playwright: bool = INSTALL_PLAYWRIGHT_OPTION,
) -> None:
    """Prepare the local environment to run ReviewBuddy."""

    result = run_setup(settings, install_playwright=install_playwright)
    console.print(format_setup_report(result.actions))
    console.print()
    console.print(format_doctor_report(result.doctor_checks))
    if has_setup_failures(result.actions) or has_doctor_failures(result.doctor_checks):
        raise typer.Exit(code=1)


@app.command()
def ask(
    run_id: str = RUN_ID_ARGUMENT,
    question: str = QUESTION_ARGUMENT,
    output_dir: Path = OUTPUT_DIR_OPTION,
    sub_agent_model: str | None = SUB_AGENT_MODEL_OPTION,
) -> None:
    """Answer a follow-up question from a previous session."""

    setup_logging(settings.log_level)

    async def _run() -> None:
        _report, followup_state = await _load_followup_state_for_run(run_id, output_dir)
        memory = await _ensure_followup_memory(followup_state)
        answer = await answer_followup_question(
            memory,
            question,
            model_name=sub_agent_model,
        )
        console.print(answer)

    asyncio.run(_run())


@tap_app.command("export")
def export_tap(
    output_dir: Path | None = TAP_OUTPUT_OPTION,
    github_owner: str | None = typer.Option(None, help="GitHub owner for source and tap repos"),
    source_repo: str | None = typer.Option(None, help="Source repository name"),
    tap_repo: str = typer.Option("homebrew-reviewbuddy", help="Tap repository name"),
) -> None:
    """Generate a Homebrew tap repository for ReviewBuddy."""

    repo_root = Path(__file__).resolve().parents[1]
    remote = detect_github_remote(repo_root)
    resolved_owner = github_owner
    resolved_source_repo = source_repo
    if remote is not None:
        resolved_owner = resolved_owner or remote[0]
        resolved_source_repo = resolved_source_repo or remote[1]

    if not resolved_owner or not resolved_source_repo:
        console.print("Could not infer the GitHub owner/repo from git remote origin.")
        console.print("Pass --github-owner and --source-repo explicitly.")
        raise typer.Exit(code=1)

    target_output_dir = output_dir or (repo_root.parent / tap_repo)
    request = TapExportRequest(
        output_dir=target_output_dir,
        github_owner=resolved_owner,
        source_repo=resolved_source_repo,
        tap_repo=tap_repo,
        version=APP_VERSION,
        app_description="AI-powered review research assistant with parallel crawling and synthesis",
    )
    result = export_tap_repository(request)

    if not (result.output_dir / ".git").exists():
        subprocess.run(
            ["git", "init", "-b", "main", str(result.output_dir)],
            check=True,
            capture_output=True,
            text=True,
        )

    console.print(Panel.fit("ReviewBuddy tap repository exported", style="green"))
    console.print(f"Output: {result.output_dir}")
    for path in result.files:
        console.print(f"- {path.relative_to(result.output_dir)}")


def main() -> None:
    """CLI entrypoint."""

    app()


if __name__ == "__main__":
    main()
