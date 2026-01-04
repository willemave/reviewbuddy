"""CLI entrypoint for ReviewBuddy."""

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from app.agents.base import AgentDeps
from app.cli_ui import (
    build_followup_prompt,
    build_progress_reporter,
    build_progress_ui,
    configure_interactive_logging,
    prompt_action,
    prompt_config_update,
    prompt_followup,
    prompt_resume_choice,
    show_logs,
    show_report,
)
from app.core.logging import setup_logging
from app.core.settings import get_settings
from app.models.review import ReviewRunConfig, ReviewRunRequest, ReviewRunResult, ReviewRunStats
from app.services.storage import fetch_run, fetch_run_stats, list_runs
from app.workflows.review import run_review

app = typer.Typer(add_completion=False)
console = Console()
settings = get_settings()

PROMPT_ARGUMENT = typer.Argument(..., help="Research question or product to investigate")
MAX_URLS_OPTION = typer.Option(None, help="Maximum total URLs to process")
MAX_AGENTS_OPTION = typer.Option(None, help="Maximum parallel agents")
HEADFUL_OPTION = typer.Option(True, help="Allow headful fallback if headless is blocked")
TIMEOUT_OPTION = typer.Option(None, help="Navigation timeout in ms")
OUTPUT_DIR_OPTION = typer.Option(None, help="Base output directory")
PLANNER_MODEL_OPTION = typer.Option(None, help="Override model for the lane planner agent")
SUB_AGENT_MODEL_OPTION = typer.Option(None, help="Override model for sub-agents")


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
    console.print(f"Fetched {result.stats.fetched}/{result.stats.total_urls} URLs")
    console.print()
    console.print(result.synthesis_markdown)


@app.command()
def interactive(
    prompt: str | None = None,
    max_urls: int = MAX_URLS_OPTION,
    max_agents: int = MAX_AGENTS_OPTION,
    headful: bool = HEADFUL_OPTION,
    timeout_ms: int = TIMEOUT_OPTION,
    output_dir: Path = OUTPUT_DIR_OPTION,
    planner_model: str | None = PLANNER_MODEL_OPTION,
    sub_agent_model: str | None = SUB_AGENT_MODEL_OPTION,
) -> None:
    """Run an interactive session with progress UI and follow-ups."""

    setup_logging(settings.log_level)
    log_buffer = configure_interactive_logging()

    async def _run() -> None:
        deps = AgentDeps(session_id="cli", job_id="cli")
        root_prompt: str
        current_prompt: str
        current_report: str | None = None
        config = ReviewRunConfig(
            max_urls=max_urls or settings.max_urls,
            max_agents=max_agents or settings.max_agents,
            headful=headful,
            navigation_timeout_ms=timeout_ms or settings.navigation_timeout_ms,
            output_dir=output_dir or settings.storage_path,
            planner_model=planner_model,
            sub_agent_model=sub_agent_model,
        )

        runs = await list_runs(settings.database_path, limit=10)
        if runs and prompt is None:
            console.print("Recent runs:")
            for idx, run in enumerate(runs, start=1):
                console.print(f"{idx}. {run.run_id} [{run.status}] {run.prompt}")
            choice = prompt_resume_choice(console, [run.run_id for run in runs])
            if choice > 0 and choice <= len(runs):
                selected = runs[choice - 1]
                base_dir = selected.output_dir
                synthesis_path = base_dir / selected.run_id / "synthesis.md"
                if synthesis_path.exists():
                    synthesis_text = synthesis_path.read_text(encoding="utf-8", errors="ignore")
                    total, fetched, failed = await fetch_run_stats(
                        settings.database_path, selected.run_id
                    )
                    report = ReviewRunResult(
                        run_id=selected.run_id,
                        prompt=selected.prompt,
                        created_at=selected.created_at,
                        stats=ReviewRunStats(total_urls=total, fetched=fetched, failed=failed),
                        synthesis_markdown=synthesis_text,
                    )
                    show_report(console, report)
                    root_prompt = selected.prompt
                    current_prompt = root_prompt
                    current_report = synthesis_text
                else:
                    console.print("Saved synthesis not found; starting new run.")
                    root_prompt = typer.prompt("Research question or product to investigate")
                    current_prompt = root_prompt
            else:
                root_prompt = typer.prompt("Research question or product to investigate")
                current_prompt = root_prompt
        else:
            root_prompt = prompt or typer.prompt("Research question or product to investigate")
            current_prompt = root_prompt

        while True:
            ui = build_progress_ui(console)
            reporter = build_progress_reporter(ui)
            ui.start()
            try:
                request = ReviewRunRequest(prompt=current_prompt, **config.model_dump())
                result = await run_review(request, deps, reporter=reporter)
            finally:
                ui.stop()

            current_report = result.synthesis_markdown
            show_report(console, result)

            while True:
                action = prompt_action(console)
                if action in {"q", "quit"}:
                    return
                if action in {"l", "logs"}:
                    show_logs(console, log_buffer.get_lines())
                    continue
                if action in {"c", "config"}:
                    config = prompt_config_update(console, config)
                    continue
                if action in {"f", "followup"}:
                    followup = prompt_followup(console)
                    if not followup:
                        continue
                    if current_report is None:
                        current_report = ""
                    current_prompt = build_followup_prompt(
                        root_prompt, current_report, followup
                    )
                    break

    asyncio.run(_run())


@app.command()
def resume(
    run_id: str = typer.Argument(..., help="Run ID to resume"),
    max_urls: int = MAX_URLS_OPTION,
    max_agents: int = MAX_AGENTS_OPTION,
    headful: bool = HEADFUL_OPTION,
    timeout_ms: int = TIMEOUT_OPTION,
    output_dir: Path = OUTPUT_DIR_OPTION,
    planner_model: str | None = PLANNER_MODEL_OPTION,
    sub_agent_model: str | None = SUB_AGENT_MODEL_OPTION,
) -> None:
    """Resume a prior run, display its report, and allow follow-ups."""

    setup_logging(settings.log_level)
    log_buffer = configure_interactive_logging()

    async def _run() -> None:
        run_record = await fetch_run(settings.database_path, run_id)
        if run_record is None:
            console.print(f"Run not found: {run_id}")
            raise typer.Exit(code=1)

        base_dir = output_dir or run_record.output_dir
        synthesis_path = base_dir / run_id / "synthesis.md"
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
        show_report(console, report)

        deps = AgentDeps(session_id="cli", job_id="cli")
        config = ReviewRunConfig(
            max_urls=max_urls or settings.max_urls,
            max_agents=max_agents or settings.max_agents,
            headful=headful,
            navigation_timeout_ms=timeout_ms or settings.navigation_timeout_ms,
            output_dir=output_dir or settings.storage_path,
            planner_model=planner_model,
            sub_agent_model=sub_agent_model,
        )
        root_prompt = run_record.prompt
        current_prompt = root_prompt
        current_report = synthesis_text

        while True:
            action = prompt_action(console)
            if action in {"q", "quit"}:
                return
            if action in {"l", "logs"}:
                show_logs(console, log_buffer.get_lines())
                continue
            if action in {"c", "config"}:
                config = prompt_config_update(console, config)
                continue
            if action in {"f", "followup"}:
                followup = prompt_followup(console)
                if not followup:
                    continue
                current_prompt = build_followup_prompt(root_prompt, current_report, followup)

                ui = build_progress_ui(console)
                reporter = build_progress_reporter(ui)
                ui.start()
                try:
                    request = ReviewRunRequest(prompt=current_prompt, **config.model_dump())
                    result = await run_review(request, deps, reporter=reporter)
                finally:
                    ui.stop()

                current_report = result.synthesis_markdown
                show_report(console, result)

    asyncio.run(_run())


def main() -> None:
    """CLI entrypoint."""

    app()


if __name__ == "__main__":
    main()
