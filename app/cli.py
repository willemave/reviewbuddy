"""CLI entrypoint for ReviewBuddy."""

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from app.agents.base import AgentDeps
from app.core.logging import setup_logging
from app.core.settings import get_settings
from app.models.review import ReviewRunRequest
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

    request = ReviewRunRequest(
        prompt=prompt,
        max_urls=max_urls or settings.max_urls,
        max_agents=max_agents or settings.max_agents,
        headful=headful,
        navigation_timeout_ms=timeout_ms or settings.navigation_timeout_ms,
        output_dir=output_dir or settings.storage_path,
        planner_model=planner_model,
        sub_agent_model=sub_agent_model,
    )

    deps = AgentDeps(session_id="cli", job_id="cli")

    result = asyncio.run(run_review(request, deps))

    console.print(Panel.fit("ReviewBuddy complete", style="green"))
    console.print(f"Run ID: {result.run_id}")
    console.print(f"Fetched {result.stats.fetched}/{result.stats.total_urls} URLs")
    console.print()
    console.print(result.synthesis_markdown)


def main() -> None:
    """CLI entrypoint."""

    app()


if __name__ == "__main__":
    main()
