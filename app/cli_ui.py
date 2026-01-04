"""Interactive CLI helpers using Rich."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.prompt import Confirm, IntPrompt, Prompt

from app.models.review import ReviewRunConfig, ReviewRunResult
from app.services.reporter import RunReporter


class LogBufferHandler(logging.Handler):
    """Buffer log lines for later viewing."""

    def __init__(self, max_lines: int = 500) -> None:
        super().__init__()
        self._max_lines = max_lines
        self._lines: list[str] = []
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        line = self.format(record)
        self._lines.append(line)
        if len(self._lines) > self._max_lines:
            self._lines = self._lines[-self._max_lines :]

    def get_lines(self) -> Iterable[str]:
        """Return buffered log lines."""

        return list(self._lines)


@dataclass
class ProgressUI:
    """Progress UI wrapper."""

    progress: Progress
    lanes_task: TaskID
    urls_task: TaskID

    def start(self) -> None:
        """Start the progress display."""

        self.progress.start()

    def stop(self) -> None:
        """Stop the progress display."""

        self.progress.stop()


def configure_interactive_logging(level: str = "WARNING") -> LogBufferHandler:
    """Configure logging for interactive mode."""

    root = logging.getLogger()
    for handler in root.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setLevel(level)

    buffer_handler = LogBufferHandler()
    buffer_handler.setLevel(logging.INFO)
    root.addHandler(buffer_handler)
    return buffer_handler


def build_progress_ui(console: Console) -> ProgressUI:
    """Create progress UI with lanes and URLs tasks."""

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )
    lanes_task = progress.add_task("Lanes", total=0)
    urls_task = progress.add_task("URLs", total=0)
    return ProgressUI(progress=progress, lanes_task=lanes_task, urls_task=urls_task)


def build_progress_reporter(ui: ProgressUI) -> RunReporter:
    """Create a RunReporter that updates the progress UI."""

    def on_lanes_planned(count: int) -> None:
        ui.progress.update(ui.lanes_task, total=count)

    def on_urls_discovered(count: int) -> None:
        if count <= 0:
            return
        task = ui.urls_task
        current_total = ui.progress.tasks[task].total or 0
        ui.progress.update(task, total=current_total + count)

    def on_url_done(_: bool) -> None:
        ui.progress.advance(ui.urls_task, 1)

    def on_lane_done(_: str) -> None:
        ui.progress.advance(ui.lanes_task, 1)

    return RunReporter(
        on_lanes_planned=on_lanes_planned,
        on_urls_discovered=on_urls_discovered,
        on_url_done=on_url_done,
        on_lane_done=on_lane_done,
    )


def show_report(console: Console, report: ReviewRunResult) -> None:
    """Display a report in a scrollable pager."""

    with console.pager():
        console.print(report.synthesis_markdown)


def show_logs(console: Console, logs: Iterable[str]) -> None:
    """Display logs in a scrollable pager."""

    with console.pager():
        for line in logs:
            console.print(line)


def prompt_followup(console: Console) -> str | None:
    """Prompt for a follow-up question."""

    value = Prompt.ask(
        "Follow-up question (blank to cancel)",
        default="",
        console=console,
    )
    return value.strip() or None


def prompt_action(console: Console) -> str:
    """Prompt for an interactive action."""

    return Prompt.ask(
        "[f]ollowup, [c]onfig, [l]ogs, [q]uit",
        default="q",
        console=console,
    ).strip().lower()


def build_followup_prompt(original_prompt: str, report: str, followup: str) -> str:
    """Construct a follow-up prompt using the prior report."""

    return (
        "You are continuing a research session. Use the existing report to "
        "identify gaps and plan additional research lanes that answer the follow-up.\n\n"
        f"Original prompt:\n{original_prompt}\n\n"
        f"Existing report:\n{report}\n\n"
        f"Follow-up question:\n{followup}\n"
    )


def prompt_resume_choice(console: Console, runs: list[str]) -> int:
    """Prompt for a resume choice index."""

    prompt = "Resume a run (0 for new)"
    return IntPrompt.ask(prompt, console=console, default=0, show_default=True)


def format_run_config(config: ReviewRunConfig) -> str:
    """Format configuration details for display.

    Args:
        config: Current review run configuration.

    Returns:
        Formatted configuration string.
    """

    planner = config.planner_model or "none"
    sub_agent = config.sub_agent_model or "none"
    lines = [
        f"max_urls: {config.max_urls}",
        f"max_agents: {config.max_agents}",
        f"headful: {config.headful}",
        f"navigation_timeout_ms: {config.navigation_timeout_ms}",
        f"output_dir: {config.output_dir}",
        f"planner_model: {planner}",
        f"sub_agent_model: {sub_agent}",
    ]
    return "\n".join(lines)


def prompt_config_update(console: Console, config: ReviewRunConfig) -> ReviewRunConfig:
    """Prompt for configuration updates.

    Args:
        console: Rich console instance.
        config: Current review run configuration.

    Returns:
        Updated review run configuration.
    """

    console.print(Panel.fit("Configuration", style="cyan"))
    console.print(format_run_config(config))
    console.print("Press enter to keep current values. Type 'none' to clear overrides.")

    while True:
        max_urls = IntPrompt.ask("Max URLs", default=config.max_urls, show_default=True)
        max_agents = IntPrompt.ask(
            "Max parallel agents",
            default=config.max_agents,
            show_default=True,
        )
        headful = Confirm.ask("Allow headful fallback", default=config.headful)
        timeout_ms = IntPrompt.ask(
            "Navigation timeout (ms)",
            default=config.navigation_timeout_ms,
            show_default=True,
        )
        output_dir_value = Prompt.ask(
            "Output directory",
            default=str(config.output_dir),
            show_default=True,
            console=console,
        )
        planner_input = Prompt.ask(
            "Planner model override",
            default=config.planner_model or "",
            show_default=config.planner_model is not None,
            console=console,
        )
        sub_agent_input = Prompt.ask(
            "Sub-agent model override",
            default=config.sub_agent_model or "",
            show_default=config.sub_agent_model is not None,
            console=console,
        )

        try:
            return ReviewRunConfig(
                max_urls=max_urls,
                max_agents=max_agents,
                headful=headful,
                navigation_timeout_ms=timeout_ms,
                output_dir=Path(output_dir_value).expanduser(),
                planner_model=_coerce_optional_override(planner_input, config.planner_model),
                sub_agent_model=_coerce_optional_override(
                    sub_agent_input,
                    config.sub_agent_model,
                ),
            )
        except ValidationError as exc:
            console.print(f"Invalid configuration: {exc}")
            retry = Confirm.ask("Try again?", default=True)
            if not retry:
                return config


def _coerce_optional_override(raw: str, current: str | None) -> str | None:
    value = raw.strip()
    if not value:
        return current
    if value.lower() in {"none", "null", "clear"}:
        return None
    return value
