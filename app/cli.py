"""CLI entrypoint for ResearchBuddy."""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import click
import typer
from rich.console import Console
from rich.panel import Panel
from typer.core import TyperCommand, TyperGroup

from app.agents.base import AgentDeps
from app.cli_doctor import format_doctor_report, has_doctor_failures, run_doctor_checks
from app.cli_help import (
    build_command_guidance,
    build_unknown_command_guidance,
)
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
    RunRuntimeRecord,
)
from app.models.skills import (
    SkillInstallMethod,
    SkillInstallRequest,
    SkillInstallResult,
    SkillInstallScope,
)
from app.services.chatgpt import build_chatgpt_continue_url
from app.services.followup import answer_followup_question, load_followup_memory
from app.services.homebrew_tap import detect_github_remote, export_tap_repository
from app.services.local_audio_transcriber import transcribe_audio as transcribe_local_audio
from app.services.podcast_transcriber import (
    PodcastEpisode,
    extract_podcast_transcript,
    is_transcribable_podcast_url,
)
from app.services.reporter import RunReporter
from app.services.research_profiles import (
    ResearchModeOption,
    parse_research_mode_option,
)
from app.services.setup_runtime import format_setup_report, has_setup_failures, run_setup
from app.services.skill_installer import install_skill
from app.services.storage import (
    build_run_paths,
    create_run,
    fetch_current_run_id,
    fetch_run,
    fetch_run_runtime,
    fetch_run_stats,
    finish_run_runtime,
    init_db,
    list_in_progress_runs,
    list_runs,
    new_run_record,
    resolve_run_dir,
    set_current_run_id,
    update_run_status,
    upsert_run_runtime,
)
from app.services.youtube_transcriber import (
    extract_youtube_transcript,
    is_youtube_url,
)
from app.workflows.review import run_review


def _command_lookup_key(ctx: click.Context) -> str | None:
    """Return the shared help key for the current command context."""

    names: list[str] = []
    current = ctx
    while current.parent is not None:
        if current.info_name:
            names.append(current.info_name)
        current = current.parent
    if not names:
        return None
    return " ".join(reversed(names))


def _build_missing_parameter_guidance(ctx: click.Context, exc: click.MissingParameter) -> str:
    """Return richer command guidance for missing parameters."""

    command_name = _command_lookup_key(ctx)
    param_name = exc.param.name if exc.param is not None else "required input"
    reason = f"Missing required input: {param_name}."
    if command_name is None:
        return reason
    return build_command_guidance(command_name, reason=reason)


def _build_unknown_option_guidance(ctx: click.Context, exc: click.NoSuchOption) -> str:
    """Return richer command guidance for unknown options."""

    command_name = _command_lookup_key(ctx)
    option_name = exc.option_name or "option"
    if not option_name.startswith("-"):
        option_name = f"--{option_name}"
    reason = f"Unknown option: {option_name}."
    if command_name is None:
        return reason
    return build_command_guidance(command_name, reason=reason)


def _exit_with_command_guidance(command_name: str, *, reason: str) -> None:
    """Print guided command help and exit with a usage error code."""

    console.print(
        build_command_guidance(command_name, reason=reason), soft_wrap=True, highlight=False
    )
    raise typer.Exit(code=2)


class ResearchBuddyCommand(TyperCommand):
    """Typer command with guided usage errors."""

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        """Parse command arguments and upgrade common usage errors."""

        try:
            return super().parse_args(ctx, args)
        except click.MissingParameter as exc:
            raise click.UsageError(_build_missing_parameter_guidance(ctx, exc), ctx=ctx) from exc
        except click.NoSuchOption as exc:
            raise click.UsageError(_build_unknown_option_guidance(ctx, exc), ctx=ctx) from exc


class ResearchBuddyGroup(TyperGroup):
    """Typer group with guided command suggestions."""

    def resolve_command(
        self,
        ctx: click.Context,
        args: list[str],
    ) -> tuple[str | None, click.Command | None, list[str]]:
        """Resolve a subcommand and show richer guidance for unknown names."""

        cmd_name = str(args[0])
        original_cmd_name = cmd_name
        cmd = self.get_command(ctx, cmd_name)

        if cmd is None and ctx.token_normalize_func is not None:
            cmd_name = ctx.token_normalize_func(cmd_name)
            cmd = self.get_command(ctx, cmd_name)

        if cmd is None and not ctx.resilient_parsing:
            if cmd_name.startswith("-"):
                self.parse_args(ctx, args)
            available_commands = tuple(self.list_commands(ctx))
            ctx.fail(
                build_unknown_command_guidance(
                    original_cmd_name,
                    available_commands=available_commands,
                )
            )

        return cmd_name if cmd else None, cmd, args[1:]


app = typer.Typer(
    name="researchbuddy",
    add_completion=False,
    cls=ResearchBuddyGroup,
    help="Start one ResearchBuddy prompt at a time, follow up on it, and inspect local runtime state.",
    invoke_without_command=True,
)
tap_app = typer.Typer(
    cls=ResearchBuddyGroup,
    help="Generate and inspect Homebrew tap publishing assets for ResearchBuddy.",
    invoke_without_command=True,
)
skills_app = typer.Typer(
    cls=ResearchBuddyGroup,
    help="Install bundled ResearchBuddy skills into local agent runtimes.",
    invoke_without_command=True,
)
app.add_typer(tap_app, name="tap", hidden=True)
app.add_typer(skills_app, name="skills")
console = Console()
settings = get_settings()

OPTIONAL_PROMPT_ARGUMENT = typer.Argument(
    None,
    help="Research question or product to investigate.",
)
OPTIONAL_QUESTION_ARGUMENT = typer.Argument(
    None,
    help="Follow-up question to answer from saved memory.",
)
MAX_URLS_OPTION = typer.Option(None, help="Maximum total URLs to process")
MAX_AGENTS_OPTION = typer.Option(None, help="Maximum parallel agents")
HEADFUL_OPTION = typer.Option(True, help="Allow headful fallback if headless is blocked")
TIMEOUT_OPTION = typer.Option(None, help="Navigation timeout in ms")
OUTPUT_DIR_OPTION = typer.Option(None, help="Base output directory")
PROMPT_FILE_OPTION = typer.Option(
    None,
    "--prompt-file",
    help="Read the run prompt from a UTF-8 text file.",
)
QUESTION_FILE_OPTION = typer.Option(
    None,
    "--question-file",
    help="Read the follow-up question from a UTF-8 text file.",
)
RESULT_FILE_OPTION = typer.Option(
    None,
    "--result-file",
    help="Write the final text output to an explicit file path.",
)
WORKER_REQUEST_FILE_OPTION = typer.Option(
    ...,
    "--request-file",
    help="Serialized run request",
)
PLANNER_MODEL_OPTION = typer.Option(None, help="Override model for the lane planner agent")
SUB_AGENT_MODEL_OPTION = typer.Option(None, help="Override model for sub-agents")
TAP_OUTPUT_OPTION = typer.Option(None, help="Output directory for the generated Homebrew tap repo")
LIST_LIMIT_OPTION = typer.Option(20, "--limit", min=1, help="Maximum runs to list")
RUN_ID_OPTION = typer.Option(None, "--run-id", help="Use an explicit run instead of current")
JSON_OPTION = typer.Option(False, "--json", help="Print machine-readable JSON")
WATCH_INTERVAL_OPTION = typer.Option(5.0, "--interval", min=0.25, help="Polling interval seconds")
WATCH_TIMEOUT_OPTION = typer.Option(
    None,
    "--timeout",
    min=1.0,
    help="Maximum seconds to watch before exiting with timeout",
)
INSTALL_PLAYWRIGHT_OPTION = typer.Option(
    True,
    "--install-playwright/--skip-playwright",
    help="Install Playwright browsers during setup",
)
RESEARCH_MODE_OPTION = typer.Option(
    "auto",
    "--mode",
    help="Force the research mode: auto, product, restaurant, or research.",
)
SOURCE_TYPE_OPTION = typer.Option(
    "auto",
    "--type",
    help="Transcription input type: auto, youtube, podcast, or audio.",
)
SKILL_SCOPE_OPTION = typer.Option(
    "shared",
    "--scope",
    help="Install scope: shared or workspace.",
)
SKILL_METHOD_OPTION = typer.Option(
    "auto",
    "--method",
    help="Install method: auto, symlink, or copy.",
)
SKILL_SOURCE_OPTION = typer.Option(
    None,
    "--source",
    help="Explicit ResearchBuddy skill directory containing SKILL.md.",
)
SKILL_WORKSPACE_OPTION = typer.Option(
    None,
    "--workspace",
    help="OpenClaw workspace path when --scope workspace is used.",
)


@dataclass
class FollowupSessionState:
    """Cached memory for local follow-up answers."""

    run_id: str
    run_dir: Path
    prompt: str
    synthesis_markdown: str
    memory: FollowupMemory | None = None


def _read_utf8_text_file(path: Path, *, label: str) -> str:
    """Read a UTF-8 text file for CLI input.

    Args:
        path: File path to read.
        label: User-facing label for error messages.

    Returns:
        Trimmed file contents.

    Raises:
        typer.BadParameter: If the path is invalid or unreadable.
    """

    if not path.exists():
        raise typer.BadParameter(f"{label} file not found: {path}")
    if not path.is_file():
        raise typer.BadParameter(f"{label} path is not a file: {path}")
    try:
        value = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise typer.BadParameter(f"{label} file must be valid UTF-8 text: {path}") from exc
    except OSError as exc:
        raise typer.BadParameter(f"Could not read {label} file {path}: {exc}") from exc
    cleaned = value.strip()
    if not cleaned:
        raise typer.BadParameter(f"{label} file is empty: {path}")
    return cleaned


def _resolve_text_input(
    value: str | None,
    file_path: Path | None,
    *,
    field_name: str,
) -> str:
    """Resolve one explicit text input from CLI args/options.

    Args:
        value: Positional string value.
        file_path: Optional file path.
        field_name: User-facing field name.

    Returns:
        Resolved non-empty text input.

    Raises:
        typer.BadParameter: If the input is missing, duplicated, or invalid.
    """

    if value and file_path is not None:
        raise typer.BadParameter(
            f"Pass either the {field_name} argument or --{field_name}-file, not both."
        )
    if file_path is not None:
        return _read_utf8_text_file(file_path, label=field_name)
    cleaned = (value or "").strip()
    if not cleaned:
        raise typer.BadParameter(
            f"Missing {field_name}. Pass it as an argument or provide --{field_name}-file."
        )
    return cleaned


def _write_result_file(path: Path, contents: str) -> None:
    """Write CLI output to an explicit deterministic file path."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
    except OSError as exc:
        raise typer.BadParameter(f"Could not write result file {path}: {exc}") from exc


def _print_chatgpt_continue_action(summary: str) -> None:
    """Print a ChatGPT continuation link for the given summary."""

    url = build_chatgpt_continue_url(summary)
    console.print(f"Continue this conversation in ChatGPT: {url}", soft_wrap=True, highlight=False)


def _resolve_transcribe_type(source: str, source_type: str) -> str:
    """Resolve a concrete transcription type from CLI input."""

    if source_type != "auto":
        return source_type
    candidate_path = Path(source).expanduser()
    if candidate_path.exists() and candidate_path.is_file():
        return "audio"
    if is_youtube_url(source):
        return "youtube"
    if is_transcribable_podcast_url(source):
        return "podcast"
    raise typer.BadParameter(
        "Could not infer transcription type. Use --type audio, --type youtube, or --type podcast."
    )


def _truncate_text(value: str, limit: int) -> str:
    """Return a single-line truncated string for CLI tables."""

    compact = " ".join(value.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: max(0, limit - 3)]}..."


def _print_status_line(message: str) -> None:
    """Print an immediate stdout status line for long-running CLI work."""

    console.print(message, soft_wrap=True, highlight=False)
    flush = getattr(console.file, "flush", None)
    if callable(flush):
        flush()


def _print_json(value: object) -> None:
    """Print JSON without Rich inserting presentation line breaks."""

    console.file.write(json.dumps(value, indent=2))
    console.file.write("\n")
    flush = getattr(console.file, "flush", None)
    if callable(flush):
        flush()


def _build_cli_run_reporter() -> RunReporter:
    """Return a lightweight stdout progress reporter for CLI runs."""

    queued_any_urls = False
    processed_urls = 0
    fetched_urls = 0
    failed_urls = 0
    planned_lanes = 0
    completed_lanes = 0

    def on_lanes_planned(count: int) -> None:
        nonlocal planned_lanes
        planned_lanes = count
        noun = "lane" if count == 1 else "lanes"
        _print_status_line(f"Planned {count} {noun}. Starting crawl.")

    def on_urls_discovered(count: int) -> None:
        nonlocal queued_any_urls
        if queued_any_urls or count <= 0:
            return
        queued_any_urls = True
        noun = "URL" if count == 1 else "URLs"
        _print_status_line(f"Queued {count} {noun} for fetch.")

    def on_url_done(ok: bool) -> None:
        nonlocal processed_urls, fetched_urls, failed_urls
        processed_urls += 1
        if ok:
            fetched_urls += 1
        else:
            failed_urls += 1
        if processed_urls != 1 and processed_urls % 5 != 0:
            return
        _print_status_line(
            f"Processed {processed_urls} URLs ({fetched_urls} fetched, {failed_urls} failed)."
        )

    def on_lane_done(lane_name: str) -> None:
        nonlocal completed_lanes
        completed_lanes += 1
        if planned_lanes > 0:
            _print_status_line(f"Completed lane {completed_lanes}/{planned_lanes}: {lane_name}")
            return
        _print_status_line(f"Completed lane: {lane_name}")

    return RunReporter(
        on_lanes_planned=on_lanes_planned,
        on_urls_discovered=on_urls_discovered,
        on_url_done=on_url_done,
        on_lane_done=on_lane_done,
    )


def _build_run_request(
    prompt: str,
    *,
    mode: ResearchModeOption,
    max_urls: int | None,
    max_agents: int | None,
    headful: bool,
    timeout_ms: int | None,
    output_dir: Path | None,
    planner_model: str | None,
    sub_agent_model: str | None,
) -> ReviewRunRequest:
    """Build a validated review request from CLI options."""

    config = ReviewRunConfig(
        max_urls=max_urls or settings.max_urls,
        max_agents=max_agents or settings.max_agents,
        headful=headful,
        navigation_timeout_ms=timeout_ms or settings.navigation_timeout_ms,
        output_dir=output_dir or settings.storage_path,
        research_mode=parse_research_mode_option(mode),
        planner_model=planner_model,
        sub_agent_model=sub_agent_model,
    )
    return ReviewRunRequest(prompt=prompt, **config.model_dump())


def _pid_is_running(pid: int | None) -> bool:
    """Return whether a process id appears to still be alive."""

    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


async def _find_live_run() -> tuple[str, int] | None:
    """Return the first in-progress run with a live worker process."""

    await init_db(settings.database_path)
    for run_record in await list_in_progress_runs(settings.database_path):
        runtime = await fetch_run_runtime(settings.database_path, run_record.run_id)
        if runtime is not None and _pid_is_running(runtime.pid):
            return run_record.run_id, runtime.pid or 0
    return None


async def _resolve_run_id_or_exit(run_id: str | None) -> str:
    """Resolve explicit or current run ID, exiting with agent-friendly guidance."""

    if run_id:
        return run_id
    current_run_id = await fetch_current_run_id(settings.database_path)
    if current_run_id:
        return current_run_id
    console.print('No current run. Start one with `researchbuddy start "research prompt"`.')
    raise typer.Exit(code=1)


def _tail_text(path: Path, max_lines: int) -> str:
    """Return the last lines of a UTF-8-ish text file."""

    if max_lines <= 0 or not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(lines[-max_lines:])


async def _run_status_payload(run_id: str, output_dir: Path | None = None) -> dict[str, object]:
    """Build a status payload for one run."""

    await init_db(settings.database_path)
    run_record = await fetch_run(settings.database_path, run_id)
    if run_record is None:
        console.print(f"Run not found: {run_id}")
        raise typer.Exit(code=1)

    run_dir = resolve_run_dir(run_record.output_dir, run_id, output_dir)
    runtime = await fetch_run_runtime(settings.database_path, run_id)
    total, fetched, failed = await fetch_run_stats(settings.database_path, run_id)
    pid = runtime.pid if runtime else None
    is_running = _pid_is_running(pid)
    log_path = runtime.log_path if runtime and runtime.log_path else run_dir / "run.log"
    synthesis_path = run_dir / "synthesis.md"
    return {
        "run_id": run_record.run_id,
        "prompt": run_record.prompt,
        "created_at": run_record.created_at.isoformat(),
        "status": run_record.status,
        "pid": pid,
        "is_process_running": is_running,
        "output_dir": str(run_dir),
        "log_path": str(log_path),
        "synthesis_path": str(synthesis_path) if synthesis_path.exists() else None,
        "urls": {"total": total, "fetched": fetched, "failed": failed},
        "started_at": runtime.started_at.isoformat() if runtime else None,
        "finished_at": runtime.finished_at.isoformat() if runtime and runtime.finished_at else None,
        "exit_code": runtime.exit_code if runtime else None,
    }


def _print_status_payload(payload: dict[str, object]) -> None:
    """Print a compact human-readable status block."""

    urls = payload["urls"]
    assert isinstance(urls, dict)
    pid = payload["pid"]
    process_label = "running" if payload["is_process_running"] else "not running"
    console.print(f"Run ID: {payload['run_id']}")
    console.print(f"Status: {payload['status']}")
    console.print(f"Process: {pid or '-'} ({process_label})")
    console.print(f"Prompt: {payload['prompt']}", soft_wrap=True)
    console.print(
        f"URLs: total={urls['total']} fetched={urls['fetched']} failed={urls['failed']}"
    )
    console.print(f"Output Dir: {payload['output_dir']}")
    console.print(f"Log: {payload['log_path']}")
    if payload["synthesis_path"]:
        console.print(f"Synthesis: {payload['synthesis_path']}")


def _parse_skill_scope(value: str) -> SkillInstallScope:
    """Parse a skill install scope option."""

    if value in {"shared", "workspace"}:
        return cast(SkillInstallScope, value)
    raise typer.BadParameter("scope must be one of: shared, workspace")


def _parse_skill_method(value: str, scope: SkillInstallScope) -> SkillInstallMethod:
    """Parse a skill install method option."""

    if value == "auto":
        return "copy" if scope == "workspace" else "symlink"
    if value in {"symlink", "copy"}:
        return cast(SkillInstallMethod, value)
    raise typer.BadParameter("method must be one of: auto, symlink, copy")


def _print_skill_install_result(result: SkillInstallResult) -> None:
    """Print a compact human-readable skill install result."""

    console.print(f"Status: {result.status}")
    console.print(f"Skill: {result.skill_name}")
    console.print(f"Agent: {result.agent}")
    console.print(f"Scope: {result.scope}")
    console.print(f"Method: {result.method}")
    if result.source_path is not None:
        console.print(f"Source: {result.source_path}")
    console.print(f"Target: {result.target_path}")
    console.print(f"Detail: {result.detail}", soft_wrap=True)
    if result.next_steps:
        console.print("Next steps:")
        for step in result.next_steps:
            console.print(f"- {step}", soft_wrap=True)


async def _record_runtime_start(run_id: str, pid: int | None, log_path: Path) -> None:
    """Persist runtime metadata for a process."""

    now = datetime.now(UTC)
    await upsert_run_runtime(
        settings.database_path,
        RunRuntimeRecord(
            run_id=run_id,
            pid=pid,
            log_path=log_path,
            started_at=now,
            updated_at=now,
        ),
    )


def _spawn_worker(run_id: str, request_file: Path, log_path: Path) -> int:
    """Spawn a detached worker process and return its PID."""

    command = [
        sys.executable,
        "-m",
        "app.cli",
        "_worker",
        run_id,
        "--request-file",
        str(request_file),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return process.pid


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

    await init_db(settings.database_path)
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


@app.callback()
def cli_callback(ctx: typer.Context) -> None:
    """Render top-level help when ResearchBuddy is invoked without a subcommand."""

    if ctx.invoked_subcommand is not None:
        return
    console.print(ctx.get_help(), highlight=False)
    raise typer.Exit()


@tap_app.callback()
def tap_callback(ctx: typer.Context) -> None:
    """Render tap help when the tap group is invoked without a subcommand."""

    if ctx.invoked_subcommand is not None:
        return
    console.print(ctx.get_help(), highlight=False)
    raise typer.Exit()


@skills_app.callback()
def skills_callback(ctx: typer.Context) -> None:
    """Render skills help when the skills group is invoked without a subcommand."""

    if ctx.invoked_subcommand is not None:
        return
    console.print(ctx.get_help(), highlight=False)
    raise typer.Exit()


@app.command(
    help="Start a new research prompt in the background and make it current.",
    cls=ResearchBuddyCommand,
)
def start(
    prompt: str | None = OPTIONAL_PROMPT_ARGUMENT,
    prompt_file: Path | None = PROMPT_FILE_OPTION,
    mode: ResearchModeOption = RESEARCH_MODE_OPTION,
    max_urls: int = MAX_URLS_OPTION,
    max_agents: int = MAX_AGENTS_OPTION,
    headful: bool = HEADFUL_OPTION,
    timeout_ms: int = TIMEOUT_OPTION,
    output_dir: Path = OUTPUT_DIR_OPTION,
    planner_model: str | None = PLANNER_MODEL_OPTION,
    sub_agent_model: str | None = SUB_AGENT_MODEL_OPTION,
    json_output: bool = JSON_OPTION,
) -> None:
    """Start a new research workflow in a worker process."""

    setup_logging(settings.log_level)
    try:
        resolved_prompt = _resolve_text_input(prompt, prompt_file, field_name="prompt")
    except typer.BadParameter as exc:
        _exit_with_command_guidance("start", reason=str(exc))

    async def _run() -> None:
        live_run = await _find_live_run()
        if live_run is not None:
            live_run_id, pid = live_run
            console.print(
                f"Run already in progress: {live_run_id} (pid {pid}). "
                "Use `researchbuddy status` or `researchbuddy watch`."
            )
            raise typer.Exit(code=1)

        request = _build_run_request(
            resolved_prompt,
            mode=mode,
            max_urls=max_urls,
            max_agents=max_agents,
            headful=headful,
            timeout_ms=timeout_ms,
            output_dir=output_dir,
            planner_model=planner_model,
            sub_agent_model=sub_agent_model,
        )
        run_id = uuid.uuid4().hex
        run_paths = build_run_paths(request.output_dir, run_id)
        run_record = new_run_record(
            run_id=run_id,
            prompt=request.prompt,
            max_urls=request.max_urls,
            max_agents=request.max_agents,
            headful=request.headful,
            output_dir=run_paths["run"],
        )
        await init_db(settings.database_path)
        await create_run(settings.database_path, run_record)
        await set_current_run_id(settings.database_path, run_id)

        request_file = run_paths["run"] / "request.json"
        request_file.write_text(request.model_dump_json(indent=2), encoding="utf-8")
        log_path = run_paths["run"] / "worker.log"
        pid = _spawn_worker(run_id, request_file, log_path)
        await _record_runtime_start(run_id, pid, log_path)

        status_command = f"researchbuddy status --run-id {run_id}"
        watch_command = f"researchbuddy watch --run-id {run_id}"
        if json_output:
            _print_json(
                {
                    "run_id": run_id,
                    "pid": pid,
                    "prompt": request.prompt,
                    "status": run_record.status,
                    "output_dir": str(run_paths["run"]),
                    "log_path": str(log_path),
                    "status_command": status_command,
                    "watch_command": watch_command,
                }
            )
            return

        console.print(Panel.fit("ResearchBuddy started", style="green"))
        console.print(f"Run ID: {run_id}")
        console.print(f"PID: {pid}")
        console.print(f"Prompt: {_truncate_text(request.prompt, 120)}", soft_wrap=True)
        console.print(f"Status: {status_command}")
        console.print(f"Watch: {watch_command}")
        console.print(f"Log: {log_path}")

    asyncio.run(_run())


@app.command(
    "list",
    help="List saved prompts and mark the current run.",
    cls=ResearchBuddyCommand,
)
def list_command(
    limit: int = LIST_LIMIT_OPTION,
    json_output: bool = JSON_OPTION,
) -> None:
    """List saved runs with current-run context."""

    async def _run() -> None:
        await init_db(settings.database_path)
        records = await list_runs(settings.database_path, limit=limit)
        current_run_id = await fetch_current_run_id(settings.database_path)
        if json_output:
            _print_json(
                [
                    {
                        "run_id": record.run_id,
                        "created_at": record.created_at.isoformat(),
                        "status": record.status,
                        "prompt": record.prompt,
                        "is_current": record.run_id == current_run_id,
                    }
                    for record in records
                ]
            )
            return
        if not records:
            console.print("No runs found.")
            return
        console.print("# Runs")
        console.print()
        for record in records:
            marker = "*" if record.run_id == current_run_id else " "
            created_at = record.created_at.astimezone().strftime("%Y-%m-%d %H:%M")
            prompt = _truncate_text(record.prompt, 72)
            console.print(f"{marker} {record.run_id}  {created_at}  {record.status}  {prompt}")

    asyncio.run(_run())


@app.command(help="Print status for the current run.", cls=ResearchBuddyCommand)
def status(
    run_id: str | None = RUN_ID_OPTION,
    output_dir: Path = OUTPUT_DIR_OPTION,
    json_output: bool = JSON_OPTION,
) -> None:
    """Print status for the current or explicit run."""

    async def _run() -> None:
        resolved_run_id = await _resolve_run_id_or_exit(run_id)
        payload = await _run_status_payload(resolved_run_id, output_dir=output_dir)
        if json_output:
            _print_json(payload)
            return
        _print_status_payload(payload)

    asyncio.run(_run())


@app.command(help="Watch the current run until it completes or fails.", cls=ResearchBuddyCommand)
def watch(
    run_id: str | None = RUN_ID_OPTION,
    output_dir: Path = OUTPUT_DIR_OPTION,
    interval: float = WATCH_INTERVAL_OPTION,
    timeout: float | None = WATCH_TIMEOUT_OPTION,
    lines: int = typer.Option(40, "--lines", min=0, help="Initial log lines to print"),
) -> None:
    """Stream status and worker logs for the current or explicit run."""

    async def _run() -> None:
        resolved_run_id = await _resolve_run_id_or_exit(run_id)
        last_size = 0
        printed_initial = False
        started_at = time.monotonic()
        while True:
            payload = await _run_status_payload(resolved_run_id, output_dir=output_dir)
            log_path = Path(str(payload["log_path"]))
            if not printed_initial:
                _print_status_payload(payload)
                initial_text = _tail_text(log_path, lines)
                if initial_text:
                    console.print()
                    console.print(initial_text)
                last_size = log_path.stat().st_size if log_path.exists() else 0
                printed_initial = True
            elif log_path.exists():
                size = log_path.stat().st_size
                if size > last_size:
                    with log_path.open("r", encoding="utf-8", errors="ignore") as file:
                        file.seek(last_size)
                        chunk = file.read()
                    if chunk:
                        console.print(chunk.rstrip("\n"))
                    last_size = size

            if payload["status"] in {"completed", "failed"}:
                console.print()
                _print_status_payload(payload)
                raise typer.Exit(code=0 if payload["status"] == "completed" else 1)
            if (
                payload["status"] == "in_progress"
                and payload["pid"] is not None
                and not payload["is_process_running"]
            ):
                console.print()
                console.print(
                    "Run worker is no longer running while status is in_progress. "
                    "Run `researchbuddy doctor --fix` to mark stale runs failed."
                )
                _print_status_payload(payload)
                raise typer.Exit(code=1)
            if timeout is not None and time.monotonic() - started_at >= timeout:
                console.print()
                console.print(f"Timed out after {timeout:g}s while watching run.")
                _print_status_payload(payload)
                raise typer.Exit(code=124)
            await asyncio.sleep(interval)

    asyncio.run(_run())


@app.command(
    help="Run environment readiness checks and optionally repair local setup first.",
    cls=ResearchBuddyCommand,
)
def doctor(
    fix: bool = typer.Option(
        False,
        "--fix",
        help="Run setup actions before re-running doctor checks",
    ),
    install_playwright: bool = INSTALL_PLAYWRIGHT_OPTION,
) -> None:
    """Check whether the local environment is ready to run ResearchBuddy.

    Prints a full readiness report.

    Use ``--fix`` to run setup actions before printing the final doctor report.
    """

    async def _repair_stale_runs() -> int:
        repaired = 0
        await init_db(settings.database_path)
        for run_record in await list_in_progress_runs(settings.database_path):
            runtime = await fetch_run_runtime(settings.database_path, run_record.run_id)
            if runtime is not None and _pid_is_running(runtime.pid):
                continue
            await update_run_status(settings.database_path, run_record.run_id, "failed")
            await finish_run_runtime(settings.database_path, run_record.run_id, exit_code=1)
            repaired += 1
        return repaired

    async def _current_run_report() -> str:
        current_run_id = await fetch_current_run_id(settings.database_path)
        if current_run_id is None:
            return "Current run: none"
        run_record = await fetch_run(settings.database_path, current_run_id)
        if run_record is None:
            return f"Current run: {current_run_id} (missing run record)"
        runtime = await fetch_run_runtime(settings.database_path, current_run_id)
        pid = runtime.pid if runtime else None
        process_label = "running" if _pid_is_running(pid) else "not running"
        return f"Current run: {current_run_id} ({run_record.status}, pid {pid or '-'} {process_label})"

    setup_failed = False
    if fix:
        result = run_setup(settings, install_playwright=install_playwright)
        console.print(format_setup_report(result.actions))
        console.print()
        checks = result.doctor_checks
        setup_failed = has_setup_failures(result.actions)
        repaired_count = asyncio.run(_repair_stale_runs())
        if repaired_count:
            console.print(f"Repaired {repaired_count} stale run record(s).")
    else:
        checks = run_doctor_checks(settings)
    console.print(format_doctor_report(checks))
    console.print()
    console.print(asyncio.run(_current_run_report()))
    if setup_failed or has_doctor_failures(checks):
        raise typer.Exit(code=1)


@skills_app.command(
    "install",
    help="Install the bundled research skill into a supported local agent.",
    cls=ResearchBuddyCommand,
)
def install_agent_skill(
    agent: str = typer.Argument(..., help="Agent target: openclaw"),
    scope: str = SKILL_SCOPE_OPTION,
    method: str = SKILL_METHOD_OPTION,
    source_path: Path | None = SKILL_SOURCE_OPTION,
    workspace_path: Path | None = SKILL_WORKSPACE_OPTION,
    force: bool = typer.Option(False, "--force", help="Replace an existing target skill"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show the install plan without writing"),
    json_output: bool = JSON_OPTION,
) -> None:
    """Install the bundled ResearchBuddy skill into a local agent skill directory."""

    if agent != "openclaw":
        _exit_with_command_guidance("skills install", reason="agent must be openclaw.")
    try:
        parsed_scope = _parse_skill_scope(scope)
        parsed_method = _parse_skill_method(method, parsed_scope)
    except typer.BadParameter as exc:
        _exit_with_command_guidance("skills install", reason=str(exc))

    result = install_skill(
        SkillInstallRequest(
            agent="openclaw",
            scope=parsed_scope,
            method=parsed_method,
            source_path=source_path,
            workspace_path=workspace_path,
            force=force,
            dry_run=dry_run,
        )
    )
    if json_output:
        _print_json(result.model_dump(mode="json"))
    else:
        _print_skill_install_result(result)
    if not result.ok:
        raise typer.Exit(code=1)


@app.command(
    "followup",
    help="Answer a follow-up question from the current completed run.",
    cls=ResearchBuddyCommand,
)
def followup(
    question: str | None = OPTIONAL_QUESTION_ARGUMENT,
    run_id: str | None = RUN_ID_OPTION,
    question_file: Path | None = QUESTION_FILE_OPTION,
    output_dir: Path = OUTPUT_DIR_OPTION,
    result_file: Path | None = RESULT_FILE_OPTION,
    sub_agent_model: str | None = SUB_AGENT_MODEL_OPTION,
) -> None:
    """Answer a follow-up question from the current or explicit run."""

    setup_logging(settings.log_level)
    try:
        resolved_question = _resolve_text_input(question, question_file, field_name="question")
    except typer.BadParameter as exc:
        _exit_with_command_guidance("followup", reason=str(exc))

    async def _run() -> None:
        resolved_run_id = await _resolve_run_id_or_exit(run_id)
        _print_status_line(f"Loading run {resolved_run_id} for follow-up.")
        _report, followup_state = await _load_followup_state_for_run(resolved_run_id, output_dir)
        memory = await _ensure_followup_memory(followup_state)
        _print_status_line(
            f'Running follow-up answer for: "{_truncate_text(resolved_question, 80)}"'
        )
        answer = await answer_followup_question(
            memory,
            resolved_question,
            model_name=sub_agent_model,
        )
        if result_file is not None:
            _write_result_file(result_file, answer)
            console.print(f"Result file: {result_file}", soft_wrap=True)
        console.print(answer)
        console.print()
        _print_chatgpt_continue_action(answer)

    asyncio.run(_run())


@app.command("_worker", hidden=True, cls=ResearchBuddyCommand)
def worker(
    run_id: str = typer.Argument(..., help="Precreated run identifier"),
    request_file: Path = WORKER_REQUEST_FILE_OPTION,
) -> None:
    """Run a precreated research workflow in a background worker."""

    setup_logging(settings.log_level)
    request = ReviewRunRequest.model_validate_json(request_file.read_text(encoding="utf-8"))

    async def _run() -> None:
        exit_code = 0
        try:
            await _record_runtime_start(run_id, os.getpid(), request.output_dir / run_id / "worker.log")
            await run_review(
                request,
                AgentDeps(session_id="cli", job_id=run_id),
                reporter=_build_cli_run_reporter(),
                run_id=run_id,
                create_record=False,
            )
        except Exception:
            exit_code = 1
            raise
        finally:
            await finish_run_runtime(settings.database_path, run_id, exit_code=exit_code)

    asyncio.run(_run())


@app.command(
    help="Transcribe a local audio file, YouTube URL, or podcast URL with local Whisper.",
    hidden=True,
    cls=ResearchBuddyCommand,
)
def transcribe(
    source: str = typer.Argument(..., help="Local audio file path or supported URL"),
    source_type: str = SOURCE_TYPE_OPTION,
    whisper_model: str = typer.Option(None, "--whisper-model", help="Override Whisper model name"),
    result_file: Path | None = RESULT_FILE_OPTION,
) -> None:
    """Transcribe one source locally and print the transcript."""

    resolved_type = _resolve_transcribe_type(source, source_type)
    model_name = whisper_model or settings.whisper_model

    with tempfile.TemporaryDirectory(prefix="researchbuddy-transcribe-") as temp_dir:
        temp_path = Path(temp_dir)
        if resolved_type == "audio":
            transcript = transcribe_local_audio(Path(source).expanduser(), model_name)
        elif resolved_type == "youtube":
            _transcript_id, _title, transcript = extract_youtube_transcript(
                source,
                temp_path / "audio",
                temp_path / "transcripts",
                model_name,
            )
        elif resolved_type == "podcast":
            _transcript_id, _title, transcript = extract_podcast_transcript(
                PodcastEpisode(url=source),
                temp_path / "audio",
                temp_path / "transcripts",
                model_name,
            )
        else:
            raise typer.BadParameter(
                "Unsupported transcription type. Use auto, youtube, podcast, or audio."
            )

    if result_file is not None:
        _write_result_file(result_file, transcript)
        console.print(f"Result file: {result_file}", soft_wrap=True)
    console.print(transcript)


@tap_app.command(
    "export",
    help="Write a Homebrew tap repo with formula, docs, workflow, and maintainer skill.",
    cls=ResearchBuddyCommand,
)
def export_tap(
    output_dir: Path | None = TAP_OUTPUT_OPTION,
    github_owner: str | None = typer.Option(None, help="GitHub owner for source and tap repos"),
    source_repo: str | None = typer.Option(None, help="Source repository name"),
    tap_repo: str = typer.Option("homebrew-researchbuddy", help="Tap repository name"),
) -> None:
    """Generate a Homebrew tap repository and print the output path plus written files."""

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

    console.print(Panel.fit("ResearchBuddy tap repository exported", style="green"))
    console.print(f"Output: {result.output_dir}")
    for path in result.files:
        console.print(f"- {path.relative_to(result.output_dir)}")


def _registered_command_name(command_info) -> str:
    """Return a stable command name for Typer registration ordering."""

    explicit_name = getattr(command_info, "name", None)
    if explicit_name:
        return explicit_name
    callback = getattr(command_info, "callback", None)
    callback_name = getattr(callback, "__name__", "")
    if callback_name == "list_command":
        return "list"
    return callback_name


def _order_registered_commands() -> None:
    """Keep help output aligned with the agent-first command contract."""

    command_order = {
        "start": 0,
        "followup": 1,
        "status": 2,
        "watch": 3,
        "doctor": 4,
        "list": 5,
    }
    app.registered_commands.sort(
        key=lambda command_info: command_order.get(
            _registered_command_name(command_info),
            100,
        )
    )


_order_registered_commands()


def main() -> None:
    """CLI entrypoint."""

    app()


if __name__ == "__main__":
    main()
