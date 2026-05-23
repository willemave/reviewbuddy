"""Shared CLI help text for users and agents."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from difflib import get_close_matches


@dataclass(frozen=True)
class CliCommandHelp:
    """Structured help for one CLI command."""

    name: str
    usage: str
    summary: str
    details: tuple[str, ...]
    examples: tuple[str, ...]


CLI_COMMANDS: tuple[CliCommandHelp, ...] = (
    CliCommandHelp(
        name="start",
        usage='researchbuddy start "<prompt>" [--mode auto|product|restaurant|research]',
        summary="Start a new research prompt in the background and make it current.",
        details=(
            "Refuses to start when another ResearchBuddy run is already active.",
            "Returns the run ID, PID, log path, and exact status/watch commands.",
            "Writes artifacts under ~/.researchbuddy/storage/<run_id>/ by default.",
            "Use --json when another agent or script needs the startup response.",
            "Use --mode to force product-review, restaurant, or general-research behavior instead of automatic prompt inference.",
            "Use --prompt-file only when you explicitly want to read the prompt from a UTF-8 text file.",
        ),
        examples=(
            'researchbuddy start "best dishwasher for quiet apartment"',
            'researchbuddy start "best sushi in portland" --mode restaurant',
            "researchbuddy start --prompt-file prompt.txt",
        ),
    ),
    CliCommandHelp(
        name="followup",
        usage='researchbuddy followup "<question>" [--run-id RUN_ID] [--result-file answer.txt]',
        summary="Answer a follow-up question from the current completed run.",
        details=(
            "Defaults to the current run selected by researchbuddy start.",
            "Use --run-id only when you intentionally want an older run.",
            "Loads persisted follow-up memory and does not re-crawl sources.",
            "Prints the follow-up answer to stdout, and can also write the answer to --result-file.",
            "Prints a ChatGPT continuation link based on the answer so you can keep the thread going in ChatGPT.",
            "Use the positional question for normal CLI usage.",
            "Use --question-file only when you explicitly want to read the question from a UTF-8 text file.",
            "Use --result-file when you want the answer written to a deterministic path.",
            "Useful for previous-session Q&A in scripts or agent workflows.",
        ),
        examples=(
            'researchbuddy followup "What were the main warranty concerns?"',
            "researchbuddy followup --run-id abc123 --question-file question.txt --result-file answer.txt",
        ),
    ),
    CliCommandHelp(
        name="status",
        usage="researchbuddy status [--run-id RUN_ID] [--json]",
        summary="Print status for the current run.",
        details=(
            "Defaults to the current run selected by researchbuddy start.",
            "Shows run ID, status, PID health, URL counts, output directory, log path, and synthesis path when available.",
            "Use --json for scripts and agents.",
        ),
        examples=(
            "researchbuddy status",
            "researchbuddy status --json",
        ),
    ),
    CliCommandHelp(
        name="watch",
        usage="researchbuddy watch [--run-id RUN_ID] [--interval SECONDS] [--timeout SECONDS]",
        summary="Watch the current run until it completes or fails.",
        details=(
            "Defaults to the current run selected by researchbuddy start.",
            "Prints a compact status block and streams worker log output.",
            "Exits non-zero when the worker process disappears while the run is still in progress.",
            "Exits zero when the run completes and non-zero when it fails.",
        ),
        examples=(
            "researchbuddy watch",
            "researchbuddy watch --interval 2 --timeout 900",
        ),
    ),
    CliCommandHelp(
        name="doctor",
        usage="researchbuddy doctor [--fix] [--skip-playwright]",
        summary="Check whether the current machine is ready to run the CLI, and optionally fix local setup gaps.",
        details=(
            "Validates required API keys, required binaries, Codex auth, Playwright browser launch, and writable storage paths.",
            "Use --fix to run the setup flow first so local storage, config, and Playwright browsers are repaired before the final report.",
            "Reports current-run health and repairs stale in-progress run records when --fix is used.",
            "Use this before handing the tool to another bot or promoting a runtime to production.",
            "Prints a full readiness report and exits non-zero when setup or runtime checks fail.",
        ),
        examples=("researchbuddy doctor", "researchbuddy doctor --fix"),
    ),
    CliCommandHelp(
        name="skills install",
        usage=(
            "researchbuddy skills install openclaw "
            "[--scope shared|workspace] [--method auto|symlink|copy] [--json]"
        ),
        summary="Install the bundled research skill into OpenClaw.",
        details=(
            "Defaults to a shared install at ~/.openclaw/skills/research.",
            "Use --scope workspace to install into <workspace>/skills/research.",
            "Uses --method auto by default: shared installs use a symlink, workspace installs use a copy.",
            "Use --method symlink only when the target OpenClaw config allows that symlink target.",
            "Refuses to replace an existing different skill unless --force is passed.",
            "Use --dry-run to preview the install plan without writing.",
            "Use --json for scripts and agents.",
        ),
        examples=(
            "researchbuddy skills install openclaw",
            "researchbuddy skills install openclaw --scope workspace --workspace /path/to/workspace",
            "researchbuddy skills install openclaw --dry-run --json",
        ),
    ),
    CliCommandHelp(
        name="list",
        usage="researchbuddy list [--limit 20] [--json]",
        summary="List saved prompts and mark the current run.",
        details=(
            "Reads the recent run history from the local SQLite database.",
            "Outputs run ID, created time, status, and a truncated prompt summary.",
            "Marks the current run with * in human output.",
            "Use --json for scripts and agents.",
        ),
        examples=("researchbuddy list --limit 10",),
    ),
)

CLI_REFERENCE_PATH = "docs/cli-reference.md"
AGENT_REFERENCE_PATH = "docs/agent-cli-reference.md"
RENAMED_COMMANDS: dict[str, str] = {}
CLI_COMMANDS_BY_NAME = {command.name: command for command in CLI_COMMANDS}


def build_command_reference(*, agent: bool = False) -> str:
    """Build CLI reference text.

    Args:
        agent: Whether to render a flatter agent-oriented format.

    Returns:
        Markdown command reference.
    """

    if agent:
        return _build_agent_reference()
    return _build_user_reference()


def get_command_help(name: str) -> CliCommandHelp | None:
    """Return shared help metadata for one command."""

    return CLI_COMMANDS_BY_NAME.get(name)


def suggest_command_names(
    value: str,
    *,
    limit: int = 3,
    available_commands: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Return the best available command suggestions for a bad command name."""

    if value in RENAMED_COMMANDS:
        return (RENAMED_COMMANDS[value],)

    searchable = tuple(available_commands or CLI_COMMANDS_BY_NAME)
    matches = get_close_matches(value, searchable, n=limit, cutoff=0.34)
    if matches:
        return tuple(matches)
    return searchable[:limit]


def build_command_guidance(command_name: str, *, reason: str | None = None) -> str:
    """Build guided help text for a specific command."""

    command = get_command_help(command_name)
    if command is None:
        fallback_reason = reason or f"No help found for command '{command_name}'."
        return "\n".join(
            (fallback_reason, "", "Use `researchbuddy --help` for the full command list.")
        )

    lines: list[str] = []
    if reason:
        lines.extend((reason, ""))
    lines.extend(
        (
            f"Usage: {command.usage}",
            command.summary,
            "",
            "Details:",
            *[f"- {detail}" for detail in command.details],
            "",
            "Examples:",
            *[f"- {example}" for example in command.examples],
        )
    )
    return "\n".join(lines).strip()


def build_unknown_command_guidance(
    command_name: str,
    *,
    available_commands: Sequence[str] | None = None,
) -> str:
    """Build guided help text for an unknown command."""

    if command_name in RENAMED_COMMANDS:
        replacement = RENAMED_COMMANDS[command_name]
        return build_command_guidance(
            replacement,
            reason=f"Command '{command_name}' was renamed to '{replacement}'.",
        )

    suggestions = suggest_command_names(
        command_name,
        available_commands=available_commands,
    )
    lines = [f"No such command '{command_name}'."]
    if suggestions:
        lines.extend(("", "Closest matches:"))
        for suggestion_name in suggestions:
            command = get_command_help(suggestion_name)
            if command is None:
                continue
            lines.append(f"- {command.usage}")
            lines.append(f"  {command.summary}")
    lines.extend(
        (
            "",
            "Use `researchbuddy --help` for the full command list.",
            "See docs/cli-reference.md for the extended reference.",
        )
    )
    return "\n".join(lines).strip()


def _build_user_reference() -> str:
    lines = [
        "# ResearchBuddy CLI",
        "",
        "Primary entry points:",
        "- Installed command: `researchbuddy`",
        "- Local wrapper: `scripts/researchbuddy`",
        "",
        "Commands:",
    ]
    for command in CLI_COMMANDS:
        lines.extend(
            [
                f"## `{command.usage}`",
                command.summary,
                "",
                *[f"- {detail}" for detail in command.details],
                "",
                "Example:",
                *[f"- `{example}`" for example in command.examples],
                "",
            ]
        )
    lines.extend(
        [
            "Reference files:",
            f"- `{CLI_REFERENCE_PATH}`",
            f"- `{AGENT_REFERENCE_PATH}`",
        ]
    )
    return "\n".join(lines).strip()


def _build_agent_reference() -> str:
    lines = [
        "# ResearchBuddy CLI For Agents",
        "",
        "Entrypoints:",
        "- `researchbuddy`",
        "- `scripts/researchbuddy`",
        "",
    ]
    for command in CLI_COMMANDS:
        lines.extend(
            [
                f"## {command.name}",
                f"Usage: `{command.usage}`",
                f"Purpose: {command.summary}",
                "Behavior:",
                *[f"- {detail}" for detail in command.details],
                "Example:",
                *[f"- `{example}`" for example in command.examples],
                "",
            ]
        )
    lines.extend(
        [
            "Docs:",
            f"- `{CLI_REFERENCE_PATH}`",
            f"- `{AGENT_REFERENCE_PATH}`",
        ]
    )
    return "\n".join(lines).strip()
