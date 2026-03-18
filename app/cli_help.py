"""Shared CLI help text for users and agents."""

from __future__ import annotations

from dataclasses import dataclass


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
        name="run",
        usage='reviewbuddy run "<prompt>"',
        summary="Execute a new one-shot research run and print the synthesis.",
        details=(
            "Runs planning, search, crawl, synthesis, and writes artifacts under data/storage/<run_id>/.",
            "Best when you want a final answer without staying in an interactive session.",
        ),
        examples=('reviewbuddy run "best dishwasher for quiet apartment"',),
    ),
    CliCommandHelp(
        name="interactive",
        usage="reviewbuddy interactive",
        summary="Start an interactive session that can create new runs or resume saved ones.",
        details=(
            "Shows recent runs on startup.",
            "Supports local Q&A with [a]sk and deeper research with [f]ollowup.",
        ),
        examples=("reviewbuddy interactive --max-urls 50",),
    ),
    CliCommandHelp(
        name="ask",
        usage='reviewbuddy ask <run_id> "<question>"',
        summary="Answer a follow-up question from a saved run without re-crawling.",
        details=(
            "Loads persisted follow-up memory from the saved session.",
            "Useful for previous-session Q&A in scripts or agent workflows.",
        ),
        examples=('reviewbuddy ask abc123 "What were the main warranty concerns?"',),
    ),
    CliCommandHelp(
        name="resume",
        usage="reviewbuddy resume <run_id>",
        summary="Open a saved run, show its synthesis, and continue with questions or more research.",
        details=(
            "Use [a]sk to query saved material locally.",
            "Use [f]ollowup to launch additional research based on the existing report.",
        ),
        examples=("reviewbuddy resume abc123",),
    ),
    CliCommandHelp(
        name="commands",
        usage="reviewbuddy commands [--agent]",
        summary="Print a compact command reference.",
        details=(
            "Use --agent for a flatter, machine-friendly reference format.",
            "Points to the markdown reference files under docs/.",
        ),
        examples=("reviewbuddy commands --agent",),
    ),
    CliCommandHelp(
        name="doctor",
        usage="reviewbuddy doctor",
        summary="Check whether the current machine is ready to run the CLI.",
        details=(
            "Validates required API keys, required binaries, and writable storage paths.",
            "Use this before handing the tool to another bot or promoting a runtime to production.",
        ),
        examples=("reviewbuddy doctor",),
    ),
)

CLI_REFERENCE_PATH = "docs/cli-reference.md"
AGENT_REFERENCE_PATH = "docs/agent-cli-reference.md"


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


def _build_user_reference() -> str:
    lines = [
        "# ReviewBuddy CLI",
        "",
        "Primary entry points:",
        "- Installed command: `reviewbuddy`",
        "- Local wrapper: `scripts/reviewbuddy`",
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
        "# ReviewBuddy CLI For Agents",
        "",
        "Entrypoints:",
        "- `reviewbuddy`",
        "- `scripts/reviewbuddy`",
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
