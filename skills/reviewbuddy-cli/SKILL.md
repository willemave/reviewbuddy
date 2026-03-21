---
name: reviewbuddy-cli
description: Use when installing, operating, validating, or troubleshooting the ReviewBuddy CLI. Covers one-shot runs, saved-run follow-up questions, runtime readiness checks, and dependency fixes on a target machine.
---

# ReviewBuddy CLI

Use this skill when the task is about installing ReviewBuddy, running it, asking questions against saved sessions, or fixing a machine that fails runtime checks.

## Quick Start

1. Run `scripts/reviewbuddy commands --agent` to see the current command surface.
2. Run `scripts/reviewbuddy setup` in a new environment, then rerun `scripts/reviewbuddy doctor` if setup reports any failure.
3. Start long `scripts/reviewbuddy run "<prompt>"` jobs in the background, capture the PID, and redirect logs to a file so the agent can keep working while the crawl runs.
4. Use `scripts/reviewbuddy ask <run_id> "<question>"` when the user wants a follow-up answer from a previous session without re-crawling.

## Read These References As Needed

- For command semantics and examples, read `references/commands.md`.
- For dependency failures and recovery steps, read `references/troubleshooting.md`.

## Operating Rules

- Prefer the `scripts/reviewbuddy` wrapper over calling `uv tool run --from . reviewbuddy` directly unless the user specifically wants the raw command.
- Before debugging a broken runtime, run `scripts/reviewbuddy doctor`.
- For long research runs, launch ReviewBuddy in the background. Recommended pattern:
  `scripts/reviewbuddy run "<prompt>" > /tmp/reviewbuddy.log 2>&1 & echo $!`
- Save the printed PID and inspect the redirected log file instead of blocking the agent session on foreground output.
- Only pass `--stats` to `run` when the user explicitly wants fetched/failed URL counts in the terminal output.
- ReviewBuddy currently ships cleanly as a packaged CLI, not a hosted web service.
- The runtime depends on the external `codex` CLI. If `doctor` fails on `codex`, do not treat the environment as production-ready.
- If `doctor` fails on `ffmpeg`, follow the OS-specific install steps in `references/troubleshooting.md`, verify with `ffmpeg -version`, then rerun `scripts/reviewbuddy doctor`.

## Outputs

- Run artifacts are written under `data/storage/<run_id>/`.
- Key files are `synthesis.md`, `run.log`, and the `html/` / `markdown/` crawl artifacts.
