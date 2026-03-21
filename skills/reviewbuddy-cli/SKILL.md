---
name: reviewbuddy-cli
description: Use when installing, operating, validating, or troubleshooting the ReviewBuddy CLI. Covers one-shot runs, saved-run follow-up questions, runtime readiness checks, and dependency fixes on a target machine.
---

# ReviewBuddy CLI

Use this skill when the task is about installing ReviewBuddy, running it, asking questions against saved sessions, or fixing a machine that fails runtime checks.

## Quick Start

1. Run `scripts/reviewbuddy commands --agent` to see the current command surface.
2. Run `scripts/reviewbuddy doctor` before using ReviewBuddy in a new environment.
3. Use `scripts/reviewbuddy run "<prompt>"` for a one-shot research run.
4. Use `scripts/reviewbuddy ask <run_id> "<question>"` when the user wants a follow-up answer from a previous session without re-crawling.

## Read These References As Needed

- For command semantics and examples, read `references/commands.md`.
- For dependency failures and recovery steps, read `references/troubleshooting.md`.

## Operating Rules

- Prefer the `scripts/reviewbuddy` wrapper over calling `uv tool run --from . reviewbuddy` directly unless the user specifically wants the raw command.
- Before debugging a broken runtime, run `scripts/reviewbuddy doctor`.
- ReviewBuddy currently ships cleanly as a packaged CLI, not a hosted web service.
- The runtime depends on the external `codex` CLI. If `doctor` fails on `codex`, do not treat the environment as production-ready.
- If `doctor` fails on `ffmpeg`, follow the OS-specific install steps in `references/troubleshooting.md`, verify with `ffmpeg -version`, then rerun `scripts/reviewbuddy doctor`.

## Outputs

- Run artifacts are written under `data/storage/<run_id>/`.
- Key files are `synthesis.md`, `run.log`, and the `html/` / `markdown/` crawl artifacts.
