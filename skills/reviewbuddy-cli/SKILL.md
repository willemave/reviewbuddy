---
name: reviewbuddy-cli
description: Use when operating, validating, releasing, or troubleshooting the ReviewBuddy CLI. Covers one-shot runs, interactive sessions, saved-run follow-up questions, runtime readiness checks, and CLI release workflows.
---

# ReviewBuddy CLI

Use this skill when the task is about running ReviewBuddy, asking questions against saved sessions, checking whether an environment is ready, or preparing a release that another agent can install.

## Quick Start

1. Run `scripts/reviewbuddy commands --agent` to see the current command surface.
2. Run `scripts/reviewbuddy doctor` before using ReviewBuddy in a new environment.
3. Use `scripts/reviewbuddy run "<prompt>"` for a one-shot research run.
4. Use `scripts/reviewbuddy interactive` when the user wants to browse recent runs or ask local follow-up questions from the terminal UI.
5. Use `scripts/reviewbuddy ask <run_id> "<question>"` when the user wants a follow-up answer from a previous session without re-crawling.
6. Use `scripts/reviewbuddy resume <run_id>` when the user wants to reopen a saved run and continue interactively.

## Read These References As Needed

- For command semantics and examples, read `references/commands.md`.
- For release, bot install, and production constraints, read `references/release.md`.

## Operating Rules

- Prefer the `scripts/reviewbuddy` wrapper over calling `uv run reviewbuddy` directly unless the user specifically wants the raw command.
- Before debugging a broken runtime, run `scripts/reviewbuddy doctor`.
- Before preparing a release, run `./scripts/release-check.sh`.
- ReviewBuddy currently ships cleanly as a packaged CLI, not a hosted web service.
- The runtime depends on the external `codex` CLI. If `doctor` fails on `codex`, do not treat the environment as production-ready.

## Outputs

- Run artifacts are written under `data/storage/<run_id>/`.
- Key files are `synthesis.md`, `run.log`, and the `html/` / `markdown/` crawl artifacts.
