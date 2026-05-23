---
name: research
description: Use when a user asks to research something with ResearchBuddy, run web research or deep research, produce a source-backed answer, follow up on a saved research session, or install, validate, and troubleshoot the Homebrew-installed researchbuddy CLI. Covers research runs, current-run follow-up, runtime readiness checks, OpenClaw search config reuse, and dependency fixes on a target machine.
---

# Research

Use this skill when the task is about using ResearchBuddy for research, installing it with Homebrew, asking questions against saved sessions, or fixing a machine that fails runtime checks.

## Quick Start

1. Install with `brew tap willemave/researchbuddy` and `brew install researchbuddy` unless the user explicitly wants a source checkout.
2. Run `researchbuddy --help` to confirm the installed command surface.
3. For OpenClaw, install this skill with `researchbuddy skills install openclaw --scope shared`, or use `--scope workspace --workspace /path/to/workspace` for one workspace only.
4. In OpenClaw environments, inspect `~/.openclaw/openclaw.json` before asking for search-provider credentials. If it already has `plugins.entries.<provider>.config.webSearch.apiKey`, legacy `tools.web.search.<provider>.apiKey`, or `tools.web.search.apiKey` for `exa`, `tavily`, or `firecrawl`, ask once whether ResearchBuddy should reuse it.
5. If the user approves reuse, do not ask them to paste a duplicate API key. Run `researchbuddy doctor --fix`; ResearchBuddy auto-loads OpenClaw config and does not copy provider credentials into local `.env`.
6. If OpenClaw has no usable provider config, or the user declines reuse, ask for one provider key and set it through the runtime environment or local `.env`, then rerun `researchbuddy doctor --fix`.
7. Start long research jobs with `researchbuddy start "<prompt>" --json`; the command returns immediately with the run ID, PID, status command, watch command, and log path.
8. Use `researchbuddy status`, `researchbuddy watch`, and `researchbuddy followup "<question>"` against the current run by default.

## Read These References As Needed

- For every public command and option, read `references/commands.md`.
- For dependency failures and recovery steps, read `references/troubleshooting.md`.

## Operating Rules

- Prefer the Homebrew-installed `researchbuddy` command. Use `scripts/researchbuddy` only inside a source checkout.
- During first-time OpenClaw installs, prefer reusing the provider configured in `~/.openclaw/openclaw.json` over asking the user for a fresh `EXA_API_KEY`, `TAVILY_API_KEY`, or `FIRECRAWL_API_KEY`.
- Do not copy OpenClaw secrets into skill files, chat logs, docs, command output, or ResearchBuddy `.env`. Let ResearchBuddy load them directly from OpenClaw or Hermes.
- Before debugging a broken runtime, run `researchbuddy doctor`.
- For long research work, use `researchbuddy start "<prompt>" --json`; do not manually background a worker process.
- Use `researchbuddy status --json` when another agent or script needs machine-readable progress.
- Use `researchbuddy watch --timeout 900` when you need to stream the active worker log until completion.
- ResearchBuddy currently ships cleanly as a packaged CLI, not a hosted web service.
- The runtime depends on the external `codex` CLI. If `doctor` fails on `codex`, do not treat the environment as production-ready.
- If `doctor` fails on `ffmpeg`, follow the OS-specific install steps in `references/troubleshooting.md`, verify with `ffmpeg -version`, then rerun `researchbuddy doctor`.

## Public Commands

- `researchbuddy start`
- `researchbuddy followup`
- `researchbuddy status`
- `researchbuddy watch`
- `researchbuddy doctor`
- `researchbuddy skills install openclaw`
- `researchbuddy list`

Read `references/commands.md` before using less-common options such as `--prompt-file`, `--result-file`, `--output-dir`, model overrides, `--skip-playwright`, or JSON output.

## Config Model

- ResearchBuddy is env-driven. It reads process environment variables, `~/.hermes/.env`, `~/.openclaw/.env`, `~/.openclaw/openclaw.json`, and local `.env`.
- Shared agent configs are canonical for search-provider credentials; local `.env` is only for manual fallback config.
- `researchbuddy doctor --fix` validates detected search-provider config, prepares storage/database, installs Playwright browsers, and repairs stale runs. It does not copy provider credentials into `.env`.
- Homebrew installs should not edit files under the Homebrew prefix or uv cache for config.

## Outputs

- Run artifacts are written under `~/.researchbuddy/storage/<run_id>/` by default.
- Key files are `synthesis.md`, `worker.log`, `run.log`, and the `html/` / `markdown/` crawl artifacts.
