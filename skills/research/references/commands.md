# ResearchBuddy Commands

Primary entry points:
- Homebrew/install path: `researchbuddy`
- Source checkout fallback: `scripts/researchbuddy`

Use the Homebrew command unless the user explicitly asks to work inside a source checkout.

## `researchbuddy start [PROMPT]`
- Starts a new research run in the background.
- Makes the new run current.
- Refuses to start if another run has a live worker process.
- Prints the run ID, PID, log path, and exact `status` / `watch` commands.

Options:
- `--prompt-file PATH`: read the prompt from a UTF-8 file instead of the positional prompt.
- `--mode auto|product|restaurant|research`: force research mode.
- `--max-urls INTEGER`: cap total URLs.
- `--max-agents INTEGER`: cap parallel agents.
- `--headful` / `--no-headful`: allow or disable headful browser fallback.
- `--timeout-ms INTEGER`: navigation timeout.
- `--output-dir PATH`: base output directory; defaults to `~/.researchbuddy/storage`.
- `--planner-model TEXT`: override the planner model.
- `--sub-agent-model TEXT`: override sub-agent models.
- `--json`: print machine-readable startup output.

Agent pattern:

```bash
researchbuddy start "<research prompt>" --json
researchbuddy status --json
researchbuddy watch --timeout 900
```

## `researchbuddy followup [QUESTION]`
- Answers a follow-up question from persisted run memory.
- Defaults to the current run.
- Does not re-crawl or re-download sources.

Options:
- `--run-id RUN_ID`: use an explicit older run instead of current.
- `--question-file PATH`: read the question from a UTF-8 file.
- `--output-dir PATH`: base output directory.
- `--result-file PATH`: write the answer to a file.
- `--sub-agent-model TEXT`: override the answering model.

## `researchbuddy status`
- Prints current run status by default.
- Shows PID health, URL counts, output directory, log path, and synthesis path.

Options:
- `--run-id RUN_ID`: inspect an explicit older run instead of current.
- `--output-dir PATH`: base output directory.
- `--json`: print machine-readable output.

## `researchbuddy watch`
- Streams the current worker log and waits until the run completes or fails.
- Exits non-zero when the worker process disappears while the run is still in progress.
- Exits zero on completion and non-zero on failure.

Options:
- `--run-id RUN_ID`: watch an explicit older run instead of current.
- `--output-dir PATH`: base output directory.
- `--interval SECONDS`: polling interval.
- `--timeout SECONDS`: maximum watch time before exit code 124.
- `--lines INTEGER`: initial log lines to print.

## `researchbuddy doctor`
- Validates required API keys, binaries, Codex auth, Playwright browser launch, writable storage paths, and current-run health.
- Use `--fix` when local setup or stale run records should be repaired before the final doctor report.
- If it fails on a dependency such as `ffmpeg`, read `references/troubleshooting.md`.

Options:
- `--fix`: validate detected search config, prepare storage/database, install Playwright browsers, repair stale runs, then rerun checks.
- `--install-playwright` / `--skip-playwright`: control browser installation during `--fix`.

## `researchbuddy list`
- Lists recent saved runs.
- Marks the current run in human output.

Options:
- `--limit INTEGER`: maximum runs to list.
- `--json`: print machine-readable output.

## `researchbuddy skills install openclaw`
- Installs the bundled `research` skill into OpenClaw.
- Defaults to a shared symlink at `~/.openclaw/skills/research`.
- Use the shared scope for all OpenClaw agents on the machine.
- Use workspace scope for one OpenClaw workspace only; the default `auto` method copies there because OpenClaw rejects workspace symlinks unless the target path is explicitly allowed.
- Refuses to replace an existing different skill unless `--force` is passed.

Options:
- `--scope shared|workspace`: choose `~/.openclaw/skills` or `<workspace>/skills`.
- `--workspace PATH`: workspace path when `--scope workspace` is used.
- `--method auto|symlink|copy`: default `auto`; shared uses symlink and workspace uses copy.
- `--source PATH`: explicit skill directory containing `SKILL.md`.
- `--force`: replace an existing target skill.
- `--dry-run`: preview the target and source without writing.
- `--json`: print machine-readable output.

## Command Defaults

- `followup`, `status`, and `watch` default to the current run selected by `start`.
- Only one live research run is allowed at a time.
- Use `--json` on `start`, `status`, `list`, and `skills install` for agent automation.
