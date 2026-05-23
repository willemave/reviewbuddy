# ResearchBuddy CLI For Agents

Entrypoints:
- `researchbuddy`
- `scripts/researchbuddy`

## start
Usage: `researchbuddy start "<prompt>" [--mode auto|product|restaurant|research] [--json]`
Purpose: Start a new research prompt in the background and make it current.
Behavior:
- Refuses to start when another ResearchBuddy run is already active.
- Returns the run ID, PID, log path, and exact status/watch commands.
- Writes artifacts under `~/.researchbuddy/storage/<run_id>/` by default.
- Use `--json` for machine-readable startup output.
- Use `--prompt-file` for file-backed prompts.
Example:
- `researchbuddy start "best dishwasher for quiet apartment"`

## followup
Usage: `researchbuddy followup "<question>" [--run-id RUN_ID] [--result-file answer.txt]`
Purpose: Answer a follow-up question from the current completed run.
Behavior:
- Defaults to the current run selected by `researchbuddy start`.
- Use `--run-id` only when intentionally targeting an older run.
- Does not re-crawl sources.
Example:
- `researchbuddy followup "What were the main warranty concerns?"`

## status
Usage: `researchbuddy status [--run-id RUN_ID] [--json]`
Purpose: Print status for the current run.
Behavior:
- Defaults to the current run.
- Shows status, PID health, URL counts, output directory, log path, and synthesis path.
- Use `--json` for machine-readable output.
Example:
- `researchbuddy status --json`

## watch
Usage: `researchbuddy watch [--run-id RUN_ID] [--interval SECONDS] [--timeout SECONDS]`
Purpose: Watch the current run until it completes or fails.
Behavior:
- Defaults to the current run.
- Streams worker logs.
- Exits non-zero if the worker process disappears while the run is still in progress.
- Exits zero on completion and non-zero on failure.
Example:
- `researchbuddy watch --timeout 900`

## doctor
Usage: `researchbuddy doctor [--fix] [--skip-playwright]`
Purpose: Check whether the current machine is ready to run the CLI.
Behavior:
- Validates API keys, binaries, Codex auth, Playwright launch, and writable storage paths.
- Use `--fix` to repair setup and stale in-progress run records.
- Run this before starting research on a new machine.
Example:
- `researchbuddy doctor`
- `researchbuddy doctor --fix`

## skills install
Usage: `researchbuddy skills install openclaw [--scope shared|workspace] [--method auto|symlink|copy] [--json]`
Purpose: Install the bundled research skill into OpenClaw.
Behavior:
- Defaults to a shared install at `~/.openclaw/skills/research`.
- Use `--scope workspace` to install into `<workspace>/skills/research`.
- Uses `--method auto` by default: shared installs use a symlink, workspace installs use a copy.
- Refuses to replace an existing different skill unless `--force` is passed.
- Use `--dry-run` to preview the install plan without writing.
- Use `--json` for machine-readable output.
Example:
- `researchbuddy skills install openclaw --json`

## list
Usage: `researchbuddy list [--limit 20] [--json]`
Purpose: List saved prompts and mark the current run.
Behavior:
- Reads local SQLite run history.
- Marks the current run with `*` in human output.
- Use `--json` for machine-readable output.
Example:
- `researchbuddy list --json`
