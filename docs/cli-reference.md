# ResearchBuddy CLI Reference

Primary entry points:
- Installed command: `researchbuddy`
- Local wrapper: `scripts/researchbuddy`

## `researchbuddy start "<prompt>" [--mode auto|product|restaurant|research] [--json]`
Start a new research prompt in the background and make it current.

- Refuses to start when another ResearchBuddy run is already active.
- Returns the run ID, PID, log path, and exact status/watch commands.
- Writes artifacts under `~/.researchbuddy/storage/<run_id>/` by default.
- Use `--json` when another agent or script needs the startup response.
- Use `--mode` to force product-review, restaurant, or general-research behavior.
- Use `--prompt-file` only when you explicitly want to read the prompt from a UTF-8 text file.

Example:
- `researchbuddy start "best dishwasher for quiet apartment"`
- `researchbuddy start "best sushi in portland" --mode restaurant`

## `researchbuddy followup "<question>" [--run-id RUN_ID] [--result-file answer.txt]`
Answer a follow-up question from the current completed run.

- Defaults to the current run selected by `researchbuddy start`.
- Use `--run-id` only when you intentionally want an older run.
- Loads persisted follow-up memory and does not re-crawl sources.
- Prints the answer to stdout and can also write it to `--result-file`.

Example:
- `researchbuddy followup "What were the main warranty concerns?"`

## `researchbuddy status [--run-id RUN_ID] [--json]`
Print status for the current run.

- Defaults to the current run selected by `researchbuddy start`.
- Shows run ID, status, PID health, URL counts, output directory, log path, and synthesis path when available.
- Use `--json` for scripts and agents.

Example:
- `researchbuddy status`
- `researchbuddy status --json`

## `researchbuddy watch [--run-id RUN_ID] [--interval SECONDS] [--timeout SECONDS]`
Watch the current run until it completes or fails.

- Defaults to the current run selected by `researchbuddy start`.
- Prints a compact status block and streams worker log output.
- Exits non-zero when the worker process disappears while the run is still in progress.
- Exits zero when the run completes and non-zero when it fails.

Example:
- `researchbuddy watch`
- `researchbuddy watch --interval 2 --timeout 900`

## `researchbuddy doctor [--fix] [--skip-playwright]`
Check whether the current machine is ready to run the CLI, and optionally fix local setup gaps.

- Validates required API keys, required binaries, Codex auth, Playwright browser launch, and writable storage paths.
- Use `--fix` to run setup before the final report when local dependencies need remediation.
- Reports current-run health and repairs stale in-progress run records when `--fix` is used.

Example:
- `researchbuddy doctor`
- `researchbuddy doctor --fix`

## `researchbuddy skills install openclaw [--scope shared|workspace] [--method auto|symlink|copy] [--json]`
Install the bundled research skill into OpenClaw.

- Defaults to a shared install at `~/.openclaw/skills/research`.
- Use `--scope workspace` to install into `<workspace>/skills/research`.
- Uses `--method auto` by default: shared installs use a symlink, workspace installs use a copy.
- Use `--method symlink` for workspace installs only when the target OpenClaw config allows that symlink target.
- Refuses to replace an existing different skill unless `--force` is passed.
- Use `--dry-run` to preview the install plan without writing.
- Use `--json` for scripts and agents.

Example:
- `researchbuddy skills install openclaw`
- `researchbuddy skills install openclaw --scope workspace --workspace /path/to/workspace`
- `researchbuddy skills install openclaw --dry-run --json`

## `researchbuddy list [--limit 20] [--json]`
List saved prompts and mark the current run.

- Reads the recent run history from the local SQLite database.
- Outputs run ID, created time, status, and a truncated prompt summary.
- Marks the current run with `*` in human output.
- Use `--json` for scripts and agents.

Example:
- `researchbuddy list --limit 10`
