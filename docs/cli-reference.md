# ReviewBuddy CLI Reference

Primary entry points:
- Installed command: `reviewbuddy`
- Local wrapper: `scripts/reviewbuddy`

## `reviewbuddy run "<prompt>" [--stats]`
Execute a new one-shot research run and print the synthesis.

- Runs planning, search, crawl, synthesis, and writes artifacts under `data/storage/<run_id>/`.
- Use `--stats` when you also want the fetched/failed URL counts printed in the terminal output.
- Best when you want a final answer in a single command.

Example:
- `reviewbuddy run "best dishwasher for quiet apartment"`

## `reviewbuddy ask <run_id> "<question>"`
Answer a follow-up question from a saved run without re-crawling.

- Loads persisted follow-up memory from the saved session.
- Useful for previous-session Q&A in scripts or agent workflows.

Example:
- `reviewbuddy ask abc123 "What were the main warranty concerns?"`

## `reviewbuddy commands [--agent]`
Print a compact command reference.

- Use `--agent` for a flatter, machine-friendly reference format.
- Points to the markdown reference files under `docs/`.

Example:
- `reviewbuddy commands --agent`

## `reviewbuddy setup [--skip-playwright]`
Prepare the local machine to run the CLI, then rerun doctor checks.

- Persists detected search-provider settings into the local `.env` when possible.
- Creates the storage/database paths and optionally installs Playwright browsers.

Example:
- `reviewbuddy setup`

## `reviewbuddy doctor`
Check whether the current machine is ready to run the CLI.

- Validates required API keys, required binaries, and writable storage paths.
- Use this before handing the tool to another bot or promoting a runtime to production.

Example:
- `reviewbuddy doctor`

## `reviewbuddy tap export [--output-dir PATH]`
Generate a Homebrew tap repository for publishing ReviewBuddy.

- Writes `Formula/`, `README.md`, a validation workflow, and a tap-maintainer skill.
- Defaults to the GitHub origin remote and writes to `../homebrew-reviewbuddy` when possible.

Example:
- `reviewbuddy tap export`
