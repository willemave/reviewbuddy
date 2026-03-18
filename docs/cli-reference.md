# ReviewBuddy CLI Reference

Primary entry points:
- Installed command: `reviewbuddy`
- Local wrapper: `scripts/reviewbuddy`

## `reviewbuddy run "<prompt>"`
Execute a new one-shot research run and print the synthesis.

- Runs planning, search, crawl, synthesis, and writes artifacts under `data/storage/<run_id>/`.
- Best when you want a final answer without staying in an interactive session.

Example:
- `reviewbuddy run "best dishwasher for quiet apartment"`

## `reviewbuddy interactive`
Start an interactive session that can create new runs or resume saved ones.

- Shows recent runs on startup.
- Supports local Q&A with `[a]sk` and deeper research with `[f]ollowup`.

Example:
- `reviewbuddy interactive --max-urls 50`

## `reviewbuddy ask <run_id> "<question>"`
Answer a follow-up question from a saved run without re-crawling.

- Loads persisted follow-up memory from the saved session.
- Useful for previous-session Q&A in scripts or agent workflows.

Example:
- `reviewbuddy ask abc123 "What were the main warranty concerns?"`

## `reviewbuddy resume <run_id>`
Open a saved run, show its synthesis, and continue with questions or more research.

- Use `[a]sk` to query saved material locally.
- Use `[f]ollowup` to launch additional research based on the existing report.

Example:
- `reviewbuddy resume abc123`

## `reviewbuddy commands [--agent]`
Print a compact command reference.

- Use `--agent` for a flatter, machine-friendly reference format.
- Points to the markdown reference files under `docs/`.

Example:
- `reviewbuddy commands --agent`

## `reviewbuddy doctor`
Check whether the current machine is ready to run the CLI.

- Validates required API keys, required binaries, and writable storage paths.
- Use this before handing the tool to another bot or promoting a runtime to production.

Example:
- `reviewbuddy doctor`
