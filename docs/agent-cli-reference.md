# ReviewBuddy CLI For Agents

Entrypoints:
- `reviewbuddy`
- `scripts/reviewbuddy`

## run
Usage: `reviewbuddy run "<prompt>"`
Purpose: Execute a new one-shot research run and print the synthesis.
Behavior:
- Runs planning, search, crawl, synthesis, and writes artifacts under `data/storage/<run_id>/`.
- Best when you want a final answer in a single command.
Example:
- `reviewbuddy run "best dishwasher for quiet apartment"`

## ask
Usage: `reviewbuddy ask <run_id> "<question>"`
Purpose: Answer a follow-up question from a saved run without re-crawling.
Behavior:
- Loads persisted follow-up memory from the saved session.
- Useful for previous-session Q&A in scripts or agent workflows.
Example:
- `reviewbuddy ask abc123 "What were the main warranty concerns?"`

## commands
Usage: `reviewbuddy commands [--agent]`
Purpose: Print a compact command reference.
Behavior:
- Use `--agent` for a flatter, machine-friendly reference format.
- Points to the markdown reference files under `docs/`.
Example:
- `reviewbuddy commands --agent`

## doctor
Usage: `reviewbuddy doctor`
Purpose: Check whether the current machine is ready to run the CLI.
Behavior:
- Validates required API keys, required binaries, and writable storage paths.
- Use this before handing the tool to another bot or promoting a runtime to production.
Example:
- `reviewbuddy doctor`
