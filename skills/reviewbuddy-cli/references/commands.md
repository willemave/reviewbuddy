# ReviewBuddy Commands

Primary entry points:
- `reviewbuddy`
- `scripts/reviewbuddy`

## `reviewbuddy run "<prompt>"`
- Executes a new one-shot research run.
- Writes artifacts under `data/storage/<run_id>/`.

## `reviewbuddy interactive`
- Starts the terminal UI.
- Lets the user resume a previous run or start a new one.
- Supports `[a]sk` for local follow-up Q&A and `[f]ollowup` for deeper research.

## `reviewbuddy ask <run_id> "<question>"`
- Answers a follow-up question from persisted run memory.
- Does not re-crawl or re-download sources.

## `reviewbuddy resume <run_id>`
- Opens a saved run and displays its synthesis.
- Supports more local questions or additional research.

## `reviewbuddy commands [--agent]`
- Prints a compact command reference.
- Use `--agent` for the flatter machine-readable variant.

## `reviewbuddy doctor`
- Validates required API keys, binaries, and writable storage paths.
- Run this before handing the tool to another bot.
