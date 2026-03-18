# ReviewBuddy Commands

Primary entry points:
- `reviewbuddy`
- `scripts/reviewbuddy`

## `reviewbuddy run "<prompt>"`
- Executes a new one-shot research run.
- Writes artifacts under `data/storage/<run_id>/`.

## `reviewbuddy ask <run_id> "<question>"`
- Answers a follow-up question from persisted run memory.
- Does not re-crawl or re-download sources.

## `reviewbuddy commands [--agent]`
- Prints a compact command reference.
- Use `--agent` for the flatter machine-readable variant.

## `reviewbuddy doctor`
- Validates required API keys, binaries, and writable storage paths.
- If it fails on a dependency such as `ffmpeg`, read `references/troubleshooting.md`.
- Run this before handing the tool to another bot.
