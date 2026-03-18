# ReviewBuddy Troubleshooting

Use this file when `reviewbuddy doctor` fails or a target machine is missing runtime dependencies.

## `ffmpeg` missing

Symptom:
- `reviewbuddy doctor` reports `ffmpeg not found in PATH`.

Fix:
- macOS with Homebrew: `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt-get update && sudo apt-get install -y ffmpeg`
- Fedora: `sudo dnf install -y ffmpeg`
- Arch: `sudo pacman -S ffmpeg`
- Windows with winget: `winget install Gyan.FFmpeg`

Verify:
- Run `ffmpeg -version`
- Then rerun `scripts/reviewbuddy doctor`

If package-manager access is blocked:
- Report the exact install command needed for the host OS.
- Do not treat the runtime as ready until `doctor` passes.

## `codex` missing or not authenticated

Symptom:
- `reviewbuddy doctor` fails the `codex` check.
- Runtime commands fail when `codex exec` cannot start.

Fix:
- Install the `codex` CLI for that machine.
- Authenticate it with `codex login`.
- Rerun `scripts/reviewbuddy doctor`.

## Search provider key missing

Symptom:
- `reviewbuddy doctor` fails the selected search provider check.

Fix:
- Set the API key for the configured provider:
  - `EXA_API_KEY`
  - `TAVILY_API_KEY`
  - `FIRECRAWL_API_KEY`
- Rerun `scripts/reviewbuddy doctor`.
