# ReviewBuddy Deploy And Release

## Current production shape

This repository is production-ready as a packaged CLI.

It is not yet a generic hosted web service:
- There is no FastAPI or frontend entrypoint in the current tree.
- Core agent execution depends on the external `codex` CLI in [app/services/codex_exec.py](/Users/willem/Development/reviewbuddy/app/services/codex_exec.py).
- Any runtime that executes ReviewBuddy must have `codex` installed and authenticated.

For bots, the shortest path is:
1. Push the repo to GitHub.
2. Let the bot install the CLI from Git or from a release artifact.
3. Let the bot install the skill from `skills/reviewbuddy-cli` if it supports SKILL.md skills.
4. Run `reviewbuddy doctor` inside the bot environment.

For Hermes/OpenClaw-style agents, the easiest install prompt is:

```text
Install ReviewBuddy with Homebrew:

brew tap willemave/reviewbuddy
brew install reviewbuddy
reviewbuddy commands --agent
reviewbuddy doctor

If you support SKILL.md directories, load:
$(brew --prefix)/opt/reviewbuddy/share/reviewbuddy/skills/reviewbuddy-cli

Do not start research runs until `reviewbuddy doctor` passes.
```

## Skill repository format

Based on the public Agent Skills examples and documentation, the safest GitHub layout for installable skills is:

```text
skills/
  <skill-name>/
    SKILL.md
    references/
    scripts/
    assets/
```

Only `SKILL.md` is required. `references/`, `scripts/`, and `assets/` are optional.

This repository now exposes the skill at:

```text
skills/reviewbuddy-cli
```

Example Codex install URL:

```text
$skill-installer https://github.com/<owner>/<repo>/tree/main/skills/reviewbuddy-cli
```

## Bot install paths

From Git:
```bash
uv tool install git+https://github.com/<owner>/<repo>.git
reviewbuddy commands --agent
reviewbuddy doctor
```

From a release wheel:
```bash
uv tool install reviewbuddy-0.1.0-py3-none-any.whl
reviewbuddy doctor
```

## Local pre-release check

Run:
```bash
./scripts/release-check.sh
```

This performs:
- dependency sync
- `ruff check`
- `pytest`
- `uv build`

## GitHub CI

The repository now includes:
- `.github/workflows/ci.yml`
- `.github/workflows/release.yml`

Behavior:
- CI runs on pushes and pull requests.
- Release runs when you push a tag like `v0.1.0`.
- The release workflow builds `dist/` artifacts and attaches them to a GitHub Release.

## Production push flow

Push to GitHub:
```bash
git add .
git commit -m "Prepare production release"
git push origin <branch>
```

Create a tagged release:
```bash
git tag v0.1.0
git push origin v0.1.0
```

After the tag push:
- GitHub Actions runs release checks.
- A GitHub Release is created.
- Built wheel and sdist files are attached under `dist/`.

## Runtime requirements for the bot

Minimum:
- Python 3.13
- `uv`
- `codex` installed and logged in
- Playwright browsers installed
- required API keys in environment

Recommended setup sequence:
```bash
uv sync
uv run playwright install
reviewbuddy doctor
reviewbuddy commands --agent
```

## Important limitation

If you want a normal hosted production deployment on Railway, Render, Fly, or Kubernetes, the next step is not release automation. The next step is to replace or wrap `codex exec` with a first-party API integration and then add a real server entrypoint.
