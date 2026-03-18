# ReviewBuddy Release And Bot Install

## Production Shape

ReviewBuddy is currently production-ready as a packaged CLI.

Important constraint:
- It depends on the external `codex` CLI for agent execution.
- A runtime is not ready unless `scripts/reviewbuddy doctor` passes.

## Bot Install From GitHub

If the repository is published on GitHub, install the skill from this path:

```text
skills/reviewbuddy-cli
```

Codex skill installer example:

```text
$skill-installer https://github.com/<owner>/<repo>/tree/main/skills/reviewbuddy-cli
```

After installation, restart the agent runtime if the installer requires it.

## CLI Install For Bots

From Git:

```bash
uv tool install git+https://github.com/<owner>/<repo>.git
reviewbuddy doctor
reviewbuddy commands --agent
```

From a built wheel:

```bash
uv tool install reviewbuddy-0.1.0-py3-none-any.whl
reviewbuddy doctor
```

## Release Procedure

1. Run `./scripts/release-check.sh`.
2. Commit and push the branch.
3. Create and push a tag like `v0.1.0`.
4. GitHub Actions builds `dist/` artifacts and attaches them to the release.
