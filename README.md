# ReviewBuddy

ReviewBuddy is an AI-powered research assistant that turns a question or product query into a concise recommendation. It plans research lanes, searches the web, crawls pages locally, summarizes sources, and produces a final report.

## What it does
- Plans 4–8 research lanes per query
- Searches and crawls sources (forums, blogs, discussions, reviews)
- Adds YouTube transcripts (local Whisper) and Reddit summaries when available
- Synthesizes a short, cited recommendation

## Requirements
- Python 3.13 (recommended)
- `uv` for dependency management
- Playwright browsers

## Setup
```bash
uv sync
uv run playwright install
cp .env.example .env   # add API keys
```

Simple local entrypoint:
```bash
scripts/reviewbuddy --help
```

Required env vars:
- `OPENAI_API_KEY`
- `SEARCH_PROVIDER` (`exa`, `tavily`, or `firecrawl`)
- Provider API key for the selected search backend:
  - `EXA_API_KEY`
  - `TAVILY_API_KEY`
  - `FIRECRAWL_API_KEY`

Optional:
- `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`
- Provider tuning:
  - `SEARCH_NUM_RESULTS`, `SEARCH_MIN_RESULTS_PER_QUERY`
  - `EXA_SEARCH_TYPE`, `EXA_USER_LOCATION`
  - `TAVILY_SEARCH_DEPTH`, `TAVILY_TOPIC`, `TAVILY_AUTO_PARAMETERS`
  - `FIRECRAWL_COUNTRY`, `FIRECRAWL_LOCATION`

## Run (one-shot)
```bash
scripts/reviewbuddy run "best dishwasher for quiet apartment"
```

## Ask a previous session a question
```bash
scripts/reviewbuddy ask <run_id> "What were the main reliability complaints?"
```
The `run` command prints the `run_id`, which you can reuse with `ask`.

## Command reference
```bash
scripts/reviewbuddy commands
scripts/reviewbuddy commands --agent
scripts/reviewbuddy doctor
```

Reference files:
- `docs/cli-reference.md`
- `docs/agent-cli-reference.md`
- `docs/deploy.md`
- `docs/homebrew.md`

## Install the ReviewBuddy skill

Skill path in this repository:
```text
skills/reviewbuddy-cli
```

Codex skill installer example:
```text
$skill-installer https://github.com/<owner>/<repo>/tree/main/skills/reviewbuddy-cli
```

Bundled skill references:
- `skills/reviewbuddy-cli/SKILL.md`
- `skills/reviewbuddy-cli/references/commands.md`
- `skills/reviewbuddy-cli/references/troubleshooting.md`

## Hermes / OpenClaw quick install

Paste this into Hermes or another OpenClaw-style agent:

```text
Install ReviewBuddy with Homebrew:

brew tap willemave/reviewbuddy
brew install reviewbuddy
reviewbuddy commands --agent
reviewbuddy doctor

If you support SKILL.md directories, load this bundled skill:
$(brew --prefix)/opt/reviewbuddy/share/reviewbuddy/skills/reviewbuddy-cli

Read:
- $(brew --prefix)/opt/reviewbuddy/share/reviewbuddy/skills/reviewbuddy-cli/SKILL.md

Do not start research runs until `reviewbuddy doctor` passes.
```

## Agent install guidance

If you are an agent or automation installing this repository, use this order:

1. Install the CLI or clone the repository.
2. Install the skill from `skills/reviewbuddy-cli`.
3. Run `scripts/reviewbuddy doctor`.
4. Run `scripts/reviewbuddy commands --agent`.
5. Only start research runs after `doctor` passes.

Recommended GitHub skill install target:
```text
https://github.com/<owner>/<repo>/tree/main/skills/reviewbuddy-cli
```

Recommended CLI preflight:
```bash
uv sync
uv run playwright install
scripts/reviewbuddy doctor
scripts/reviewbuddy commands --agent
```

Agent operating rules:
- Prefer `scripts/reviewbuddy` over raw `uv tool run --from . reviewbuddy`.
- The local wrapper delegates to `uv tool run --from . reviewbuddy`.
- Use `scripts/reviewbuddy ask <run_id> "<question>"` for previous-session follow-up questions without re-crawling.
- Treat a failing `doctor` command as a hard stop for production use.
- This repository currently ships as a packaged CLI, not a hosted web service.

## Homebrew tap export

Generate a sibling tap repository:

```bash
reviewbuddy tap export
```

This writes a `homebrew-reviewbuddy` repository next to the source repo with:
- `Formula/reviewbuddy.rb`
- a tap README
- a GitHub Actions validation workflow
- a tap-maintainer skill under `skills/reviewbuddy-tap-maintainer`

## Output
Each run writes files under:
```
data/storage/<run_id>/
```
Key files:
- `synthesis.md` — final report
- `run.log` — detailed logs
- `html/` and `markdown/` — crawled sources

## Notes
- Run `codex login` locally before using the LLM-backed workflows.
- `codex` must be installed and authenticated for agent execution paths.
- YouTube transcripts are produced locally via Whisper.
- Reddit uses the API when credentials are provided.
- PDF summaries are generated through `codex exec` after local text extraction.

## Release and production
For the current repo shape, production means shipping the CLI cleanly.

Local preflight:
```bash
./scripts/release-check.sh
```

Bot/runtime preflight:
```bash
scripts/reviewbuddy doctor
```

Release guide:
- `docs/deploy.md`
