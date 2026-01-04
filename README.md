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

Required env vars:
- `OPENAI_API_KEY`
- `EXA_API_KEY`

Optional:
- `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`
- `GOOGLE_API_KEY` (for PDF summaries)

## Run (one-shot)
```bash
uv run reviewbuddy run "best dishwasher for quiet apartment"
```

## Interactive mode
```bash
uv run reviewbuddy interactive
```
Interactive mode shows progress, lets you view logs, and ask follow-up questions. On startup it lists recent runs so you can resume an old session or start a new one.

## Resume a run
```bash
uv run reviewbuddy resume <run_id>
```

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
- YouTube transcripts are produced locally via Whisper.
- Reddit uses the API when credentials are provided.
- PDF summaries are generated via Gemini when `GOOGLE_API_KEY` is set.
