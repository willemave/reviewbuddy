# Review Buddy

Review Buddy is an AI-powered review research assistant that searches the web, crawls pages locally,
converts them to markdown, and synthesizes a tight recommendation.

## Quick start

```bash
uv sync
playwright install
reviewbuddy run "best dishwasher for quiet apartment"
```

Outputs are stored under `data/storage/<run_id>/` alongside a SQLite metadata database.
