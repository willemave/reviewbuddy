# Repository Map

## Scope

This file summarizes the repository as it exists on disk on April 9, 2026.

Notes:
- The actual product in this repo is `ResearchBuddy`, a local CLI research assistant.
- `AGENTS.md` and `CLAUDE.md` are stale and describe a different app (`Writing Buddy` / FastAPI).
- `docs/architecture.md` is referenced by repo instructions but does not exist.
- This document lists source-controlled files individually and groups large generated/runtime trees where repeating every artifact would add no value.

## Root

- `.benchmarks/`: empty local benchmark scratch directory.
- `.env`: local machine configuration and secrets; runtime-only, not something the rewrite should depend on structurally.
- `.env.example`: sample environment template for search providers, semantic dedupe, and Reddit credentials.
- `.github/`: GitHub Actions configuration.
- `.github/workflows/release.yml`: tagged-release workflow; runs `scripts/release-check.sh` and uploads `dist/*`.
- `.gitignore`: ignore rules for Python, build, and runtime artifacts.
- `.pytest_cache/`: pytest cache; generated.
- `.ruff_cache/`: Ruff cache; generated.
- `AGENTS.md`: stale repo instructions copied from another project; does not match the current tree.
- `CLAUDE.md`: same stale project instructions as `AGENTS.md`.
- `README.md`: the real product overview; explains the research pipeline, CLI, configuration, output bundle shape, and installation paths.
- `pyproject.toml`: Python package metadata, dependencies, Ruff/Pytest config, and CLI entrypoint.
- `uv.lock`: locked Python dependency graph for `uv`.

## Application Code

### `app/`

- `app/__init__.py`: package marker.
- `app/cli.py`: main Typer CLI; wires commands for `run`, `followup`, `inspect`, `transcribe`, `setup`, `doctor`, `commands`, and tap export.
- `app/cli_doctor.py`: environment validation checks for binaries, search provider config, storage, and local coding-agent readiness.
- `app/cli_help.py`: structured command metadata and richer help/error guidance.
- `app/constants.py`: global constants for app version, storage layout, filenames, and run/url statuses.

### `app/agents/`

- `app/agents/__init__.py`: package marker.
- `app/agents/base.py`: shared Pydantic schemas for agent inputs/outputs such as lane plans, search queries, and synthesis payloads.
- `app/agents/lane_planner.py`: calls the local coding-agent harness to decompose a prompt into parallel research lanes, then semantically dedupes/merges them.
- `app/agents/lane_refiner.py`: asks the local coding-agent harness for follow-up search queries based on lane evidence, then dedupes them.
- `app/agents/rlm_agents.py`: thin Codex-backed adapters for the RLM-style REPL loop.
- `app/agents/synthesizer.py`: builds prompts and runs lane, merge-node, and final synthesis steps.

### `app/core/`

- `app/core/__init__.py`: package marker.
- `app/core/logging.py`: root logging setup plus per-run file handler attachment.
- `app/core/settings.py`: Pydantic settings model and config loading; merges `.env`, environment variables, and local agent config from Hermes/OpenClaw.

### `app/models/`

- `app/models/__init__.py`: package marker.
- `app/models/homebrew.py`: validated request/result models for generating a Homebrew tap repository.
- `app/models/review.py`: core workflow data contracts for runs, search responses, URL tasks, persistence records, and follow-up memory.
- `app/models/rlm.py`: models for context documents and RLM-style interactive runs/refinement loops.
- `app/models/skills.py`: validated request/result models for installing bundled skills into agent runtimes.

### `app/services/`

- `app/services/__init__.py`: package marker.
- `app/services/chatgpt.py`: builds a ChatGPT continuation URL from a saved synthesis.
- `app/services/codex_exec.py`: subprocess bridge to local coding-agent CLIs (`codex`, `claude`, `amp`) with optional structured output validation.
- `app/services/exa_client.py`: direct Exa API client and response parsing.
- `app/services/followup.py`: follow-up memory reconstruction/loading plus local Q&A from saved evidence.
- `app/services/homebrew_tap.py`: renders a Homebrew tap repo, formula, workflow, and bundled maintenance skill.
- `app/services/local_audio_transcriber.py`: shared Whisper model/audio loading and local file transcription helpers.
- `app/services/markdown_converter.py`: converts captured HTML into markdown via Crawl4AI with BM25/pruning filters.
- `app/services/playwright_fetcher.py`: Playwright navigation/content capture and heuristics for headful retry.
- `app/services/podcast_transcriber.py`: podcast URL detection, RSS/audio resolution, download, and Whisper transcription.
- `app/services/query_shaper.py`: appends/removes search suffix clauses to bias toward forums/blogs/discussions without duplicating existing terms.
- `app/services/reporter.py`: optional callback container for run progress reporting.
- `app/services/research_profiles.py`: prompt intent detection and mode-specific research guidance for product, restaurant, and general research.
- `app/services/rlm_engine.py`: executes the REPL-like “RLM” loop, including subquery calls and `FINAL(...)` extraction.
- `app/services/search_provider.py`: provider abstraction plus concrete Exa/Tavily/Firecrawl search adapters and normalized result parsing.
- `app/services/semantic_dedupe.py`: local embeddings, clustering, semantic dedupe, and MMR ranking.
- `app/services/setup_runtime.py`: local setup workflow for search config validation, storage, database, and Playwright browser installation.
- `app/services/skill_installer.py`: safe symlink/copy installation of bundled ResearchBuddy skills into OpenClaw skill directories.
- `app/services/storage.py`: SQLite schema/init plus run/url persistence and artifact path helpers.
- `app/services/token_estimator.py`: token counting/budgeting based on `tiktoken`.
- `app/services/transcript_summarizer.py`: compresses podcast/YouTube transcripts through the coding-agent harness.
- `app/services/url_handlers.py`: custom ingestion paths for Reddit, YouTube, and PDFs before generic browser crawling.
- `app/services/usage_tracker.py`: aggregates model/search/source usage for a run.
- `app/services/youtube_transcriber.py`: YouTube search, metadata lookup, caption extraction, audio download, and Whisper fallback.

### `app/workflows/`

- `app/workflows/__init__.py`: package marker.
- `app/workflows/review.py`: the main orchestration pipeline; plans lanes, searches providers, crawls pages, distills evidence, transcribes media, synthesizes summaries, persists artifacts, and records usage.

## Tests

### `app/tests/`

- `app/tests/__init__.py`: test package marker.
- `app/tests/test_chatgpt.py`: tests ChatGPT continuation URL generation and clipping.
- `app/tests/test_cli_doctor.py`: tests doctor report formatting and environment checks.
- `app/tests/test_cli_help.py`: tests CLI help/reference rendering and top-level help text.
- `app/tests/test_codex_agents.py`: tests planner/refiner/synthesizer integration with the Codex runner abstraction.
- `app/tests/test_codex_exec.py`: tests subprocess command construction, parsing, and error handling for local agent execution.
- `app/tests/test_dev_wrapper.py`: tests the shell wrapper uses `uv run`.
- `app/tests/test_exa_client.py`: tests Exa request/response parsing.
- `app/tests/test_followup.py`: tests follow-up prompt building, ranking, and memory reconstruction behavior.
- `app/tests/test_homebrew_tap.py`: tests Homebrew tap rendering and export paths.
- `app/tests/test_lane_budgeting.py`: tests lane and search query budget allocation.
- `app/tests/test_lane_prompts.py`: tests prompt constants and research profile inference.
- `app/tests/test_markdown_converter.py`: tests Crawl4AI markdown selection/filter construction.
- `app/tests/test_playwright_fetcher.py`: tests headful retry heuristics.
- `app/tests/test_podcast_transcriber.py`: tests podcast URL detection, episode selection, audio resolution, and transcription helpers.
- `app/tests/test_query_shaper.py`: tests query suffix shaping and dedupe behavior.
- `app/tests/test_refinement_targets.py`: tests refinement target scheduling.
- `app/tests/test_research_compression.py`: tests evidence ranking, distillation, and merge-input packing.
- `app/tests/test_review_provider_ingestion.py`: tests the workflow prefers provider-supplied content before Playwright crawling.
- `app/tests/test_review_workflow.py`: tests synthesis formatting, search result request sizing, and usage logging.
- `app/tests/test_rlm_engine.py`: tests the RLM interactive loop and subquery execution.
- `app/tests/test_search_provider.py`: tests provider selection plus Exa/Tavily/Firecrawl normalization.
- `app/tests/test_semantic_dedupe.py`: tests clustering, dedupe, and MMR ranking semantics.
- `app/tests/test_settings.py`: tests loading provider config from Hermes/OpenClaw files.
- `app/tests/test_setup_runtime.py`: tests `.env` persistence and workspace setup behavior.
- `app/tests/test_skill_installer.py`: tests OpenClaw shared/workspace skill install behavior.
- `app/tests/test_storage.py`: tests SQLite run persistence and run-dir resolution.
- `app/tests/test_transcript_summarizer.py`: tests transcript summarization fallback behavior.
- `app/tests/test_url_handlers.py`: tests Reddit/YouTube/PDF URL detection and markdown formatting helpers.
- `app/tests/test_usage_tracker.py`: tests usage aggregation across model and search calls.
- `app/tests/test_youtube_transcriber.py`: tests YouTube selection, caption parsing, and transcription edge cases.

## Documentation

### `docs/`

- `docs/agent-cli-reference.md`: flat command reference tuned for agents/scripts.
- `docs/cli-reference.md`: human-oriented CLI reference.
- `docs/deploy.md`: packaging/deploy notes for the packaged CLI production shape.
- `docs/homebrew.md`: Homebrew publishing workflow and install instructions.

## Scripts

### `scripts/`

- `scripts/release-check.sh`: release gate that runs `uv sync`, `ruff check`, `pytest`, and `uv build`.
- `scripts/researchbuddy`: local wrapper that executes the installed CLI through `uv run --project`.

## Skills

### `skills/`

- `skills/research/`: bundled agent skill for installing and operating the CLI.
- `skills/research/SKILL.md`: main usage instructions for agents.
- `skills/research/references/commands.md`: compact command semantics.
- `skills/research/references/troubleshooting.md`: recovery steps for missing dependencies or credentials.

## Runtime / Generated Trees

These are present in the working tree but are generated state, not primary source.

- `app/**/__pycache__/`: compiled Python bytecode caches.
- `data/`: local runtime persistence.
- `~/.researchbuddy/researchbuddy.db`: default SQLite database for saved runs.
- `data/reviewbuddy.db`: stale/legacy database name that suggests an earlier rename from `reviewbuddy` to `researchbuddy`.
- `~/.researchbuddy/storage/<run_id>/`: default saved research run bundles; each directory contains run logs, synthesis, lane artifacts, captured HTML/markdown, and transcript files.
- `dist/`: built package artifacts.
- `dist/reviewbuddy-0.1.0-py3-none-any.whl`: stale wheel artifact under the old app name.
- `dist/reviewbuddy-0.1.0.tar.gz`: stale source distribution under the old app name.
- `tmp/`: scratch workspace.
- `tmp/yt-test/`: temporary YouTube-related scratch directory.

## Structural Findings Relevant To A Rust Rewrite

- The real product boundary is a local CLI plus library code, not a web server.
- The workflow is already library-shaped: `settings -> providers -> workflow -> storage -> CLI`.
- The cleanest Rust port target is the current behavior, not the stale FastAPI/“Writing Buddy” instructions.
- There is naming drift (`reviewbuddy` vs `researchbuddy`) in docs, databases, and dist artifacts; the rewrite should normalize this early.
