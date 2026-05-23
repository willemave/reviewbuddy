# Rust Rewrite Plan

## Goal

Rewrite `ResearchBuddy` as an idiomatic Rust CLI that preserves the current behavior:
- plan research lanes
- search multiple providers
- crawl and normalize source material
- transcribe YouTube/podcast/local audio
- synthesize cited answers
- persist run artifacts for follow-up questions

The target is not a line-by-line port. The target is a Rust-native architecture with the same product surface.

## What Exists Today

The current Python app is a local CLI with one main workflow:

`prompt -> plan lanes -> search -> crawl/transcribe -> distill evidence -> synthesize -> persist run bundle`

Core facts from the repo:
- The real entrypoint is `app/cli.py`.
- The real orchestration lives in `app/workflows/review.py`.
- Storage is local SQLite plus filesystem artifacts under `~/.researchbuddy/storage/<run_id>/` by default.
- “Agents” are not a framework; they are prompt wrappers around a local coding-agent subprocess (`codex`, `claude`, `amp`).
- The repo docs contain stale references to another project. Do not port those assumptions.

## Rust Design Principles

1. Prefer `serde` types plus explicit validation over Pydantic-style magic.
2. Keep the app library-first: put business logic in `src/lib.rs` modules and keep `src/bin/researchbuddy.rs` thin.
3. Use async only where it buys concurrency:
   - network I/O
   - crawling
   - provider requests
   - subprocess coordination
4. Keep boundaries explicit with traits:
   - `SearchProvider`
   - `BrowserFetcher`
   - `Transcriber`
   - `LlmRunner`
5. Keep state on disk simple:
   - SQLite for run/url metadata
   - plain files for captured artifacts
6. Do not force a 1:1 port of every Python dependency. Some should disappear into stdlib or cleaner Rust composition.

## Recommended Target Layout

Use one Cargo package with a library and one binary.

```text
src/
  lib.rs
  bin/
    researchbuddy.rs
  cli/
    mod.rs
    commands.rs
    output.rs
  config.rs
  constants.rs
  error.rs
  domain/
    mod.rs
    research.rs
    run.rs
    transcript.rs
    usage.rs
  llm/
    mod.rs
    codex_exec.rs
    prompts.rs
  providers/
    mod.rs
    search/
      mod.rs
      exa.rs
      tavily.rs
      firecrawl.rs
    media/
      mod.rs
      youtube.rs
      podcast.rs
      reddit.rs
      pdf.rs
    browser/
      mod.rs
      playwright.rs
      fallback.rs
  pipeline/
    mod.rs
    planning.rs
    refinement.rs
    synthesis.rs
    review.rs
    followup.rs
  storage/
    mod.rs
    sqlite.rs
    files.rs
  text/
    mod.rs
    markdown.rs
    tokenize.rs
    dedupe.rs
  telemetry/
    mod.rs
    logging.rs
    usage.rs
tests/
  integration/
```

Why this shape:
- one crate keeps compile and packaging simple
- modules separate workflow from adapters
- the CLI becomes an adapter instead of the application core

## Runtime Stack

- Async runtime: `tokio`
- HTTP client: `reqwest` with `rustls`
- Serialization: `serde`, `serde_json`
- Validation: `garde`
- Config loading: `figment` + `dotenvy`
- Error handling: `thiserror` for library errors, `anyhow` at command boundaries
- Logging/tracing: `tracing`, `tracing-subscriber`
- CLI parsing: `clap`
- Terminal UX: `owo-colors`, `indicatif`
- SQLite: `sqlx` with SQLite feature
- Testing: `cargo test`, `insta`, `assert_cmd`, `predicates`, `wiremock`, `tempfile`

## Dependency Translation

This is the dependency-level rewrite recommendation. The choices below are the ones I would actually build with.

| Python dependency | Current role | Rust choice | Recommendation |
|---|---|---|---|
| `pydantic` | typed models and validation | `serde` + `garde` | Use `serde` for shape, `garde` for boundary validation. More idiomatic than chasing a Pydantic clone. |
| `pydantic-settings` | env/.env config | `figment` + `dotenvy` | `figment` handles layered config cleanly; `dotenvy` covers local `.env`. |
| `python-dotenv` | load `.env` | `dotenvy` | Keep this tiny and explicit. |
| `aiosqlite` | async SQLite | `sqlx` (`sqlite`) | Best fit if the app stays async end-to-end. |
| `httpx` | async HTTP | `reqwest` | Standard Rust choice. |
| `transformers` | local embeddings for semantic dedupe | `fastembed` | Best direct replacement for local embeddings; supports Qwen3 embeddings. |
| `openai-whisper` | local transcription | `whisper-rs` | Mature Rust bindings to `whisper.cpp`; good fit for local-only transcription. |
| `praw` | Reddit API wrapper | `roux` or raw `reqwest` | Closest wrapper is `roux`; for read-only fetches, raw `reqwest` may be cleaner. |
| `pypdf` | PDF text extraction | `unpdf` | Good structured extraction target; keep `pdf-text-extract` as fallback if markdown-heavy extraction wins in testing. |
| `tiktoken` | token counting | `tiktoken-rs` | Strong OpenAI-tokenizer parity and broad existing use. |
| `typer` | CLI | `clap` | Default Rust answer. |
| `rich` | terminal formatting | `owo-colors` + `indicatif` | Enough for this CLI; do not add a full TUI unless the product changes. |
| `playwright` | browser rendering/capture | `playwright-rs` behind a trait | Viable, but isolate it because the crate is still pre-1.0. |
| `crawl4ai` | HTML cleanup + markdown extraction | compose `dom_smoothie` + `html-to-markdown-rs` + `scraper`/`lol_html` | There is no single Rust equivalent; compose small crates instead. |
| `yt-dlp` | YouTube search/download/captions | `rusty_ytdl` | Prefer this over `rustube` based on more recent maintenance signals. |
| `tokonomics` | optional token/cost utility | remove | Implement pricing/token usage locally; no need for a direct equivalent. |
| `pytest` | tests | `cargo test` + `assert_cmd` + `wiremock` + `insta` | Match test type to problem, not framework familiarity. |
| `ruff` | lint/format | `cargo fmt` + `clippy` | Standard Rust toolchain. |

## Dependency Notes That Matter

### 1. Do not use a Rust “Pydantic clone” as the foundation

I found Rust crates inspired by Pydantic, but that is not the idiomatic target here.

Use:
- `serde` for serialization/deserialization
- `garde` for validation
- explicit constructors/validation at boundaries

That matches normal Rust design better than importing a Python mental model.

### 2. Prefer `sqlx` over `rusqlite` here

`rusqlite` is good, but this app is network-heavy and already concurrency-oriented. `sqlx` keeps the whole system on one async runtime and avoids blocking islands.

If the rewrite later proves SQLite access is tiny and synchronous, `rusqlite` is still a valid simplification. Start with `sqlx`.

### 3. `transformers` should become `fastembed`, not a generic transformer stack

The Python repo uses `transformers` only for local embeddings in semantic dedupe.

That means the Rust port should not pull in a general transformer runtime first. Use an embedding-focused crate:
- `fastembed` first
- optional `candle` only if you later need custom model execution outside the supported matrix

### 4. YouTube is the unstable edge

You specifically mentioned `rustube`. The current signals are:
- `rustube` latest crate release found: `0.6.0` on October 16, 2022
- `rusty_ytdl` latest crate release found: `0.7.4` on August 10, 2024
- `rusty_ytdl` repository activity surfaced as January 18, 2026

Recommendation:
- prefer `rusty_ytdl` over `rustube`
- keep a strict trait boundary around YouTube ingestion
- if parity becomes a problem, keep a temporary subprocess fallback for `yt-dlp` until Rust behavior is proven

### 5. Browser automation must be isolated

`playwright-rs` is promising, but its docs still describe it as pre-1.0 and “not yet ready for production use.”

Recommendation:
- keep browser capture behind `trait BrowserFetcher`
- start with a single implementation module
- ensure the rest of the pipeline works from saved HTML, so browser choice can change without touching synthesis/storage/search

### 6. `crawl4ai` should become composition, not a monolith

Rust is better here if you decompose the job:
- fetch/render HTML
- extract readable content
- normalize DOM if needed
- convert HTML to markdown

Suggested composition:
- `dom_smoothie` for Readability-style extraction
- `html-to-markdown-rs` for conversion
- `scraper` for CSS-selector queries
- `lol_html` only where streaming cleanup or rewrite passes are useful

## Proposed Domain Model Changes

Port the concepts, but simplify the type system.

### Keep

- `RunRecord`
- `UrlRecord`
- `ReviewRunRequest`
- `ReviewRunResult`
- `LaneSpec`
- `LaneSynthesis`
- `ReviewSynthesis`
- `FollowupMemory`
- `SearchResult`
- `SearchUsage`

### Change

- Replace Pydantic inheritance/validation patterns with plain Rust structs.
- Replace many “service” modules with smaller traits + implementations.
- Separate persisted models from in-memory workflow models.
- Make source kind a real enum, not a free-form string.
- Make run status and URL status enums, not string constants.

### Add

- `enum ResearchMode`
- `enum SourceKind`
- `enum SearchProviderName`
- `enum RunStatus`
- `enum UrlStatus`
- `struct RunPaths`
- `struct BrowserCapture`
- `struct DistilledSourceCard`

## Rewrite Phases

### Phase 0: Freeze Behavior

Before writing Rust:
- audit current CLI outputs
- capture example run bundles
- preserve golden outputs for:
  - lane planning
  - URL ranking
  - markdown distillation
  - follow-up answer prompt packing
- document current naming drift (`reviewbuddy` artifacts)

Output:
- fixture corpus under `tests/fixtures/`
- behavioral checklist

### Phase 1: Bootstrap Rust Skeleton

Build:
- Cargo package
- `clap` CLI with command stubs
- `tracing` logging
- config loading
- error types

Acceptance:
- `researchbuddy --help`
- `researchbuddy doctor`
- config can load `.env` and environment variables

### Phase 2: Port Storage And Filesystem Layout

Build:
- run path helpers
- SQLite schema
- run/url record persistence
- artifact layout writer

Acceptance:
- create/read run records
- write run bundles under `~/.researchbuddy/storage/<run_id>/` by default
- parity with current output filenames where worth keeping

### Phase 3: Port Search Layer

Build:
- provider trait
- Exa/Tavily/Firecrawl adapters
- query shaping
- normalized result types

Acceptance:
- integration tests with mocked APIs
- same provider auto-selection semantics
- usage tracking for search calls

### Phase 4: Port Non-LLM Text Utilities

Build:
- research mode inference
- token estimation
- URL-to-filename hashing
- ranking helpers
- basic distillation helpers

Acceptance:
- direct parity tests against current Python fixtures

### Phase 5: Port Local Embeddings And Dedupe

Build:
- local embedding wrapper over `fastembed`
- clustering
- MMR ranking

Acceptance:
- dedupe/ranking regression tests pass
- settings expose model/device thresholds cleanly

### Phase 6: Port LLM Runner Boundary

Build:
- subprocess adapter for `codex`/other local harnesses
- structured result parsing
- timeout/error handling

Acceptance:
- equivalent to current `codex_exec.py`
- planner/refiner/synthesizer prompts can execute end-to-end

### Phase 7: Port Media Ingestion

Build:
- YouTube detection/search/metadata/transcription path
- podcast resolution/transcription path
- local audio transcription path
- Reddit ingestion
- PDF extraction

Acceptance:
- fixture-backed tests for URL detection and parsing
- successful local transcription on test media
- saved transcript metadata matches current product needs

### Phase 8: Port Browser Capture And Markdown Pipeline

Build:
- browser fetcher trait
- initial `playwright-rs` adapter
- readability/HTML-to-markdown pipeline

Acceptance:
- fetch and convert saved HTML fixtures
- browserless markdown conversion works independently
- browser adapter can be swapped without touching workflow code

### Phase 9: Port Main Review Workflow

Build:
- lane planning
- search/crawl budget allocation
- crawl orchestration with bounded concurrency
- lane synthesis, merge synthesis, final synthesis
- usage logging and artifact writes

Acceptance:
- one real run completes locally
- follow-up artifacts are persisted
- usage snapshots are emitted

### Phase 10: Port Follow-Up And Inspect Commands

Build:
- follow-up memory rebuild/load
- follow-up answer generation
- inspect/runs/transcribe/setup/tap export command parity

Acceptance:
- CLI surface is feature-complete
- Python implementation is no longer required for core flows

### Phase 11: Cutover

Do this only after parity is good enough:
- ship Rust binary
- move Python app to `legacy/` or delete it
- update Homebrew/release flow
- rewrite stale docs

## Command-Level Rewrite Plan

### `run`

Keep:
- prompt input
- research mode selection
- optional stats output

Rust shape:
- thin CLI command -> `pipeline::review::run_review(request, deps)`

### `followup`

Keep:
- no re-crawl
- answer only from saved synthesis + source cards

Rust shape:
- load `FollowupMemory`
- rank cards
- build LLM prompt

### `inspect`

Keep:
- read-only local artifact inspection

Rust shape:
- pure filesystem + SQLite reads
- no provider/network dependency

### `transcribe`

Keep:
- local audio, YouTube, podcast support

Rust shape:
- `Transcriber` trait plus source-specific adapters

### `setup` / `doctor`

Keep:
- environment diagnosis and remediation

Rust shape:
- deterministic checks, pure output structs, simple terminal rendering

### `tap export`

Keep:
- repo/template generation

Rust shape:
- mostly stdlib + string/template rendering; no heavy dependencies needed

## Testing Plan

### Unit tests

Use for:
- query shaping
- research profile inference
- token estimation
- ranking/dedupe
- prompt builders

Crates:
- built-in `#[test]`
- `rstest` optionally if parameterization starts paying off

### Integration tests

Use for:
- CLI command behavior
- SQLite persistence
- provider adapters
- workflow happy paths with fixtures

Crates:
- `assert_cmd`
- `predicates`
- `tempfile`
- `wiremock`

### Snapshot tests

Use for:
- prompt templates
- markdown output
- synthesis input packing
- doctor/setup output

Crates:
- `insta`

### Golden artifact tests

Use for:
- replaying a captured Python run bundle through Rust readers
- ensuring follow-up ranking and packing remain stable

## Risks

### 1. Browser parity

This is the biggest technical risk. Browser rendering is the least mature part of the Rust ecosystem compared to Python/Node.

Mitigation:
- trait boundary
- saved HTML fixtures
- temporary subprocess fallback if needed

### 2. YouTube volatility

YouTube scraping/downloading breaks often.

Mitigation:
- prefer `rusty_ytdl`
- isolate implementation
- allow a fallback path during migration

### 3. Local ML portability

Embeddings and Whisper behavior vary by platform and accelerator.

Mitigation:
- make device selection explicit
- add CPU-only default path
- test on macOS first because current repo is clearly macOS-centric

### 4. Behavioral drift caused by stale docs

The repo contains instructions for another product.

Mitigation:
- treat code + README as source of truth
- ignore stale FastAPI assumptions

## What I Would Not Port 1:1

- Pydantic settings patterns
- service-layer sprawl where a trait + module is cleaner
- stringly-typed status values
- heavy dynamic validation at every layer
- `crawl4ai` as a monolith
- Python packaging/release assumptions

## First Concrete Milestone

The best first milestone is not “rewrite everything.”

It is:
1. Rust config loader
2. Rust SQLite storage
3. Rust search provider layer
4. Rust CLI `doctor` + `inspect`

Why:
- low-risk
- proves packaging/tooling
- creates stable infrastructure for the harder workflow port

## Recommended Source Links

These are the sources I used for the dependency choices:

- `config-rs`: https://crates.io/crates/config
- `figment`: https://docs.rs/figment/latest/figment/
- `dotenvy`: https://crates.io/crates/dotenvy
- `garde`: https://docs.rs/garde
- `validator`: https://crates.io/crates/validator
- `reqwest`: https://docs.rs/reqwest/latest/
- `sqlx`: https://docs.rs/sqlx/
- `rusqlite`: https://crates.io/crates/rusqlite
- `fastembed`: https://crates.io/crates/fastembed
- `candle-transformers`: https://crates.io/crates/candle-transformers
- `whisper-rs`: https://crates.io/crates/whisper-rs
- `roux`: https://crates.io/crates/roux
- `unpdf`: https://docs.rs/unpdf/latest/unpdf/
- `tiktoken-rs`: https://crates.io/crates/tiktoken-rs
- `clap`: https://crates.io/crates/clap
- `owo-colors`: https://crates.io/crates/owo-colors
- `indicatif`: https://crates.io/crates/indicatif
- `playwright-rs`: https://docs.rs/playwright-rs/latest/playwright_rs/
- `dom_smoothie`: https://docs.rs/dom_smoothie/latest/dom_smoothie/
- `scraper`: https://docs.rs/scraper
- `lol_html`: https://crates.io/crates/lol_html
- `html-to-markdown-rs`: https://crates.io/crates/html-to-markdown-rs
- `rustube`: https://crates.io/crates/rustube
- `rusty_ytdl`: https://crates.io/crates/rusty_ytdl
