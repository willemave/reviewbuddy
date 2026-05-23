<p align="center">
  <strong>ResearchBuddy</strong><br>
  <em>Parallel research lanes. Dense evidence. Cited answers.</em>
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> &middot;
  <a href="#how-it-works">How It Works</a> &middot;
  <a href="#cli-reference">CLI</a> &middot;
  <a href="#research-foundations">Research</a> &middot;
  <a href="#agent-install">Agent Install</a>
</p>

---

ResearchBuddy turns a messy research question into a short, cited answer. Instead of one giant prompt, it plans parallel research lanes, crawls the web locally, distills each source into dense evidence cards, and merges lane summaries into one final answer.

```
prompt ── plan 4-8 lanes ── crawl in parallel ── compress evidence ── synthesize with citations
```

It adapts automatically to three research modes:

| Mode | Biases toward | Example |
|------|---------------|---------|
| **Product reviews** | Blog posts, Reddit, YouTube reviews, owner discussions | *"best dishwasher for quiet apartment"* |
| **Restaurant picks** | Local magazines, neighborhood guides, Reddit, forums | *"best sushi restaurants in portland"* |
| **General research** | Podcasts, YouTube, interviews, broader web analysis | *"history and current debate around nuclear fusion"* |

---

## Quickstart

**Requirements:** Python 3.11+, [uv](https://docs.astral.sh/uv/), one search provider key (Exa, Tavily, or Firecrawl).

```bash
# Install
brew tap willemave/researchbuddy && brew install researchbuddy

# Or from source
git clone https://github.com/willemave/researchbuddy && cd researchbuddy
uv sync && uv run playwright install chromium

# Validate
researchbuddy doctor         # validate everything is ready
researchbuddy doctor --fix   # repair local setup gaps

# Start, monitor, and follow up
researchbuddy start "best dishwasher for quiet apartment"
researchbuddy status
researchbuddy watch --timeout 900
researchbuddy followup "What were the strongest recommendations?"
```

Each run writes artifacts to `~/.researchbuddy/storage/<run_id>/` by default, including `synthesis.md` (the final report), per-lane snapshots, captured source material, and follow-up memory. Set `RESEARCHBUDDY_HOME` or pass `--output-dir` when you need an explicit alternate location.

---

## How It Works

ResearchBuddy uses a **hierarchical task decomposition** architecture. The pipeline breaks a single question into independent lanes, crawls and compresses evidence in parallel, and synthesizes upward through a merge tree.

```
                        ┌─────────────────────────────────────┐
                        │           User Prompt               │
                        └──────────────┬──────────────────────┘
                                       │
                        ┌──────────────▼──────────────────────┐
                        │         Lane Planner                │
                        │   breaks prompt into 4-8 lanes      │
                        │   with seed queries per lane        │
                        └──────────────┬──────────────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
     ┌────────▼────────┐     ┌────────▼────────┐     ┌────────▼────────┐
     │   Lane: Owner   │     │  Lane: Expert   │     │ Lane: Alterna-  │
     │   Feedback      │     │  Analysis       │     │ tives           │
     │                 │     │                 │     │                 │
     │ search → crawl  │     │ search → crawl  │     │ search → crawl  │
     │ → refine query  │     │ → refine query  │     │ → refine query  │
     │ → crawl again   │     │ → crawl again   │     │ → crawl again   │
     │ → evidence cards│     │ → evidence cards│     │ → evidence cards│
     └────────┬────────┘     └────────┬────────┘     └────────┬────────┘
              │                        │                        │
     ┌────────▼────────┐     ┌────────▼────────┐     ┌────────▼────────┘
     │ Lane Summary    │     │ Lane Summary    │     │ Lane Summary    │
     └────────┬────────┘     └────────┬────────┘     └────────┬────────┘
              │                        │                        │
              └────────────────────────┼────────────────────────┘
                                       │
                        ┌──────────────▼──────────────────────┐
                        │     Hierarchical Merge              │
                        │  (intermediate nodes for large runs)│
                        └──────────────┬──────────────────────┘
                                       │
                        ┌──────────────▼──────────────────────┐
                        │       Final Synthesis               │
                        │   cited markdown answer             │
                        └─────────────────────────────────────┘
```

### Key mechanisms

**Parallel lanes.** The planner breaks a query into 4-8 independent research lanes based on the detected mode -- owner feedback, expert analysis, competing alternatives, local consensus, complaints, etc. Each lane runs its own search-crawl-refine cycle.

**Refinement loops.** After the initial seed queries in each lane, early evidence generates better follow-up queries before the full crawl is finished. This catches early misses and improves relevance without exhausting the search budget.

**Dense evidence cards.** Each crawled source is compressed into structured highlights, quantitative signals, caveats, and URLs. Sources are ranked using BM25 + semantic matching on mode-specific signal keywords. This keeps synthesis prompts small and grounded.

**Hierarchical synthesis.** Lane summaries are produced first, then merged through intermediate nodes for large runs, then a final synthesis step generates the user-facing answer. This avoids a single giant prompt and keeps long evidence sets usable.

**Follow-up memory.** After a run, `researchbuddy followup "..."` answers new questions from the current run's stored evidence instead of re-crawling. Evidence cards and synthesis are persisted for later Q&A.

**Local ingestion.** Playwright crawling with headful fallback, Reddit API integration, YouTube captions with Whisper fallback, podcast RSS parsing and local Whisper transcription, and PDF extraction -- all running locally.

---

## CLI Reference

```bash
researchbuddy start "<prompt>" [--mode auto|product|restaurant|research] [--json]
```
Start a research run in the background, make it current, and write all artifacts to `~/.researchbuddy/storage/<run_id>/` by default.

```bash
researchbuddy status [--run-id RUN_ID] [--json]
```
Print progress, process health, URL counts, and output paths for the current run.

```bash
researchbuddy watch [--run-id RUN_ID] [--interval SECONDS] [--timeout SECONDS]
```
Stream the current run's worker log until completion or failure.

```bash
researchbuddy followup "<question>" [--run-id RUN_ID]
```
Answer a follow-up from stored evidence without re-crawling.

```bash
researchbuddy doctor [--fix]               # validate readiness
researchbuddy list [--limit 20] [--json]   # list saved prompts
```

Full reference: [`docs/cli-reference.md`](docs/cli-reference.md) and [`docs/agent-cli-reference.md`](docs/agent-cli-reference.md).

---

## Configuration

ResearchBuddy requires **one search provider API key**. If `SEARCH_PROVIDER` is unset, it auto-selects from `EXA_API_KEY`, `TAVILY_API_KEY`, then `FIRECRAWL_API_KEY`.

It also auto-loads search settings from local agent installs:
- **Hermes:** `~/.hermes/.env`
- **OpenClaw env refs:** `~/.openclaw/.env`
- **OpenClaw:** `~/.openclaw/openclaw.json`

Hermes/OpenClaw search config is treated as the shared credential source and is not copied into ResearchBuddy `.env`. Copy `.env.example` to `.env` only for manual fallback configuration. Common settings:

| Variable | Default | Purpose |
|----------|---------|---------|
| `RESEARCHBUDDY_HOME` | `~/.researchbuddy` | Per-user state directory for the SQLite DB and run artifacts |
| `SEARCH_PROVIDER` | auto | `exa`, `tavily`, or `firecrawl` |
| `EXA_API_KEY` | -- | Exa search key |
| `TAVILY_API_KEY` | -- | Tavily search key |
| `FIRECRAWL_API_KEY` | -- | Firecrawl search key |
| `REDDIT_CLIENT_ID` | -- | Reddit API (optional, improves forum sourcing) |
| `PLANNER_MODEL` | `gpt-5.4` | LLM for lane planning |
| `SYNTHESIZER_MODEL` | `gpt-5.4` | LLM for synthesis |
| `MAX_URLS` | `100` | URL budget per run |
| `YOUTUBE_MAX_VIDEOS` | `6` | YouTube transcript cap |
| `PODCAST_MAX_EPISODES` | `4` | Podcast episode cap |
| `WHISPER_MODEL` | `base` | Local Whisper model size |
| `SEMANTIC_EMBEDDING_LOCAL_FILES_ONLY` | `true` | Avoid runtime Hugging Face downloads for semantic dedupe; falls back gracefully when the model is not cached |

---

## Output

Each run produces a research bundle at `~/.researchbuddy/storage/<run_id>/` by default:

```
~/.researchbuddy/storage/<run_id>/
  synthesis.md              # final cited report
  run.log                   # detailed execution log
  followup_memory.json      # persisted evidence for Q&A
  lanes/                    # per-lane crawl snapshots
  markdown/                 # converted source material
  html/                     # raw captured pages
  youtube_transcripts.json  # transcript metadata
  podcast_transcripts.json  # podcast transcript metadata
```

---

## Research Foundations

ResearchBuddy does not implement these papers directly, but the design aligns with several strong ideas from the literature:

> **Task decomposition** -- breaking a complex question into smaller, solvable sub-problems that can be addressed independently.
>
> Zhou et al., [*Least-to-Most Prompting Enables Complex Reasoning in Large Language Models*](https://arxiv.org/abs/2205.10625), 2022

> **Relevance + diversity** -- using Maximal Marginal Relevance to balance information gain against redundancy when selecting and ranking documents.
>
> Carbonell & Goldstein, [*The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries*](https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf), 1998

> **Hierarchical merging for long inputs** -- compressing book-length content through staged summarization rather than a single pass.
>
> Chang et al., [*BooookScore: A Systematic Exploration of Book-Length Summarization in the Era of LLMs*](https://arxiv.org/abs/2310.00785), 2023

> **Hierarchical LLM-agent summarization** -- using multiple specialized agents in a tree structure to summarize long-form narratives.
>
> [*NexusSum: Hierarchical LLM Agents for Long-Form Narrative Summarization*](https://arxiv.org/abs/2505.24575), 2025

> **Evidence-grounded generation** -- augmenting generation with retrieved documents to reduce hallucination in knowledge-intensive tasks.
>
> Lewis et al., [*Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*](https://arxiv.org/abs/2005.11401), 2020

The core thesis: split the job, collect diverse evidence, compress aggressively, then synthesize with citations. Single-shot prompts are faster. This architecture is harder to fool.

---

## Agent Install

ResearchBuddy installs as a skill in Hermes, OpenClaw, or any agent that supports `SKILL.md` directories.

```text
Install ResearchBuddy with Homebrew:

brew tap willemave/researchbuddy
brew install researchbuddy

Before asking for any new search-provider key, check ~/.openclaw/openclaw.json.
If exa, tavily, or firecrawl is already configured there, ask whether
ResearchBuddy should reuse that existing provider/key.

researchbuddy doctor
researchbuddy doctor --fix
researchbuddy list

Install the bundled OpenClaw skill:
researchbuddy skills install openclaw --scope shared

For a single OpenClaw workspace instead:
researchbuddy skills install openclaw --scope workspace --workspace /path/to/workspace

Do not start research runs until researchbuddy doctor passes.
Use researchbuddy start, researchbuddy status, researchbuddy watch, and
researchbuddy followup for normal agent operation.
```

Repository skill path: `skills/research`

---

## Development

```bash
uv sync                          # install dependencies
source .venv/bin/activate        # activate venv
cp .env.example .env             # add your API keys
pytest app/tests/ -v             # run tests
ruff check . && ruff format .    # lint and format
```

---

<p align="center">
  <sub>MIT License</sub>
</p>
