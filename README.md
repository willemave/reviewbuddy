# ReviewBuddy

ReviewBuddy turns a messy product question into a short, cited recommendation.

Instead of one giant prompt, it plans parallel research lanes, crawls the web locally, distills each source into dense evidence cards, and merges lane summaries into one final answer. It is built for agents first, but it is also a solid local CLI.

## Agent install

If you are installing ReviewBuddy into Hermes, OpenClaw, or another agent that supports `SKILL.md` directories, install the CLI first and then install the bundled skill from `skills/reviewbuddy-cli`.

Paste this into Hermes or another OpenClaw-style agent:

```text
Install ReviewBuddy with Homebrew:

brew tap willemave/reviewbuddy
brew install reviewbuddy
reviewbuddy commands --agent
reviewbuddy doctor

Install the bundled skill from:
$(brew --prefix)/opt/reviewbuddy/share/reviewbuddy/skills/reviewbuddy-cli

Read:
- $(brew --prefix)/opt/reviewbuddy/share/reviewbuddy/skills/reviewbuddy-cli/SKILL.md

Do not start research runs until `reviewbuddy doctor` passes.
```

Repository skill path:
```text
skills/reviewbuddy-cli
```

Codex skill installer example:
```text
$skill-installer https://github.com/<owner>/<repo>/tree/main/skills/reviewbuddy-cli
```

## Why it works

- `Parallel lanes`: ReviewBuddy breaks a query into 4-8 independent lanes like owner feedback, reliability, value, alternatives, and complaints.
- `Refinement loops`: early evidence inside a lane generates better follow-up queries before the full crawl is finished.
- `Dense evidence cards`: each source is compressed into highlights, quantitative signals, caveats, and URLs before synthesis.
- `Hierarchical summaries`: each lane is summarized first, and larger runs are merged again before the final answer so long evidence sets stay usable.
- `Follow-up memory`: `reviewbuddy ask <run_id> "..."` answers new questions from stored evidence instead of re-crawling.
- `Local ingestion`: Playwright crawling, Reddit handling, YouTube captions with Whisper fallback, PDF summaries, and headful retry when sites block headless browsers.

Single-shot prompts are faster. This architecture is usually harder to fool.

## Research precedent

ReviewBuddy does not implement these papers directly, but the design lines up with a few strong ideas:

- `Task decomposition`: [Least-to-Most Prompting Enables Complex Reasoning in Large Language Models](https://arxiv.org/abs/2205.10625)
- `Relevance + diversity`: [The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries](https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf)
- `Hierarchical merging for long inputs`: [BooookScore: A systematic exploration of book-length summarization in the era of LLMs](https://arxiv.org/html/2310.00785v4)
- `Hierarchical LLM-agent summarization`: [NexusSum: Hierarchical LLM Agents for Long-Form Narrative Summarization](https://arxiv.org/html/2505.24575v1)
- `Evidence-grounded generation`: [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)

That is the basic shape: split the job, collect diverse evidence, compress aggressively, then synthesize with citations.

## Quickstart

Requirements:
- Python 3.13
- `uv`
- Playwright browsers

Setup:

```bash
scripts/reviewbuddy setup
```

What `setup` does:
- Detects search-provider settings from `~/.hermes/.env` or `~/.openclaw/openclaw.json` when available
- Creates storage and database paths
- Installs Playwright browsers by default
- Reruns `doctor` checks

Manual equivalent:

```bash
uv sync
uv run playwright install
```

If auto-detection is unavailable, copy `.env.example` to `.env` and add your provider keys.

Run a research pass:

```bash
scripts/reviewbuddy run "best dishwasher for quiet apartment"
```

Ask a saved run a follow-up:

```bash
scripts/reviewbuddy ask <run_id> "What were the main reliability complaints?"
```

The `run` command prints the `run_id`.

## Common commands

```bash
scripts/reviewbuddy commands
scripts/reviewbuddy commands --agent
scripts/reviewbuddy setup
scripts/reviewbuddy doctor
```

## Environment

Required:
- One search provider API key. If `SEARCH_PROVIDER` is unset, ReviewBuddy auto-selects from `EXA_API_KEY`, `TAVILY_API_KEY`, then `FIRECRAWL_API_KEY`.

Common optional settings:
- `SEARCH_PROVIDER` (`exa`, `tavily`, or `firecrawl`)
- `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`

ReviewBuddy also auto-loads search settings from local agent installs:
- Hermes: `~/.hermes/.env`
- OpenClaw: `~/.openclaw/openclaw.json`

## Output

Each run writes a research bundle under:

```text
data/storage/<run_id>/
```

Useful files:
- `synthesis.md`: final report
- `run.log`: detailed logs
- `lanes/`: per-lane crawl snapshots
- `markdown/` and `html/`: captured source material

## Notes

- Run `codex login` locally before using LLM-backed workflows.
- `codex` must be installed and authenticated for agent execution paths.
- `reviewbuddy doctor` is the hard stop before production or automation use.

## Homebrew tap export

Generate a sibling tap repository:

```bash
reviewbuddy tap export
```

This writes a `homebrew-reviewbuddy` repository next to the source repo with the formula, README, validation workflow, and tap-maintainer skill.

## Docs

- `docs/cli-reference.md`
- `docs/agent-cli-reference.md`
- `docs/deploy.md`
- `docs/homebrew.md`
