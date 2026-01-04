#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install uv first: https://docs.astral.sh/uv/" >&2
  exit 1
fi

uv sync
uv run playwright install

if [ -f ".env.example" ] && [ ! -f ".env" ]; then
  cp .env.example .env
  echo "Created .env from .env.example"
fi

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "OPENAI_API_KEY is not set. Add it to .env or export it before running." >&2
  exit 1
fi

if [ -z "${EXA_API_KEY:-}" ]; then
  echo "EXA_API_KEY is not set. Add it to .env or export it before running." >&2
  exit 1
fi

if [ "$#" -eq 0 ]; then
  echo "Usage: scripts/startup.sh \"your research query\""
  echo "Example: scripts/startup.sh \"best dishwasher for quiet apartment\""
  exit 0
fi

has_max_urls=false
has_max_agents=false
has_headful_flag=false

for arg in "$@"; do
  case "$arg" in
    --max-urls)
      has_max_urls=true
      ;;
    --max-agents)
      has_max_agents=true
      ;;
    --headful|--no-headful)
      has_headful_flag=true
      ;;
  esac
done

args=("$@")
if [ "$has_max_urls" = false ]; then
  args+=(--max-urls "${REVIEWBUDDY_MAX_URLS:-100}")
fi
if [ "$has_max_agents" = false ]; then
  args+=(--max-agents "${REVIEWBUDDY_MAX_AGENTS:-3}")
fi
if [ "$has_headful_flag" = false ]; then
  if [ "${REVIEWBUDDY_HEADFUL:-true}" = "true" ]; then
    args+=(--headful)
  else
    args+=(--no-headful)
  fi
fi

uv run reviewbuddy "${args[@]}"
